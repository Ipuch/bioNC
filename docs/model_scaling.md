# Scaling / personalizing a model from a C3D static trial

This document explains how `bionc` turns a *generic* model template into a *personalized*
(scaled) model using experimental marker data (typically a static reference trial such as
[`examples/model_creation/statref.c3d`](../examples/model_creation/statref.c3d)), how the
**mean over data points (frames)** is used, and — importantly — **what is scaled and what is
not**.

TL;DR: today, scaling personalizes **segment geometry (the rigid-body parameters)** and
**marker positions** from the data. **Joint geometric properties (joint length, `theta`,
sphere radius, ellipsoid semi-axes, …) are NOT scaled from the data** — they are the constants
you pass at model-definition time. So your belief is correct.

Two further things `update(data)` does not give you, both easy to assume it does:
**inertial parameters are silently dropped** (§4.1 — a C3D-scaled model has `mass = None`), and
**muscles are not part of the pipeline at all** (§8). If you are starting from an OpenSim `.osim`
model rather than from scratch, go to **§7** — the template route is not the one you want on its
own.

---

## 1. The entry point: `BiomechanicalModelTemplate.update(data)`

You describe the model *generically* (axes/points defined as functions of marker names), then
collapse it onto real data:

```python
model_template = BiomechanicalModelTemplate()
# ... add SegmentTemplate / MarkerTemplate / joints ...
model = model_template.update(C3dData("statref.c3d"))   # <-- the scaling step
```

`update(data)` lives in
[`bionc/model_creation/biomechanical_model_template.py`](../bionc/model_creation/biomechanical_model_template.py#L99).
For each segment it does three things, then adds the joints:

```python
for name in self.segments:
    s = self.segments[name]
    Q_xp = s.natural_segment.experimental_Q(data, model)     # (1) per-frame coordinates
    natural_segment = s.natural_segment.update()             # (2) collapse -> scaled segment
    model[s.name] = natural_segment
    for marker in s.markers:                                 # (3) scaled marker positions
        model.segments[name].add_natural_marker(marker.to_natural_marker(data, model, Q_xp))

for key, joint in self.joints.items():
    model._add_joint(joint)                                  # (4) joints added as-is (no data)
```

The critical detail is that steps (1)–(3) *see the data* and average over it, while step (4)
does **not** receive `data` at all.

---

## 2. How segment scaling uses the mean over frames

### 2.1 Per-frame natural coordinates

[`NaturalSegmentTemplate.experimental_Q`](../bionc/model_creation/natural_segment_template.py#L41)
evaluates the segment's `u` axis, proximal point `rp`, distal point `rd`, and `w` axis from the
marker trajectories, for **every frame** in the trial:

```python
self.Q = SegmentNaturalCoordinates.from_components(
    u=self.u_axis.to_axis(data, kinematic_chain).axis()[:3, :],   # (3 x n_frames)
    rp=self.proximal_point.to_marker(...).position[:3, :],         # (3 x n_frames)
    rd=self.distal_point.to_marker(...).position[:3, :],           # (3 x n_frames)
    w=self.w_axis.to_axis(...).axis()[:3, :],                      # (3 x n_frames)
)
```

So `Q_xp` is a `12 x n_frames` array — one segment pose per data point.

### 2.2 Collapsing to four scalar shape parameters, averaged over frames

A `NaturalSegment`'s *shape* is fully described by four scalars:

| parameter | meaning |
|-----------|---------|
| `length`  | length of the segment (‖v‖, from `rp` to `rd`)               |
| `alpha`   | angle between `v` (rp→rd) and `w`                            |
| `beta`    | angle between `u` and `w`                                    |
| `gamma`   | angle between `u` and `v`                                    |

[`NaturalSegment.from_experimental_Q`](../bionc/bionc_numpy/natural_segment.py#L267) computes
these **for each frame** and then takes the **mean across frames**:

```python
for i, Qif in enumerate(Qi.vector.T):
    alpha[i], beta[i], gamma[i], length[i] = cls.parameters_from_Q(Qif)   # per frame

return cls(
    alpha=np.mean(alpha, axis=0),      # <-- mean over data points
    beta=np.mean(beta, axis=0),
    gamma=np.mean(gamma, axis=0),
    length=np.mean(length, axis=0),
)
```

**This is the "scaling".** The averaged `(alpha, beta, gamma, length)` are exactly the
quantities that define the segment's **rigid-body constraints** (`Phi_r`): the rigid-body
constraint is what enforces that these four shape parameters stay constant during motion.
Personalizing them from the static trial *is* personalizing the rigid-body constraints.

> Why the mean? A static trial has small residual movement/soft-tissue noise, so each frame
> gives a slightly different segment shape. Averaging over frames gives a single, robust
> subject-specific geometry.

### 2.3 Markers are also personalized (also by averaging)

[`NaturalMarker.from_data`](../bionc/bionc_numpy/natural_marker.py#L74) expresses each
experimental marker in the segment's natural (local) coordinate system and averages over
frames — `natural_positions.mean(axis=1)` and, in the interpolation helpers, `np.nanmean`
(so occluded/NaN samples are ignored). The marker's local position is therefore data-driven
and scaled too.

---

## 3. What is NOT scaled: joints (including ellipsoids)

Joints are added by step (4),
[`_add_joint`](../bionc/protocols/biomechanical_model_joints.py#L75), which simply instantiates
the joint class from the template dictionary:

```python
self.joints[joint["name"]] = joint_type.value(**joint)
```

Note that `_add_joint` **never receives `data`**. Every geometric property of a joint is taken
verbatim from the constants you supplied when defining the joint:

- `Hinge` / `Universal` / `Spherical`: `theta`, `length`, axis choices — fixed constants.
- [`SphereOnPlane`](../bionc/bionc_numpy/joints.py#L612): `sphere_radius` is stored as-is
  (`self.sphere_radius = sphere_radius`).
- [`EllipsoidOnPlane`](../bionc/bionc_numpy/joints.py#L776): the semi-axis lengths are cast
  straight from the argument, never estimated from data:

  ```python
  self.semi_axis_lengths = tuple(float(length) for length in semi_axis_lengths)  # a, b, c
  ```

So for an ellipsoid (e.g. scapulothoracic) joint, the **size and shape of the ellipsoid
(`a, b, c`) are fixed inputs, not scaled**.

### A nuance: joint *anchors* do move with scaling

The joint's geometry references *markers/vectors on the segments* — e.g. the ellipsoid centre
and its three principal axes are `parent.marker_from_name(...)` / `parent.vector_from_name(...)`,
and the plane point/normal are markers/vectors on the child. Those anchor **markers are
personalized** (Section 2.3), so their *placement* follows the scaled segment. But the
**intrinsic dimensions** — the semi-axes `a, b, c`, a sphere `radius`, a joint `length`, an
angle `theta` — stay at whatever constant you passed. Only the frame they are attached to is
subject-specific; the numbers themselves are not.

---

## 4. Summary table

| Quantity                                   | Source                     | Scaled from C3D? | How                          |
|--------------------------------------------|----------------------------|------------------|------------------------------|
| Segment length                             | data                       | ✅ yes           | mean over frames             |
| Segment angles `alpha`, `beta`, `gamma`    | data                       | ✅ yes           | mean over frames             |
| → Rigid-body constraints `Phi_r`           | the above                  | ✅ yes           | defined by scaled parameters |
| Marker local positions                     | data                       | ✅ yes           | (nan)mean over frames        |
| Joint `length`, `theta`, axes              | your constants             | ❌ no            | passed as-is                 |
| Sphere `radius` (SphereOnPlane)            | your constants             | ❌ no            | passed as-is                 |
| Ellipsoid semi-axes `a, b, c`              | your constants             | ❌ no            | passed as-is                 |
| Joint anchor markers/axes (centre, plane)  | data (via natural markers) | ✅ placement only| markers are personalized     |
| Mass, centre of mass, inertia              | —                          | ❌ **dropped**   | see §4.1                     |
| Muscles (via points)                       | —                          | ❌ not handled   | see §8                       |

### 4.1 Inertia is not merely unscaled — it is discarded

`SegmentTemplate` accepts an `inertia_parameters` argument, but `update()` never uses it. The call
is commented out in
[`biomechanical_model_template.py`](../bionc/model_creation/biomechanical_model_template.py#L153):

```python
# inertia_parameters = None
# if s.inertia_parameters is not None:
# todo: this is not working yet
# natural_segment.set_inertia_parameters(s.inertia_parameters)
```

and [`from_experimental_Q`](../bionc/bionc_numpy/natural_segment.py#L267) returns only
`cls(alpha, beta, gamma, length)`. **A model built from a C3D therefore has `mass = None` and no
mass matrix, so dynamics are unavailable on it.** To get inertial parameters you must set them
imperatively after the fact, with
[`NaturalSegment.with_cartesian_inertial_parameters`](../bionc/bionc_numpy/natural_segment.py#L224)
or `set_inertial_parameters(...)`. This is a second, independent reason the OpenSim route of §7 is
attractive: a `.osim` carries real mass/CoM/inertia per body.

---

## 5. Practical guidance: personalizing joint geometry yourself

Because the framework does not estimate joint dimensions from the static trial, if you want a
subject-specific ellipsoid (or sphere radius, or joint length) you currently must compute it
yourself and pass it in. Typical options:

1. **Anthropometric scaling** — scale `a, b, c` by a subject measurement (e.g. thorax
   width/depth from the same static markers) relative to the generic model's reference size,
   and pass the scaled tuple to the joint.
2. **Geometric fit** — fit the ellipsoid/sphere to the relevant marker cloud from the static
   (or a range-of-motion) trial, then pass the fitted parameters.
3. **Literature/regression values** — use a regression (analogous to the `harrington2007` hip
   joint centre in
   [`examples/model_creation/right_side_lower_limb.py`](../examples/model_creation/right_side_lower_limb.py))
   to derive the dimensions from subject data.

In all three cases the *scaling logic is yours*; `bionc` will then keep those values fixed and
enforce the corresponding joint constraint. If you want data-driven joint dimensions to become
a first-class feature, that would mean extending `_add_joint`/`update` so joints also receive
`data` and can collapse their own parameters the way segments do in `from_experimental_Q`.
Section 6 sketches how that could be done.

---

## 6. Implementation plan & API draft: scaling joint parameters in `update`

### 6.1 Design principle

Reuse the pattern the framework already uses everywhere else for scaling: a **parameter is
either a constant or a callable of the data**. Segment axes and markers are already defined as
`Callable | str` (functions of marker names) and collapsed against `data` inside `update`. We
extend the *same* idea to joint parameters (`length`, `theta`, `sphere_radius`,
`semi_axis_lengths`, …).

The two invariants to preserve:

1. **Backward compatibility** — passing a plain scalar/tuple must keep working unchanged.
2. **Same collapse semantics as segments** — a callable returns a per-frame array
   `(n_frames,)` or `(k, n_frames)`, and `update` reduces it over frames with `np.nanmean`,
   exactly like [`from_experimental_Q`](../bionc/bionc_numpy/natural_segment.py#L267).

### 6.2 Public API draft

`BiomechanicalModelTemplate.add_joint` gains the currently-missing geometric parameters, and
every geometric parameter accepts a callable in addition to a constant:

```python
# type alias (bionc/model_creation/protocols.py)
JointParameter = float | tuple[float, ...] | np.ndarray | Callable[[Data, "BiomechanicalModel"], np.ndarray]

def add_joint(
    self,
    name: str,
    joint_type: "JointType",
    parent: str,
    child: str,
    *,
    # --- existing ---
    parent_axis=None, child_axis=None, parent_point=None, child_point=None,
    projection_basis=None, parent_basis=None, child_basis=None,
    # --- now accept constant OR callable(data, model) -> per-frame values ---
    length: JointParameter = None,
    theta: JointParameter = None,
    sphere_radius: JointParameter = None,          # newly exposed
    semi_axis_lengths: JointParameter = None,      # newly exposed (a, b, c)
): ...
```

Example — a subject-specific scapulothoracic ellipsoid scaled from thorax markers:

```python
def thorax_semi_axes(data, model):
    # returns (3, n_frames): a, b, c per frame; update() averages over frames
    width = np.linalg.norm(data["IJ"][:3] - data["T1"][:3], axis=0)   # per-frame widths
    return np.vstack([0.5 * width, 0.6 * width, 0.4 * width])

model_template.add_joint(
    name="scapulothoracic",
    joint_type=JointType.ELLIPSOID_ON_PLANE,
    parent="thorax", child="scapula",
    semi_axis_lengths=thorax_semi_axes,     # callable  -> scaled
    # or semi_axis_lengths=(0.09, 0.11, 0.07)  # constant -> unchanged, as today
    ...
)
```

### 6.3 Where the collapse happens — this already exists

⚠️ **This section originally sketched a `resolve_joint_parameters` function in a new
`joint_template.py`. That is not what was built.** Commit `12f310c` added the resolver as a static
method on the template instead —
[`BiomechanicalModelTemplate._resolve_joint_callables`](../bionc/model_creation/biomechanical_model_template.py#L168):

```python
@staticmethod
def _resolve_joint_callables(joint: dict, data: Data, model: "BiomechanicalModel") -> dict:
    return {key: np.mean(value(data.values, model)) if callable(value) else value
            for key, value in joint.items()}
```

It applies to *every* key rather than a whitelist (harmless: a `str` marker name is not callable),
and is called from `update` at
[`biomechanical_model_template.py:164`](../bionc/model_creation/biomechanical_model_template.py#L164)
— after the segments exist, so a callable can use already-built geometry via `model`.

Three divergences from the design above remain open, and the first is a latent bug:

1. **`np.mean` has no `axis`.** It collapses a `(k, n_frames)` return to a *single scalar*. A
   callable `semi_axis_lengths` — the motivating example of §6.2 — would reach `EllipsoidOnPlane`
   as a 0-d array and fail on `tuple(float(length) for length in semi_axis_lengths)`. The correct
   form is `np.nanmean(per_frame, axis=-1)`. Only scalar parameters (`theta`, `length`) work today.
2. **`np.mean`, not `np.nanmean`** — this contradicts the NaN-robustness goal stated in §6.6 and
   the `np.nanmean` already used for markers (§2.3). One occluded frame poisons the whole average.
3. **Calling convention.** The resolver passes `value(data.values, model)` — the `(m, bio)` marker
   convention, where `m` is the dict of marker arrays. §6.2 above writes `value(data, model)`. The
   `thorax_semi_axes` example is correct as written (it indexes `data["IJ"]`), but its parameter is
   the marker *dict*, not a `Data` object. Read the signature as `(m, bio)`.

### 6.4 Change to `update` — already applied

The joint loop already reads as designed; segment scaling is untouched:

```python
# bionc/model_creation/biomechanical_model_template.py:163-164
for key, joint in self.joints.items():
    model._add_joint(self._resolve_joint_callables(joint, data, model))
```

`_add_joint` needed **no change**: it does `joint_type.value(**joint)` after popping `joint_type`
and filtering out `None` values
([`biomechanical_model_joints.py:100-113`](../bionc/protocols/biomechanical_model_joints.py#L100)),
so by the time it runs every parameter is a concrete value. Note the consequence of that `**joint`
splat: **dict keys must match constructor kwarg names exactly**, and any unexpected non-`None` key
raises `TypeError`. Data-handling stays in the `model_creation` layer and the numpy joint classes
stay purely numeric — consistent with how segments are collapsed before `NaturalSegment` sees them.

### 6.5 Incremental steps — current status

Steps 1 and 2 have partly landed since this section was written. Status as of `158027f`:

1. **Expose the missing constants — 🟡 half done** (`d55fac2 feat(ellipsoid): extra args`).
   The *ellipsoid* half is complete: `semi_axis_lengths`, `ellipsoid_center`,
   `ellipsoid_axis_a/b/c`, `plane_point` and `plane_normal` are now in the `add_joint` signature
   ([`:38`](../bionc/model_creation/biomechanical_model_template.py#L38)) and threaded through the
   dict ([`:117`](../bionc/model_creation/biomechanical_model_template.py#L117)). These names match
   the `EllipsoidOnPlane` constructor 1:1, so `ELLIPSOID_ON_PLANE` works end-to-end through the
   template.
   **Remaining:** `sphere_radius` and `sphere_center` were *not* added. `SphereOnPlane` requires
   both and raises `ValueError` if either is `None`
   ([`joints.py:643`](../bionc/bionc_numpy/joints.py#L643)), so `JointType.SPHERE_ON_PLANE` still
   cannot be defined through `add_joint` at all. Mirroring lines 38 and 117 is the whole fix.
2. **Resolver + the one line in `update` — 🟡 done, with three caveats** (`12f310c`). See §6.3:
   the `np.mean` call is missing its `axis` argument, which silently breaks exactly the
   vector-valued case (`semi_axis_lengths`) that step 1 unblocked; it should be `nanmean`; and the
   callable signature is `(m, bio)`, not `(data, model)`.
3. **Validate resolved values — ❌ open.** Reuse the joints' existing checks (e.g.
   `EllipsoidOnPlane` already raises on non-positive semi-axes); validating the `(k, n_frames)`
   shape in the resolver would turn caveat 1 above into a clear error instead of a confusing one.
4. **Docs + one example — ❌ open.** Note that the step-1 plumbing is currently **unexercised**:
   no test or example calls `add_joint` with `semi_axis_lengths`. Every existing user of
   ellipsoid/sphere joints bypasses the template, either through raw
   `model._add_joint(dict(...))` (e.g.
   [`knee_feikes.py:193`](../examples/knee_parallel_mechanism/knee_feikes.py#L193)) or by
   constructing `Joint.*` directly
   ([`plane_on_ellipsoid.py:78`](../examples/play_with_joints/plane_on_ellipsoid.py#L78)). The
   resolver's own coverage
   ([`test_model_creation_joint_lambda_function.py`](../tests/test_model_creation_joint_lambda_function.py#L69))
   only tests scalar-returning `theta` and `length` — which is why caveat 1 went unnoticed.

### 6.6 Testing

- **Backward-compat:** a model built with constant joint params yields byte-identical joints
  before/after the change (parametrize over each joint type).
- **Scaling:** a callable returning a known per-frame array collapses to its `nanmean`
  (e.g. constant-across-frames input ⇒ that constant; a ramp ⇒ its mean), verified directly on
  `resolve_joint_parameters` and end-to-end through `update`.
- **NaN robustness:** frames with NaN markers are ignored (`np.nanmean`), matching marker
  scaling in §2.3.
- **Validation:** a callable producing a non-positive semi-axis raises the same `ValueError`
  as the constant path.

### 6.7 Open questions / trade-offs

- **Reduction choice.** `nanmean` matches segments, but some parameters may want `median`
  (robust to spikes) or the value at a chosen reference frame. Could be a per-joint
  `reduction=` option, defaulting to `nanmean` for consistency.
- **Callable signature.** `(data, model)` mirrors `MarkerTemplate`/`AxisTemplate` and lets a
  joint parameter depend on already-scaled segments. If a parameter only needs raw markers,
  `model` is simply ignored — no extra cost.
- **A dedicated `JointTemplate` class?** For now a resolver over the existing joint dict is the
  lightest change. If joint personalization grows (per-parameter reductions, validation,
  derived quantities), promoting joints to a `JointTemplate` with a `.collapse(data, model)`
  method — symmetric with `SegmentTemplate` — would be the natural next refactor.

---

## 7. Starting from an OpenSim `.osim` model

There is **no `.osim` reader in the `bionc` package** — the only one in the repository is
[`parse_osim`](../sandbox/scapulothoraric_seth/build_bionc_model.py#L120), an `xml.etree` parser in
`sandbox/`. It is unpackaged, untested, hard-coded to a 7-body shoulder, and reads only bodies,
joint offset frames, markers and the thoracic ellipsoid radii — it skips `<ForceSet>` entirely.
Treat it as a worked reference, not an API.

### 7.1 The enabling trick: make the segment frame *be* the OpenSim body frame

This is the load-bearing idea, documented at
[`build_bionc_model.py:14-22`](../sandbox/scapulothoraric_seth/build_bionc_model.py#L14). Build each
segment as a **unit orthonormal frame** — `alpha = beta = gamma = pi/2`, `length = 1` — so that
`u`, `v`, `w` coincide with the body's X, Y, Z:

```python
model[name] = NaturalSegment.with_cartesian_inertial_parameters(
    name=name, alpha=np.pi / 2, beta=np.pi / 2, gamma=np.pi / 2, length=1.0,
    mass=b["mass"], center_of_mass=b["com"], inertia=b["inertia"],
    inertial_transformation_matrix=TransformationMatrixType.Buv,
)
```

Every quantity the `.osim` stores in body coordinates then transfers *verbatim*: a point is exactly
the `location` argument of
[`add_natural_marker_from_segment_coordinates`](../bionc/bionc_numpy/natural_segment.py#L753), and a
direction is exactly the `direction` argument of
[`add_natural_vector_from_segment_coordinates`](../bionc/bionc_numpy/natural_segment.py#L685). No
change of basis to get wrong.

The cost is that the body frame is not aligned with the proximal→distal axis, so `Hinge`/`Universal`
(which constrain the segment's own `rp`/`rd` to the joint centre) are unusable. Anchor every joint
on explicit named points instead — which is also what lets one segment carry several joints and
close a loop.

### 7.2 Three routes

**Route A — scale in OpenSim, then import.** Run OpenSim's ScaleTool against your static trial and
parse the *scaled* `.osim`. This is the only route that personalizes muscle geometry (§8).
Diffing the two models bundled in `sandbox/scapulothoraric_seth/Model/` shows what the ScaleTool
actually touches:

| Quantity                | generic vs `…ModelSubject.osim` |
|-------------------------|---------------------------------|
| Muscle path points      | 70 / 73 differ                  |
| Markers                 | 20 / 21 differ                  |
| Body mass and inertia   | 7 / 7 differ                    |
| Body centres of mass    | 4 / 7 differ                    |
| Joint offset frames     | 5 / 10 differ                   |
| Wrap surfaces           | 10 / 10 differ                  |
| **Thoracic ellipsoid radii** | **identical**              |

That last row is worth dwelling on: the `ScapulothoracicJoint` ellipsoid is *not* scaled by
OpenSim either. The gap described in §3 is an upstream limitation, not a bioNC omission — if you
want a subject-specific ellipsoid you must compute it yourself in either framework (§5).

**Route B — the C3D template alone.** The instinctive approach: export a static trial and program a
`BiomechanicalModelTemplate` over it. This personalizes segment shape and measured markers well
(§2), but it cannot recover anything the `.osim` knows that a mocap file does not contain — joint
centres, the ellipsoid centre and axes, muscle attachments. Those are *virtual* points. The
supported way to synthesize one is a callable `MarkerTemplate` flagged non-technical, exactly as
`harrington2007` does for the hip joint centre in
[`right_side_lower_limb.py:24`](../examples/model_creation/right_side_lower_limb.py#L24):

```python
model["PELVIS"].add_marker(MarkerTemplate(
    name="RIGHT_HIP_JOINT", function=right_hip_joint, parent_name="PELVIS",
    is_technical=False, is_anatomical=True))
```

`is_technical=False` means the point rides rigidly on the segment but contributes no marker
residual to IK. Route B is the right answer only if you are willing to re-derive every virtual
point from a regression.

**Route C — hybrid (recommended).** Keep the `.osim` as the source of anatomy, and the C3D as the
source of size:

1. Parse the generic `.osim` for topology, inertia, and virtual geometry (`parse_osim`).
2. Derive per-segment scale factors from the static C3D — the ratio of subject inter-marker
   distances to the same distances computed on the generic model's markers, which is what the
   OpenSim ScaleTool does internally. `C3dData.mean_marker_positions(...)`
   ([`c3d_data.py:19`](../bionc/model_creation/c3d_data.py#L19)) gives the averaged subject
   positions directly.
3. Apply each segment's factor to *all* of that body's coordinates before construction — marker
   locations, joint offsets, CoM, ellipsoid radii, and muscle path points alike. Because §7.1 keeps
   the segment frame identical to the body frame, this is a componentwise multiply on the raw
   `.osim` vectors, with no basis change.
4. Build the model as in
   [`model_creation_scapulothoracic_from_trc.py`](../sandbox/scapulothoraric_seth/model_creation_scapulothoracic_from_trc.py),
   and scale mass/inertia separately (OpenSim's convention: preserve the mass distribution, scale
   the total to the subject's measured mass).

This gets you the `.osim`'s anatomical knowledge *and* subject size, and — unlike Route A — lets
you scale the quantities OpenSim refuses to, the thoracic ellipsoid among them.

**Caveats common to A and C.** The joint mapping is lossy: bionc has no non-ground weld and no
generic 1-dof `CustomJoint`, so `build_bionc_model.py` downgrades 1-dof joints and welds to
`SPHERICAL`, which its docstring
([`:27-42`](../sandbox/scapulothoraric_seth/build_bionc_model.py#L27)) records per joint. Verify
what you built: that script reports a marker reconstruction error of `3.4e-17 m` at the reference
pose, which is the check to reproduce after any import.

---

## 8. Muscles are outside the scaling pipeline

**Muscle geometry cannot be scaled by bionc.** Two independent reasons, and it is worth being clear
about which is a design decision and which is a limitation.

### 8.1 No template path — by design

`BiomechanicalModelTemplate` holds only `self.segments` and `self.joints`
([`:11-13`](../bionc/model_creation/biomechanical_model_template.py#L11)); `update()` never touches
muscles. They are attached imperatively to an already-built model:

```python
origin = MuscleViaPoint(name="origin", parent_name="GROUND", position=np.array([0.3, 0.0, 0.2]))
insertion = MuscleViaPoint(name="insertion", parent_name="pendulum", position=(0.0, -0.5, 0.0))
model.add_muscle(Muscle(name="muscle1", via_points=[origin, insertion]))
```

This is deliberate, not an oversight. The `*Template` classes exist to map experimental marker data
onto a model, and via-point positions are **not observable from mocap** — they are anatomical
knowledge. Scaling them therefore means *transforming known coordinates* (§7.2), never fitting them
to data. There should be no `MuscleTemplate`.

### 8.2 The muscle model is a polyline — a real limitation

[`Muscle`](../bionc/bionc_numpy/muscle.py#L100) is an ordered list of at least two
[`MuscleViaPoint`](../bionc/bionc_numpy/muscle.py#L12)s, exposing `length()` (the summed Euclidean
distance between consecutive points) and `moment_arm()` (`-dL/dQ`). There are **no wrapping
surfaces, no conditional or moving via points, and no force/activation model** anywhere in
`bionc/` — no tendon slack length, no optimal fibre length, no maximum isometric force.

The `PathPoint → MuscleViaPoint` mapping itself is clean:
[`MuscleViaPoint.from_cartesian`](../bionc/bionc_numpy/muscle.py#L49) takes exactly the
segment-Cartesian coordinate an `.osim` stores. What does not survive is everything else. In the
bundled `ThoracoscapularShoulderModel.osim`:

| Element                          | count | maps to bionc?          |
|----------------------------------|-------|-------------------------|
| `Millard2012EquilibriumMuscle`   | 33    | geometry only, no forces |
| `PathPoint`                      | 73    | ✅ `MuscleViaPoint`      |
| `MovingPathPoint`                | 5     | ❌                       |
| `PathWrap`                       | 37    | ❌                       |
| `WrapEllipsoid` / `Cylinder` / `Sphere` | 4 / 4 / 2 | ❌            |

**30 of the 33 muscles use at least one wrap.** So an importer that reads `<ForceSet>` would
produce faithful paths for only 3 of them; the other 30 would cut straight through the bone they
are supposed to wrap around, giving wrong lengths and wrong moment arms.

### 8.3 Practical recommendation

Use Route A or C of §7 for muscles specifically: let OpenSim place the scaled path points, import
them as via points, and **treat the resulting lengths and moment arms as approximate** for any
muscle that wraps. If you need correct muscle-tendon lengths across a range of motion, compute them
in OpenSim and bring the *results* into bionc rather than trying to reproduce the paths.
