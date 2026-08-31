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

### 6.3 Where the collapse happens

A small resolver, mirroring `from_experimental_Q`, turns callables into scalars/tuples by
averaging over frames. It lives next to the other collapse logic and is called from `update`
*after* the segments exist (so a callable can use already-built segment geometry via `model`):

```python
# bionc/model_creation/joint_template.py  (new, tiny)
JOINT_PARAMETER_KEYS = ("length", "theta", "sphere_radius", "semi_axis_lengths")

def resolve_joint_parameters(joint: dict, data: Data, model: "BiomechanicalModel") -> dict:
    resolved = dict(joint)
    for key in JOINT_PARAMETER_KEYS:
        value = joint.get(key)
        if callable(value):
            per_frame = np.asarray(value(data, model), dtype=float)   # (n_frames,) or (k, n_frames)
            resolved[key] = np.nanmean(per_frame, axis=-1)            # mean over data points
    return resolved
```

### 6.4 Change to `update`

Only the joint loop changes; segment scaling is untouched:

```python
# bionc/model_creation/biomechanical_model_template.py  (update)
for key, joint in self.joints.items():
    joint = resolve_joint_parameters(joint, data, model)   # <-- new line
    model._add_joint(joint)
```

`_add_joint` itself needs **no change**: it already does `joint_type.value(**joint)`, and by
the time it runs, every parameter is a concrete scalar/tuple. This keeps the data-handling in
the `model_creation` layer and leaves the numpy joint classes purely numeric — consistent with
how segments are collapsed before `NaturalSegment` ever sees them.

### 6.5 Incremental steps (smallest reviewable PRs)

1. **Expose the missing constants** — add `sphere_radius` and `semi_axis_lengths` to
   `add_joint` and thread them through the dict. Pure plumbing, no behavior change; unblocks
   defining ellipsoid/sphere joints through the template at all (see the gap noted in §3).
2. **Add `resolve_joint_parameters` + the one line in `update`** — enables callables while
   scalars keep flowing through untouched (`callable(value)` is `False` for them).
3. **Validate resolved values** — reuse the joints' existing checks (e.g. `EllipsoidOnPlane`
   already raises on non-positive semi-axes); optionally validate shape `(k, n_frames)` in the
   resolver to give a clear error instead of a confusing broadcast later.
4. **Docs + one example** — extend
   [`right_side_lower_limb.py`](../examples/model_creation/right_side_lower_limb.py) or a new
   upper-limb example with a scaled ellipsoid, analogous to the `harrington2007` HJC callable.

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
