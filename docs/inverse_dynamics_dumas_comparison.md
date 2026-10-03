# Inverse dynamics: comparison with Dumas' toolbox

bioNC's recursive inverse dynamics (`BiomechanicalModel.inverse_dynamics`) is checked against the
"generalized coordinates" method of R. Dumas' *3D Kinematics and Inverse Dynamics* MATLAB toolbox
(v2.4.0, `Inverse_Dynamics_GC.m`), on the toolbox's own lower limb gait example.

**Result:** ankle, knee and hip forces and moments match to `1e-8` (absolute, N and N·m) on all 131
frames. With the library code of `main` (before the fixes of `fix/inverse-dynamics`), every compared
value fails, by up to ~1164.

The test is [`tests/test_inverse_dynamics_dumas.py`](../tests/test_inverse_dynamics_dumas.py):

```
conda run -n bionc python -m pytest tests/test_inverse_dynamics_dumas.py -rxX
```

---

## 1. What is compared

| | |
|---|---|
| Data | `tests/data/dumas_lower_limb_gait/` — `Segment.mat`, `Joint.mat` (BSD, © R. Dumas) |
| Segments | toolbox `Segment(2..4)` = foot, shank, thigh; `Segment(1).Q(4:6)` = centre of pressure |
| Input wrench | `Joint(1).F`, `Joint(1).M`: force and moment of the foot **on the ground**, at the CoP |
| Accelerations | `Q̈ = filt(d/dt(filt(d/dt Q)))`, 4th order Butterworth, `f = 100 Hz`, `fc = 6 Hz` |
| Outputs | `Joint(i).F`, `Joint(i).M`, i = 2, 3, 4: wrench applied on segment i by segment i+1, at `rP_i` |

The reference is a line by line Python port of `Inverse_Dynamics_GC.m`
(`dumas_inverse_dynamics_gc`, [test file l.77](../tests/test_inverse_dynamics_dumas.py#L77)), because
MATLAB is not required to run the test suite. Section 5 explains how to freeze genuine MATLAB outputs.

## 2. The equations

For each segment `i`, natural coordinates `Qᵢ = (u, rP, rD, w)`, `v = rP − rD`. From the foot up to the
thigh, Dumas solves one 12×12 linear system per frame:

```
[ N_Pᵀ   N*ᵀ   −Kᵀ ] [ Fᵢ ]
                     [ Mᵢ ]  =  Gᵢ Q̈ᵢ  −  N_Cᵀ mᵢ g  −  N_Pᵀ (−F_{i−1})  −  N*ᵀ ( −M_{i−1} + (rP_{i−1} − rPᵢ) × (−F_{i−1}) )
                     [ λᵢ ]
```

- `N_Pᵀ = [0; E; 0; 0]`: interpolation of a force at `rP`.
- `N_Cᵀ = [n_C1 E; (1 + n_C2) E; −n_C2 E; n_C3 E]`: interpolation at the centre of mass, `n_C = inv(B) r_C`.
- `N*ᵀ = [0, v, 0; 0, 0, −w; 0, 0, w; u, 0, 0] · inv(B*)`, with `B* = [w × u, u × v, −v × w]`: pseudo-interpolation
  of a moment, as the forces equivalent to it.
- `Kᵀ`: transpose of the Jacobian of the six rigid body constraints.
- `Gᵢ`: generalized mass matrix, built from `m`, `m n_C` and the pseudo-inertia `J` (section 4).
- The last two terms are the **reaction** of the distal neighbour: it receives `F_{i−1}, M_{i−1}` from segment `i`,
  so it applies `−F_{i−1}, −M_{i−1}` on it, at `rP_{i−1}`, transported to `rPᵢ` with
  `M_B = M_A + (A − B) × F`. For the foot, the distal "neighbour" is the ground, at the CoP.

bioNC solves exactly the same system, distal to proximal by recursion on the kinematic tree:

| Term | Dumas, `Inverse_Dynamics_GC.m` | Python port (test file) | bioNC |
|---|---|---|---|
| `Gᵢ` | l.123-138 | l.102-114 | `NaturalInertialParameters._update_mass_matrix`, [natural_inertial_parameters.py:155](../bionc/bionc_numpy/natural_inertial_parameters.py#L155) |
| `N_Pᵀ`, `N_Cᵀ` | l.147, l.150 | l.115-116 | `NaturalVector.proximal().interpolate()`, `NaturalSegment.gravity_force`, [natural_segment.py:592](../bionc/bionc_numpy/natural_segment.py#L592) |
| `Kᵀ` | l.141-144 | l.122-129 | `NaturalSegment.rigid_body_constraint_jacobian` |
| `B*`, `N*ᵀ` | l.154-159 | l.130-141 | `SegmentNaturalCoordinates.compute_pseudo_interpolation_matrix`, [natural_coordinates.py:132](../bionc/bionc_numpy/natural_coordinates.py#L132) |
| linear solve | l.186-191 | l.144-150 | `NaturalSegment.inverse_dynamics`, [natural_segment.py:837](../bionc/bionc_numpy/natural_segment.py#L837) |
| reaction of the child, `−F, −M` | l.190-191 | l.147-148 | [biomechanical_model.py:561-566](../bionc/bionc_numpy/biomechanical_model.py#L561-L566) |
| moment transport `(A − B) × F` | `cross(rP_{i−1} − rPᵢ, ·)` | l.148 | `ExternalForceInGlobalOnProximal.transport_to_another_segment`, [external_force_global_on_proximal.py:143](../bionc/bionc_numpy/external_force_global_on_proximal.py#L143) |
| ground reaction at the CoP | `Joint(1)`, `Segment(1).Q(4:6)` | l.85-86, l.143 | `ExternalForceSet.add_in_global`, [external_force_global.py:89](../bionc/bionc_numpy/external_force_global.py#L89) |

## 3. Making the two setups identical

[`test_inverse_dynamics_reproduces_dumas_gait_example`](../tests/test_inverse_dynamics_dumas.py#L196):

- **Gravity.** The toolbox uses `g = (0, −9.81, 0)`, bioNC hard-codes `(0, 0, −9.81)`
  ([natural_segment.py:602](../bionc/bionc_numpy/natural_segment.py#L602)). All positions, vectors, accelerations
  and wrenches are rotated by `R = Rx(+90°)` (`R_TOOLBOX_TO_BIONC`, l.41) before bioNC, and the outputs by `Rᵀ`.
- **Segment geometry.** `Q` is not rigid in the dataset (lengths vary by up to 2 cm), so, as Dumas does, the
  segment parameters `L, α, β, γ` are the means over the trial (`mean_segment_geometry`, l.66), while `Kᵀ`, `N*ᵀ`
  use the instantaneous `Q`.
- **Tree.** bioNC's recursion starts at segment 0: the model is thigh (root) → shank → foot
  (`build_bionc_lower_limb`, l.164). Joint types do not enter the inverse dynamics.
- **Ground reaction.** Added on the foot as `−Joint(1).M, −Joint(1).F` at the CoP (l.227).
- **Same `Q̈`.** Both sides receive the same filtered second derivative (`filtered_second_derivative`, l.54).

## 4. Inertial parameters: same natural parameters on both sides

The test gives bioNC the natural inertial parameters computed as in `Inverse_Dynamics_GC.m`
(`dumas_natural_inertial_parameters`, l.156), so that it checks the **inverse dynamics algorithm** alone:

```
n_C = inv(Buv) r_C
J   = inv(Buv) · I_P · inv(Buv)ᵀ,     I_P = I_C + m ((r_C·r_C) E − r_C r_Cᵀ)        (Inverse_Dynamics_GC.m l.117-120)
```

This `J` is not the right pseudo-inertia. `G = ∫ Nᵀ N dm` needs the **second moment of mass**
`J = ∫ n nᵀ dm`, that is

```
J = inv(Buv) · (½ tr(I_P) E − I_P) · inv(Buv)ᵀ
```

which is what Dumas' own `Inverse_Dynamics_HM.m` builds (l.73-76: `(trace(Is)/2)·eye(3) − Is`), and what bioNC
computes from cartesian parameters
([natural_inertial_parameters.py:381-383](../bionc/bionc_numpy/natural_inertial_parameters.py#L381-L383)).
`tests/test_inertial_paramaters.py::test_generalized_kinetic_energy_equals_rigid_body_kinetic_energy` checks it
against the rigid body kinetic energy.

Building the segments with `NaturalSegment.with_cartesian_inertial_parameters` is not yet possible for these
non-orthogonal segments: `NaturalSegment.compute_transformation_matrix` returns `Buv.T`, which misplaces `n_C`
and `J`. The strict xfail
[`test_with_cartesian_inertial_parameters_non_orthogonal_segment`](../tests/test_inverse_dynamics_dumas.py#L250)
pins it (foot: `n_C` off by 0.21), and passes once the four `.T` are removed.

## 5. Freezing genuine MATLAB outputs

The reference is a port, so a transcription error would hide in both the port and bioNC's agreement with it.
To compare with MATLAB itself, run the toolbox once and save the outputs **and** the accelerations
(MATLAB's and SciPy's `filtfilt` pad differently near the trial ends):

```matlab
load Segment.mat; load Joint.mat
f = 100; fc = 6; dt = 1/f; n = size(Segment(2).Q, 3);
[Joint, Segment] = Inverse_Dynamics_GC(Joint, Segment, f, fc, n);
F = cat(2, Joint(2:4).F);   % 3 x 3 x n: ankle, knee, hip
M = cat(2, Joint(2:4).M);
for i = 2:4
    d2Qdt2{i-1} = Vfilt_array3(Derive_array3(Vfilt_array3(Derive_array3(Segment(i).Q, dt), f, fc), dt), f, fc);
end
save('dumas_gc_reference.mat', 'F', 'M', 'd2Qdt2', '-v7');
```

Then put `dumas_gc_reference.mat` in `tests/data/dumas_lower_limb_gait/`, feed `d2Qdt2` to bioNC in place of
`filtered_second_derivative`, and compare with `F`, `M` instead of `dumas_inverse_dynamics_gc`.
