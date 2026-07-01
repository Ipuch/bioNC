"""
Compare the three scapulothoracic joint models on the SAME (noised) marker trajectory.

Following Naaim (2016/2017), the scapula is linked to a thorax ellipsoid by either:
  * a tangent contact          -> ``Joint.EllipsoidOnPlane``       (the scapula plane is tangent,
                                                                     no penetration by definition),
  * one fixed contact point    -> ``Joint.PointOnEllipsoid``,
  * two fixed contact points   -> ``Joint.TwoPointsOnEllipsoid``.

We synthesise a "true" scapula motion gliding on the ellipsoid (tangent contact, zero penetration),
sample three scapula markers (like angulus acromialis / trigonum spinae / angulus inferior) plus
three thorax markers, add Gaussian noise to the scapula markers only (the thorax markers are kept
exact so the thorax -- and therefore its ellipsoid -- reconstructs at a fixed pose), then run
differential inverse kinematics (``method="dik"``) with each of the three joint models tracking the
very same markers. We report, per model, the marker tracking error and the scapula penetration into
the ellipsoid, and (optionally) display the three reconstructions side by side in a single pyorerun
scene with one fixed ellipsoid per model.

------------------------------------------------------------------------------------------------
Natural-coordinate poses used to synthesise the reference motion
------------------------------------------------------------------------------------------------
A natural-coordinate segment is described by ``Q = (u, rp, rd, w)``: two unit direction vectors
``u`` and ``w`` and two points ``rp`` (proximal) and ``rd`` (distal), with ``v = rp - rd``. For an
orthonormal segment (alpha=beta=gamma=pi/2) ``(u, v, w)`` is a right-handed orthonormal frame and a
point given in segment coordinates ``L`` maps to the world as ``rp + [u v w] @ L``.

To make the scapula glide while staying tangent to the ellipsoid we place, at each instant, a contact
on the ellipsoid surface and align the scapula frame with the local surface geometry:
  * surface point     P(theta, phi) = (a sin th cos ph, b sin th sin ph, c cos th)
  * outward normal    n proportional to (P_x/a^2, P_y/b^2, P_z/c^2), then normalised
  * scapula frame     u = n  (so the scapula x-axis is the contact normal), and (v, w) span the
                      tangent plane; rp = P (the contact point sits at the scapula origin).
With the three scapula markers placed in the scapula y-z plane (x = 0), they lie in that tangent
plane, i.e. exactly on the tangent-contact surface and just outside the (convex) ellipsoid elsewhere.
"""

import numpy as np

from bionc import (
    BiomechanicalModel,
    NaturalSegment,
    NaturalCoordinates,
    SegmentNaturalCoordinates,
    JointType,
    InverseKinematics,
    TransformationMatrixType,
)
from bionc.bionc_numpy.enums import InitialGuessModeType

SEMI_AXES = (0.20, 0.15, 0.10)  # ellipsoid semi-axes a, b, c [m]
# scapula markers in segment coordinates, all in the x = 0 plane (the scapula plane)
SCAPULA_MARKERS = {
    "AA": np.array([0.0, 0.06, 0.00]),  # angulus acromialis
    "TS": np.array([0.0, -0.03, 0.05]),  # trigonum spinae
    "AI": np.array([0.0, -0.03, -0.05]),  # angulus inferior
}
THORAX_MARKERS = {  # arbitrary fixed thorax landmarks (keep the thorax determined)
    "T1": np.array([0.10, 0.10, 0.00]),
    "T2": np.array([-0.10, 0.10, 0.05]),
    "T3": np.array([0.00, -0.10, 0.10]),
}
MODELS = ("tangent", "one_point", "two_point")


def _segment(name: str) -> NaturalSegment:
    return NaturalSegment.with_cartesian_inertial_parameters(
        name=name,
        alpha=np.pi / 2,
        beta=np.pi / 2,
        gamma=np.pi / 2,
        length=1,
        mass=1,
        center_of_mass=np.array([0, 0, 0]),
        inertia=np.eye(3),
        inertial_transformation_matrix=TransformationMatrixType.Buv,
    )


def surface_point(theta: float, phi: float) -> np.ndarray:
    a, b, c = SEMI_AXES
    return np.array([a * np.sin(theta) * np.cos(phi), b * np.sin(theta) * np.sin(phi), c * np.cos(theta)])


def build_model(model_type: str) -> BiomechanicalModel:
    """Thorax ellipsoid + scapula with shared markers and the requested scapulothoracic joint."""
    model = BiomechanicalModel()
    model["thorax"] = _segment("thorax")
    model["scapula"] = _segment("scapula")

    # ellipsoid geometry on the thorax (anatomical, not tracked)
    model["thorax"].add_natural_marker_from_segment_coordinates(
        name="ELLIPSOID_CENTER", location=np.zeros(3), is_anatomical=True, is_technical=False
    )
    for name, direction in zip(("AXIS_A", "AXIS_B", "AXIS_C"), ([1, 0, 0], [0, 1, 0], [0, 0, 1])):
        model["thorax"].add_natural_vector_from_segment_coordinates(name=name, direction=np.array(direction, float))

    # tracked markers
    for name, loc in THORAX_MARKERS.items():
        model["thorax"].add_natural_marker_from_segment_coordinates(name=name, location=loc, is_technical=True)
    for name, loc in SCAPULA_MARKERS.items():
        model["scapula"].add_natural_marker_from_segment_coordinates(name=name, location=loc, is_technical=True)

    # contact geometry on the scapula (derived from the markers, like the paper)
    model["scapula"].add_natural_marker_from_segment_coordinates(
        name="SCAP_CONTACT", location=np.zeros(3), is_anatomical=True, is_technical=False
    )
    model["scapula"].add_natural_vector_from_segment_coordinates(
        name="SCAP_NORMAL", direction=np.array([1.0, 0.0, 0.0])
    )

    ellipsoid = dict(
        semi_axis_lengths=SEMI_AXES,
        ellipsoid_center="ELLIPSOID_CENTER",
        ellipsoid_axis_a="AXIS_A",
        ellipsoid_axis_b="AXIS_B",
        ellipsoid_axis_c="AXIS_C",
    )
    if model_type == "tangent":
        joint = dict(joint_type=JointType.ELLIPSOID_ON_PLANE, plane_point="SCAP_CONTACT", plane_normal="SCAP_NORMAL")
    elif model_type == "one_point":
        joint = dict(joint_type=JointType.POINT_ON_ELLIPSOID, contact_point="SCAP_CONTACT")
    elif model_type == "two_point":
        joint = dict(joint_type=JointType.TWO_POINTS_ON_ELLIPSOID, contact_point_1="AA", contact_point_2="AI")
    else:
        raise ValueError(f"unknown model_type {model_type!r}")

    # Weld the thorax to ground at the world origin so the ellipsoid is genuinely fixed: the weld
    # holds rp/rd exactly (a hard constraint in the inverse kinematics) and the three noise-free
    # thorax markers pin the remaining rotation, so the thorax cannot be traded off against the
    # scapula markers. This is what makes the static ellipsoid in the viz exact.
    model._add_joint(
        dict(
            name="thorax_to_ground",
            joint_type=JointType.GROUND_WELD,
            parent="GROUND",
            child="thorax",
            rp_child_ref=np.array([0.0, 0.0, 0.0]),
            rd_child_ref=np.array([0.0, -1.0, 0.0]),
        )
    )
    model._add_joint(dict(name="scapulothoracic", parent="thorax", child="scapula", **ellipsoid, **joint))
    return model


def thorax_pose() -> SegmentNaturalCoordinates:
    return SegmentNaturalCoordinates.from_components(u=[1, 0, 0], rp=[0, 0, 0], rd=[0, -1, 0], w=[0, 0, 1])


def reference_scapula_pose(theta: float, phi: float) -> SegmentNaturalCoordinates:
    """Tangent-contact scapula pose at the surface point (theta, phi) -- see module docstring."""
    a, b, c = SEMI_AXES
    point = surface_point(theta, phi)
    normal = np.array([point[0] / a**2, point[1] / b**2, point[2] / c**2])
    normal /= np.linalg.norm(normal)
    reference = np.array([0.0, 0.0, 1.0]) if abs(normal[2]) < 0.9 else np.array([0.0, 1.0, 0.0])
    y_axis = np.cross(reference, normal)
    y_axis /= np.linalg.norm(y_axis)
    z_axis = np.cross(normal, y_axis)
    return SegmentNaturalCoordinates.from_components(u=normal, rp=point, rd=point - y_axis, w=z_axis)


def reference_motion(n_frames: int):
    """Synthesise the true (tangent-contact) thorax+scapula trajectory, [24, n_frames]."""
    q_thorax = np.array(thorax_pose()).reshape(-1)
    all_q = np.zeros((24, n_frames))
    for k, t in enumerate(np.linspace(0, 1, n_frames)):
        Q_scap = reference_scapula_pose(theta=np.pi / 2 + 0.5 * np.sin(2 * np.pi * t), phi=1.5 * np.pi * t)
        all_q[:12, k] = q_thorax
        all_q[12:, k] = np.array(Q_scap).reshape(-1)
    return all_q


def technical_markers(model: BiomechanicalModel, all_q: np.ndarray) -> np.ndarray:
    """World positions of the technical markers along a trajectory, ordered like the model, [3, m, T]."""
    all_names = list(model.marker_names)
    idx = [all_names.index(name) for name in model.marker_names_technical]
    out = np.zeros((3, len(idx), all_q.shape[1]))
    for k in range(all_q.shape[1]):
        m = np.array(model.markers(NaturalCoordinates(all_q[:, k])))
        m = m[:3, :, 0] if m.ndim == 3 else m[:3, :]
        out[:, :, k] = m[:, idx]
    return out


def _radial_penetration(point: np.ndarray, center: np.ndarray, axes: list, s: np.ndarray) -> float:
    """
    Depth (m, positive when inside) of a world point below the ellipsoid, measured along the ray from
    the ellipsoid centre. ``axes`` are the (reconstructed) unit principal axes, ``s`` the squared
    semi-axes. t = sqrt(sum_i (a_i . (P - C))^2 / s_i); t < 1 means inside.
    """
    w = point - center
    t = np.sqrt(sum((axes[i] @ w) ** 2 / s[i] for i in range(3)))
    return float(np.linalg.norm(w) * (1.0 / t - 1.0)) if t < 1.0 else 0.0


def _penetration_depths(model: BiomechanicalModel, all_q: np.ndarray) -> np.ndarray:
    """
    Penetration into the ellipsoid (m, positive = inside) of the scapula markers and their centroid,
    measured against the *reconstructed* ellipsoid (its centre and principal axes follow the thorax,
    which the inverse kinematics also moves). The centroid matters for the two-contact-point model:
    the chord between two surface points passes inside the convex ellipsoid, so the scapula plane dips
    below the surface there.
    """
    joint = model.joints["scapulothoracic"]
    N_C = joint.ellipsoid_center.interpolation_matrix.to_array()
    N_axes = [np.array(axis.interpolation_matrix) for axis in joint.ellipsoid_axes]
    s = [length**2 for length in joint.semi_axis_lengths]

    all_names = list(model.marker_names)
    scap_idx = [all_names.index(name) for name in SCAPULA_MARKERS]
    depths = np.zeros((len(scap_idx) + 1, all_q.shape[1]))
    for k in range(all_q.shape[1]):
        q_thorax = all_q[:12, k]
        center = N_C @ q_thorax
        axes = [N @ q_thorax for N in N_axes]
        m = np.array(model.markers(NaturalCoordinates(all_q[:, k])))
        m = m[:3, :, 0] if m.ndim == 3 else m[:3, :]
        scap_points = [m[:, mi] for mi in scap_idx]
        for j, point in enumerate(scap_points):
            depths[j, k] = _radial_penetration(point, center, axes, s)
        depths[-1, k] = _radial_penetration(np.mean(scap_points, axis=0), center, axes, s)
    return depths


def solve_model(model_type: str, markers: np.ndarray, q_init: np.ndarray):
    model = build_model(model_type)
    ik = InverseKinematics(model, experimental_markers=markers)
    Qopt = ik.solve(
        method="dik",
        Q_init=NaturalCoordinates(q_init),
        initial_guess_mode=InitialGuessModeType.USER_PROVIDED,
    )
    reconstructed = technical_markers(model, Qopt)
    marker_rmse_mm = 1e3 * np.sqrt(np.mean((reconstructed - markers) ** 2))
    penetration_mm = 1e3 * float(np.max(_penetration_depths(model, Qopt)))
    return model, Qopt, marker_rmse_mm, penetration_mm


def main(n_frames: int = 60, noise_std: float = 0.003, seed: int = 0, show_results: bool = True):
    # 1) synthesise the true motion and the noisy markers (shared by all three models)
    reference_model = build_model("tangent")
    q_true = reference_motion(n_frames)
    markers_true = technical_markers(reference_model, q_true)

    # Noise is added only to the scapula markers: the thorax markers stay exact so the thorax (and
    # therefore its ellipsoid) reconstructs exactly at the world origin -- which is what lets us draw
    # a single, fixed (static) ellipsoid in the viz.
    rng = np.random.default_rng(seed)
    tech_names = reference_model.marker_names_technical
    scapula_mask = np.array([name in SCAPULA_MARKERS for name in tech_names])
    noise = rng.normal(0, noise_std, markers_true.shape)
    noise[:, ~scapula_mask, :] = 0.0
    markers_noisy = markers_true + noise

    # 2) track the same markers with each joint model
    results = {}
    for model_type in MODELS:
        model, Qopt, rmse_mm, penetration_mm = solve_model(model_type, markers_noisy, q_true)
        results[model_type] = dict(model=model, Qopt=Qopt, rmse_mm=rmse_mm, penetration_mm=penetration_mm)

    if show_results:
        print(f"tracking {n_frames} frames, marker noise std = {1e3 * noise_std:.1f} mm")
        print(f"{'model':>10} | {'marker RMSE [mm]':>16} | {'max penetration [mm]':>20}")
        for model_type in MODELS:
            r = results[model_type]
            print(f"{model_type:>10} | {r['rmse_mm']:>16.2f} | {r['penetration_mm']:>20.2f}")
        animate(results, markers_noisy)

    return results, markers_noisy


def animate(results: dict, markers_noisy: np.ndarray) -> None:
    """Show the three reconstructions side by side (offset along z) in one pyorerun scene."""
    import rerun as rr
    from pyorerun import PhaseRerun, DisplayModelOptions
    from bionc.vizualization.pyorerun_interface import BioncModelNoMesh

    n_frames = markers_noisy.shape[2]
    time_steps = np.linspace(0, 4, n_frames)
    gap = 0.7  # lateral offset between models [m]
    colors = {
        "tangent": (80, 200, 120),
        "one_point": (230, 170, 60),
        "two_point": (110, 150, 240),
    }
    print("pyorerun layout (along +z): " + ", ".join(f"{m} {colors[m]}" for m in MODELS))

    prr = PhaseRerun(t_span=time_steps)
    for i, model_type in enumerate(MODELS):
        offset = np.array([0.0, 0.0, i * gap])
        Q = _offset_q(results[model_type]["Qopt"], offset)
        markers = markers_noisy + offset[:, None, None]
        options = DisplayModelOptions(_markers_color=colors[model_type], _show_marker_labels=False)
        prr.add_animated_model(BioncModelNoMesh(results[model_type]["model"], options), Q, tracked_markers=markers)

    prr.rerun()
    for i, model_type in enumerate(MODELS):
        center = [0.0, 0.0, i * gap]
        rr.log(
            f"ellipsoids/{model_type}",
            rr.Ellipsoids3D(half_sizes=[list(SEMI_AXES)], centers=[center], colors=[list(colors[model_type]) + [70]]),
            static=True,
        )


def _offset_q(Qopt: np.ndarray, offset: np.ndarray) -> np.ndarray:
    """Translate every segment's points (rp, rd) by a world offset for side-by-side display."""
    Q = Qopt.copy()
    for seg in range(Q.shape[0] // 12):
        base = 12 * seg
        Q[base + 3 : base + 6, :] += offset[:, None]  # rp
        Q[base + 6 : base + 9, :] += offset[:, None]  # rd
    return Q


if __name__ == "__main__":
    main(show_results=True)
