"""
Minimal example of the one- and two-contact-point scapulothoracic joints, with a pyorerun viz.

Following Naaim (2016/2017), the scapula (child) is linked to a thorax ellipsoid (parent) by one
or two *fixed contact points* constrained to lie on the ellipsoid. Each point yields one scalar
constraint ``phi = sum_i (a_i . (P - C))^2 / s_i - 1`` (phi = 0 on the surface).

Two kinematic demos that keep every contact point exactly on the surface:
  * model_type="one": a single contact point glides over the ellipsoid (the scapula is free to
    re-orient around it),
  * model_type="two": two contact points stay fixed on the surface while the scapula spins about
    the chord joining them (the remaining rotational degree of freedom).

------------------------------------------------------------------------------------------------
Building the trajectories in natural coordinates
------------------------------------------------------------------------------------------------
A natural-coordinate segment is ``Q = (u, rp, rd, w)``: two unit direction vectors ``u`` and ``w``
and two points ``rp`` (proximal) and ``rd`` (distal), with ``v = rp - rd``. With alpha=beta=gamma=
pi/2 the frame ``(u, v, w)`` is orthonormal, and a point given in segment coordinates ``L`` maps to
the world as ``rp + [u v w] @ L``. So to place the segment we choose:
  * its orientation by picking the world directions of ``u`` (segment x), ``v`` (segment y) and
    ``w`` (segment z), and
  * its position by choosing ``rp`` (segment origin) and setting ``rd = rp - v``.

The ellipsoid surface is sampled with the standard parametrisation
    P(theta, phi) = (a sin th cos ph, b sin th sin ph, c cos th).
Each contact point is built to sit exactly on the surface (see the two trajectory helpers), so the
joint defect ``phi`` stays at machine precision along the whole motion.
"""

import numpy as np

from bionc import (
    BiomechanicalModel,
    NaturalSegment,
    NaturalCoordinates,
    SegmentNaturalCoordinates,
    TransformationMatrixType,
)
from bionc.bionc_numpy import Joint

# Ellipsoid semi-axes (a along x, b along y, c along z), in meters.
SEMI_AXES = (0.20, 0.15, 0.10)
# Reference surface points (theta, phi) used by the two-contact-point model.
TWO_POINT_PARAMS = ((np.pi / 2, 0.0), (np.pi / 2, 0.6))


def surface_point(theta: float, phi: float) -> np.ndarray:
    a, b, c = SEMI_AXES
    return np.array([a * np.sin(theta) * np.cos(phi), b * np.sin(theta) * np.sin(phi), c * np.cos(theta)])


def _rodrigues(axis: np.ndarray, angle: float) -> np.ndarray:
    k = axis / np.linalg.norm(axis)
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)


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


def build_model(model_type: str) -> BiomechanicalModel:
    """Thorax ellipsoid (parent) + scapula (child) with one or two fixed contact markers."""
    model = BiomechanicalModel()
    model["thorax"] = _segment("thorax")
    model["scapula"] = _segment("scapula")

    model["thorax"].add_natural_marker_from_segment_coordinates(
        name="ELLIPSOID_CENTER", location=np.array([0.0, 0.0, 0.0]), is_anatomical=True
    )
    for name, direction in zip(("AXIS_A", "AXIS_B", "AXIS_C"), ([1, 0, 0], [0, 1, 0], [0, 0, 1])):
        model["thorax"].add_natural_vector_from_segment_coordinates(name=name, direction=np.array(direction, float))

    if model_type == "one":
        # contact point at the scapula origin (it will be parked on the surface frame by frame)
        model["scapula"].add_natural_marker_from_segment_coordinates(
            name="CONTACT_POINT", location=np.array([0.0, 0.0, 0.0]), is_anatomical=True
        )
    elif model_type == "two":
        # two contact markers placed so the identity scapula at P1 maps them onto the two surface points
        p1 = surface_point(*TWO_POINT_PARAMS[0])
        p2 = surface_point(*TWO_POINT_PARAMS[1])
        model["scapula"].add_natural_marker_from_segment_coordinates(
            name="CONTACT_POINT_1", location=np.array([0.0, 0.0, 0.0]), is_anatomical=True
        )
        model["scapula"].add_natural_marker_from_segment_coordinates(
            name="CONTACT_POINT_2", location=p2 - p1, is_anatomical=True
        )
    else:
        raise ValueError("model_type must be 'one' or 'two'")

    return model


def build_joint(model: BiomechanicalModel, model_type: str):
    common = dict(
        parent=model["thorax"],
        child=model["scapula"],
        index=0,
        semi_axis_lengths=SEMI_AXES,
        ellipsoid_center="ELLIPSOID_CENTER",
        ellipsoid_axis_a="AXIS_A",
        ellipsoid_axis_b="AXIS_B",
        ellipsoid_axis_c="AXIS_C",
    )
    if model_type == "one":
        return Joint.PointOnEllipsoid(name="one_contact", contact_point="CONTACT_POINT", **common)
    return Joint.TwoPointsOnEllipsoid(
        name="two_contact", contact_point_1="CONTACT_POINT_1", contact_point_2="CONTACT_POINT_2", **common
    )


def thorax_pose() -> SegmentNaturalCoordinates:
    return SegmentNaturalCoordinates.from_components(u=[1, 0, 0], rp=[0, 0, 0], rd=[0, -1, 0], w=[0, 0, 1])


def _one_contact_trajectory(n_frames: int):
    """
    A single contact point glides over the ellipsoid surface.

    The scapula contact marker is the segment origin, so we just put ``rp`` on the surface point
    ``P(theta, phi)`` and keep a fixed (identity) orientation. With ``contact = origin`` the joint
    constraint reads ``P on the ellipsoid`` and is satisfied exactly. The orientation here is
    arbitrary (the one-contact joint leaves the scapula free to re-orient around the point); we keep
    it constant for clarity.
    """
    poses = []
    for t in np.linspace(0, 1, n_frames):
        point = surface_point(theta=np.pi / 2 + 0.6 * np.sin(2 * np.pi * t), phi=2 * np.pi * t)
        poses.append(
            SegmentNaturalCoordinates.from_components(
                u=[1, 0, 0], rp=point, rd=point - np.array([0, 1.0, 0]), w=[0, 0, 1]
            )
        )
    return poses


def _two_contact_trajectory(n_frames: int):
    """
    Two contact points stay fixed on the surface while the scapula spins about their chord.

    Two points removed from the surface leave 4 dof; one of them is a pure rotation about the line
    through the two contacts. A rigid rotation ``rot`` about the chord axis ``p2 - p1`` (Rodrigues)
    leaves both p1 and p2 fixed -- they are on the axis -- so both stay exactly on the surface while
    the scapula body swings. The reference pose is the identity scapula whose origin is at ``p1`` and
    whose second contact marker sits at ``p2 - p1`` in segment coordinates (set in ``build_model``),
    so at the reference the two markers map to p1 and p2. Rotating the whole frame by ``rot`` keeps
    that true for every frame:
        u = rot @ x,   w = rot @ z,   rp = p1,   rd = p1 - rot @ y .
    """
    p1 = surface_point(*TWO_POINT_PARAMS[0])
    p2 = surface_point(*TWO_POINT_PARAMS[1])
    axis = p2 - p1
    poses = []
    for t in np.linspace(0, 1, n_frames):
        rot = _rodrigues(axis, 2 * np.pi * t)
        poses.append(
            SegmentNaturalCoordinates.from_components(
                u=rot @ np.array([1.0, 0, 0]),
                rp=p1,
                rd=p1 - rot @ np.array([0, 1.0, 0]),
                w=rot @ np.array([0, 0, 1.0]),
            )
        )
    return poses


def build_trajectory(model_type: str, n_frames: int = 120):
    """Return (time_steps, stacked natural coordinates [24, n_frames], max contact defect)."""
    model = build_model(model_type)
    joint = build_joint(model, model_type)
    Q_thorax = thorax_pose()
    q_thorax = np.array(Q_thorax).reshape(-1)
    poses = _one_contact_trajectory(n_frames) if model_type == "one" else _two_contact_trajectory(n_frames)

    time_steps = np.linspace(0, 4, n_frames)
    all_q = np.zeros((24, n_frames))
    max_defect = 0.0
    for k, Q_scap in enumerate(poses):
        all_q[:12, k] = q_thorax
        all_q[12:, k] = np.array(Q_scap).reshape(-1)
        max_defect = max(max_defect, float(np.max(np.abs(np.array(joint.constraint(Q_thorax, Q_scap))))))
    return model, joint, time_steps, all_q, max_defect


def animate(model: BiomechanicalModel, time_steps: np.ndarray, all_q: np.ndarray) -> None:
    import rerun as rr
    from pyorerun import PhaseRerun
    from bionc.vizualization.pyorerun_interface import BioncModelNoMesh

    prr = PhaseRerun(t_span=time_steps)
    prr.add_animated_model(BioncModelNoMesh(model), NaturalCoordinates(all_q))
    prr.rerun()
    rr.log(
        "thorax/ellipsoid",
        rr.Ellipsoids3D(half_sizes=[list(SEMI_AXES)], centers=[[0.0, 0.0, 0.0]], colors=[[120, 160, 255, 90]]),
        static=True,
    )


def main(model_type: str = "one", show_results: bool = True):
    model, joint, time_steps, all_q, max_defect = build_trajectory(model_type)
    np.testing.assert_allclose(max_defect, 0.0, atol=1e-9)

    if show_results:
        n = {"one": 1, "two": 2}[model_type]
        print(f"{model_type}-contact-point model: {n} constraint(s), ellipsoid semi-axes {SEMI_AXES} m")
        print(f"  frames = {time_steps.size}, max contact defect along path = {max_defect:.2e}")
        animate(model, time_steps, all_q)

    return model, joint, max_defect


if __name__ == "__main__":
    # main("one", show_results=True)
    main("two", show_results=True)
