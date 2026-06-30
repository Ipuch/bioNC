"""
Minimal example of a segment "riding" on an ellipsoid-on-plane joint, with a pyorerun viz.

This mimics a scapulothoracic joint: the *parent* segment carries an ellipsoid (the thorax)
and the *child* segment carries a plane (the scapula). The single holonomic constraint keeps
the child plane tangent to the parent ellipsoid:

    phi = sqrt(u^T R^T B R u) + u^T (C - A)

with ``u`` the plane normal, ``C`` the ellipsoid centre, ``A`` the plane contact point,
``B = diag(a^2, b^2, c^2)`` and ``R`` the ellipsoid orientation (its three principal axes).

We first evaluate the constraint at three static poses to show its geometric meaning
(tangent -> 0, tangential slide -> 0, pushed along the normal -> signed distance), then build a
trajectory of tangent poses so the scapula plane glides over the thorax ellipsoid, and animate it
with pyorerun (the thorax ellipsoid is drawn as a translucent surface).
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


def build_model() -> BiomechanicalModel:
    """Two orthonormal segments: a thorax ellipsoid (parent) and a scapula plane (child)."""
    model = BiomechanicalModel()
    for name in ("thorax", "scapula"):
        model[name] = NaturalSegment.with_cartesian_inertial_parameters(
            name=name,
            alpha=np.pi / 2,  # orthonormal natural basis
            beta=np.pi / 2,
            gamma=np.pi / 2,
            length=1,
            mass=1,
            center_of_mass=np.array([0, 0, 0]),
            inertia=np.eye(3),
            inertial_transformation_matrix=TransformationMatrixType.Buv,
        )

    # Parent (thorax): ellipsoid centred on the segment origin, principal axes = segment axes.
    model["thorax"].add_natural_marker_from_segment_coordinates(
        name="ELLIPSOID_CENTER", location=np.array([0.0, 0.0, 0.0]), is_anatomical=True
    )
    model["thorax"].add_natural_vector_from_segment_coordinates(
        name="ELLIPSOID_AXIS_A", direction=np.array([1.0, 0.0, 0.0]), normalize=True
    )
    model["thorax"].add_natural_vector_from_segment_coordinates(
        name="ELLIPSOID_AXIS_B", direction=np.array([0.0, 1.0, 0.0]), normalize=True
    )
    model["thorax"].add_natural_vector_from_segment_coordinates(
        name="ELLIPSOID_AXIS_C", direction=np.array([0.0, 0.0, 1.0]), normalize=True
    )

    # Child (scapula): contact point at the segment origin, plane normal along +x.
    model["scapula"].add_natural_marker_from_segment_coordinates(
        name="CONTACT_POINT",
        location=np.array([0.0, 0.0, 0.0]),
        is_anatomical=True,
        is_technical=True,
    )
    model["scapula"].add_natural_vector_from_segment_coordinates(
        name="PLANE_NORMAL", direction=np.array([1.0, 0.0, 0.0]), normalize=True
    )

    return model


def build_joint(model: BiomechanicalModel) -> Joint.EllipsoidOnPlane:
    return Joint.EllipsoidOnPlane(
        name="scapulothoracic",
        parent=model["thorax"],
        child=model["scapula"],
        index=0,
        semi_axis_lengths=SEMI_AXES,
        ellipsoid_center="ELLIPSOID_CENTER",
        ellipsoid_axis_a="ELLIPSOID_AXIS_A",
        ellipsoid_axis_b="ELLIPSOID_AXIS_B",
        ellipsoid_axis_c="ELLIPSOID_AXIS_C",
        plane_point="CONTACT_POINT",
        plane_normal="PLANE_NORMAL",
    )


def thorax_pose() -> SegmentNaturalCoordinates:
    """Thorax as an identity frame at the world origin (ellipsoid centred there, axes global)."""
    return SegmentNaturalCoordinates.from_components(u=[1, 0, 0], rp=[0, 0, 0], rd=[0, -1, 0], w=[0, 0, 1])


def tangent_scapula_pose(theta: float, phi: float) -> SegmentNaturalCoordinates:
    """
    Scapula natural coordinates so its plane is tangent to the thorax ellipsoid.

    Recall a segment is ``Q = (u, rp, rd, w)`` with ``v = rp - rd``; for an orthonormal segment
    ``(u, v, w)`` is a right-handed frame and ``rp`` is the segment origin. Here the scapula plane is
    the segment's ``y-z`` plane (normal = segment x = ``u``) and ``SCAP_CONTACT`` is the origin.

    Steps:
      1. pick a surface point ``P(theta, phi) = (a sin th cos ph, b sin th sin ph, c cos th)``;
      2. the outward ellipsoid normal there is ``grad`` of ``x^2/a^2 + y^2/b^2 + z^2/c^2``, i.e.
         proportional to ``(Px/a^2, Py/b^2, Pz/c^2)``, then normalised -> ``n``;
      3. set ``u = n`` so the scapula plane is perpendicular to ``n`` and therefore tangent at ``P``;
      4. complete an orthonormal frame: ``y = ref x n`` (ref chosen non-parallel to ``n``), ``z = n x y``;
      5. put the contact at the surface: ``rp = P`` and ``rd = rp - y``.
    The plane through ``rp`` with normal ``u`` is the tangent plane, so the joint constraint is met.
    """
    a, b, c = SEMI_AXES
    point = np.array([a * np.sin(theta) * np.cos(phi), b * np.sin(theta) * np.sin(phi), c * np.cos(theta)])
    normal = np.array([point[0] / a**2, point[1] / b**2, point[2] / c**2])
    normal = normal / np.linalg.norm(normal)
    # build an orthonormal scapula frame whose x-axis is the outward normal
    reference = np.array([0.0, 0.0, 1.0]) if abs(normal[2]) < 0.9 else np.array([0.0, 1.0, 0.0])
    y_axis = np.cross(reference, normal)
    y_axis /= np.linalg.norm(y_axis)
    z_axis = np.cross(normal, y_axis)
    # u = normal (segment x), v = rp - rd = y_axis, w = z_axis, rp = contact point
    return SegmentNaturalCoordinates.from_components(u=normal, rp=point, rd=point - y_axis, w=z_axis)


def riding_trajectory(n_frames: int = 120):
    """
    Build a trajectory where the scapula glides over the thorax ellipsoid while staying tangent.

    Returns
    -------
    time_steps : np.ndarray, shape (n_frames,)
    all_q : np.ndarray, shape (24, n_frames)  -- stacked [thorax; scapula] natural coordinates
    max_defect : float                         -- max |phi| along the path (should be ~0)
    """
    time_steps = np.linspace(0, 4, n_frames)
    q_thorax = np.array(thorax_pose()).reshape(-1)
    all_q = np.zeros((24, n_frames))
    model = build_model()
    joint = build_joint(model)
    Q_thorax = thorax_pose()
    max_defect = 0.0
    for k, t in enumerate(time_steps):
        s = t / time_steps[-1]
        # a wandering path: full turn in longitude, gentle latitude sweep around the equator
        Q_scap = tangent_scapula_pose(theta=np.pi / 2 + 0.6 * np.sin(2 * np.pi * s), phi=2 * np.pi * s)
        all_q[:12, k] = q_thorax
        all_q[12:, k] = np.array(Q_scap).reshape(-1)
        max_defect = max(max_defect, abs(float(np.array(joint.constraint(Q_thorax, Q_scap)))))
    return time_steps, all_q, max_defect


def animate(model: BiomechanicalModel, time_steps: np.ndarray, all_q: np.ndarray) -> None:
    """Animate the riding scapula with pyorerun, drawing the thorax ellipsoid as a surface."""
    import rerun as rr
    from pyorerun import PhaseRerun
    from bionc.vizualization.pyorerun_interface import BioncModelNoMesh

    prr = PhaseRerun(t_span=time_steps)
    prr.add_animated_model(BioncModelNoMesh(model), NaturalCoordinates(all_q))
    prr.rerun()

    # the thorax ellipsoid is fixed at the world origin with axes aligned to the global frame
    rr.log(
        "thorax/ellipsoid",
        rr.Ellipsoids3D(half_sizes=[list(SEMI_AXES)], centers=[[0.0, 0.0, 0.0]], colors=[[120, 160, 255, 90]]),
        static=True,
    )


def main(show_results: bool = True):
    a = SEMI_AXES[0]
    model = build_model()
    joint = build_joint(model)

    # --- Static poses: geometric meaning of the constraint ---
    Q_thorax = thorax_pose()
    Q_tangent = SegmentNaturalCoordinates.from_components(u=[1, 0, 0], rp=[a, 0, 0], rd=[a, -1, 0], w=[0, 0, 1])
    Q_slide = SegmentNaturalCoordinates.from_components(u=[1, 0, 0], rp=[a, 0.05, 0], rd=[a, -0.95, 0], w=[0, 0, 1])
    Q_pushed = SegmentNaturalCoordinates.from_components(
        u=[1, 0, 0], rp=[a + 0.02, 0, 0], rd=[a + 0.02, -1, 0], w=[0, 0, 1]
    )
    phi = {
        "tangent": float(np.array(joint.constraint(Q_thorax, Q_tangent))),
        "slide_+y": float(np.array(joint.constraint(Q_thorax, Q_slide))),
        "pushed_+x": float(np.array(joint.constraint(Q_thorax, Q_pushed))),
    }
    np.testing.assert_allclose(phi["tangent"], 0.0, atol=1e-9)
    np.testing.assert_allclose(phi["slide_+y"], 0.0, atol=1e-9)
    np.testing.assert_allclose(phi["pushed_+x"], -0.02, atol=1e-9)

    # --- Riding trajectory: scapula glides over the ellipsoid, staying tangent ---
    time_steps, all_q, max_defect = riding_trajectory()
    np.testing.assert_allclose(max_defect, 0.0, atol=1e-9)

    if show_results:
        print(f"ellipsoid semi-axes (a, b, c) = {SEMI_AXES} m")
        for pose, value in phi.items():
            print(f"  phi[{pose:>9}] = {value:+.6f}")
        print(f"  riding trajectory frames = {time_steps.size}, max tangency defect = {max_defect:.2e}")
        animate(model, time_steps, all_q)

    return model, joint, phi


if __name__ == "__main__":
    main(show_results=True)
