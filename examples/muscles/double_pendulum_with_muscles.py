"""
Two-segment pendulum with four muscles, showing several muscle topologies:

* ``m_ground_to_seg1``: GROUND -> pendulum_0 (straight line, 2 via points)
* ``m_ground_to_seg2``: GROUND -> pendulum_1 (straight line, 2 via points)
* ``m_seg1_to_seg2``:   pendulum_0 -> pendulum_1 (straight line, 2 via points)
* ``m_multi_via``:      GROUND -> pendulum_0 -> pendulum_1 (4 via points)
"""

import numpy as np

from bionc import (
    BiomechanicalModel,
    CartesianAxis,
    JointType,
    Muscle,
    MuscleViaPoint,
    NaturalAxis,
    NaturalSegment,
    NaturalCoordinates,
    SegmentNaturalCoordinates,
    TransformationMatrixType,
)


def build_model() -> BiomechanicalModel:
    model = BiomechanicalModel()
    for i in range(2):
        name = f"pendulum_{i}"
        model[name] = NaturalSegment.with_cartesian_inertial_parameters(
            name=name,
            alpha=np.pi / 2,
            beta=np.pi / 2,
            gamma=np.pi / 2,
            length=1.0,
            mass=1.0,
            center_of_mass=np.array([0.0, -0.5, 0.0]),
            inertia=np.diag([0.05, 0.05, 0.05]),
            inertial_transformation_matrix=TransformationMatrixType.Buv,
        )

    model._add_joint(
        dict(
            name="hinge_0",
            joint_type=JointType.GROUND_REVOLUTE,
            parent="GROUND",
            child="pendulum_0",
            parent_axis=[CartesianAxis.X, CartesianAxis.X],
            child_axis=[NaturalAxis.V, NaturalAxis.W],
            theta=[np.pi / 2, np.pi / 2],
        )
    )
    model._add_joint(
        dict(
            name="hinge_1",
            joint_type=JointType.REVOLUTE,
            parent="pendulum_0",
            child="pendulum_1",
            parent_axis=[NaturalAxis.U, NaturalAxis.U],
            child_axis=[NaturalAxis.V, NaturalAxis.W],
            theta=[np.pi / 2, np.pi / 2],
        )
    )

    # 1) GROUND -> pendulum_0
    model.add_muscle(
        Muscle(
            name="m_ground_to_seg1",
            via_points=[
                MuscleViaPoint(name="m1_origin", parent_name="GROUND", position=np.array([0.3, 0.0, 0.2])),
                MuscleViaPoint(name="m1_insertion", parent_name="pendulum_0", position=(0.0, -0.5, 0.0)),
            ],
        )
    )

    # 2) GROUND -> pendulum_1
    model.add_muscle(
        Muscle(
            name="m_ground_to_seg2",
            via_points=[
                MuscleViaPoint(name="m2_origin", parent_name="GROUND", position=np.array([0.3, -1.0, 0.2])),
                MuscleViaPoint(name="m2_insertion", parent_name="pendulum_1", position=(0.0, -0.5, 0.0)),
            ],
        )
    )

    # 3) pendulum_0 -> pendulum_1 (bi-articular straight line)
    model.add_muscle(
        Muscle(
            name="m_seg1_to_seg2",
            via_points=[
                MuscleViaPoint(name="m3_origin", parent_name="pendulum_0", position=(0.05, -0.8, 0.0)),
                MuscleViaPoint(name="m3_insertion", parent_name="pendulum_1", position=(0.05, -0.2, 0.0)),
            ],
        )
    )

    # 4) GROUND -> pendulum_0 -> pendulum_1, 4 via points
    model.add_muscle(
        Muscle(
            name="m_multi_via",
            via_points=[
                MuscleViaPoint(name="m4_origin", parent_name="GROUND", position=np.array([-0.2, 0.1, 0.0])),
                MuscleViaPoint(name="m4_via_seg1", parent_name="pendulum_0", position=(-0.05, -0.4, 0.0)),
                MuscleViaPoint(name="m4_via_seg2", parent_name="pendulum_1", position=(-0.05, -0.3, 0.0)),
                MuscleViaPoint(name="m4_insertion", parent_name="pendulum_1", position=(0.0, -0.7, 0.0)),
            ],
        )
    )
    return model


def main():
    model = build_model()
    print(model)
    print(f"Muscles: {model.muscle_names}")

    # Both segments hanging vertically.
    Q0 = SegmentNaturalCoordinates.from_components(u=[1, 0, 0], rp=[0, 0, 0], rd=[0, -1, 0], w=[0, 0, 1])
    Q1 = SegmentNaturalCoordinates.from_components(u=[1, 0, 0], rp=[0, -1, 0], rd=[0, -2, 0], w=[0, 0, 1])
    Q = NaturalCoordinates.from_qi((Q0, Q1))

    print("\n=== Numpy ===")
    L = model.muscle_lengths(Q)
    MA = model.muscle_moment_arms(Q)
    print(f"length: {L}")
    print(f"moment arm shape: {MA.shape}")
    for name, row in zip(model.muscle_names, MA):
        print(f"  {name}: {row}")

    print("\n=== Casadi ===")
    from casadi import Function

    mx_model = model.to_mx()
    from bionc.bionc_casadi import NaturalCoordinates as MXNaturalCoordinates

    Q_sym = MXNaturalCoordinates.sym(mx_model.nb_segments)
    L_sym = mx_model.muscle_lengths(Q_sym)
    MA_sym = mx_model.muscle_moment_arms(Q_sym)
    f_L = Function("muscle_length", [Q_sym], [L_sym])
    f_MA = Function("muscle_moment_arm", [Q_sym], [MA_sym])
    Q_num = np.array(Q).reshape(-1)
    L_mx = np.array(f_L(Q_num)).squeeze()
    MA_mx = np.array(f_MA(Q_num))
    print(f"length (casadi): {L_mx}")
    print(f"moment arm shape (casadi): {MA_mx.shape}")

    return dict(
        model=model,
        Q=Q,
        length_numpy=L,
        moment_arm_numpy=MA,
        length_casadi=L_mx,
        moment_arm_casadi=MA_mx,
    )


def _segment_Q_about_x(theta: float, rp: np.ndarray) -> SegmentNaturalCoordinates:
    """Q for a unit segment hinged about the global X axis, at angle ``theta`` from -Y."""
    c, s = np.cos(theta), np.sin(theta)
    rd = rp + np.array([0.0, -c, -s])
    w = np.array([0.0, -s, c])
    return SegmentNaturalCoordinates.from_components(u=[1, 0, 0], rp=rp, rd=rd, w=w)


def animate(nb_frames: int = 120):
    """Animate the double pendulum + muscles by sweeping both hinge angles."""
    model = build_model()

    t = np.linspace(0.0, 2.0 * np.pi, nb_frames)
    theta0 = 0.6 * np.sin(t)
    theta1 = 0.9 * np.sin(t + np.pi / 3)

    Q_traj = np.zeros((model.nb_Q, nb_frames))
    for k in range(nb_frames):
        Q0 = _segment_Q_about_x(theta0[k], rp=np.zeros(3))
        rp1 = np.array(Q0.rd).reshape(3)
        Q1 = _segment_Q_about_x(theta0[k] + theta1[k], rp=rp1)
        Q_traj[:, k] = np.array(NaturalCoordinates.from_qi((Q0, Q1))).reshape(-1)

    from pyorerun import PhaseRerun
    from bionc.vizualization.pyorerun_interface import BioncModelNoMesh
    from bionc.vizualization.pyorerun_natural_vectors import add_natural_vectors

    prr = PhaseRerun(t_span=t)
    prr.add_animated_model(BioncModelNoMesh(model), Q_traj)
    add_natural_vectors(prr, model, Q_traj, scale_u=0.25, scale_v=1.0, scale_w=0.25)
    prr.rerun()


if __name__ == "__main__":
    main()
    animate()
