"""
Minimal example showing how to add a muscle to a pendulum model and how to query
its length and moment arm.

The pendulum is a single segment hinged on the global X axis. A single muscle
spans from a fixed point in the global frame (origin, anchored to GROUND) to
a via point on the pendulum segment (insertion).
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
    model["pendulum"] = NaturalSegment.with_cartesian_inertial_parameters(
        name="pendulum",
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
            name="hinge",
            joint_type=JointType.GROUND_REVOLUTE,
            parent="GROUND",
            child="pendulum",
            parent_axis=[CartesianAxis.X, CartesianAxis.X],
            child_axis=[NaturalAxis.V, NaturalAxis.W],
            theta=[np.pi / 2, np.pi / 2],
        )
    )

    # One muscle: origin = fixed point in global, insertion = mid-segment of pendulum.
    # Local coordinates on the pendulum are expressed as (u, v, w) with v ∈ [0, -1]
    # going from rp (proximal) to rd (distal). v = -0.5 places the via point at mid-segment.
    origin = MuscleViaPoint(name="origin", parent_name="GROUND", position=np.array([0.3, 0.0, 0.2]))
    insertion = MuscleViaPoint(name="insertion", parent_name="pendulum", position=(0.0, -0.5, 0.0))
    model.add_muscle(Muscle(name="muscle1", via_points=[origin, insertion]))
    return model


def main():
    model = build_model()
    print(model)
    print(f"Muscles: {model.muscle_names}")

    # Pendulum hanging vertically: rp at the origin, rd one meter below.
    Qi = SegmentNaturalCoordinates.from_components(u=[1, 0, 0], rp=[0, 0, 0], rd=[0, -1, 0], w=[0, 0, 1])
    Q = NaturalCoordinates(Qi)

    print("\n=== Numpy ===")
    L = model.muscle_lengths(Q)
    MA = model.muscle_moment_arms(Q)
    print(f"length: {L}")
    print(f"moment arm shape: {MA.shape}")
    print(f"moment arm (-dL/dQ): {MA.squeeze()}")

    # Casadi: build symbolic length and moment arm, then evaluate at Q.
    print("\n=== Casadi ===")
    from casadi import Function

    mx_model = model.to_mx()
    from bionc.bionc_casadi import NaturalCoordinates as MXNaturalCoordinates

    Q_sym = MXNaturalCoordinates.sym(mx_model.nb_segments)
    L_sym = mx_model.muscle_lengths(Q_sym)
    MA_sym = mx_model.muscle_moment_arms(Q_sym)
    f_L = Function("muscle_length", [Q_sym], [L_sym])
    f_MA = Function("muscle_moment_arm", [Q_sym], [MA_sym])
    L_mx = np.array(f_L(Q.to_array())).squeeze()
    MA_mx = np.array(f_MA(Q.to_array())).squeeze()
    print(f"length (casadi): {L_mx}")
    print(f"moment arm (casadi): {MA_mx}")

    return dict(
        model=model,
        Q=Q,
        length_numpy=L,
        moment_arm_numpy=MA,
        length_casadi=L_mx,
        moment_arm_casadi=MA_mx,
    )


if __name__ == "__main__":
    main()
