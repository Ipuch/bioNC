import numpy as np

from bionc import NaturalAxis, CartesianAxis, TransformationMatrixType
from bionc.bionc_numpy import (
    NaturalCoordinates,
    NaturalAccelerations,
    BiomechanicalModel,
    SegmentNaturalCoordinates,
    SegmentNaturalAccelerations,
    NaturalSegment,
    JointType,
)


def build_n_link_pendulum(nb_segments: int = 1) -> BiomechanicalModel:
    """Build a n-link pendulum model"""
    if nb_segments < 1:
        raise ValueError("The number of segment must be greater than 1")
    # Let's create a model
    model = BiomechanicalModel()
    # number of segments
    # fill the biomechanical model with the segment
    for i in range(nb_segments):
        name = f"pendulum_{i}"
        model[name] = NaturalSegment.with_cartesian_inertial_parameters(
            name=name,
            alpha=np.pi / 2,  # setting alpha, beta, gamma to pi/2 creates a orthogonal coordinate system
            beta=np.pi / 2,
            gamma=np.pi / 2,
            length=1,
            mass=1 * i,
            center_of_mass=np.array([0, -0.5, 0]),  # in segment coordinates system
            inertia=np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),  # in segment coordinates system
            inertial_transformation_matrix=TransformationMatrixType.Buv,
        )
    # add a revolute joint (still experimental)
    # if you want to add a revolute joint,
    # you need to ensure that x is always orthogonal to u and v
    model._add_joint(
        dict(
            name="hinge_0",
            joint_type=JointType.GROUND_REVOLUTE,
            parent="GROUND",
            child="pendulum_0",
            parent_axis=[CartesianAxis.X, CartesianAxis.X],
            child_axis=[NaturalAxis.V, NaturalAxis.W],  # meaning we pivot around the cartesian x-axis
            theta=[np.pi / 2, np.pi / 2],
        )
    )
    for i in range(1, nb_segments):
        model._add_joint(
            dict(
                name=f"hinge_{i}",
                joint_type=JointType.REVOLUTE,
                parent=f"pendulum_{0}",
                child=f"pendulum_{i}",
                parent_axis=[NaturalAxis.U, NaturalAxis.U],
                child_axis=[NaturalAxis.V, NaturalAxis.W],
                theta=[np.pi / 2, np.pi / 2],
            )
        )

    return model


def main(mode: str = "horizontal", with_fext: bool = False):
    nb_segments = 3

    model = build_n_link_pendulum(nb_segments=nb_segments)

    if mode == "horizontal":
        tuple_of_Q = [
            SegmentNaturalCoordinates.from_components(
                u=[1, 0, 0], rp=[0, -i if i <= 1 else -1, 0], rd=[0, -i - 1 if i <= 1 else -2, 0], w=[0, 0, 1]
            )
            for i in range(0, nb_segments)
        ]

    elif mode == "vertical":
        tuple_of_Q = [
            SegmentNaturalCoordinates.from_components(
                u=[1, 0, 0], rp=[0, 0, -i if i <= 1 else -1], rd=[0, 0, -i - 1 if i <= 1 else -2], w=[0, -1, 0]
            )
            for i in range(0, nb_segments)
        ]
    else:
        raise ValueError("mode must be horizontal or vertical")

    Q = NaturalCoordinates.from_qi(tuple(tuple_of_Q))

    tuple_of_Qddot = [
        SegmentNaturalAccelerations.from_components(
            uddot=[0, 0, 0], rpddot=[0, 0, 0], rdddot=[0, 0, 0], wddot=[0, 0, 0]
        )
        for i in range(0, nb_segments)
    ]
    Qddot = NaturalAccelerations.from_qddoti(tuple(tuple_of_Qddot))

    if with_fext:
        fext = model.external_force_set()
        fext.add_in_global_local_point(1, np.array([0.1, 0.2, 0.3, 0.01, 0.02, 0.03]), np.array([0.01, 0.02, 0.03]))
        fext.add_in_global(2, np.array([0.1, 0.2, 0.3, 0.01, 0.02, 0.03]) * 0.1, np.array([0.01, 0.02, 0.03]) * 0.1)

    else:
        fext = None

    torques, forces, lambdas = model.inverse_dynamics(Q=Q, Qddot=Qddot, external_forces=fext)

    print(torques)
    print(forces)
    print(lambdas)

    return torques, forces, lambdas


if __name__ == "__main__":
    main(mode="vertical")
    # main(mode="horizontal")
