from bionc import (
    BiomechanicalModel,
    NaturalSegment,
    CartesianAxis,
    NaturalAxis,
    JointType,
)
from casadi import MX

import numpy as np
import pytest

from .utils import TestUtils


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
        model[name] = NaturalSegment(
            name=name,
            alpha=np.pi / 2,  # setting alpha, beta, gamma to pi/2 creates a orthogonal coordinate system
            beta=np.pi / 2,
            gamma=np.pi / 2,
            length=1,
            mass=1 * i,
            center_of_mass=np.array([0, -0.5, 0]),  # in segment coordinates system
            inertia=np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),  # in segment coordinates system
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


@pytest.mark.parametrize(
    "bionc_type",
    [
        "numpy",
        "casadi",
    ],
)
def test_inverse_dynamics(bionc_type):
    if bionc_type == "casadi":
        from bionc.bionc_casadi import (
            SegmentNaturalCoordinates,
            SegmentNaturalAccelerations,
            NaturalCoordinates,
            NaturalAccelerations,
        )
    else:
        from bionc.bionc_numpy import (
            SegmentNaturalCoordinates,
            SegmentNaturalAccelerations,
            NaturalCoordinates,
            NaturalAccelerations,
        )

    nb_segments = 3

    model = build_n_link_pendulum(nb_segments=nb_segments)
    if bionc_type == "casadi":
        model = model.to_mx()

    # vertical
    tuple_of_Q = [
        SegmentNaturalCoordinates.from_components(
            u=[1, 0, 0], rp=[0, 0, -i if i <= 1 else -1], rd=[0, 0, -i - 1 if i <= 1 else -2], w=[0, -1, 0]
        )
        for i in range(0, nb_segments)
    ]

    Q = NaturalCoordinates.from_qi(tuple(tuple_of_Q))

    tuple_of_Qddot = [
        SegmentNaturalAccelerations.from_components(
            uddot=[0, 0, 0], rpddot=[0, 0, 0], rdddot=[0, 0, 0], wddot=[0, 0, 0]
        )
        for i in range(0, nb_segments)
    ]
    Qddot = NaturalAccelerations.from_qddoti(tuple(tuple_of_Qddot))

    (torques, forces, lambdas) = model.inverse_dynamics(Q, Qddot)

    print(torques)

    TestUtils.assert_equal(torques, np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [-29.43, 9.81, 19.62]]))

    print(forces)

    TestUtils.assert_equal(
        forces,
        np.array(
            [
                [9.01033882e-16, -3.00344627e-16, -6.00689255e-16],
                [0.00000000e00, 0.00000000e00, 0.00000000e00],
                [0.00000000e00, 0.00000000e00, 0.00000000e00],
            ]
        ),
    )

    print(lambdas)

    TestUtils.assert_equal(
        lambdas,
        np.array(
            [
                [0.00000000e00, 0.00000000e00, 0.00000000e00],
                [0.00000000e00, 0.00000000e00, 0.00000000e00],
                [0.00000000e00, 0.00000000e00, 0.00000000e00],
                [0.00000000e00, 2.45250000e00, 4.90500000e00],
                [0.00000000e00, -3.00344627e-16, -6.00689255e-16],
                [0.00000000e00, 0.00000000e00, 0.00000000e00],
            ]
        ),
    )

@pytest.mark.parametrize(
    "bionc_type",
    [
        "numpy",
        "casadi",
    ],
)
def test_inverse_dynamics_segment(bionc_type):
    if bionc_type == "casadi":
        from bionc.bionc_casadi import (
            SegmentNaturalCoordinates,
            SegmentNaturalAccelerations,
        )
    else:
        from bionc.bionc_numpy import (
            SegmentNaturalCoordinates,
            SegmentNaturalAccelerations,
        )

    nb_segments = 3
    model = build_n_link_pendulum(nb_segments=nb_segments)
    if bionc_type == "casadi":
        model = model.to_mx()

    Qi = SegmentNaturalCoordinates.from_components(u=[1, 0.001, 0], rp=[0, 0.001, 0], rd=[0, 0, 0.001], w=[0, -1, 0.001])
    Qddoti = SegmentNaturalAccelerations.from_components(uddot=[0.01, 0.02, 0.03], rpddot=[0.04, 0.05, 0.06], rdddot=[0.07, 0.08, 0.09], wddot=[0.010, 0.011, 0.012])
    subtree_forces = np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.01, 0.011, 0.013])
    external_forces = np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.01, 0.011, 0.013])

    if bionc_type == "casadi":
        subtree_forces = MX(subtree_forces)
        external_forces = MX(external_forces)

    gf = model.segments["pendulum_1"].inverse_dynamics(Qi, Qddoti, subtree_intersegmental_generalized_forces=subtree_forces, segment_external_forces=external_forces)

    # print(gf)

    TestUtils.assert_equal(gf[0], np.array([0.015, 0.025, 9.845]), expand=False)
    TestUtils.assert_equal(gf[1], np.array([-0.01198125, -0.020465  ,  0.010285  ]), expand=False)
    TestUtils.assert_equal(gf[2], np.array([-1.73472348e-18,  3.50000000e-02,
                              3.07807808e-02, -2.48025161e+03,
                              7.01051051e+00,  5.25525526e-03]), expand=False)
