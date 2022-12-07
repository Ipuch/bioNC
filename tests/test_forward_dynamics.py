from sys import platform
import numpy as np
import pytest

from .utils import TestUtils


@pytest.mark.parametrize(
    "bionc_type",
    [
        "numpy",
    ],
)
def test_biomech_model(bionc_type):
    from bionc.bionc_numpy import (
        SegmentNaturalVelocities,
        NaturalSegment,
        SegmentNaturalCoordinates,
    )

    bionc = TestUtils.bionc_folder()
    module = TestUtils.load_module(bionc + "/examples/forward_dynamics/forward_dynamics.py")

    # Let's create a segment
    my_segment = NaturalSegment(
        name="box",
        alpha=np.pi / 2,  # setting alpha, beta, gamma to pi/2 creates a orthogonal coordinate system
        beta=np.pi / 2,
        gamma=np.pi / 2,
        length=1,
        mass=1,
        center_of_mass=np.array([0, 0, 0]),  # in segment coordinates system
        inertia=np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),  # in segment coordinates system
    )

    # Let's create a motion now
    # One can comment, one of the following Qi to pick, one initial condition
    # u as x-axis, w as z axis
    Qi = SegmentNaturalCoordinates.from_components(u=[1, 0, 0], rp=[0, 0, 0], rd=[0, -1, 0], w=[0, 0, 1])
    # u as y-axis
    # Qi = SegmentNaturalCoordinates.from_components(
    #     u=[0, 1, 0], rp=[0, 0, 0], rd=[0, 0, -1], w=[1, 0, 0]
    #     )
    # u as z-axis
    # Qi = SegmentNaturalCoordinates.from_components(u=[0, 0, 1], rp=[0, 0, 0], rd=[-1, 0, 0], w=[0, 1, 0])

    # Velocities are set to zero at t=0
    Qidot = SegmentNaturalVelocities.from_components(
        udot=np.array([0, 0, 0]), rpdot=np.array([0, 0, 0]), rddot=np.array([0, 0, 0]), wdot=np.array([0, 0, 0])
    )

    t_final = 2
    time_steps, all_states, dynamics = module.drop_the_box(
        my_segment=my_segment,
        Q_init=Qi,
        Qdot_init=Qidot,
        t_final=t_final,
    )

    defects, defects_dot, all_lambdas, center_of_mass = module.post_computations(
        my_segment, time_steps, all_states, dynamics
    )

    # Let's check the results
    TestUtils.assert_equal(
        time_steps[0:11],
        np.array(
            [
                0.0,
                0.02,
                0.04,
                0.06,
                0.08,
                0.1,
                0.12,
                0.14,
                0.16,
                0.18,
                0.2,
            ]
        ),
    )

    # only test on linux
    if platform == "linux":
        xdot, lambdas = dynamics(
            0,
            np.concatenate(
                (
                    SegmentNaturalCoordinates(np.linspace(0, 11, 12)).to_array(),
                    SegmentNaturalVelocities(np.linspace(0, 11, 12)).to_array(),
                ),
                axis=0,
            ),
        )

        TestUtils.assert_equal(
            xdot,
            np.array(
                [
                    0.0,
                    1.0,
                    2.0,
                    3.0,
                    4.0,
                    5.0,
                    6.0,
                    7.0,
                    8.0,
                    9.0,
                    1.0e01,
                    1.1e01,
                    -1.97372982e-16,
                    -1.0,
                    -2.0,
                    1.77635684e-15,
                    -1.59872116e-15,
                    -9.81,
                    -3.0,
                    -3.0,
                    -1.281e01,
                    -9.0,
                    -1.0e01,
                    -1.1e01,
                ]
            ),
        )
        TestUtils.assert_equal(
            lambdas, np.array([0.71294616, -1.27767695, -0.42589232, 2.41651543, 1.27767695, 0.71294616])
        )

    TestUtils.assert_equal(
        all_states[:, 0],
        np.array(
            [
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                -1.0,
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ]
        ),
    )
    TestUtils.assert_equal(
        all_states[:, 50],
        np.array(
            [
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                -4.905,
                0.0,
                -1.0,
                -4.905,
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                -9.81,
                0.0,
                0.0,
                -9.81,
                0.0,
                0.0,
                0.0,
            ]
        ),
    )
    TestUtils.assert_equal(
        all_states[:, -1],
        np.array(
            [
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                -19.62,
                0.0,
                -1.0,
                -19.62,
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                -19.62,
                0.0,
                0.0,
                -19.62,
                0.0,
                0.0,
                0.0,
            ]
        ),
    )

    TestUtils.assert_equal(defects[:, 0], np.zeros(6))
    TestUtils.assert_equal(defects[:, 50], np.zeros(6))
    TestUtils.assert_equal(defects[:, -1], np.zeros(6))

    TestUtils.assert_equal(defects_dot[:, 0], np.zeros(6))
    TestUtils.assert_equal(defects_dot[:, 50], np.zeros(6))
    TestUtils.assert_equal(defects_dot[:, -1], np.zeros(6))
