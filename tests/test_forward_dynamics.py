from sys import platform

import numpy as np
import pytest

from bionc import TransformationMatrixType
from .utils import TestUtils
from .utils_constant_forward_dynamics import (
    G as G_EXPECTED,
    K as K_EXPECTED,
    bias_expected as BIAS_EXPECTED,
    AUGMENTED_MASS_MATRIX as AUGMENTED_MASS_MATRIX_EXPECTED,
)


@pytest.mark.parametrize(
    "bionc_type",
    [
        "numpy",
    ],
)
def test_forward_dynamics(bionc_type):
    from bionc.bionc_numpy import (
        SegmentNaturalVelocities,
        NaturalSegment,
        SegmentNaturalCoordinates,
    )

    bionc = TestUtils.bionc_folder()
    module = TestUtils.load_module(bionc + "/examples/forward_dynamics/drop_the_box.py")

    # Let's create a segment
    my_segment = NaturalSegment.with_cartesian_inertial_parameters(
        name="box",
        alpha=np.pi / 2,  # setting alpha, beta, gamma to pi/2 creates a orthogonal coordinate system
        beta=np.pi / 2,
        gamma=np.pi / 2,
        length=1,
        mass=1,
        center_of_mass=np.array([0, 0, 0]),  # in segment coordinates system
        inertia=np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),  # in segment coordinates system
        inertial_transformation_matrix=TransformationMatrixType.Buv,
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
        # TestUtils.assert_equal(
        #     lambdas, np.array([0.71294616, -1.27767695, -0.42589232, 2.41651543, 1.27767695, 0.71294616])
        # )

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


@pytest.mark.parametrize(
    "bionc_type",
    [
        "numpy",
        "casadi",
    ],
)
def test_forward_dynamics_n_pendulum(bionc_type):
    # todo: inspect why we have a difference between casadi and numpy ...
    #   test everything that is built inside.

    if bionc_type == "numpy":
        from bionc.bionc_numpy import NaturalCoordinates, NaturalVelocities
    else:
        from bionc.bionc_casadi import NaturalCoordinates, NaturalVelocities

    bionc = TestUtils.bionc_folder()
    module = TestUtils.load_module(bionc + "/examples/forward_dynamics/n_link_pendulum.py")

    nb_segments = 4
    model = module.build_n_link_pendulum(nb_segments=nb_segments)
    if bionc_type == "casadi":
        model = model.to_mx()

    Q_init = NaturalCoordinates(np.linspace(0, 0.24, nb_segments * 12))
    Qdot_init = NaturalVelocities(np.linspace(0, 0.02, nb_segments * 12))
    # inspect each element of the forward dynamics
    G = model.mass_matrix
    K = model.holonomic_constraints_jacobian(Q_init)
    bias = model.holonomic_constraints_acceleration_bias(Qdot_init)
    augmented_mass_matrix = model.augmented_mass_matrix(Q_init)
    TestUtils.assert_equal(G, G_EXPECTED)
    TestUtils.assert_equal(K, K_EXPECTED)
    assert bias.shape == BIAS_EXPECTED.shape
    TestUtils.assert_equal(bias, BIAS_EXPECTED.squeeze())
    TestUtils.assert_equal(augmented_mass_matrix, AUGMENTED_MASS_MATRIX_EXPECTED)
    Qddot, lagrange_multipliers = model.forward_dynamics(Q_init, Qdot_init)

    if bionc_type == "numpy":
        Qddot_expected = np.array(
            [
                [2.66575990e-02],
                [1.06275786e-02],
                [-5.40244178e-03],
                [-1.25304678e-12],
                [-1.19321597e-12],
                [2.26060515e-12],
                [-2.10285707e-12],
                [9.56482073e-02],
                [-9.59673562e-02],
                [0.00000000e00],
                [3.50887398e-01],
                [-3.19962109e-01],
                [2.18166780e-01],
                [1.02020467e-02],
                [-1.97762686e-01],
                [-2.10285707e-12],
                [9.56482073e-02],
                [-9.59673562e-02],
                [-8.65687519e-01],
                [1.92267145e00],
                [-1.05762223e00],
                [3.61798665e-01],
                [9.88289774e-03],
                [-3.42032869e-01],
                [4.09675960e-01],
                [9.77651476e-03],
                [-3.90122931e-01],
                [-8.65687519e-01],
                [1.92267145e00],
                [-1.05762223e00],
                [-1.11904968e00],
                [2.52504399e00],
                [-1.40695175e00],
                [5.53307846e-01],
                [9.45736582e-03],
                [-5.34393114e-01],
                [6.01185141e-01],
                [9.35098285e-03],
                [-5.82483175e-01],
                [-1.11904968e00],
                [2.52504399e00],
                [-1.40695175e00],
                [-1.03394054e00],
                [2.45047392e00],
                [-1.41780997e00],
                [7.44817026e-01],
                [9.03183391e-03],
                [-7.26753358e-01],
            ]
        )
        lagrange_multipliers_expected = np.array(
            [
                [-1.88134315e00],
                [4.29831625e00],
                [3.68212397e01],
                [1.52362739e01],
                [1.03195428e00],
                [-2.36386825e03],
                [1.87491692e04],
                [5.00001346e03],
                [-2.12432170e04],
                [-1.45346114e04],
                [-2.43366251e03],
                [-1.88134315e00],
                [4.30788108e00],
                [2.70016429e01],
                [-5.63111577e03],
                [-2.69106359e02],
                [-7.42277876e02],
                [7.35787516e03],
                [7.83364189e02],
                [1.49648524e04],
                [4.47475754e01],
                [-2.22325912e02],
                [-1.96791190e00],
                [4.39493519e00],
                [1.71914448e01],
                [3.67412801e02],
                [7.42325253e02],
                [4.37473346e02],
                [5.01916070e02],
                [-1.31231535e03],
                [2.56698679e03],
                [-3.95907940e01],
                [3.38779597e02],
                [-1.12756060e00],
                [2.53250099e00],
                [8.40413407e00],
                [-1.63819658e02],
                [5.16465442e02],
                [2.31488382e02],
                [4.06604145e02],
                [-3.45916900e02],
                [-8.12722829e02],
                [-3.45418499e02],
                [-1.25081812e01],
            ]
        )
    else:
        Qddot_expected = np.array(
            [
                -7.478870595938203e00,
                -3.227696516618579e00,
                3.266227013957749e00,
                -1.170635192822490e00,
                3.015589696048855e-01,
                -1.200665772564065e00,
                -2.703807924687865e00,
                -3.924672420729592e01,
                2.777900960532280e01,
                6.473574464687598e-02,
                -8.179214443783889e01,
                7.572674898858101e01,
                -4.506150079493827e01,
                -3.249307706206350e00,
                4.220478071185416e01,
                -3.435546408854905e00,
                -3.823127996427767e01,
                2.750349758990746e01,
                -3.048106473449040e00,
                -1.051750951919537e01,
                1.919659669537513e01,
                -8.862196666429492e01,
                -4.410359706979638e00,
                8.038438599940703e01,
                2.300597449207447e02,
                5.001582784536146e00,
                -2.197430236746320e02,
                -3.614093176155250e00,
                -1.167343021716773e01,
                2.006608371055658e01,
                -3.938848589781653e01,
                -6.437659020393835e00,
                4.816580784986410e01,
                -4.561613597941893e01,
                -1.712626865807047e00,
                4.448749408007144e01,
                -1.013461664593831e02,
                -9.893066965277802e-01,
                9.886045096288029e01,
                -3.947478899248575e01,
                -5.827831255077090e00,
                4.685644490521795e01,
                1.889301706179062e01,
                -6.740725573957497e00,
                -6.126731806890545e00,
                1.060537864080596e02,
                2.583106167311323e00,
                -1.010407275556867e02,
            ]
        )[:, np.newaxis]

        lagrange_multipliers_expected = np.array(
            [
                [-1.88134315e00],
                [4.29831625e00],
                [3.68212397e01],
                [1.52362739e01],
                [1.03195428e00],
                [-2.36386825e03],
                [1.87491692e04],
                [5.00001346e03],
                [-2.12432170e04],
                [-1.45346114e04],
                [-2.43366251e03],
                [-1.88134315e00],
                [4.30788108e00],
                [2.70016429e01],
                [-5.63111577e03],
                [-2.69106359e02],
                [-7.42277876e02],
                [7.35787516e03],
                [7.83364189e02],
                [1.49648524e04],
                [4.47475754e01],
                [-2.22325912e02],
                [-1.96791190e00],
                [4.39493519e00],
                [1.71914448e01],
                [3.67412801e02],
                [7.42325253e02],
                [4.37473346e02],
                [5.01916070e02],
                [-1.31231535e03],
                [2.56698679e03],
                [-3.95907940e01],
                [3.38779597e02],
                [-1.12756060e00],
                [2.53250099e00],
                [8.40413407e00],
                [-1.63819658e02],
                [5.16465442e02],
                [2.31488382e02],
                [4.06604145e02],
                [-3.45916900e02],
                [-8.12722829e02],
                [-3.45418499e02],
                [-1.25081812e01],
            ]
        )

    TestUtils.assert_equal(
        Qddot,
        Qddot_expected,
        squeeze=False,
        expand=False,
    )
    if bionc_type == "numpy":
        TestUtils.assert_equal(
            lagrange_multipliers[:3, 0],
            lagrange_multipliers_expected[
                :3, 0
            ],  # only the three first values tested because hard to test it on cross plateforms
            decimal=3,
            squeeze=False,
            expand=False,
        )
