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
                [0.05107325],
                [0.02039384],
                [-0.01028557],
                [-1.25304678e-12],
                [-1.19321597e-12],
                [2.26060515e-12],
                [-2.10285707e-12],
                [0.18354455],
                [-0.18386369],
                [0.00000000e00],
                [0.67317397],
                [-0.6129499],
                [0.4183751],
                [0.01996831],
                [-0.37843849],
                [-2.10285707e-12],
                [0.18354455],
                [-0.18386369],
                [-0.97359434],
                [2.31427777],
                [-1.34132173],
                [0.6938515],
                [0.01964916],
                [-0.65455318],
                [0.78567696],
                [0.01954277],
                [-0.74659141],
                [-0.97359434],
                [2.31427777],
                [-1.34132173],
                [-1.16398421],
                [2.87860206],
                [-1.71557529],
                [1.06115335],
                [0.01922363],
                [-1.0227061],
                [1.15297882],
                [0.01911724],
                [-1.11474433],
                [-1.16398421],
                [2.87860206],
                [-1.71557529],
                [-1.03381769],
                [2.80181356],
                [-1.76927246],
                [1.42845521],
                [0.01879809],
                [-1.39085902],
            ]
        )
        lagrange_multipliers_expected = np.array(
            [
                [-2.03419678e+00],
                [5.09624302e+00],
                [3.61761665e+01],
                [1.28224025e+01],
                [9.79648921e-01],
                [-2.36719638e+04],
                [1.31102839e+05],
                [4.25384100e+04],
                [-1.90275320e+05],
                [-1.27173482e+05],
                [-2.12062385e+04],
                [-2.03419678e+00],
                [5.11459747e+00],
                [2.63477802e+01],
                [3.01737083e+04],
                [4.80852153e+03],
                [7.98849954e+03],
                [-4.69809100e+04],
                [-1.72377893e+04],
                [-2.89913861e+04],
                [1.85384731e+04],
                [6.24764169e+03],
                [-2.13155622e+00],
                [5.14412625e+00],
                [1.66058981e+01],
                [4.53584613e+03],
                [1.29980085e+03],
                [6.09685306e+02],
                [-8.95303508e+02],
                [-1.70263297e+03],
                [-1.00571629e+04],
                [-2.83382703e+03],
                [2.52427202e+02],
                [-1.17700086e+00],
                [2.88628091e+00],
                [8.09979443e+00],
                [3.21144304e+02],
                [5.58279661e+02],
                [4.24498807e+02],
                [-8.44001705e+02],
                [-7.37973667e+02],
                [-6.12745457e+02],
                [4.23020454e+02],
                [1.58861466e+02],
            ]
        )
    else:
        Qddot_expected = np.array(
            [
                -8.875152106000000e+00,
                -1.060780657800000e+00,
                7.997378626100000e+00,
                -3.081796260500000e-01,
                1.090889824600000e-01,
                -5.107305552000000e-01,
                3.147498086700000e-01,
                1.587150331900000e+02,
                -1.636923576000000e+02,
                7.133136723800000e-02,
                1.465503981000000e+01,
                -1.354616124000000e+01,
                -1.154550842500000e+02,
                -3.526304764500000e+00,
                1.067411857500000e+02,
                3.108998372100000e-01,
                1.587795241700000e+02,
                -1.637051969700000e+02,
                -5.500120227000000e+01,
                4.388714176600000e+01,
                -1.140079951300000e+01,
                -5.139119375200000e+01,
                -1.099816206700000e+00,
                4.764823776800000e+01,
                -3.110075901500000e+01,
                -7.970060417800000e-01,
                2.820364389300000e+01,
                -5.581759035900000e+01,
                4.362403574200000e+01,
                -1.137894329800000e+01,
                -2.948286759800000e+02,
                9.373473707800001e+00,
                2.317054767900000e+02,
                -2.528078850000000e+02,
                -8.134806280199999e+00,
                2.427119835200000e+02,
                3.668244035900000e+02,
                6.885679489200000e+00,
                -3.545332385500000e+02,
                -2.957231783000000e+02,
                8.766504957300000e+00,
                2.316034696300000e+02,
                1.171672013100000e+01,
                1.835106089500000e+01,
                -6.637048794600000e+01,
                -5.054327375900000e+00,
                3.686418779700000e-01,
                3.863359369900000e+00,
            ]
        )[:, np.newaxis]

        lagrange_multipliers_expected = np.array(
            [
                [-2.03419678e+00],
                [5.09624302e+00],
                [3.61761665e+01],
                [1.28224025e+01],
                [9.79648921e-01],
                [-2.36719638e+04],
                [1.31102839e+05],
                [4.25384100e+04],
                [-1.90275320e+05],
                [-1.27173482e+05],
                [-2.12062385e+04],
                [-2.03419678e+00],
                [5.11459747e+00],
                [2.63477802e+01],
                [3.01737083e+04],
                [4.80852153e+03],
                [7.98849954e+03],
                [-4.69809100e+04],
                [-1.72377893e+04],
                [-2.89913861e+04],
                [1.85384731e+04],
                [6.24764169e+03],
                [-2.13155622e+00],
                [5.14412625e+00],
                [1.66058981e+01],
                [4.53584613e+03],
                [1.29980085e+03],
                [6.09685306e+02],
                [-8.95303508e+02],
                [-1.70263297e+03],
                [-1.00571629e+04],
                [-2.83382703e+03],
                [2.52427202e+02],
                [-1.17700086e+00],
                [2.88628091e+00],
                [8.09979443e+00],
                [3.21144304e+02],
                [5.58279661e+02],
                [4.24498807e+02],
                [-8.44001705e+02],
                [-7.37973667e+02],
                [-6.12745457e+02],
                [4.23020454e+02],
                [1.58861466e+02],
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
