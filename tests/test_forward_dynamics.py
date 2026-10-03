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
                [-2.03419678e00],
                [5.09624302e00],
                [3.61761665e01],
                [1.28224025e01],
                [9.79648921e-01],
                [-2.36719638e04],
                [1.31102839e05],
                [4.25384100e04],
                [-1.90275320e05],
                [-1.27173482e05],
                [-2.12062385e04],
                [-2.03419678e00],
                [5.11459747e00],
                [2.63477802e01],
                [3.01737083e04],
                [4.80852153e03],
                [7.98849954e03],
                [-4.69809100e04],
                [-1.72377893e04],
                [-2.89913861e04],
                [1.85384731e04],
                [6.24764169e03],
                [-2.13155622e00],
                [5.14412625e00],
                [1.66058981e01],
                [4.53584613e03],
                [1.29980085e03],
                [6.09685306e02],
                [-8.95303508e02],
                [-1.70263297e03],
                [-1.00571629e04],
                [-2.83382703e03],
                [2.52427202e02],
                [-1.17700086e00],
                [2.88628091e00],
                [8.09979443e00],
                [3.21144304e02],
                [5.58279661e02],
                [4.24498807e02],
                [-8.44001705e02],
                [-7.37973667e02],
                [-6.12745457e02],
                [4.23020454e02],
                [1.58861466e02],
            ]
        )
    else:
        Qddot_expected = np.array(
            [
                -8.875152106000000e00,
                -1.060780657800000e00,
                7.997378626100000e00,
                -3.081796260500000e-01,
                1.090889824600000e-01,
                -5.107305552000000e-01,
                3.147498086700000e-01,
                1.587150331900000e02,
                -1.636923576000000e02,
                7.133136723800000e-02,
                1.465503981000000e01,
                -1.354616124000000e01,
                -1.154550842500000e02,
                -3.526304764500000e00,
                1.067411857500000e02,
                3.108998372100000e-01,
                1.587795241700000e02,
                -1.637051969700000e02,
                -5.500120227000000e01,
                4.388714176600000e01,
                -1.140079951300000e01,
                -5.139119375200000e01,
                -1.099816206700000e00,
                4.764823776800000e01,
                -3.110075901500000e01,
                -7.970060417800000e-01,
                2.820364389300000e01,
                -5.581759035900000e01,
                4.362403574200000e01,
                -1.137894329800000e01,
                -2.948286759800000e02,
                9.373473707800001e00,
                2.317054767900000e02,
                -2.528078850000000e02,
                -8.134806280199999e00,
                2.427119835200000e02,
                3.668244035900000e02,
                6.885679489200000e00,
                -3.545332385500000e02,
                -2.957231783000000e02,
                8.766504957300000e00,
                2.316034696300000e02,
                1.171672013100000e01,
                1.835106089500000e01,
                -6.637048794600000e01,
                -5.054327375900000e00,
                3.686418779700000e-01,
                3.863359369900000e00,
            ]
        )[:, np.newaxis]

        lagrange_multipliers_expected = np.array(
            [
                [-2.03419678e00],
                [5.09624302e00],
                [3.61761665e01],
                [1.28224025e01],
                [9.79648921e-01],
                [-2.36719638e04],
                [1.31102839e05],
                [4.25384100e04],
                [-1.90275320e05],
                [-1.27173482e05],
                [-2.12062385e04],
                [-2.03419678e00],
                [5.11459747e00],
                [2.63477802e01],
                [3.01737083e04],
                [4.80852153e03],
                [7.98849954e03],
                [-4.69809100e04],
                [-1.72377893e04],
                [-2.89913861e04],
                [1.85384731e04],
                [6.24764169e03],
                [-2.13155622e00],
                [5.14412625e00],
                [1.66058981e01],
                [4.53584613e03],
                [1.29980085e03],
                [6.09685306e02],
                [-8.95303508e02],
                [-1.70263297e03],
                [-1.00571629e04],
                [-2.83382703e03],
                [2.52427202e02],
                [-1.17700086e00],
                [2.88628091e00],
                [8.09979443e00],
                [3.21144304e02],
                [5.58279661e02],
                [4.24498807e02],
                [-8.44001705e02],
                [-7.37973667e02],
                [-6.12745457e02],
                [4.23020454e02],
                [1.58861466e02],
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


def test_actuated_3d_pendulum_example_runs(monkeypatch, tmp_path):
    # the example saves pendulum_3d.nmod in the cwd, so run it from a temporary directory
    monkeypatch.chdir(tmp_path)

    from bionc.bionc_numpy import NaturalCoordinates, NaturalVelocities

    bionc = TestUtils.bionc_folder()
    module = TestUtils.load_module(bionc + "/examples/forward_dynamics/actuated_3d_pendulum.py")

    model, time_steps, all_states, dynamics = module.apply_force_and_drop_pendulum(t_final=1)
    assert not np.isnan(all_states).any()

    # the pendulum is released at rest, pivot at the origin, center of mass at (0, -1, -0.5), gravity along -z:
    # the torque about the pivot is r_C x (0, 0, -m g) = (m g, 0, 0), so the rotation is about X only,
    # alpha = m g / I_pivot with I_pivot = I_xx + m (y_C^2 + z_C^2) = 0.01 + 1 * 1.25 = 1.26
    # and the acceleration of the center of mass is alpha_vec x r_C
    g, mass = 9.81, 1.0
    r_C = np.array([0, -1, -0.5])
    I_pivot = 0.01 + mass * (r_C[1] ** 2 + r_C[2] ** 2)
    alpha_vec = np.array([mass * g / I_pivot, 0, 0])
    a_C_expected = np.cross(alpha_vec, r_C)

    Q0 = NaturalCoordinates(all_states[: model.nb_Q, 0])
    Qdot0 = NaturalVelocities(all_states[model.nb_Q : model.nb_Q + model.nb_Qdot, 0])
    Qddot0, _ = model.forward_dynamics(Q0, Qdot0)
    a_C = model.segments["pendulum"].natural_center_of_mass.interpolate() @ np.array(Qddot0).reshape(-1)
    np.testing.assert_allclose(a_C, a_C_expected, atol=1e-3)

    # the angular motion stays about X: x-coordinates of rp, rd and the center of mass stay at zero
    # natural coordinates are [u, rp, rd, w]
    interpolation = model.segments["pendulum"].natural_center_of_mass.interpolate()
    np.testing.assert_allclose(all_states[3, :], 0, atol=1e-8)  # rp_x
    np.testing.assert_allclose(all_states[6, :], 0, atol=1e-8)  # rd_x
    com_x = interpolation[0, :] @ all_states[: model.nb_Q, :]
    np.testing.assert_allclose(com_x, 0, atol=1e-8)


def test_forward_dynamics_refuses_joint_generalized_forces(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    from bionc.bionc_numpy import NaturalCoordinates, NaturalVelocities

    bionc = TestUtils.bionc_folder()
    module = TestUtils.load_module(bionc + "/examples/forward_dynamics/actuated_3d_pendulum.py")
    model = module.build_3d_pendulum()

    Q = NaturalCoordinates(np.array([1, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 1], dtype=float))
    Qdot = NaturalVelocities(np.zeros(12))
    with pytest.raises(NotImplementedError):
        model.forward_dynamics(Q, Qdot, joint_generalized_forces=np.zeros(3))
