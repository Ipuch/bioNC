from sys import platform

import numpy as np
import pytest

from bionc import TransformationMatrixType
from .utils import TestUtils
from .utils_constant_forward_dynamics import (
    G as G_EXPECTED,
    K as K_EXPECTED,
    PHIR_DOT as PHIR_DOT_EXPECTED,
    PHIJ_DOT as PHIJ_DOT_EXPECTED,
    KDOT as KDOT_EXPECTED,
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
    Phirdot = model.rigid_body_constraint_jacobian_derivative(Qdot_init)
    Phijdot = model.joint_constraints_jacobian_derivative(Qdot_init)
    Kdot = model.holonomic_constraints_jacobian_derivative(Qdot_init)
    augmented_mass_matrix = model.augmented_mass_matrix(Q_init)
    TestUtils.assert_equal(G, G_EXPECTED)
    TestUtils.assert_equal(K, K_EXPECTED)
    TestUtils.assert_equal(Phirdot, PHIR_DOT_EXPECTED)
    TestUtils.assert_equal(Phijdot, PHIJ_DOT_EXPECTED)
    TestUtils.assert_equal(Kdot, KDOT_EXPECTED)
    TestUtils.assert_equal(-Kdot @ Qdot_init, -KDOT_EXPECTED @ Qdot_init)

    TestUtils.assert_equal(augmented_mass_matrix, AUGMENTED_MASS_MATRIX_EXPECTED)
    Qddot, lagrange_multipliers = model.forward_dynamics(Q_init, Qdot_init)

    if bionc_type == "numpy":
        Qddot_expected = np.array(
            [
                [2.58507235e-02],
                [1.32910061e-02],
                [-6.73415550e-03],
                [-1.25304678e-12],
                [-1.19321597e-12],
                [2.26060515e-12],
                [-2.10285707e-12],
                [9.72227221e-02],
                [-9.75418711e-02],
                [0.00000000e00],
                [3.56660619e-01],
                [-3.25210492e-01],
                [2.30977136e-01],
                [-8.07093002e-03],
                [-1.91775227e-01],
                [-2.10285707e-12],
                [9.72227221e-02],
                [-9.75418711e-02],
                [-8.64507625e-01],
                [1.92346070e00],
                [-1.05959137e00],
                [3.76970793e-01],
                [-8.39007896e-03],
                [-3.38407183e-01],
                [4.19452730e-01],
                [3.86877079e-03],
                [-3.93467118e-01],
                [-8.64507625e-01],
                [1.92346070e00],
                [-1.05959137e00],
                [-1.11693623e00],
                [2.52554063e00],
                [-1.40956184e00],
                [5.65446387e-01],
                [3.54962185e-03],
                [-5.40099073e-01],
                [6.10693690e-01],
                [1.02777376e-02],
                [-5.92393641e-01],
                [-1.11693623e00],
                [2.52554063e00],
                [-1.40956184e00],
                [-1.03105715e00],
                [2.45100519e00],
                [-1.42122463e00],
                [7.56687348e-01],
                [9.95858867e-03],
                [-7.39025597e-01],
            ]
        )
        lagrange_multipliers_expected = np.array(
            [
                [-1.87833814e00],
                [4.30112353e00],
                [3.68154274e01],
                [1.52121962e01],
                [1.04137695e00],
                [7.46735734e04],
                [-4.52558461e05],
                [-1.52099897e05],
                [6.85711724e05],
                [4.56769278e05],
                [7.61168860e04],
                [-1.87833814e00],
                [4.31084580e00],
                [2.69956732e01],
                [1.55504808e04],
                [2.75571749e03],
                [8.90108082e03],
                [-5.64761788e04],
                [-2.04910795e04],
                [3.65730303e04],
                [4.26952118e04],
                [8.90254551e03],
                [-1.96478890e00],
                [4.39624687e00],
                [1.71870101e01],
                [1.42883176e04],
                [2.72970742e03],
                [2.47266579e03],
                [-1.15994530e04],
                [-5.34228157e03],
                [-2.80180098e04],
                [-1.86157536e03],
                [1.36006250e03],
                [-1.12552414e00],
                [2.53299417e00],
                [8.40160443e00],
                [-4.42723787e02],
                [4.75561376e02],
                [2.59249586e02],
                [2.30428206e02],
                [-4.02065912e02],
                [4.06612288e02],
                [1.06911074e02],
                [3.59386244e01],
            ]
        )
    else:
        Qddot_expected = np.array(
            [
                -3.127055953974438e01,
                -7.634299328825800e00,
                1.752921988988392e01,
                -1.969305277148914e-01,
                -6.189472049527404e-01,
                1.459175592599798e00,
                1.966307999466519e-01,
                -1.610980906256362e02,
                1.252492953328899e02,
                5.010638437812892e-01,
                -1.705533528914464e02,
                1.552372362596975e02,
                -4.861082024818751e01,
                -4.915061574278707e00,
                4.278657313089258e01,
                -8.842506628224632e-01,
                -1.608466301525380e02,
                1.252954111767938e02,
                -1.088127132577218e02,
                -7.601776907913624e01,
                1.405623155123574e02,
                -2.707370781658236e02,
                -5.095478601674613e00,
                2.610192605773854e02,
                -3.249880396360093e00,
                1.145875296973965e00,
                5.802542533327520e00,
                -1.105323125874475e02,
                -7.525106573481152e01,
                1.393220338349717e02,
                -3.169088444362398e02,
                -5.193860625852796e01,
                3.121301933629575e02,
                4.590988198732746e01,
                1.207461398512568e00,
                -4.525386632519989e01,
                4.678842037268220e01,
                1.385951433885823e00,
                -4.367955951750544e01,
                -3.180333242786281e02,
                -5.057470816674911e01,
                3.123225449956934e02,
                -4.947718180781757e02,
                -5.965532390589068e01,
                4.901190785845675e02,
                3.066540503567427e02,
                4.229444121156359e00,
                -2.999039561101401e02,
            ]
        )[:, np.newaxis]

        lagrange_multipliers_expected = np.array(
            [
                [-1.87833814e00],
                [4.30112353e00],
                [3.68154274e01],
                [1.52121962e01],
                [1.04137695e00],
                [7.46735734e04],
                [-4.52558461e05],
                [-1.52099897e05],
                [6.85711724e05],
                [4.56769278e05],
                [7.61168860e04],
                [-1.87833814e00],
                [4.31084580e00],
                [2.69956732e01],
                [1.55504808e04],
                [2.75571749e03],
                [8.90108082e03],
                [-5.64761788e04],
                [-2.04910795e04],
                [3.65730303e04],
                [4.26952118e04],
                [8.90254551e03],
                [-1.96478890e00],
                [4.39624687e00],
                [1.71870101e01],
                [1.42883176e04],
                [2.72970742e03],
                [2.47266579e03],
                [-1.15994530e04],
                [-5.34228157e03],
                [-2.80180098e04],
                [-1.86157536e03],
                [1.36006250e03],
                [-1.12552414e00],
                [2.53299417e00],
                [8.40160443e00],
                [-4.42723787e02],
                [4.75561376e02],
                [2.59249586e02],
                [2.30428206e02],
                [-4.02065912e02],
                [4.06612288e02],
                [1.06911074e02],
                [3.59386244e01],
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
