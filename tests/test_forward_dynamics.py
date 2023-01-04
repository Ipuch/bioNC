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
def test_forward_dynamics(bionc_type):
    from bionc.bionc_numpy import (
        SegmentNaturalVelocities,
        NaturalSegment,
        SegmentNaturalCoordinates,
    )

    bionc = TestUtils.bionc_folder()
    module = TestUtils.load_module(bionc + "/examples/forward_dynamics/drop_the_box.py")

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

    Qddot, lagrange_multipliers = model.forward_dynamics(Q_init, Qdot_init)

    if bionc_type == "numpy":
        Qddot_expected = np.array(
            [
                [2.88108716e-02],
                [6.25804959e-03],
                [-3.21767728e-03],
                [5.10424440e-13],
                [9.57610887e-14],
                [-4.36186081e-13],
                [4.44839026e-13],
                [9.55537316e-02],
                [-9.58728806e-02],
                [0.00000000e00],
                [3.50540988e-01],
                [-3.19647191e-01],
                [2.26997638e-01],
                [-7.90055666e-03],
                [-1.88522433e-01],
                [4.44839026e-13],
                [9.55537316e-02],
                [-9.58728806e-02],
                [-8.65821298e-01],
                [1.92275006e00],
                [-1.05756706e00],
                [3.70487810e-01],
                [-8.21970560e-03],
                [-3.32650903e-01],
                [4.12254502e-01],
                [3.80064148e-03],
                [-3.86757091e-01],
                [-8.65821298e-01],
                [1.92275006e00],
                [-1.05756706e00],
                [-1.11924273e00],
                [2.52514665e00],
                [-1.40686137e00],
                [5.55744674e-01],
                [3.48149254e-03],
                [-5.30885561e-01],
                [6.00223389e-01],
                [1.00777941e-02],
                [-5.82279727e-01],
                [-1.11924273e00],
                [2.52514665e00],
                [-1.40686137e00],
                [-1.03417859e00],
                [2.45057210e00],
                [-1.41767011e00],
                [7.43713561e-01],
                [9.75864519e-03],
                [-7.26408196e-01],
            ]
        )
        lagrange_multipliers_expected = np.array(
            [
                [-1.88164617e00],
                [4.29839323e00],
                [3.68214657e01],
                [1.52262782e01],
                [1.02346820e00],
                [-2.13441260e04],
                [1.29893131e05],
                [4.20471930e04],
                [-1.87961378e05],
                [-1.25679835e05],
                [-2.09577772e04],
                [-1.88164617e00],
                [4.30794860e00],
                [2.70018784e01],
                [7.62542077e02],
                [6.44601954e02],
                [5.07782581e02],
                [9.75349099e02],
                [-1.34519036e03],
                [2.15700951e03],
                [3.25731696e01],
                [3.84969070e02],
                [-1.96822830e00],
                [4.39511450e00],
                [1.71915819e01],
                [-2.23289349e03],
                [3.71093899e02],
                [1.55101341e02],
                [2.23927183e03],
                [-7.34264491e02],
                [9.05175016e03],
                [8.21547606e02],
                [2.35223022e02],
                [-1.12774914e00],
                [2.53260411e00],
                [8.40421950e00],
                [-2.55607946e02],
                [5.03456228e02],
                [2.55353069e02],
                [2.67405840e02],
                [-3.93386005e02],
                [-3.02535819e02],
                [-1.17022793e02],
                [1.75654711e01],
            ]
        )
    else:
        Qddot_expected = np.array(
            [
                [-3.17049930e00],
                [-3.17183087e00],
                [-7.45179050e00],
                [8.69048517e-01],
                [2.53704764e00],
                [-7.07367821e-01],
                [3.23965585e-01],
                [1.43240566e01],
                [-2.38622035e01],
                [-1.67091082e00],
                [-1.31686929e02],
                [1.18044824e02],
                [4.32087333e00],
                [-3.63601541e-01],
                [-5.62071571e00],
                [-4.62006593e-02],
                [1.44548913e01],
                [-2.40941739e01],
                [1.81049422e01],
                [5.59138990e00],
                [-3.12017629e01],
                [-8.37144660e01],
                [-3.50826164e00],
                [7.59451220e01],
                [-8.32493231e01],
                [-2.25625411e00],
                [8.01679070e01],
                [1.93727277e01],
                [6.22982526e00],
                [-3.09457500e01],
                [4.90638751e01],
                [6.04192306e00],
                [-6.41352877e01],
                [-8.76640663e01],
                [-2.15307192e00],
                [8.51377465e01],
                [3.15586319e02],
                [4.64892848e00],
                [-3.06757218e02],
                [4.82819438e01],
                [6.77525817e00],
                [-6.51515001e01],
                [3.19858993e02],
                [1.30605183e01],
                [-3.24919516e02],
                [-8.69633111e01],
                [-9.70274983e-01],
                [8.39994165e01],
            ]
        )
        lagrange_multipliers_expected = np.array(
            [
                [4.17088539e01],
                [3.15933935e01],
                [-3.05202267e01],
                [-1.08805562e02],
                [-3.78573397e02],
                [-2.54578665e16],
                [1.71937901e17],
                [5.73126337e16],
                [-2.57906851e17],
                [-1.71937901e17],
                [-2.86563168e16],
                [4.27859502e01],
                [3.26823294e01],
                [-4.65235954e01],
                [-4.47783041e16],
                [-6.39690058e15],
                [9.62650851e15],
                [-3.10891374e16],
                [-1.03630458e16],
                [2.03357770e17],
                [7.58674414e16],
                [8.37997318e15],
                [4.21843813e01],
                [1.49094379e01],
                [-3.52372496e01],
                [-6.22297986e16],
                [-8.88997123e15],
                [-5.49081785e15],
                [2.02308300e16],
                [6.74361001e15],
                [1.87458050e17],
                [4.19989686e16],
                [1.07318061e15],
                [2.29315148e01],
                [7.96642666e00],
                [-2.36214875e01],
                [2.96661799e16],
                [4.23802569e15],
                [4.59626513e15],
                [-2.75775908e16],
                [-9.19253025e15],
                [-6.24652434e16],
                [-2.08858911e15],
                [2.47725228e15],
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
            lagrange_multipliers,
            lagrange_multipliers_expected,
            decimal=3,
            squeeze=False,
            expand=False,
        )
