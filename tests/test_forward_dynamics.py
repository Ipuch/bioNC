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
        Qddot_expected =         np.array([[ 2.88108716e-02],
                      [ 6.25804959e-03],
                      [-3.21767728e-03],
                      [ 5.10424440e-13],
                      [ 9.57610887e-14],
                      [-4.36186081e-13],
                      [ 4.44839026e-13],
                      [ 9.55537316e-02],
                      [-9.58728806e-02],
                      [ 0.00000000e+00],
                      [ 3.50540988e-01],
                      [-3.19647191e-01],
                      [ 2.26997638e-01],
                      [-7.90055666e-03],
                      [-1.88522433e-01],
                      [ 4.44839026e-13],
                      [ 9.55537316e-02],
                      [-9.58728806e-02],
                      [-8.65821298e-01],
                      [ 1.92275006e+00],
                      [-1.05756706e+00],
                      [ 3.70487810e-01],
                      [-8.21970560e-03],
                      [-3.32650903e-01],
                      [ 4.12254502e-01],
                      [ 3.80064148e-03],
                      [-3.86757091e-01],
                      [-8.65821298e-01],
                      [ 1.92275006e+00],
                      [-1.05756706e+00],
                      [-1.11924273e+00],
                      [ 2.52514665e+00],
                      [-1.40686137e+00],
                      [ 5.55744674e-01],
                      [ 3.48149254e-03],
                      [-5.30885561e-01],
                      [ 6.00223389e-01],
                      [ 1.00777941e-02],
                      [-5.82279727e-01],
                      [-1.11924273e+00],
                      [ 2.52514665e+00],
                      [-1.40686137e+00],
                      [-1.03417859e+00],
                      [ 2.45057210e+00],
                      [-1.41767011e+00],
                      [ 7.43713561e-01],
                      [ 9.75864519e-03],
                      [-7.26408196e-01]])
        lagrange_multipliers_expected = np.array([[-1.88164617e+00],
       [ 4.29839323e+00],
       [ 3.68214657e+01],
       [ 1.52262782e+01],
       [ 1.02346820e+00],
       [-2.13441260e+04],
       [ 1.29893131e+05],
       [ 4.20471930e+04],
       [-1.87961378e+05],
       [-1.25679835e+05],
       [-2.09577772e+04],
       [-1.88164617e+00],
       [ 4.30794860e+00],
       [ 2.70018784e+01],
       [ 7.62542077e+02],
       [ 6.44601954e+02],
       [ 5.07782581e+02],
       [ 9.75349099e+02],
       [-1.34519036e+03],
       [ 2.15700951e+03],
       [ 3.25731696e+01],
       [ 3.84969070e+02],
       [-1.96822830e+00],
       [ 4.39511450e+00],
       [ 1.71915819e+01],
       [-2.23289349e+03],
       [ 3.71093899e+02],
       [ 1.55101341e+02],
       [ 2.23927183e+03],
       [-7.34264491e+02],
       [ 9.05175016e+03],
       [ 8.21547606e+02],
       [ 2.35223022e+02],
       [-1.12774914e+00],
       [ 2.53260411e+00],
       [ 8.40421950e+00],
       [-2.55607946e+02],
       [ 5.03456228e+02],
       [ 2.55353069e+02],
       [ 2.67405840e+02],
       [-3.93386005e+02],
       [-3.02535819e+02],
       [-1.17022793e+02],
       [ 1.75654711e+01]])
    else:
        Qddot_expected = np.array([[-3.17049930e+00],
       [-3.17183087e+00],
       [-7.45179050e+00],
       [ 8.69048517e-01],
       [ 2.53704764e+00],
       [-7.07367821e-01],
       [ 3.23965585e-01],
       [ 1.43240566e+01],
       [-2.38622035e+01],
       [-1.67091082e+00],
       [-1.31686929e+02],
       [ 1.18044824e+02],
       [ 4.32087333e+00],
       [-3.63601541e-01],
       [-5.62071571e+00],
       [-4.62006593e-02],
       [ 1.44548913e+01],
       [-2.40941739e+01],
       [ 1.81049422e+01],
       [ 5.59138990e+00],
       [-3.12017629e+01],
       [-8.37144660e+01],
       [-3.50826164e+00],
       [ 7.59451220e+01],
       [-8.32493231e+01],
       [-2.25625411e+00],
       [ 8.01679070e+01],
       [ 1.93727277e+01],
       [ 6.22982526e+00],
       [-3.09457500e+01],
       [ 4.90638751e+01],
       [ 6.04192306e+00],
       [-6.41352877e+01],
       [-8.76640663e+01],
       [-2.15307192e+00],
       [ 8.51377465e+01],
       [ 3.15586319e+02],
       [ 4.64892848e+00],
       [-3.06757218e+02],
       [ 4.82819438e+01],
       [ 6.77525817e+00],
       [-6.51515001e+01],
       [ 3.19858993e+02],
       [ 1.30605183e+01],
       [-3.24919516e+02],
       [-8.69633111e+01],
       [-9.70274983e-01],
       [ 8.39994165e+01]])
        lagrange_multipliers_expected = np.array([[ 4.17088539e+01],
       [ 3.15933935e+01],
       [-3.05202267e+01],
       [-1.08805562e+02],
       [-3.78573397e+02],
       [-2.54578665e+16],
       [ 1.71937901e+17],
       [ 5.73126337e+16],
       [-2.57906851e+17],
       [-1.71937901e+17],
       [-2.86563168e+16],
       [ 4.27859502e+01],
       [ 3.26823294e+01],
       [-4.65235954e+01],
       [-4.47783041e+16],
       [-6.39690058e+15],
       [ 9.62650851e+15],
       [-3.10891374e+16],
       [-1.03630458e+16],
       [ 2.03357770e+17],
       [ 7.58674414e+16],
       [ 8.37997318e+15],
       [ 4.21843813e+01],
       [ 1.49094379e+01],
       [-3.52372496e+01],
       [-6.22297986e+16],
       [-8.88997123e+15],
       [-5.49081785e+15],
       [ 2.02308300e+16],
       [ 6.74361001e+15],
       [ 1.87458050e+17],
       [ 4.19989686e+16],
       [ 1.07318061e+15],
       [ 2.29315148e+01],
       [ 7.96642666e+00],
       [-2.36214875e+01],
       [ 2.96661799e+16],
       [ 4.23802569e+15],
       [ 4.59626513e+15],
       [-2.75775908e+16],
       [-9.19253025e+15],
       [-6.24652434e+16],
       [-2.08858911e+15],
       [ 2.47725228e+15]])

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
