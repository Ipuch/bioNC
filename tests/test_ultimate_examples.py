import numpy as np

from .utils import TestUtils


def test_play_with_joints_constant_length():
    bionc = TestUtils.bionc_folder()
    module = TestUtils.load_module(bionc + "/examples/play_with_joints/constant_length.py")

    module.main("hanged", False)
    module.main("ready_to_swing", False)


def test_play_with_joints_two_constant_length():
    bionc = TestUtils.bionc_folder()
    module = TestUtils.load_module(bionc + "/examples/play_with_joints/two_constant_length.py")

    module.main("hanged", False)
    module.main("ready_to_swing", False)


def test_inverse_kinematics_one_frame():
    # import the lower limb model
    bionc = TestUtils.bionc_folder()
    module = TestUtils.load_module(bionc + "/examples/model_creation/right_side_lower_limb.py")
    module_ik = TestUtils.load_module(bionc + "/examples/inverse_kinematics/one_frame_inverse_kinematics.py")

    # Generate c3d file
    filename = module.generate_c3d_file()
    # Generate model
    model = module.model_creation_from_measured_data(filename)

    module_ik.main(model, filename, show_animation=False)


def test_forward_dynamics_with_force():
    bionc = TestUtils.bionc_folder()
    module_fd = TestUtils.load_module(bionc + "/examples/forward_dynamics/pendulum_with_force.py")

    model, all_states = module_fd.main(mode="moment_equilibrium")

    np.testing.assert_almost_equal(
        all_states[:, -1],
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

    model, all_states = module_fd.main(mode="force_equilibrium")

    np.testing.assert_almost_equal(
        all_states[:, -1],
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

    model, all_states = module_fd.main(mode="no_equilibrium")

    np.testing.assert_almost_equal(
        all_states[:, -1],
        np.array(
            [
                1.00000000e00,
                -1.25752847e-15,
                -2.58263690e-16,
                4.26089219e-16,
                1.62086166e-16,
                1.14800688e-15,
                6.25950500e-16,
                9.99998625e-01,
                -1.29093221e-03,
                -1.02667903e-17,
                -1.29099764e-03,
                -9.99999167e-01,
                -5.60086390e-31,
                -4.23925536e-16,
                -1.04410614e-16,
                1.25862608e-16,
                1.74366566e-17,
                2.62623414e-16,
                2.31835366e-16,
                7.25566569e-05,
                5.62890649e-02,
                -1.23324768e-18,
                5.62889248e-02,
                -7.25601194e-05,
            ]
        ),
    )


def test_forward_dynamics_with_force_double_pendulum():
    bionc = TestUtils.bionc_folder()
    module_fd = TestUtils.load_module(bionc + "/examples/forward_dynamics/double_pendulum_with_force.py")

    model, all_states = module_fd.main(mode="force_equilibrium")

    np.testing.assert_almost_equal(
        all_states[:, -1],
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
                1.0,
                0.0,
                0.0,
                0.0,
                -1.0,
                0.0,
                0.0,
                -2.0,
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

    model, all_states = module_fd.main(mode="no_equilibrium")

    np.testing.assert_almost_equal(
        all_states[:, -1],
        np.array(
            [
                1.00000000e00,
                -2.52239560e-16,
                -1.73873256e-16,
                2.31152175e-16,
                2.95333882e-16,
                7.08351108e-16,
                9.04104726e-17,
                -8.52201460e-01,
                5.23213380e-01,
                -2.07680249e-18,
                5.23213359e-01,
                8.52201725e-01,
                1.00000000e00,
                1.52225682e-15,
                -2.28291765e-15,
                9.04104726e-17,
                -8.52201460e-01,
                5.23213380e-01,
                -1.31951335e-15,
                -1.71527649e-01,
                -2.09372903e-01,
                -5.50765736e-16,
                -7.32586547e-01,
                -6.80673895e-01,
                -1.59966942e-32,
                2.04722487e-18,
                -9.47976382e-17,
                4.42267945e-17,
                4.07752930e-17,
                1.31157361e-16,
                2.44895674e-17,
                -1.64051562e-01,
                -2.67204615e-01,
                1.19086981e-19,
                -2.67204780e-01,
                1.64051606e-01,
                -1.96343239e-30,
                -8.13124634e-16,
                -1.40222374e-15,
                2.44895674e-17,
                -1.64051562e-01,
                -2.67204615e-01,
                -4.94674778e-16,
                -5.55382157e-01,
                -6.30804712e-01,
                -1.46529245e-16,
                -3.63600172e-01,
                3.91330768e-01,
            ]
        ),
    )
