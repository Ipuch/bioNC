import numpy as np
import pytest

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


@pytest.mark.parametrize(
    "mode",
    ["x_revolute", "y_revolute", "z_revolute"],
)
def test_single_pendulum_dofs(mode):
    bionc = TestUtils.bionc_folder()
    module_fd = TestUtils.load_module(bionc + "/examples/forward_dynamics/pendulum.py")

    model, all_states = module_fd.main(mode=mode, show_results=False)

    Qf = all_states[:12, -1]
    if mode == "x_revolute":
        TestUtils.assert_equal(
            Qf,
            np.array(
                [
                    1.000000e00,
                    0.000000e00,
                    0.000000e00,
                    0.000000e00,
                    -3.297850e-15,
                    -1.641582e-14,
                    0.000000e00,
                    -5.151967e-01,
                    8.568669e-01,
                    0.000000e00,
                    8.568669e-01,
                    5.151967e-01,
                ]
            ),
        )
    elif mode == "y_revolute":
        TestUtils.assert_equal(
            Qf,
            np.array(
                [
                    5.151967e-01,
                    0.000000e00,
                    -8.568669e-01,
                    -3.661819e-16,
                    4.391503e-31,
                    -7.966007e-15,
                    -3.427154e-16,
                    -1.000000e00,
                    -9.281921e-15,
                    8.568669e-01,
                    0.000000e00,
                    5.151967e-01,
                ]
            ),
        )
    elif mode == "z_revolute":
        TestUtils.assert_equal(
            Qf,
            np.array(
                [
                    0.000000e00,
                    -8.568669e-01,
                    -5.151967e-01,
                    -2.328333e-17,
                    1.398126e-15,
                    -8.541665e-15,
                    -9.373376e-18,
                    -5.151967e-01,
                    8.568669e-01,
                    1.000000e00,
                    0.000000e00,
                    0.000000e00,
                ]
            ),
        )
    else:
        raise ValueError("Invalid mode")


@pytest.mark.parametrize(
    "mode",
    ["moment_equilibrium", "force_equilibrium", "no_equilibrium"],
)
def test_forward_dynamics_with_force(mode):
    bionc = TestUtils.bionc_folder()
    module_fd = TestUtils.load_module(bionc + "/examples/forward_dynamics/pendulum_with_force.py")

    model, all_states = module_fd.main(mode=mode)

    mode_to_states_expected = {
        "moment_equilibrium": np.array(
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
        "force_equilibrium": np.array(
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
        "no_equilibrium": np.array(
            [
                1.00000000e00,
                -1.35070921e-14,
                -5.74195276e-15,
                4.31085832e-15,
                6.18406905e-17,
                2.11272817e-15,
                6.29819871e-15,
                -9.42305961e-01,
                -3.34739950e-01,
                -7.85747160e-18,
                -3.34742001e-01,
                9.42309818e-01,
                -4.31132552e-30,
                7.14524221e-15,
                -1.75584183e-14,
                7.70496015e-16,
                -5.53946285e-18,
                3.70702716e-16,
                2.21909111e-15,
                4.28930797e-01,
                -1.20745434e00,
                1.24168228e-17,
                -1.20746022e00,
                -4.28933759e-01,
            ]
        ),
    }

    np.testing.assert_almost_equal(
        all_states[:, -1],
        mode_to_states_expected[mode],
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
