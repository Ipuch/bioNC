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


def test_play_with_joints_plane_on_ellipsoid():
    bionc = TestUtils.bionc_folder()
    module = TestUtils.load_module(bionc + "/examples/play_with_joints/plane_on_ellipsoid.py")

    model, joint, phi = module.main(show_results=False)

    assert joint.nb_constraints == 1
    np.testing.assert_allclose(phi["tangent"], 0.0, atol=1e-9)
    np.testing.assert_allclose(phi["slide_+y"], 0.0, atol=1e-9)
    np.testing.assert_allclose(phi["pushed_+x"], -0.02, atol=1e-9)


@pytest.mark.parametrize("model_type", ["one", "two"])
def test_play_with_joints_points_on_ellipsoid(model_type):
    bionc = TestUtils.bionc_folder()
    module = TestUtils.load_module(bionc + "/examples/play_with_joints/points_on_ellipsoid.py")

    model, joint, max_defect = module.main(model_type=model_type, show_results=False)

    assert joint.nb_constraints == (1 if model_type == "one" else 2)
    np.testing.assert_allclose(max_defect, 0.0, atol=1e-9)


def test_play_with_joints_compare_scapulothoracic_models():
    bionc = TestUtils.bionc_folder()
    module = TestUtils.load_module(bionc + "/examples/play_with_joints/compare_scapulothoracic_models.py")

    results, markers_noisy = module.main(n_frames=30, noise_std=0.003, seed=0, show_results=False)

    # all three joint models track the markers, with constraints satisfied
    for model_type in ("tangent", "one_point", "two_point"):
        assert results[model_type]["rmse_mm"] < 10.0
    # the tangent-contact model does not penetrate; the two-point model does
    np.testing.assert_allclose(results["tangent"]["penetration_mm"], 0.0, atol=1e-6)
    assert results["two_point"]["penetration_mm"] > results["tangent"]["penetration_mm"]


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

    model, all_states, _ = module_fd.main(mode=mode, show_results=False)

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
                    -0.88815999,
                    0.4584195,
                    0.000000e00,
                    0.4584195,
                    0.88815999,
                ]
            ),
        )
    elif mode == "y_revolute":
        TestUtils.assert_equal(
            Qf,
            np.array(
                [
                    0.88815999,
                    0.000000e00,
                    -0.4584195,
                    -3.661819e-16,
                    4.391503e-31,
                    -7.966007e-15,
                    -3.427154e-16,
                    -1.000000e00,
                    -9.281921e-15,
                    0.4584195,
                    0.000000e00,
                    0.88815999,
                ]
            ),
        )
    elif mode == "z_revolute":
        TestUtils.assert_equal(
            Qf,
            np.array(
                [
                    0.000000e00,
                    -0.4584195,
                    -0.88815999,
                    -2.328333e-17,
                    1.398126e-15,
                    -8.541665e-15,
                    -9.373376e-18,
                    -0.88815999,
                    0.4584195,
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

    model, all_states, _ = module_fd.main(mode=mode)

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
                -5.48992170e-15,
                7.93602786e-16,
                -2.43822452e-16,
                -5.04689367e-16,
                2.26393412e-15,
                7.98564420e-15,
                0.70561899,
                -0.70856647,
                3.23937407e-17,
                -0.70858146,
                -0.70562902,
                -5.34376529e-30,
                -8.70706413e-16,
                7.09099939e-16,
                -1.16542899e-16,
                -2.84650809e-17,
                4.42731431e-16,
                1.58346121e-15,
                -1.6709541,
                -1.66399832,
                8.27191143e-18,
                -1.66402369,
                1.6709912,
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

    model, all_states, _ = module_fd.main(mode="force_equilibrium")

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

    model, all_states, _ = module_fd.main(mode="no_equilibrium")

    np.testing.assert_almost_equal(
        all_states[:, -1],
        np.array(
            [
                1.00000000e00,
                6.43108418e-33,
                -9.19912107e-32,
                -1.48235566e-31,
                -6.29941693e-17,
                2.90641835e-15,
                2.23258064e-31,
                -0.44617318,
                0.89492306,
                -2.17211235e-31,
                0.89493725,
                0.44619202,
                1.00000000e00,
                7.15480431e-31,
                -1.73088630e-30,
                2.23258064e-31,
                -0.44617318,
                0.89492306,
                -6.73626060e-31,
                -0.26281001,
                -0.08809376,
                -2.43822417e-31,
                -0.98304462,
                -0.18336648,
                -5.33418427e-63,
                -2.23349320e-31,
                -7.35352937e-32,
                -3.00907544e-32,
                -1.09328269e-16,
                5.00891912e-16,
                2.21544037e-32,
                0.21494607,
                0.10715871,
                -3.54200617e-32,
                0.10715797,
                -0.21493892,
                -5.13544513e-61,
                -2.46078849e-30,
                -1.31388051e-30,
                2.21544037e-32,
                0.21494607,
                0.10715871,
                -9.34505060e-32,
                1.07851989,
                0.2682469,
                -1.59637020e-31,
                0.161091,
                -0.86359775,
            ]
        ),
    )


def test_natural_vs_segment_coordinates_example():
    bionc = TestUtils.bionc_folder()
    module = TestUtils.load_module(bionc + "/examples/transformation_matrix/natural_vs_segment_coordinates.py")

    results = module.main(show=False)

    # inv(B) p then [u v w] n brings p back exactly, B.T misplaces it by ~126 mm
    assert results["error"] < 1e-12
    assert results["error_bionc"] < 1e-12
    assert results["error_transposed"] > 0.1
    # the mistake is invisible on an orthogonal segment and grows with the deviation from orthogonality
    assert results["errors"][0] < 1e-12
    assert np.all(np.diff(results["errors"]) > 0)
