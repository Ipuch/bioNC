import numpy as np
import pytest
from casadi import Function, sumsqr
from pyomeca import Markers

from bionc import InverseKinematics
from tests.utils import TestUtils

TWO_SIDE_SEGMENT_NAMES = [
    "PELVIS",
    "RTHIGH",
    "LTHIGH",
    "RSHANK",
    "LSHANK",
    "RFOOT",
    "LFOOT",
]
TWO_SIDE_TECHNICAL_MARKERS = [
    "RFWT",
    "LFWT",
    "RBWT",
    "LBWT",
    "RKNE",
    "RKNI",
    "LKNE",
    "LKNI",
    "RANE",
    "RANI",
    "LANE",
    "LANI",
    "RHEE",
    "RTARI",
    "RTAR",
    "LHEE",
    "LTARI",
    "LTAR",
]

IK_METHOD_TOLERANCES = {
    "ipopt": {"marker": 0.03, "joint": 1e-8, "rigid": 1e-8},
    "dik": {"marker": 0.03, "joint": 1e-8, "rigid": 1e-8},
    "sqpmethod": {"marker": 0.08, "joint": 0.02, "rigid": 0.13},
}


def _assert_inverse_kinematics_solution(ik_solver, Qopt, model, markers, tolerances):
    stats = ik_solver.sol()

    assert Qopt.shape == (model.nb_Q, markers.shape[2])
    np.testing.assert_allclose(ik_solver.Qopt, Qopt)
    assert stats["success"] == [True] * markers.shape[2]
    assert np.isfinite(Qopt).all()

    assert np.max(stats["marker_residuals_norm"]) < tolerances["marker"]
    assert np.max(np.abs(stats["total_joint_residuals"])) < tolerances["joint"]
    assert np.max(np.abs(stats["total_rigidity_residuals"])) < tolerances["rigid"]
    assert ik_solver.segment_determinants.shape == (model.nb_segments, markers.shape[2])
    assert np.min(ik_solver.segment_determinants) > 0


@pytest.mark.parametrize(
    ("method", "active_direct_frame_constraints"),
    [
        pytest.param("ipopt", False, id="ipopt"),
        pytest.param("sqpmethod", False, id="sqpmethod"),
        pytest.param("dik", False, id="dik"),
        pytest.param("ipopt", True, id="ipopt-active-direct-frame-constraints"),
    ],
)
def test_ik_example_solves_two_side_lower_limbs(method, active_direct_frame_constraints):
    bionc = TestUtils.bionc_folder()
    module = TestUtils.load_module(bionc + "/examples/inverse_kinematics/inverse_kinematics.py")

    results, model, markers = module.main(
        methods=(method,),
        n_repeats=1,
        active_direct_frame_constraints=active_direct_frame_constraints,
        print_timing=False,
    )
    result = results[method]

    assert model.segment_names == TWO_SIDE_SEGMENT_NAMES
    assert model.marker_names_technical == TWO_SIDE_TECHNICAL_MARKERS
    assert markers.shape == (3, len(TWO_SIDE_TECHNICAL_MARKERS), 2)
    assert result["elapsed"] >= 0
    assert result["solver"]._active_direct_frame_constraints is active_direct_frame_constraints

    _assert_inverse_kinematics_solution(
        result["solver"],
        result["Qopt"],
        model,
        markers,
        IK_METHOD_TOLERANCES[method],
    )


def test_inverse_kinematics_class():
    bionc = TestUtils.bionc_folder()
    module = TestUtils.load_module(bionc + "/examples/model_creation/right_side_lower_limb.py")

    filename = module.generate_c3d_file()
    natural_model = module.model_creation_from_measured_data(filename)

    markers = Markers.from_c3d(filename).to_numpy()[:3, :, :]
    markers = np.repeat(markers, 2, axis=2)
    rng = np.random.default_rng(42)
    markers = markers + rng.normal(0, 0.01, markers.shape)

    ik = InverseKinematics(natural_model, markers)
    assert ik.nb_frames == markers.shape[2]
    assert ik.nb_markers == markers.shape[1]

    with pytest.raises(
        TypeError,
        match=f"experimental_markers must be a path as a string or a numpy array of size 3xNxM. Got {type(1)} instead.",
    ):
        InverseKinematics(natural_model, 1)

    with pytest.raises(ValueError):
        InverseKinematics(1, markers)


def test_ik_frame_per_frame_extra_obj():
    bionc = TestUtils.bionc_folder()
    module = TestUtils.load_module(bionc + "/examples/model_creation/right_side_lower_limb.py")

    filename = module.generate_c3d_file()
    natural_model = module.model_creation_from_measured_data(filename)

    markers = Markers.from_c3d(filename).to_numpy()[:3, :, :]
    markers = np.repeat(markers, 2, axis=2)
    rng = np.random.default_rng(42)
    markers = markers + rng.normal(0, 0.01, markers.shape)

    ik = InverseKinematics(natural_model, markers)
    extra_objective_function = Function(
        "extra_objective",
        [ik._Q_sym, ik._markers_sym],
        [sumsqr(ik._Q_sym[[0, 13, 14, 19, 22, 30, 47]])],
    )
    ik.add_objective(extra_objective_function)

    Qopt = ik.solve(method="ipopt")
    stats = ik.sol()

    assert Qopt.shape == (natural_model.nb_Q, markers.shape[2])
    assert stats["success"] == [True] * markers.shape[2]
    assert np.isfinite(Qopt).all()
    assert np.max(np.abs(stats["total_joint_residuals"])) < 1e-8
    assert np.max(np.abs(stats["total_rigidity_residuals"])) < 1e-8
    assert np.min(ik.segment_determinants) > 0


def test_dik_rejects_extra_objectives():
    bionc = TestUtils.bionc_folder()
    module = TestUtils.load_module(bionc + "/examples/model_creation/right_side_lower_limb.py")

    filename = module.generate_c3d_file()
    natural_model = module.model_creation_from_measured_data(filename)
    markers = Markers.from_c3d(filename).to_numpy()[:3, :, :]

    ik = InverseKinematics(natural_model, markers)
    extra_objective_function = Function(
        "extra_objective",
        [ik._Q_sym, ik._markers_sym],
        [sumsqr(ik._Q_sym[[0]])],
    )
    ik.add_objective(extra_objective_function)

    with pytest.raises(ValueError, match='method="dik" only supports the default marker objective.'):
        ik.solve(method="dik")


def test_ik_example():
    """Run the example's main() with all solvers at once and check every solution is valid."""
    bionc = TestUtils.bionc_folder()
    module = TestUtils.load_module(bionc + "/examples/inverse_kinematics/inverse_kinematics.py")

    methods = ("ipopt", "sqpmethod", "dik")
    results, model, markers = module.main(methods=methods, n_repeats=1, print_timing=False)

    assert set(results) == set(methods)
    assert model.segment_names == TWO_SIDE_SEGMENT_NAMES
    assert model.marker_names_technical == TWO_SIDE_TECHNICAL_MARKERS
    assert markers.shape == (3, len(TWO_SIDE_TECHNICAL_MARKERS), 2)

    for method, result in results.items():
        _assert_inverse_kinematics_solution(
            result["solver"],
            result["Qopt"],
            model,
            markers,
            IK_METHOD_TOLERANCES[method],
        )
