"""
Tests for the native (C++/Eigen + nanobind) differential IK backend, method="dik_native".

These tests exercise the full installation path: they trigger the CasADi codegen,
the C++ compilation (via CMake/nanobind) and the native solve, then check that the
result matches the pure-python ``method="dik"`` solver.

They are skipped (not failed) when the C++ toolchain is unavailable, so that
environments without a compiler can still run the rest of the suite.
"""

import shutil

import numpy as np
import pytest
from pyomeca import Markers

from bionc import InverseKinematics
from tests.utils import TestUtils


def _toolchain_available() -> bool:
    if shutil.which("cmake") is None:
        return False
    return any(shutil.which(cc) is not None for cc in ("c++", "g++", "clang++", "cc", "gcc", "cl"))


requires_toolchain = pytest.mark.skipif(
    not _toolchain_available(),
    reason="C++ toolchain (cmake + compiler) not available; see bionc/utils/dik_native/README.md",
)


def _two_side_model_and_markers(n_frames: int = 3):
    bionc = TestUtils.bionc_folder()
    right = TestUtils.load_module(bionc + "/examples/model_creation/right_side_lower_limb.py")
    two_side = TestUtils.load_module(bionc + "/examples/model_creation/two_side_lower_limbs.py")

    filename = right.generate_c3d_file(two_side=True)
    model = two_side.model_creation_from_measured_data(filename)

    markers = Markers.from_c3d(filename, usecols=model.marker_names_technical).to_numpy()[:3, :, :]
    markers = np.repeat(markers, n_frames, axis=2)
    rng = np.random.default_rng(42)
    markers = markers + rng.normal(0, 0.01, markers.shape)
    return model, markers


@requires_toolchain
def test_native_module_builds_and_imports():
    """The C++ module compiles (CMake + nanobind) and exposes solve_native."""
    pytest.importorskip("nanobind", reason="nanobind not installed; see dik_native/README.md")
    from bionc.utils import native_dik

    module = native_dik._ensure_module_built()
    assert hasattr(module, "solve_native")


@requires_toolchain
def test_dik_native_matches_python_dik():
    """method="dik_native" yields the same solution as method="dik"."""
    pytest.importorskip("nanobind", reason="nanobind not installed; see dik_native/README.md")
    model, markers = _two_side_model_and_markers(n_frames=3)

    ik_ref = InverseKinematics(model, markers)
    Q_ref = ik_ref.solve(method="dik")
    stats_ref = ik_ref.sol()

    ik = InverseKinematics(model, markers)
    Qopt = ik.solve(method="dik_native")  # triggers the C++ build on first run
    stats = ik.sol()

    # same maths as the python dik solver -> same solution and same residuals
    np.testing.assert_allclose(Qopt, Q_ref, atol=1e-4)
    np.testing.assert_allclose(stats["marker_residuals_norm"], stats_ref["marker_residuals_norm"], atol=1e-4)

    assert Qopt.shape == (model.nb_Q, markers.shape[2])
    assert stats["success"] == [True] * markers.shape[2]
    assert np.isfinite(Qopt).all()
    # loose absolute sanity bound (the noise floor is ~0.03); the parity checks above are the real test
    assert np.max(stats["marker_residuals_norm"]) < 0.05
    assert np.max(np.abs(stats["total_joint_residuals"])) < 1e-6
    assert np.max(np.abs(stats["total_rigidity_residuals"])) < 1e-6
    assert ik.segment_determinants.shape == (model.nb_segments, markers.shape[2])
    assert np.min(ik.segment_determinants) > 0
