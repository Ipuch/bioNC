import numpy as np
import pytest

pytest.importorskip("pyorerun")

from .utils import TestUtils
from bionc import TransformationMatrixType


def _load_module():
    bionc = TestUtils.bionc_folder()
    return TestUtils.load_module(bionc + "/examples/transformation_matrix/compare_transformation_matrices.py")


def test_implemented_types_excludes_unimplemented_ones():
    module = _load_module()
    types = module._implemented_types()
    assert TransformationMatrixType.Buv in types
    assert TransformationMatrixType.Bvu in types
    assert TransformationMatrixType.Bwu in types
    assert TransformationMatrixType.Buw in types
    assert TransformationMatrixType.Bvw not in types
    assert TransformationMatrixType.Bwv not in types


def test_angle_sweep_uses_cosine():
    module = _load_module()
    nb_frames = 60
    t, alpha, beta, gamma = module._angle_sweep(nb_frames)
    assert t.shape == (nb_frames,)
    assert alpha.shape == (nb_frames,)
    # Cosine-periodic: first frame value ~ last frame value.
    np.testing.assert_almost_equal(alpha[0], alpha[-1], decimal=10)
    np.testing.assert_almost_equal(beta[0], beta[-1], decimal=10)
    np.testing.assert_almost_equal(gamma[0], gamma[-1], decimal=10)
    # Values vary, so the sweep is not constant.
    assert alpha.std() > 0
    assert beta.std() > 0
    assert gamma.std() > 0


def test_buv_orthogonal_frame_is_identity_per_frame():
    """With our reference choice (global = Buv segment-Cartesian), the Buv tile's
    orthogonal axes are always the global axes."""
    module = _load_module()
    types = module._implemented_types()
    _, alpha, beta, gamma = module._angle_sweep(40)
    _, ortho = module._natural_and_orthogonal_axes(types, alpha, beta, gamma)
    rot_buv = ortho[TransformationMatrixType.Buv]
    for k in range(rot_buv.shape[2]):
        np.testing.assert_almost_equal(rot_buv[:, :, k], np.eye(3), decimal=10)


@pytest.mark.parametrize(
    "type_enum",
    [
        TransformationMatrixType.Buv,
        TransformationMatrixType.Bvu,
        TransformationMatrixType.Bwu,
        TransformationMatrixType.Buw,
    ],
)
def test_orthogonal_axes_have_unit_norm(type_enum):
    """For every implemented type and every frame, the three columns of the
    orthogonal rotation matrix are (approximately) unit vectors."""
    module = _load_module()
    types = module._implemented_types()
    _, alpha, beta, gamma = module._angle_sweep(20)
    _, ortho = module._natural_and_orthogonal_axes(types, alpha, beta, gamma)
    rot = ortho[type_enum]
    for k in range(rot.shape[2]):
        for col in range(3):
            np.testing.assert_almost_equal(np.linalg.norm(rot[:, col, k]), 1.0, decimal=2)


def test_main_registers_one_arrow_per_axis_per_tile():
    """Six arrows per tile: u, v, w (natural) + x, y, z (orthogonal, always RGB)."""
    import pyorerun

    module = _load_module()
    captured = {}

    def fake_rerun(self):
        captured["names"] = list(self.xp_data.component_names)
        captured["arrows"] = list(self.xp_data.xp_data)

    monkey = pyorerun.PhaseRerun.rerun
    pyorerun.PhaseRerun.rerun = fake_rerun
    try:
        module.main(nb_frames=8)
    finally:
        pyorerun.PhaseRerun.rerun = monkey

    nb_types = len(module._implemented_types())
    assert len(captured["names"]) == 6 * nb_types

    # Tiles are spaced along x.
    spacing = 2.5
    for i, type_enum in enumerate(module._implemented_types()):
        for axis in ("u", "v", "w", "x", "y", "z"):
            assert any(f"{type_enum.name}/{axis}" in n for n in captured["names"])

    # Origins of the first frame land at (i*spacing, 0, 0) for tile i.
    origins_per_tile = {}
    for arrow in captured["arrows"]:
        type_label = arrow.name.split("/")[-3]  # ".../<TYPE>/<axis>/force_vector_<num>"
        origins_per_tile.setdefault(type_label, set()).add(tuple(arrow.vector_origins[:, 0]))
    for i, type_enum in enumerate(module._implemented_types()):
        # All six arrows of a tile share the same origin.
        assert origins_per_tile[type_enum.name] == {(i * spacing, 0.0, 0.0)}


def test_orthogonal_axes_use_pure_rgb():
    """Orthogonal axes (x, y, z) must be drawn in pure red, green, blue."""
    from pyorerun.abstract.markers import rgb255_to_hex_rgba
    import pyorerun

    module = _load_module()
    captured = {}

    def fake_rerun(self):
        captured["arrows"] = list(self.xp_data.xp_data)

    monkey = pyorerun.PhaseRerun.rerun
    pyorerun.PhaseRerun.rerun = fake_rerun
    try:
        module.main(nb_frames=4)
    finally:
        pyorerun.PhaseRerun.rerun = monkey

    expected = {
        "x": rgb255_to_hex_rgba(np.array([255, 0, 0])),
        "y": rgb255_to_hex_rgba(np.array([0, 255, 0])),
        "z": rgb255_to_hex_rgba(np.array([0, 0, 255])),
    }
    for arrow in captured["arrows"]:
        # Names look like ".../<TYPE>/<axis>/force_vector_<num>"
        parts = arrow.name.split("/")
        axis = parts[-2]
        if axis in expected:
            assert arrow._color == expected[axis], f"{arrow.name}: {arrow._color} != {expected[axis]}"
