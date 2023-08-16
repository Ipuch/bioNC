from bionc.bionc_numpy.transformation_matrix import check_plane
from bionc import NaturalAxis
import pytest


def test_check_plane():
    plane = (NaturalAxis.U, NaturalAxis.U)
    with pytest.raises(ValueError, match=f"Plane must be a tuple of different axis, got \(<NaturalAxis.U: 'U'>, <NaturalAxis.U: 'U'>\)"):
        check_plane(plane)
    plane = (NaturalAxis.U, NaturalAxis.U, NaturalAxis.U)
    with pytest.raises(ValueError, match="Plane must be a tuple of length 2, got 3"):
        check_plane(plane)
    plane = (NaturalAxis.U, 1)
    with pytest.raises(ValueError, match=f"Plane must be a tuple of NaturalAxis, got \(<NaturalAxis.U: 'U'>, 1\)"):
        check_plane(plane)
    plane = (NaturalAxis.V, NaturalAxis.U)
    with pytest.raises(ValueError,
                       match=f"Invert Axis in plane, because it would lead to an indirect frame, got \(<NaturalAxis.V: 'V'>, <NaturalAxis.U: 'U'>\)"):
        check_plane(plane)
    plane = (NaturalAxis.U, NaturalAxis.W)
    with pytest.raises(ValueError,
                       match=f"Invert Axis in plane, because it would lead to an indirect frame, got \(<NaturalAxis.U: 'U'>, <NaturalAxis.W: 'W'>\)"):
        check_plane(plane)
    plane = (NaturalAxis.W, NaturalAxis.V)
    with pytest.raises(ValueError,
                       match=f"Invert Axis in plane, because it would lead to an indirect frame, got \(<NaturalAxis.W: 'W'>, <NaturalAxis.V: 'V'>\)"):
        check_plane(plane)
