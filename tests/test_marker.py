import pytest
import numpy as np

from bionc import SegmentMarker, Marker


def test_natural_marker():

    with pytest.raises(ValueError, match="Either a position or an interpolation matrix must be provided"):
        segment_marker = SegmentMarker(
            name="my_marker",
            parent_name="Thigh",
            position=None,
            interpolation_matrix=None,
            is_technical=True,
            is_anatomical=False,
        )

    with pytest.raises(ValueError, match="The position must be a 3d vector"):
        segment_marker = SegmentMarker(
            name="my_marker",
            parent_name="Thigh",
            position=np.zeros(2),
            interpolation_matrix=None,
            is_technical=True,
            is_anatomical=False,
        )

    with pytest.raises(ValueError, match="The interpolation matrix must be a 3x12 matrix"):
        segment_marker = SegmentMarker(
            name="my_marker",
            parent_name="Thigh",
            position=None,
            interpolation_matrix=np.zeros((1, 2)),
            is_technical=True,
            is_anatomical=False,
        )

    with pytest.raises(ValueError, match="position and interpolation matrix cannot both be provided"):
        segment_marker = SegmentMarker(
            name="my_marker",
            parent_name="Thigh",
            position=np.zeros(3),
            interpolation_matrix=np.zeros((1, 2)),
            is_technical=True,
            is_anatomical=False,
        )
