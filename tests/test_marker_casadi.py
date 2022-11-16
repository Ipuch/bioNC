import pytest
import numpy as np

from bionc.bionc_casadi import SegmentMarker, SegmentNaturalCoordinates
from .utils import TestUtils


def test_segment_marker():

    with pytest.raises(ValueError, match="Either a position or an interpolation matrix must be provided"):
        SegmentMarker(
            name="my_marker",
            parent_name="Thigh",
            position=None,
            interpolation_matrix=None,
            is_technical=True,
            is_anatomical=False,
        )

    with pytest.raises(ValueError, match="The position must be a 3d vector"):
        SegmentMarker(
            name="my_marker",
            parent_name="Thigh",
            position=np.zeros(2),
            interpolation_matrix=None,
            is_technical=True,
            is_anatomical=False,
        )

    with pytest.raises(ValueError, match="The interpolation matrix must be a 3x12 matrix"):
        SegmentMarker(
            name="my_marker",
            parent_name="Thigh",
            position=None,
            interpolation_matrix=np.zeros((1, 2)),
            is_technical=True,
            is_anatomical=False,
        )

    with pytest.raises(ValueError, match="position and interpolation matrix cannot both be provided"):
        SegmentMarker(
            name="my_marker",
            parent_name="Thigh",
            position=np.zeros(3),
            interpolation_matrix=np.zeros((1, 2)),
            is_technical=True,
            is_anatomical=False,
        )

    segment_marker = SegmentMarker(
        name="my_marker",
        parent_name="Thigh",
        position=np.ones(3),
        is_technical=True,
        is_anatomical=False,
    )

    TestUtils.mx_assert_equal(segment_marker.position, np.ones(3))
    TestUtils.mx_assert_equal(
        segment_marker.interpolation_matrix,
        np.array(
            [
                [1.0, 0.0, 0.0, 2.0, 0.0, 0.0, -1.0, -0.0, -0.0, 1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 2.0, 0.0, -0.0, -1.0, -0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0, 2.0, -0.0, -0.0, -1.0, 0.0, 0.0, 1.0],
            ]
        ),
    )
    assert segment_marker.is_technical
    assert not segment_marker.is_anatomical

    marker_location = np.array([1, 2, 3])
    Qi = SegmentNaturalCoordinates.from_components(
        u=[1, 2, 3],
        rp=[1, 1, 3],
        rd=[1, 2, 4],
        w=[1, 2, 5],
    )

    constraint = segment_marker.constraint(marker_location=marker_location, Qi=Qi)
    TestUtils.mx_assert_equal(constraint, np.array([-2, -2, -7]))

    constraint = segment_marker.constraint(marker_location=marker_location[:, np.newaxis], Qi=Qi)
    TestUtils.mx_assert_equal(constraint, np.array([-2, -2, -7]))

    with pytest.raises(ValueError, match="The marker location must be a 3d vector"):
        segment_marker.constraint(marker_location=np.zeros(2), Qi=Qi)

    with pytest.raises(ValueError, match="The marker location must be a 3d vector with only one column"):
        segment_marker.constraint(marker_location=np.zeros((3, 2)), Qi=Qi)
