from bionc.bionc_casadi import NaturalSegment, SegmentMarker, SegmentNaturalCoordinates
import numpy as np
import pytest
from .utils import TestUtils


def test_natural_segment_casadi():
    # Let's create a segment
    my_segment = NaturalSegment(
        name="box",
        alpha=np.pi / 2,
        beta=np.pi / 2,
        gamma=np.pi / 2,
        length=1,
        mass=1,
        center_of_mass=np.array([0, 0.01, 0]),
        inertia=np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
    )
    # test the name of the segment
    assert my_segment.name == "box"
    TestUtils.mx_assert_equal(my_segment.alpha, np.pi / 2)
    TestUtils.mx_assert_equal(my_segment.beta, np.pi / 2)
    TestUtils.mx_assert_equal(my_segment.gamma, np.pi / 2)
    TestUtils.mx_assert_equal(my_segment.length, 1)
    TestUtils.mx_assert_equal(my_segment.mass, 1)
    TestUtils.mx_assert_equal(my_segment.center_of_mass, np.array([0, 0.01, 0]))
    TestUtils.mx_assert_equal(my_segment.inertia, np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]))


def test_marker_features_casadi():
    # Let's create a segment
    my_segment = NaturalSegment(
        name="Thigh",
        alpha=np.pi / 2,
        beta=np.pi / 2,
        gamma=np.pi / 2,
        length=1,
        mass=1,
        center_of_mass=np.array([0, 0.01, 0]),
        inertia=np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
    )

    # Let's add a marker
    with pytest.raises(ValueError, match="The position must be a 3d vector with only one column"):
        SegmentMarker(
            name="my_marker1",
            parent_name="Thigh",
            position=np.eye(3),
            is_technical=True,
            is_anatomical=False,
        )

    marker1 = SegmentMarker(
        name="my_marker1",
        parent_name="Thigh",
        position=np.ones(3),
        is_technical=True,
        is_anatomical=False,
    )
    marker2 = SegmentMarker(
        name="my_marker2",
        parent_name="Thigh",
        position=np.array([0, 1, 2]),
        is_technical=True,
        is_anatomical=False,
    )
    my_segment.add_marker(marker1)
    my_segment.add_marker(marker2)

    Qi = SegmentNaturalCoordinates.from_components(
        u=[1, 2, 3],
        rp=[1, 1, 3],
        rd=[1, 2, 4],
        w=[1, 2, 5],
    )

    np.testing.assert_array_equal(my_segment.nb_markers(), 2)
    TestUtils.mx_assert_equal(
        my_segment.marker_constraints(
            marker_locations=np.array([[1, 2, 3], [1, 2, 3]]).T,
            Qi=Qi,
        ),
        np.array([[-2, -2, -7], [-2, -2, -9]]).T,
    )

    with pytest.raises(
        ValueError,
        # match=f"marker_locations should be of shape (3, {my_segment.nb_markers()})"  # don't know why this doesn't work
    ):
        my_segment.marker_constraints(
            marker_locations=np.array([[1, 2, 3]]).T,
            Qi=Qi,
        )

    TestUtils.mx_assert_equal(
        my_segment.marker_jacobian(),
        np.array(
            [
                [-1.0, -0.0, -0.0, -2.0, -0.0, -0.0, 1.0, 0.0, 0.0, -1.0, -0.0, -0.0],
                [-0.0, -1.0, -0.0, -0.0, -2.0, -0.0, 0.0, 1.0, 0.0, -0.0, -1.0, -0.0],
                [-0.0, -0.0, -1.0, -0.0, -0.0, -2.0, 0.0, 0.0, 1.0, -0.0, -0.0, -1.0],
                [-0.0, -0.0, -0.0, -2.0, -0.0, -0.0, 1.0, 0.0, 0.0, -2.0, -0.0, -0.0],
                [-0.0, -0.0, -0.0, -0.0, -2.0, -0.0, 0.0, 1.0, 0.0, -0.0, -2.0, -0.0],
                [-0.0, -0.0, -0.0, -0.0, -0.0, -2.0, 0.0, 0.0, 1.0, -0.0, -0.0, -2.0],
            ]
        ),
    )
