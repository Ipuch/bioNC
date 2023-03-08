import numpy as np
import pytest

from .utils import TestUtils


@pytest.mark.parametrize(
    "bionc_type",
    ["numpy", "casadi"],
)
def test_natural_segment(bionc_type):
    if bionc_type == "casadi":
        from bionc.bionc_casadi import (
            NaturalSegment,
            SegmentNaturalCoordinates,
            SegmentNaturalVelocities,
        )
    else:
        from bionc.bionc_numpy import (
            NaturalSegment,
            SegmentNaturalCoordinates,
            SegmentNaturalVelocities,
        )

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
    TestUtils.assert_equal(my_segment.alpha, np.pi / 2)
    TestUtils.assert_equal(my_segment.beta, np.pi / 2)
    TestUtils.assert_equal(my_segment.gamma, np.pi / 2)
    TestUtils.assert_equal(my_segment.length, 1)
    TestUtils.assert_equal(my_segment.mass, 1)
    TestUtils.assert_equal(my_segment.center_of_mass, np.array([0, 0.01, 0]))
    TestUtils.assert_equal(my_segment.inertia, np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]))

    TestUtils.assert_equal(my_segment.natural_center_of_mass, np.array([0, 0.01, 0]))
    N = np.array(
        [
            [0.0, 0.0, 0.0, 1.010, 0.0, 0.0, -1.0e-02, -0.0, -0.0, -6.123234e-19, -0.0, -0.0],
            [0.0, 0.0, 0.0, 0.0, 1.010, 0.0, -0.0, -1.0e-02, -0.0, -0.0, -6.123234e-19, -0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.010, -0.0, -0.0, -1.0e-02, -0.0, -0.0, -6.123234e-19],
        ]
    )
    TestUtils.assert_equal(my_segment.natural_center_of_mass.interpolate(), N)

    M = np.array(
        [
            [1.0, 0.0, 0.0, -1.0e-04, -0.0, -0.0, 1.0e-04, 0.0, 0.0, 1.0e-04, 0.0, 0.0],
            [
                0.0,
                1.0,
                0.0,
                -0.0,
                -1.0e-04,
                -0.0,
                0.0,
                1.0e-04,
                0.0,
                0.0,
                1.0e-04,
                0.0,
            ],
            [
                0.0,
                0.0,
                1.0,
                -0.0,
                -0.0,
                -1.0e-04,
                0.0,
                0.0,
                1.0e-04,
                0.0,
                0.0,
                1.0e-04,
            ],
            [
                -1.0e-04,
                -0.0,
                -0.0,
                2.02,
                0.0,
                0.0,
                -1.01,
                -0.0,
                -0.0,
                -1.0e-04,
                -0.0,
                -0.0,
            ],
            [-0.0, -1.0e-04, -0.0, 0.0, 2.02, 0.0, -0.0, -1.01, -0.0, -0.0, -1.0e-04, -0.0],
            [-0.0, -0.0, -1.0e-04, 0.0, 0.0, 2.02, -0.0, -0.0, -1.01, -0.0, -0.0, -1.0e-04],
            [1.0e-04, 0.0, 0.0, -1.01, -0.0, -0.0, 1.0, 0.0, 0.0, 1.0e-04, 0.0, 0.0],
            [0.0, 1.0e-04, 0.0, -0.0, -1.01, -0.0, 0.0, 1.0, 0.0, 0.0, 1.0e-04, 0.0],
            [0.0, 0.0, 1.0e-04, -0.0, -0.0, -1.01, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0e-04],
            [1.0e-04, 0.0, 0.0, -1.0e-04, -0.0, -0.0, 1.0e-04, 0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 1.0e-04, 0.0, -0.0, -1.0e-04, -0.0, 0.0, 1.0e-04, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0e-04, -0.0, -0.0, -1.0e-04, 0.0, 0.0, 1.0e-04, 0.0, 0.0, 1.0],
        ]
    )

    TestUtils.assert_equal(my_segment.mass_matrix, M)

    # kinetic energy and potential energy

    Qi = SegmentNaturalCoordinates.from_components(
        u=np.array([0.11, 0.12, 0.13]),
        rp=np.array([0.21, 0.22, 0.23]),
        rd=np.array([0.31, 0.32, 0.33]),
        w=np.array([0.41, 0.42, 0.43]),
    )

    Qdoti = SegmentNaturalVelocities.from_components(
        udot=np.array([0.1, 0.2, 0.3]),
        rpdot=np.array([0.4, 0.5, 0.6]),
        rddot=np.array([0.7, 0.8, 0.9]),
        wdot=np.array([0.4, 0.5, 0.6]),
    )

    TestUtils.assert_equal(my_segment.kinetic_energy(Qdoti), np.array(0.970595))
    TestUtils.assert_equal(my_segment.potential_energy(Qi), np.array(0.229))


@pytest.mark.parametrize(
    "bionc_type",
    ["numpy", "casadi"],
)
def test_marker_features(bionc_type):
    if bionc_type == "casadi":
        from bionc.bionc_casadi import (
            NaturalSegment,
            SegmentNaturalCoordinates,
            SegmentNaturalVelocities,
            NaturalMarker,
        )
    else:
        from bionc.bionc_numpy import (
            NaturalSegment,
            SegmentNaturalCoordinates,
            SegmentNaturalVelocities,
            NaturalMarker,
        )

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

    # NaturalMarker(
    #     name="my_marker1",
    #     parent_name="Thigh",
    #     position=np.eye(3),
    #     is_technical=True,
    #     is_anatomical=False,
    # )

    # Let's add a marker
    with pytest.raises(ValueError, match="The position must be a 3d vector with only one column"):
        NaturalMarker(
            name="my_marker1",
            parent_name="Thigh",
            position=np.eye(3),
            is_technical=True,
            is_anatomical=False,
        )

    marker1 = NaturalMarker(
        name="my_marker1",
        parent_name="Thigh",
        position=np.ones(3),
        is_technical=True,
        is_anatomical=False,
    )

    marker2 = NaturalMarker(
        name="my_marker2",
        parent_name="Thigh",
        position=np.array([0, 1, 2]),
        is_technical=True,
        is_anatomical=False,
    )
    my_segment.add_natural_marker(marker1)
    my_segment.add_natural_marker(marker2)

    Qi = SegmentNaturalCoordinates.from_components(
        u=[1, 2, 3],
        rp=[1, 1, 3],
        rd=[1, 2, 4],
        w=[1, 2, 5],
    )

    TestUtils.assert_equal(my_segment.nb_markers, 2)
    TestUtils.assert_equal(
        my_segment.marker_constraints(
            marker_locations=np.array([[1, 2, 3], [1, 2, 3]]).T,
            Qi=Qi,
        ),
        np.array([[-2, -2, -7], [-2, -2, -9]]).T,
    )

    with pytest.raises(
        ValueError,
        # match=f"marker_locations should be of shape (3, {my_segment.nb_markers})"  # don't know why this doesn't work
    ):
        my_segment.marker_constraints(
            marker_locations=np.array([[1, 2, 3]]).T,
            Qi=Qi,
        )

    TestUtils.assert_equal(
        my_segment.markers_jacobian(),
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
    markers_positions = np.array([[[3.0], [3.0]], [[4.0], [4.0]], [[10.0], [12.0]]])
    TestUtils.assert_equal(
        my_segment.markers(Qi=Qi),
        markers_positions.squeeze() if bionc_type == "casadi" else markers_positions,
    )


@pytest.mark.parametrize(
    "bionc_type",
    ["numpy", "casadi"],
)
def test_marker_add_from_scs(bionc_type):
    if bionc_type == "casadi":
        from bionc.bionc_casadi import (
            NaturalSegment,
            SegmentNaturalCoordinates,
            SegmentNaturalVelocities,
            NaturalMarker,
        )
    else:
        from bionc.bionc_numpy import (
            NaturalSegment,
            SegmentNaturalCoordinates,
            SegmentNaturalVelocities,
            NaturalMarker,
        )

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

    my_segment.add_natural_marker_from_segment_coordinates(
        name="test1",
        location=np.array([1, 1, 1]),
        is_distal_location=True,
        is_technical=True,
        is_anatomical=False,
    )
    my_segment.add_natural_marker_from_segment_coordinates(
        name="test2",
        location=np.array([2, 2, 2]),
        is_distal_location=False,
        is_technical=True,
        is_anatomical=False,
    )
    my_segment.add_natural_marker_from_segment_coordinates(
        name="test3",
        location=np.array([3, 3, 3]),
        is_distal_location=True,
        is_technical=False,
        is_anatomical=False,
    )
    my_segment.add_natural_marker_from_segment_coordinates(
        name="test4",
        location=np.array([4, 4, 4]),
        is_distal_location=False,
        is_technical=False,
        is_anatomical=False,
    )
    my_segment.add_natural_marker_from_segment_coordinates(
        name="test5",
        location=np.array([5, 5, 5]),
        is_distal_location=True,
        is_technical=False,
        is_anatomical=True,
    )
    my_segment.add_natural_marker_from_segment_coordinates(
        name="test6",
        location=np.array([6, 6, 6]),
        is_distal_location=False,
        is_technical=False,
        is_anatomical=False,
    )
    my_segment.add_natural_marker_from_segment_coordinates(
        name="test7",
        location=np.array([7, 7, 7]),
        is_distal_location=True,
        is_technical=True,
        is_anatomical=True,
    )

    Qi = SegmentNaturalCoordinates.from_components(
        u=[1, 2, 3],
        rp=[1, 1, 3],
        rd=[1, 2, 4],
        w=[1, 2, 5],
    )

    TestUtils.assert_equal(my_segment.nb_markers, 7)
    TestUtils.assert_equal(my_segment.nb_markers_technical, 3)
    TestUtils.assert_equal(
        my_segment.marker_constraints(
            marker_locations=np.ones((3, 7)),
            Qi=Qi,
            only_technical=False,
        ),
        np.array(
            [
                [-2.0, -4.0, -6.0, -8.0, -10.0, -12.0, -14.0],
                [-4.0, -6.0, -10.0, -12.0, -16.0, -18.0, -22.0],
                [-10.0, -16.0, -24.0, -30.0, -38.0, -44.0, -52.0],
            ]
        ),
    )

    TestUtils.assert_equal(
        my_segment.markers_jacobian(),
        np.array(
            [
                [-1.0, -0.0, -0.0, -1.0, -0.0, -0.0, 0.0, 0.0, 0.0, -1.0, -0.0, -0.0],
                [-0.0, -1.0, -0.0, -0.0, -1.0, -0.0, 0.0, 0.0, 0.0, -0.0, -1.0, -0.0],
                [-0.0, -0.0, -1.0, -0.0, -0.0, -1.0, 0.0, 0.0, 0.0, -0.0, -0.0, -1.0],
                [-2.0, -0.0, -0.0, -3.0, -0.0, -0.0, 2.0, 0.0, 0.0, -2.0, -0.0, -0.0],
                [-0.0, -2.0, -0.0, -0.0, -3.0, -0.0, 0.0, 2.0, 0.0, -0.0, -2.0, -0.0],
                [-0.0, -0.0, -2.0, -0.0, -0.0, -3.0, 0.0, 0.0, 2.0, -0.0, -0.0, -2.0],
                [-7.0, -0.0, -0.0, -7.0, -0.0, -0.0, 6.0, 0.0, 0.0, -7.0, -0.0, -0.0],
                [-0.0, -7.0, -0.0, -0.0, -7.0, -0.0, 0.0, 6.0, 0.0, -0.0, -7.0, -0.0],
                [-0.0, -0.0, -7.0, -0.0, -0.0, -7.0, 0.0, 0.0, 6.0, -0.0, -0.0, -7.0],
            ]
        ),
    )
    markers_positions = np.array(
        [
            [[3.0], [5.0], [7.0], [9.0], [11.0], [13.0], [15.0]],
            [[5.0], [7.0], [11.0], [13.0], [17.0], [19.0], [23.0]],
            [[11.0], [17.0], [25.0], [31.0], [39.0], [45.0], [53.0]],
        ]
    )

    TestUtils.assert_equal(
        my_segment.markers(Qi=Qi),
        markers_positions.squeeze() if bionc_type == "casadi" else markers_positions,
    )

@pytest.mark.parametrize(
    "bionc_type",
    ["numpy", "casadi"],
)
def test_angle_sanity_check(bionc_type):
    if bionc_type == "casadi":
        from bionc.bionc_casadi import (
            NaturalSegment,
            SegmentNaturalCoordinates,
        )
    else:
        from bionc.bionc_numpy import (
            NaturalSegment,
            SegmentNaturalCoordinates,
        )

    with pytest.raises(
            ValueError,
            match="The angles alpha, beta, gamma, would produce a singular transformation matrix for the segment"):
        bbox = NaturalSegment(
            name="bbox",
            alpha=np.pi / 5,
            beta=np.pi / 3,
            gamma=np.pi / 2.1,
            length=1.5,
            mass=1.1,
            center_of_mass=np.array([0.1, 0.11, 0.111]),  # scs
            inertia=np.array([[1.1, 0, 0], [0, 1.2, 0], [0, 0, 1.3]]),  # scs
        )