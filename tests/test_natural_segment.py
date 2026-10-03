import numpy as np
import pytest

from bionc import TransformationMatrixType
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
    my_segment = NaturalSegment.with_cartesian_inertial_parameters(
        name="box",
        alpha=np.pi / 2,
        beta=np.pi / 2,
        gamma=np.pi / 2,
        length=1,
        mass=1,
        center_of_mass=np.array([0, 0.01, 0]),
        inertia=np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
        inertial_transformation_matrix=TransformationMatrixType.Buv,
    )
    # test the name of the segment
    assert my_segment.name == "box"
    TestUtils.assert_equal(my_segment.alpha, np.pi / 2)
    TestUtils.assert_equal(my_segment.beta, np.pi / 2)
    TestUtils.assert_equal(my_segment.gamma, np.pi / 2)
    TestUtils.assert_equal(my_segment.length, 1)
    TestUtils.assert_equal(my_segment.mass, 1)
    TestUtils.assert_equal(my_segment.center_of_mass(), np.array([0, 0.01, 0]))
    TestUtils.assert_equal(
        my_segment._natural_inertial_parameters.inertia(), np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    )

    TestUtils.assert_equal(my_segment.natural_center_of_mass, np.array([0, 0.01, 0]), expand=False)
    N = np.array(
        [
            [0.0, 0.0, 0.0, 1.010, 0.0, 0.0, -1.0e-02, -0.0, -0.0, -6.123234e-19, -0.0, -0.0],
            [0.0, 0.0, 0.0, 0.0, 1.010, 0.0, -0.0, -1.0e-02, -0.0, -0.0, -6.123234e-19, -0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.010, -0.0, -0.0, -1.0e-02, -0.0, -0.0, -6.123234e-19],
        ]
    )
    TestUtils.assert_equal(my_segment.natural_center_of_mass.interpolate(), N, expand=False)

    M = np.array(
        [
            [
                5.00000000e-01,
                0.00e00,
                0.00e00,
                -0.00e00,
                -0.00e00,
                -0.00e00,
                0.00e00,
                0.00e00,
                0.00e00,
                -0.00e00,
                -0.00e00,
                -0.00e00,
            ],
            [
                0.00e00,
                5.00000000e-01,
                0.00e00,
                -0.00e00,
                -0.00e00,
                -0.00e00,
                0.00e00,
                0.00e00,
                0.00e00,
                -0.00e00,
                -0.00e00,
                -0.00e00,
            ],
            [
                0.00e00,
                0.00e00,
                5.00000000e-01,
                -0.00e00,
                -0.00e00,
                -0.00e00,
                0.00e00,
                0.00e00,
                0.00e00,
                -0.00e00,
                -0.00e00,
                -0.00e00,
            ],
            [
                -0.00e00,
                -0.00e00,
                -0.00e00,
                1.52010000e00,
                0.00e00,
                0.00e00,
                -5.10100000e-01,
                -0.00e00,
                -0.00e00,
                -0.00e00,
                -0.00e00,
                -0.00e00,
            ],
            [
                -0.00e00,
                -0.00e00,
                -0.00e00,
                0.00e00,
                1.52010000e00,
                0.00e00,
                -0.00e00,
                -5.10100000e-01,
                -0.00e00,
                -0.00e00,
                -0.00e00,
                -0.00e00,
            ],
            [
                -0.00e00,
                -0.00e00,
                -0.00e00,
                0.00e00,
                0.00e00,
                1.52010000e00,
                -0.00e00,
                -0.00e00,
                -5.10100000e-01,
                -0.00e00,
                -0.00e00,
                -0.00e00,
            ],
            [
                0.00e00,
                0.00e00,
                0.00e00,
                -5.10100000e-01,
                -0.00e00,
                -0.00e00,
                5.00100000e-01,
                0.00e00,
                0.00e00,
                0.00e00,
                0.00e00,
                0.00e00,
            ],
            [
                0.00e00,
                0.00e00,
                0.00e00,
                -0.00e00,
                -5.10100000e-01,
                -0.00e00,
                0.00e00,
                5.00100000e-01,
                0.00e00,
                0.00e00,
                0.00e00,
                0.00e00,
            ],
            [
                0.00e00,
                0.00e00,
                0.00e00,
                -0.00e00,
                -0.00e00,
                -5.10100000e-01,
                0.00e00,
                0.00e00,
                5.00100000e-01,
                0.00e00,
                0.00e00,
                0.00e00,
            ],
            [
                -0.00e00,
                -0.00e00,
                -0.00e00,
                -0.00e00,
                -0.00e00,
                -0.00e00,
                0.00e00,
                0.00e00,
                0.00e00,
                5.00000000e-01,
                0.00e00,
                0.00e00,
            ],
            [
                -0.00e00,
                -0.00e00,
                -0.00e00,
                -0.00e00,
                -0.00e00,
                -0.00e00,
                0.00e00,
                0.00e00,
                0.00e00,
                0.00e00,
                5.00000000e-01,
                0.00e00,
            ],
            [
                -0.00e00,
                -0.00e00,
                -0.00e00,
                -0.00e00,
                -0.00e00,
                -0.00e00,
                0.00e00,
                0.00e00,
                0.00e00,
                0.00e00,
                0.00e00,
                5.00000000e-01,
            ],
        ]
    )

    TestUtils.assert_equal(my_segment.mass_matrix, M, expand=False)

    J = np.array([[0.5, -0.0, -0.0], [-0.0, 0.5001, -0.0], [-0.0, -0.0, 0.5]])

    TestUtils.assert_equal(my_segment._natural_pseudo_inertia, J, expand=False)

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

    TestUtils.assert_equal(my_segment.kinetic_energy(Qdoti), np.array(0.6755135), expand=False)
    TestUtils.assert_equal(my_segment.potential_energy(Qi), np.array(2.24649), expand=False)

    my_segment2 = NaturalSegment.with_cartesian_inertial_parameters(
        name="box",
        alpha=np.pi / 2 + 0.1,
        beta=np.pi / 2 - 0.05,
        gamma=np.pi / 2 + 0.01,
        length=1,
        mass=1,
        center_of_mass=np.array([0, 0.01, 0]),
        inertia=np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
        inertial_transformation_matrix=TransformationMatrixType.Buv,
    )
    mat = my_segment2.segment_coordinates_system(Qi)
    TestUtils.assert_equal(
        mat,
        np.array(
            [
                [0.11, -0.09890496, 0.39714038, 0.21],
                [0.12, -0.09880496, 0.40670988, 0.22],
                [0.13, -0.09870496, 0.41627937, 0.23],
                [0.0, 0.0, 0.0, 1.0],
            ]
        ),
    )


@pytest.mark.parametrize(
    "bionc_type",
    [
        "numpy",
        "casadi",
    ],
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
    my_segment = NaturalSegment.with_cartesian_inertial_parameters(
        name="Thigh",
        alpha=np.pi / 2,
        beta=np.pi / 2,
        gamma=np.pi / 2,
        length=1,
        mass=1,
        center_of_mass=np.array([0, 0.01, 0]),
        inertia=np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
        inertial_transformation_matrix=TransformationMatrixType.Buv,
    )

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
    my_segment = NaturalSegment.with_cartesian_inertial_parameters(
        name="Thigh",
        alpha=np.pi / 2,
        beta=np.pi / 2,
        gamma=np.pi / 2,
        length=1,
        mass=1,
        center_of_mass=np.array([0, 0.01, 0]),
        inertia=np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
        inertial_transformation_matrix=TransformationMatrixType.Buv,
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

    # u and v both sit pi/5 away from w, yet pi/1.5 apart from each other: impossible in 3D
    with pytest.raises(
        ValueError,
        match="The angles alpha, beta, gamma, would produce a singular transformation matrix for the segment",
    ):
        NaturalSegment(
            name="bbox",
            alpha=np.pi / 5,
            beta=np.pi / 5,
            gamma=np.pi / 1.5,
            length=1.5,
        )

    # a realisable, non-degenerate set of angles must be accepted
    NaturalSegment(
        name="bbox",
        alpha=np.pi / 5,
        beta=np.pi / 3,
        gamma=np.pi / 2.1,
        length=1.5,
    )


@pytest.mark.parametrize(
    "bionc_type",
    ["numpy", "casadi"],
)
def test_center_of_mass(bionc_type):
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

    seg = NaturalSegment.with_cartesian_inertial_parameters(
        name="bbox",
        alpha=np.pi / 2 + 0.1,
        beta=np.pi / 2 - 0.05,
        gamma=np.pi / 2.1,
        length=1.5,
        center_of_mass=np.array([0.1, 0.2, 0.3]),
        inertia=np.array([[0.01, 0.02, 0.03], [0.02, 0.04, 0.05], [0.03, 0.05, 0.06]]),
        mass=2.66,
        inertial_transformation_matrix=TransformationMatrixType.Buv,
    )

    n_com = seg.natural_center_of_mass
    gravity_force = seg.gravity_force()
    inertia = seg.natural_pseudo_inertia
    M = seg.mass_matrix

    TestUtils.assert_equal(n_com, np.array([0.1, 0.126213, 0.310178]), expand=False)
    TestUtils.assert_equal(
        gravity_force,
        np.array([0.0, 0.0, -2.60946, 0.0, 0.0, -29.388084, 0.0, 0.0, 3.293484, 0.0, 0.0, -8.093961]),
        expand=False,
    )
    TestUtils.assert_equal(
        inertia,
        np.array(
            [
                [0.0716, 0.01682972, 0.04829124],
                [0.01682972, 0.05133404, 0.07452659],
                [0.04829124, 0.07452659, 0.2477153],
            ]
        ),
    )
    TestUtils.assert_equal(
        M,
        np.array(
            [
                [
                    7.16000000e-02,
                    0.0,
                    0.0,
                    2.82829718e-01,
                    0.0,
                    0.0,
                    -1.68297179e-02,
                    0.0,
                    0.0,
                    4.82912418e-02,
                    -0.0,
                    -0.0,
                ],
                [
                    0.0,
                    7.16000000e-02,
                    0.0,
                    0.0,
                    2.82829718e-01,
                    0.0,
                    0.0,
                    -1.68297179e-02,
                    0.0,
                    -0.0,
                    4.82912418e-02,
                    -0.0,
                ],
                [
                    0.0,
                    0.0,
                    7.16000000e-02,
                    0.0,
                    0.0,
                    2.82829718e-01,
                    0.0,
                    0.0,
                    -1.68297179e-02,
                    -0.0,
                    -0.0,
                    4.82912418e-02,
                ],
                [
                    2.82829718e-01,
                    0.0,
                    0.0,
                    3.38278849e00,
                    0.0,
                    0.0,
                    -3.87061262e-01,
                    -0.0,
                    -0.0,
                    8.99599037e-01,
                    0.0,
                    0.0,
                ],
                [
                    0.0,
                    2.82829718e-01,
                    0.0,
                    0.0,
                    3.38278849e00,
                    0.0,
                    -0.0,
                    -3.87061262e-01,
                    -0.0,
                    0.0,
                    8.99599037e-01,
                    0.0,
                ],
                [
                    0.0,
                    0.0,
                    2.82829718e-01,
                    0.0,
                    0.0,
                    3.38278849e00,
                    -0.0,
                    -0.0,
                    -3.87061262e-01,
                    0.0,
                    0.0,
                    8.99599037e-01,
                ],
                [
                    -1.68297179e-02,
                    0.0,
                    0.0,
                    -3.87061262e-01,
                    -0.0,
                    -0.0,
                    5.13340392e-02,
                    0.0,
                    0.0,
                    -7.45265915e-02,
                    0.0,
                    0.0,
                ],
                [
                    0.0,
                    -1.68297179e-02,
                    0.0,
                    -0.0,
                    -3.87061262e-01,
                    -0.0,
                    0.0,
                    5.13340392e-02,
                    0.0,
                    0.0,
                    -7.45265915e-02,
                    0.0,
                ],
                [
                    0.0,
                    0.0,
                    -1.68297179e-02,
                    -0.0,
                    -0.0,
                    -3.87061262e-01,
                    0.0,
                    0.0,
                    5.13340392e-02,
                    0.0,
                    0.0,
                    -7.45265915e-02,
                ],
                [
                    4.82912418e-02,
                    -0.0,
                    -0.0,
                    8.99599037e-01,
                    0.0,
                    0.0,
                    -7.45265915e-02,
                    0.0,
                    0.0,
                    2.47715304e-01,
                    0.0,
                    0.0,
                ],
                [
                    -0.0,
                    4.82912418e-02,
                    -0.0,
                    0.0,
                    8.99599037e-01,
                    0.0,
                    0.0,
                    -7.45265915e-02,
                    0.0,
                    0.0,
                    2.47715304e-01,
                    0.0,
                ],
                [
                    -0.0,
                    -0.0,
                    4.82912418e-02,
                    0.0,
                    0.0,
                    8.99599037e-01,
                    0.0,
                    0.0,
                    -7.45265915e-02,
                    0.0,
                    0.0,
                    2.47715304e-01,
                ],
            ]
        ),
        expand=False,
    )


@pytest.mark.parametrize(
    "bionc_type",
    ["numpy", "casadi"],
)
def test_potential_energy_is_m_g_z(bionc_type):
    if bionc_type == "casadi":
        from bionc.bionc_casadi import NaturalSegment, SegmentNaturalCoordinates
    else:
        from bionc.bionc_numpy import NaturalSegment, SegmentNaturalCoordinates

    mass = 2.5
    segment = NaturalSegment.with_cartesian_inertial_parameters(
        name="box",
        alpha=np.pi / 2 + 0.1,
        beta=np.pi / 2 - 0.2,
        gamma=np.pi / 2 + 0.3,
        length=0.7,
        mass=mass,
        center_of_mass=np.array([0.1, 0.2, -0.3]),  # off-axis
        inertia=np.array([[0.05, 0, 0], [0, 0.06, 0], [0, 0, 0.07]]),
        inertial_transformation_matrix=TransformationMatrixType.Buv,
    )

    Qi = SegmentNaturalCoordinates.from_components(
        u=np.array([0.11, 0.12, 0.13]),
        rp=np.array([0.21, 0.22, 0.23]),
        rd=np.array([0.31, 0.32, 0.33]),
        w=np.array([0.41, 0.42, 0.43]),
    )

    # z coordinate of the center of mass: z_C = (N_C Q)_z
    z_C = (segment.natural_center_of_mass.interpolate() @ Qi.vector)[2]

    # (casadi: TestUtils.assert_equal would compare an MX expected value with itself, so evaluate it first)
    if bionc_type == "casadi":
        z_C = TestUtils.mx_to_array(z_C, expand=False)
    TestUtils.assert_equal(segment.potential_energy(Qi), np.array(mass * 9.81 * z_C), expand=False)


def test_passive_pendulum_conserves_energy():
    from bionc import NaturalAxis, CartesianAxis, RK4
    from bionc.bionc_numpy import (
        BiomechanicalModel,
        NaturalSegment,
        JointType,
        SegmentNaturalCoordinates,
        NaturalCoordinates,
        NaturalVelocities,
    )

    model = BiomechanicalModel()
    model["pendulum"] = NaturalSegment.with_cartesian_inertial_parameters(
        name="pendulum",
        alpha=np.pi / 2,
        beta=np.pi / 2,
        gamma=np.pi / 2,
        length=1,
        mass=1,
        center_of_mass=np.array([0.1, 0.1, -0.1]),
        inertia=np.array([[0.05, 0, 0], [0, 0.05, 0], [0, 0, 0.05]]),
        inertial_transformation_matrix=TransformationMatrixType.Buv,
    )
    model._add_joint(
        dict(
            name="hinge",
            joint_type=JointType.GROUND_REVOLUTE,
            parent="GROUND",
            child="pendulum",
            parent_axis=[CartesianAxis.X, CartesianAxis.X],
            child_axis=[NaturalAxis.V, NaturalAxis.W],
            theta=[np.pi / 2, np.pi / 2],
        )
    )

    # hanging pendulum rotated by 45 deg around x, at rest
    a = np.pi / 4
    Q = NaturalCoordinates(
        SegmentNaturalCoordinates.from_components(
            u=[1, 0, 0], rp=[0, 0, 0], rd=[0, -np.cos(a), -np.sin(a)], w=[0, -np.sin(a), np.cos(a)]
        )
    )
    states_0 = np.concatenate((Q.to_array(), np.zeros(model.nb_Qdot)))

    def dynamics(t, states):
        qddot, _ = model.forward_dynamics(
            NaturalCoordinates(states[: model.nb_Q]), NaturalVelocities(states[model.nb_Q :])
        )
        return np.concatenate((states[model.nb_Q :], qddot.to_array()))

    drifts = []
    potential_range = []
    for steps_per_second in (200, 400):
        t = np.linspace(0, 2, 2 * steps_per_second + 1)
        states = RK4(t=t, f=dynamics, y0=states_0)
        V = np.array([model.potential_energy(NaturalCoordinates(states[: model.nb_Q, i])) for i in range(len(t))])
        T = np.array([model.kinetic_energy(NaturalVelocities(states[model.nb_Q :, i])) for i in range(len(t))])
        E = T + V
        drifts.append(np.max(np.abs(E - E[0])))
        potential_range.append(V.max() - V.min())

    # the pendulum really swings, and the energy is conserved up to the integration error
    assert potential_range[0] > 1
    assert drifts[0] < 1e-3 * potential_range[0]
    # RK4 is 4th order: halving the step divides the drift by ~16
    assert drifts[0] / drifts[1] > 8
