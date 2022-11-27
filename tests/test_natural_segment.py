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
    N = np.array([[ 0.0,  0.0,  0.0,  1.010,
         0.0,  0.0, -1.0e-02, -0.0,
        -0.0, -6.123234e-19, -0.0, -0.0],
       [ 0.0,  0.0,  0.0,  0.0,
         1.010,  0.0, -0.0, -1.0e-02,
        -0.0, -0.0, -6.123234e-19, -0.0],
       [ 0.0,  0.0,  0.0,  0.0,
         0.0,  1.010, -0.0, -0.0,
        -1.0e-02, -0.0, -0.0, -6.123234e-19]])
    TestUtils.assert_equal(my_segment.natural_center_of_mass.interpolate(), N)

    M = np.array(
        [
            [
                1.0e0,
                0.0e0,
                0.0e0,
                -1.0e-04,
                -0.0e0,
                -0.0e0,
                1.0e-04,
                0.0e0,
                0.0e0,
                1.0e-04,
                0.0e0,
                0.0e0,
            ],
            [
                0.0e0,
                1.0e0,
                0.0e0,
                -0.0e0,
                -1.0e-04,
                -0.0e0,
                0.0e0,
                1.0e-04,
                0.0e0,
                0.0e0,
                1.0e-04,
                0.0e0,
            ],
            [
                0.0e0,
                0.0e0,
                1.0e0,
                -0.0e0,
                -0.0e0,
                -1.0e-04,
                0.0e0,
                0.0e0,
                1.0e-04,
                0.0e0,
                0.0e0,
                1.0e-04,
            ],
            [
                -1.0e-04,
                -0.0e0,
                -0.0e0,
                2.02e0,
                0.0e0,
                0.0e0,
                -1.01e0,
                -0.0e0,
                -0.0e0,
                -1.0e-04,
                -0.0e0,
                -0.0e0,
            ],
            [
                -0.0e0,
                -1.0e-04,
                -0.0e0,
                0.0e0,
                2.02e0,
                0.0e0,
                -0.0e0,
                -1.01e0,
                -0.0e0,
                -0.0e0,
                -1.0e-04,
                -0.0e0,
            ],
            [
                -0.0e0,
                -0.0e0,
                -1.0e-04,
                0.0e0,
                0.0e0,
                2.02e0,
                -0.0e0,
                -0.0e0,
                -1.01e0,
                -0.0e0,
                -0.0e0,
                -1.0e-04,
            ],
            [
                1.0e-04,
                0.0e0,
                0.0e0,
                -1.01e0,
                -0.0e0,
                -0.0e0,
                1.0e0,
                0.0e0,
                0.0e0,
                1.0e-04,
                0.0e0,
                0.0e0,
            ],
            [
                0.0e0,
                1.0e-04,
                0.0e0,
                -0.0e0,
                -1.01e0,
                -0.0e0,
                0.0e0,
                1.0e0,
                0.0e0,
                0.0e0,
                1.0e-04,
                0.0e0,
            ],
            [
                0.0e0,
                0.0e0,
                1.0e-04,
                -0.0e0,
                -0.0e0,
                -1.01e0,
                0.0e0,
                0.0e0,
                1.0e0,
                0.0e0,
                0.0e0,
                1.0e-04,
            ],
            [
                1.0e-04,
                0.0e0,
                0.0e0,
                -1.0e-04,
                -0.0e0,
                -0.0e0,
                1.0e-04,
                0.0e0,
                0.0e0,
                1.0e0,
                0.0e0,
                0.0e0,
            ],
            [
                0.0e0,
                1.0e-04,
                0.0e0,
                -0.0e0,
                -1.0e-04,
                -0.0e0,
                0.0e0,
                1.0e-04,
                0.0e0,
                0.0e0,
                1.0e0,
                0.0e0,
            ],
            [
                0.0e0,
                0.0e0,
                1.0e-04,
                -0.0e0,
                -0.0e0,
                -1.0e-04,
                0.0e0,
                0.0e0,
                1.0e-04,
                0.0e0,
                0.0e0,
                1.0e0,
            ],
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
            SegmentMarker,
        )
    else:
        from bionc.bionc_numpy import (
            NaturalSegment,
            SegmentNaturalCoordinates,
            SegmentNaturalVelocities,
            SegmentMarker,
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

    TestUtils.assert_equal(my_segment.nb_markers(), 2)
    TestUtils.assert_equal(
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

    TestUtils.assert_equal(
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
