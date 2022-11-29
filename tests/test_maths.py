import numpy as np
import pytest
from .utils import TestUtils


@pytest.mark.parametrize(
    "bionc_type",
    ["numpy", "casadi"],
)
def test_vnop(bionc_type):
    if bionc_type == "casadi":
        from bionc.bionc_casadi import (
            SegmentNaturalCoordinates,
        )
    else:
        from bionc.bionc_numpy import (
            SegmentNaturalCoordinates,
        )
    e1 = np.array([1, 0, 0])
    e2 = np.array([0, 1, 0])
    e3 = np.array([0, 0, 1])
    V = np.array([1, 1, 1])
    vnop = SegmentNaturalCoordinates.vnop_array(V, e1, e2, e3)
    TestUtils.assert_equal(vnop, np.array([1, 1, 1]).T[:, np.newaxis], squeeze=False)

    e1 = np.array([1, 4, 5])
    e2 = np.array([2, -3, 6])
    e3 = np.array([3, 2, -7])
    V = np.array([-1, 3, 5])
    vnop = SegmentNaturalCoordinates.vnop_array(V, e1, e2, e3)
    TestUtils.assert_equal(vnop, np.array([0.752475, -0.267327, -0.405941]).T[:, np.newaxis], squeeze=False)

    # tests errors with wrong shape
    e1 = np.array([1, 4, 5, 6])
    e2 = np.array([2, -3, 6])
    e3 = np.array([3, 2, -7])
    V = np.array([-1, 3, 5])
    with pytest.raises(ValueError):
        SegmentNaturalCoordinates.vnop_array(V, e1, e2, e3)

    # e2 is wrong shape
    e1 = np.array([1, 4, 5])
    e2 = np.array([2, -3, 6, 7])

    with pytest.raises(ValueError):
        SegmentNaturalCoordinates.vnop_array(V, e1, e2, e3)

    # e3 is wrong shape
    e3 = np.array([3, 2, -7, 8])
    e2 = np.array([2, -3, 6])
    with pytest.raises(ValueError):
        SegmentNaturalCoordinates.vnop_array(V, e1, e2, e3)

    # V is wrong shape
    V = np.array([-1, 3, 5, 6])
    e3 = np.array([3, 2, -7])
    with pytest.raises(ValueError):
        SegmentNaturalCoordinates.vnop_array(V, e1, e2, e3)

    if bionc_type == "numpy":
        e1 = np.array([[1, 4, 5], [0.5, 5, 6]]).T
        e2 = np.array([[2, -3, 6], [2, -2.3, 7]]).T
        e3 = np.array([[3, 2, -7], [1, 1.9, -8]]).T
        V = np.array([[-1, 3, 5], [-0.5, 3, 6]]).T
        vnop = SegmentNaturalCoordinates.vnop_array(V, e1, e2, e3)
        TestUtils.assert_equal(
            vnop,
            np.array([[0.752475, -0.267327, -0.405941], [0.668505, -0.203698, -0.426857]]).T,
            decimal=5,
        )


@pytest.mark.parametrize(
    "bionc_type",
    ["numpy", "casadi"],
)
def test_interpolate_natural_vector(bionc_type):
    if bionc_type == "casadi":
        from bionc.bionc_casadi import (
            NaturalVector,
        )
    else:
        from bionc.bionc_numpy import (
            NaturalVector,
        )

    vector = np.array([1, 2, 3])
    interpolation_matrix = NaturalVector(vector).interpolate()
    TestUtils.assert_equal(
        interpolation_matrix,
        np.array(
            [
                [1.0, 0.0, 0.0, 3.0, 0.0, 0.0, -2.0, -0.0, -0.0, 3.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 3.0, 0.0, -0.0, -2.0, -0.0, 0.0, 3.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0, 3.0, -0.0, -0.0, -2.0, 0.0, 0.0, 3.0],
            ]
        ),
    )
