import numpy as np
from bionc import vnop_array
import pytest
from .utils import TestUtils


def test_vnop():
    e1 = np.array([1, 0, 0])
    e2 = np.array([0, 1, 0])
    e3 = np.array([0, 0, 1])
    V = np.array([1, 1, 1])
    vnop = vnop_array(V, e1, e2, e3)
    assert np.allclose(vnop, np.array([1, 1, 1]))

    e1 = np.array([1, 4, 5])
    e2 = np.array([2, -3, 6])
    e3 = np.array([3, 2, -7])
    V = np.array([-1, 3, 5])
    vnop = vnop_array(V, e1, e2, e3)
    np.testing.assert_allclose(vnop, np.array([0.752475, -0.267327, -0.405941])[:, np.newaxis], rtol=1e-6)

    # tests errors with wrong shape
    e1 = np.array([1, 4, 5, 6])
    e2 = np.array([2, -3, 6])
    e3 = np.array([3, 2, -7])
    V = np.array([-1, 3, 5])
    with pytest.raises(ValueError):
        vnop_array(V, e1, e2, e3)

    # e2 is wrong shape
    e1 = np.array([1, 4, 5])
    e2 = np.array([2, -3, 6, 7])

    with pytest.raises(ValueError):
        vnop_array(V, e1, e2, e3)

    # e3 is wrong shape
    e3 = np.array([3, 2, -7, 8])
    e2 = np.array([2, -3, 6])
    with pytest.raises(ValueError):
        vnop_array(V, e1, e2, e3)

    # V is wrong shape
    V = np.array([-1, 3, 5, 6])
    e3 = np.array([3, 2, -7])
    with pytest.raises(ValueError):
        vnop_array(V, e1, e2, e3)

    e1 = np.array([[1, 4, 5], [0.5, 5, 6]]).T
    e2 = np.array([[2, -3, 6], [2, -2.3, 7]]).T
    e3 = np.array([[3, 2, -7], [1, 1.9, -8]]).T
    V = np.array([[-1, 3, 5], [-0.5, 3, 6]]).T
    vnop = vnop_array(V, e1, e2, e3)
    np.testing.assert_allclose(
        vnop,
        np.array([[0.752475, -0.267327, -0.405941], [0.668505, -0.203698, -0.426857]]).T,
        rtol=1e-5,
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

