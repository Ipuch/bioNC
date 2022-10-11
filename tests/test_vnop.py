import numpy as np
from bionc.utils.vnop_array import vnop_array
import pytest


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
    np.testing.assert_allclose(vnop, np.array([0.752475, -0.267327, -0.405941]), rtol=1e-6)

    # tests errors with wrong shape
    e1 = np.array([1, 4, 5, 6])
    e2 = np.array([2, -3, 6])
    e3 = np.array([3, 2, -7])
    V = np.array([-1, 3, 5])
    with pytest.raises(ValueError):
        vnop = vnop_array(V, e1, e2, e3)

    # e2 is wrong shape
    e1 = np.array([1, 4, 5])
    e2 = np.array([2, -3, 6, 7])

    with pytest.raises(ValueError):
        vnop = vnop_array(V, e1, e2, e3)

    # e3 is wrong shape
    e3 = np.array([3, 2, -7, 8])
    e2 = np.array([2, -3, 6])
    with pytest.raises(ValueError):
        vnop = vnop_array(V, e1, e2, e3)

    # V is wrong shape
    V = np.array([-1, 3, 5, 6])
    e3 = np.array([3, 2, -7])
    with pytest.raises(ValueError):
        vnop = vnop_array(V, e1, e2, e3)



