import pytest
import numpy as np
from .utils import TestUtils


@pytest.mark.parametrize(
    "bionc_type",
    ["numpy", "casadi"],
)
def test_natural_vector(bionc_type):
    if bionc_type == "casadi":
        from bionc.bionc_casadi import (
            NaturalVector,
            NaturalMarker,
        )
    else:
        from bionc.bionc_numpy import (
            NaturalVector,
            NaturalMarker,
        )

    natural_vector = NaturalVector([1, 2, 3])
    TestUtils.assert_equal(natural_vector, np.array([1, 2, 3]))
    assert natural_vector.interpolate().shape == (3, 12)
    Nrot = np.concatenate((np.eye(3), 2 * np.eye(3), -2 * np.eye(3), 3 * np.eye(3)), axis=1)
    Ntrans = np.concatenate((np.zeros((3, 3)), 1 * np.eye(3), np.zeros((3, 3)), np.zeros((3, 3))), axis=1)
    TestUtils.assert_equal(natural_vector.interpolate(), Nrot + Ntrans)
    TestUtils.assert_equal(natural_vector.interpolate().rot, Nrot)
    TestUtils.assert_equal(natural_vector.interpolate().trans, Ntrans)

    natural_vector = NaturalVector.proximal()
    TestUtils.assert_equal(natural_vector, np.array([0, 0, 0]))
    assert natural_vector.interpolate().shape == (3, 12)
    N = np.concatenate((np.zeros((3, 3)), np.eye(3), np.zeros((3, 3)), np.zeros((3, 3))), axis=1)
    TestUtils.assert_equal(natural_vector.interpolate(), N)

    natural_vector = NaturalVector.distal()
    TestUtils.assert_equal(natural_vector, np.array([0, -1, 0]))
    assert natural_vector.interpolate().shape == (3, 12)
    N = np.concatenate((np.zeros((3, 3)), np.zeros((3, 3)), np.eye(3), np.zeros((3, 3))), axis=1)
    TestUtils.assert_equal(natural_vector.interpolate(), N)
