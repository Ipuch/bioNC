import numpy as np
import pytest
from .utils import TestUtils


@pytest.mark.parametrize(
    "bionc_type",
    ["numpy", "casadi"],
)
def test_homogenous_transform(bionc_type):
    if bionc_type == "casadi":
        from bionc.bionc_casadi import HomogeneousTransform
    else:
        from bionc.bionc_numpy import HomogeneousTransform

    # test constructor
    ht = HomogeneousTransform(np.eye(4))
    assert np.allclose(TestUtils.to_array(ht.rot), np.eye(3))
    assert np.allclose(TestUtils.to_array(ht.translation), np.zeros(3))

    # test from_components
    x = np.array([[1], [0], [0]])
    y = np.array([[0], [1], [0]])
    z = np.array([[0], [0], [1]])
    t = np.array([[0], [0], [0]])
    ht = HomogeneousTransform.from_components(x, y, z, t)
    assert np.allclose(TestUtils.to_array(ht.rot), np.eye(3))
    assert np.allclose(TestUtils.to_array(ht.translation), np.zeros(3))

    # test from_rt
    r = np.eye(3)
    t = np.zeros((3, 1))
    ht = HomogeneousTransform.from_rt(r, t)
    assert np.allclose(TestUtils.to_array(ht.rot), np.eye(3))
    assert np.allclose(TestUtils.to_array(ht.translation), np.zeros(3))

    # test eye
    ht = HomogeneousTransform.eye()
    assert np.allclose(TestUtils.to_array(ht.rot), np.eye(3))
    assert np.allclose(TestUtils.to_array(ht.translation), np.zeros(3))
    assert np.allclose(TestUtils.to_array(ht), TestUtils.to_array(HomogeneousTransform(np.eye(4))))

    # test inv
    x = np.array([[1], [0], [0]])
    y = np.array([[0], [np.sqrt(2) / 2], [-np.sqrt(2) / 2]])
    z = np.array([[0], [np.sqrt(2) / 2], [np.sqrt(2) / 2]])
    t = np.array([[2], [3], [1]])
    ht = HomogeneousTransform.from_components(x, y, z, t)
    ht_inv = ht.inv()
    ht_inv_np = np.linalg.inv(TestUtils.to_array(ht))
    assert np.allclose(TestUtils.to_array(ht_inv), ht_inv_np)
    assert np.allclose(TestUtils.to_array(ht_inv.rot), TestUtils.to_array(ht.rot.T))
    assert np.allclose(TestUtils.to_array(ht_inv.translation), TestUtils.to_array(-ht_inv.rot @ ht.translation))
    assert np.allclose(TestUtils.to_array(ht.translation), t.squeeze())

    # test __getitem__
    ht = HomogeneousTransform.eye()
    assert np.allclose(TestUtils.to_array(ht[0, 0]), 1)
    assert np.allclose(TestUtils.to_array(ht[1, 1]), 1)
    assert np.allclose(TestUtils.to_array(ht[2, 2]), 1)
    assert np.allclose(TestUtils.to_array(ht[3, 3]), 1)

    # test __setitem__
    ht = HomogeneousTransform.eye()
    ht[0, 0] = 2
    ht[1, 1] = 2
    ht[2, 2] = 2
    ht[3, 3] = 2
    assert np.allclose(TestUtils.to_array(ht[0, 0]), 2)
    assert np.allclose(TestUtils.to_array(ht[1, 1]), 2)
    assert np.allclose(TestUtils.to_array(ht[2, 2]), 2)
    assert np.allclose(TestUtils.to_array(ht[3, 3]), 2)

    # test __add__
    ht1 = HomogeneousTransform.eye()
    ht2 = HomogeneousTransform.eye()
    ht3 = ht1 + ht2
    assert np.allclose(TestUtils.to_array(ht3[0, 0]), 2)
    assert np.allclose(TestUtils.to_array(ht3[1, 1]), 2)
    assert np.allclose(TestUtils.to_array(ht3[2, 2]), 2)
    assert np.allclose(TestUtils.to_array(ht3[3, 3]), 2)

    # test __sub__
    ht1 = HomogeneousTransform.eye()
    ht2 = HomogeneousTransform.eye()
    ht3 = ht1 - ht2
    assert np.allclose(TestUtils.to_array(ht3[0, 0]), 0)
    assert np.allclose(TestUtils.to_array(ht3[1, 1]), 0)
    assert np.allclose(TestUtils.to_array(ht3[2, 2]), 0)
    assert np.allclose(TestUtils.to_array(ht3[3, 3]), 0)

    # test __mul__
    ht1 = HomogeneousTransform.eye()
    ht2 = HomogeneousTransform.eye()
    ht3 = ht1 * ht2
    assert np.allclose(TestUtils.to_array(ht3[0, 0]), 1)
    assert np.allclose(TestUtils.to_array(ht3[1, 1]), 1)
    assert np.allclose(TestUtils.to_array(ht3[2, 2]), 1)
    assert np.allclose(TestUtils.to_array(ht3[3, 3]), 1)
