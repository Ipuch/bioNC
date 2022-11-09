import numpy as np

from bionc import bionc_numpy as bionc_np


# generate tests for the class to be use with pytest
def test_homogenous_transform():
    # test constructor
    ht = bionc_np.HomogeneousTransform(np.eye(4))
    assert np.allclose(ht.rot, np.eye(3))
    assert np.allclose(ht.translation, np.zeros(3))

    # test from_components
    x = np.array([[1], [0], [0]])
    y = np.array([[0], [1], [0]])
    z = np.array([[0], [0], [1]])
    t = np.array([[0], [0], [0]])
    ht = bionc_np.HomogeneousTransform.from_components(x, y, z, t)
    assert np.allclose(ht.rot, np.eye(3))
    assert np.allclose(ht.translation, np.zeros(3))

    # test from_rt
    r = np.eye(3)
    t = np.zeros((3, 1))
    ht = bionc_np.HomogeneousTransform.from_rt(r, t)
    assert np.allclose(ht.rot, np.eye(3))
    assert np.allclose(ht.translation, np.zeros(3))

    # test eye
    ht = bionc_np.HomogeneousTransform.eye()
    assert np.allclose(ht.rot, np.eye(3))
    assert np.allclose(ht.translation, np.zeros(3))
    assert np.allclose(ht.to_array(), bionc_np.HomogeneousTransform(np.eye(4)))

    # test inv
    x = np.array([[1], [0], [0]])
    y = np.array([[0], [np.sqrt(2) / 2], [-np.sqrt(2) / 2]])
    z = np.array([[0], [np.sqrt(2) / 2], [np.sqrt(2) / 2]])
    t = np.array([[2], [3], [1]])
    ht = bionc_np.HomogeneousTransform.from_components(x, y, z, t)
    ht_inv = ht.inv()
    ht_inv_np = np.linalg.inv(ht.to_array())
    assert np.allclose(ht_inv.to_array(), ht_inv_np)
    assert np.allclose(ht_inv.rot, ht.rot.T)
    assert np.allclose(ht_inv.translation, -ht_inv.rot @ ht.translation)

    # test __getitem__
    ht = bionc_np.HomogeneousTransform.eye()
    assert np.allclose(ht[0, 0], 1)
    assert np.allclose(ht[1, 1], 1)
    assert np.allclose(ht[2, 2], 1)
    assert np.allclose(ht[3, 3], 1)

    # test __setitem__
    ht = bionc_np.HomogeneousTransform.eye()
    ht[0, 0] = 2
    ht[1, 1] = 2
    ht[2, 2] = 2
    ht[3, 3] = 2
    assert np.allclose(ht[0, 0], 2)
    assert np.allclose(ht[1, 1], 2)
    assert np.allclose(ht[2, 2], 2)
    assert np.allclose(ht[3, 3], 2)

    # test __add__
    ht1 = bionc_np.HomogeneousTransform.eye()
    ht2 = bionc_np.HomogeneousTransform.eye()
    ht3 = ht1 + ht2
    assert np.allclose(ht3[0, 0], 2)
    assert np.allclose(ht3[1, 1], 2)
    assert np.allclose(ht3[2, 2], 2)
    assert np.allclose(ht3[3, 3], 2)

    # test __sub__
    ht1 = bionc_np.HomogeneousTransform.eye()
    ht2 = bionc_np.HomogeneousTransform.eye()
    ht3 = ht1 - ht2
    assert np.allclose(ht3[0, 0], 0)
    assert np.allclose(ht3[1, 1], 0)
    assert np.allclose(ht3[2, 2], 0)
    assert np.allclose(ht3[3, 3], 0)

    # test __mul__
    ht1 = bionc_np.HomogeneousTransform.eye()
    ht2 = bionc_np.HomogeneousTransform.eye()
    ht3 = ht1 * ht2
    assert np.allclose(ht3[0, 0], 1)
    assert np.allclose(ht3[1, 1], 1)
    assert np.allclose(ht3[2, 2], 1)
    assert np.allclose(ht3[3, 3], 1)

    # test __rmul__
    ht1 = bionc_np.HomogeneousTransform.eye()
    ht2 = bionc_np.HomogeneousTransform.eye()
    ht3 = ht1 * ht2
    assert np.allclose(ht3[0, 0], 1)
    assert np.allclose(ht3[1, 1], 1)
    assert np.allclose(ht3[2, 2], 1)
    assert np.allclose(ht3[3, 3], 1)
