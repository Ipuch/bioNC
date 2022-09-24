from typing import Union
import numpy as np


class HomogeneousTransform(np.ndarray):
    """
    Homogenous transform class
    """

    def __new__(cls, input_array: Union[np.ndarray, list, tuple]):
        """
        Create a new instance of the class.
        """
        if not isinstance(input_array, np.ndarray):
            raise TypeError("input_array must be a numpy array")

        if input_array.shape != (4, 4):
            raise ValueError("input_array must be a 4x4 numpy array")

        obj = np.asarray(input_array).view(cls)

        return obj

    @classmethod
    def from_components(cls, x: np.ndarray, y: np.ndarray, z: np.ndarray, t: np.ndarray):
        """
        Constructor of the class from the components of the homogenous transform

        Parameters
        ----------
        x: np.ndarray
            The x axis of the homogenous transform, a 3x1 numpy array
        y: np.ndarray
            The y axis of the homogenous transform, a 3x1 numpy array
        z: np.ndarray
            The z axis of the homogenous transform, a 3x1 numpy array
        t: np.ndarray
            translation vector, a 3x1 numpy array
        """
        if not isinstance(x, np.ndarray):
            raise TypeError("x must be a numpy array")
        if not isinstance(y, np.ndarray):
            raise TypeError("y must be a numpy array")
        if not isinstance(z, np.ndarray):
            raise TypeError("z must be a numpy array")
        if not isinstance(t, np.ndarray):
            raise TypeError("t must be a numpy array")

        if x.shape != (3, 1):
            raise ValueError("x must be a 3x1 numpy array")
        if y.shape != (3, 1):
            raise ValueError("y must be a 3x1 numpy array")
        if z.shape != (3, 1):
            raise ValueError("z must be a 3x1 numpy array")
        if t.shape != (3, 1):
            raise ValueError("t must be a 3x1 numpy array")

        input_array = np.concatenate((x, y, z, t), axis=1)
        input_array = np.concatenate((input_array, np.array([[0, 0, 0, 1]])), axis=0)
        return cls(input_array)

    @classmethod
    def from_rt(cls, rotation: np.ndarray, translation: np.ndarray):
        """
        Constructor of the class from a rotation matrix and a translation vector

        Parameters
        ----------
        rotation: np.ndarray
            A 3x3 rotation matrix
        translation: np.ndarray
            A 3x1 translation vector
        """
        if not isinstance(rotation, np.ndarray):
            raise TypeError("r must be a numpy array")
        if not isinstance(translation, np.ndarray):
            raise TypeError("t must be a numpy array")

        if rotation.shape != (3, 3):
            raise ValueError("r must be a 3x3 numpy array")
        if translation.shape != (3, 1):
            raise ValueError("t must be a 3x1 numpy array")

        input_array = np.concatenate((rotation, translation), axis=1)
        input_array = np.concatenate((input_array, np.array([[0, 0, 0, 1]])), axis=0)

        return cls(input_array)

    @classmethod
    def eye(cls):
        """
        Returns the identity homogenous transform
        """
        return cls(np.eye(4))

    @property
    def rot(self):
        return self[:3, :3].to_array()

    @property
    def translation(self):
        return self[3, 0:3].to_array()

    def inv(self):
        """
        Returns the inverse of the homogenous transform
        """
        inv_mat = np.zeros((4, 4))
        inv_mat[:3, :3] = self[:3, :3].T
        inv_mat[:3, 3] = -inv_mat[:3, :3] @ self[:3, 3]
        inv_mat[3, :] = np.array([0, 0, 0, 1])

        return HomogeneousTransform(inv_mat)

    def to_array(self):
        return np.array(self)


# generate tests for the class to be use with pytest
def test_homogenous_transform():
    # test constructor
    ht = HomogeneousTransform(np.eye(4))
    assert np.allclose(ht.rot, np.eye(3))
    assert np.allclose(ht.translation, np.zeros(3))

    # test from_components
    x = np.array([[1], [0], [0]])
    y = np.array([[0], [1], [0]])
    z = np.array([[0], [0], [1]])
    t = np.array([[0], [0], [0]])
    ht = HomogeneousTransform.from_components(x, y, z, t)
    assert np.allclose(ht.rot, np.eye(3))
    assert np.allclose(ht.translation, np.zeros(3))

    # test from_rt
    r = np.eye(3)
    t = np.zeros((3, 1))
    ht = HomogeneousTransform.from_rt(r, t)
    assert np.allclose(ht.rot, np.eye(3))
    assert np.allclose(ht.translation, np.zeros(3))

    # test eye
    ht = HomogeneousTransform.eye()
    assert np.allclose(ht.rot, np.eye(3))
    assert np.allclose(ht.translation, np.zeros(3))

    # test inv
    x = np.array([[1], [0], [0]])
    y = np.array([[0], [np.sqrt(2) / 2], [-np.sqrt(2) / 2]])
    z = np.array([[0], [np.sqrt(2) / 2], [np.sqrt(2) / 2]])
    t = np.array([[2], [3], [1]])
    ht = HomogeneousTransform.from_components(x, y, z, t)
    ht_inv = ht.inv()
    ht_inv_np = np.linalg.inv(ht.to_array())
    assert np.allclose(ht_inv.to_array(), ht_inv_np)
    assert np.allclose(ht_inv.rot, ht.rot.T)
    assert np.allclose(ht_inv.translation, -ht_inv.rot @ ht.translation)

    # test __getitem__
    ht = HomogeneousTransform.eye()
    assert np.allclose(ht[0, 0], 1)
    assert np.allclose(ht[1, 1], 1)
    assert np.allclose(ht[2, 2], 1)
    assert np.allclose(ht[3, 3], 1)

    # test __setitem__
    ht = HomogeneousTransform.eye()
    ht[0, 0] = 2
    ht[1, 1] = 2
    ht[2, 2] = 2
    ht[3, 3] = 2
    assert np.allclose(ht[0, 0], 2)
    assert np.allclose(ht[1, 1], 2)
    assert np.allclose(ht[2, 2], 2)
    assert np.allclose(ht[3, 3], 2)

    # test __add__
    ht1 = HomogeneousTransform.eye()
    ht2 = HomogeneousTransform.eye()
    ht3 = ht1 + ht2
    assert np.allclose(ht3[0, 0], 2)
    assert np.allclose(ht3[1, 1], 2)
    assert np.allclose(ht3[2, 2], 2)
    assert np.allclose(ht3[3, 3], 2)

    # test __sub__
    ht1 = HomogeneousTransform.eye()
    ht2 = HomogeneousTransform.eye()
    ht3 = ht1 - ht2
    assert np.allclose(ht3[0, 0], 0)
    assert np.allclose(ht3[1, 1], 0)
    assert np.allclose(ht3[2, 2], 0)
    assert np.allclose(ht3[3, 3], 0)

    # test __mul__
    ht1 = HomogeneousTransform.eye()
    ht2 = HomogeneousTransform.eye()
    ht3 = ht1 * ht2
    assert np.allclose(ht3[0, 0], 1)
    assert np.allclose(ht3[1, 1], 1)
    assert np.allclose(ht3[2, 2], 1)
    assert np.allclose(ht3[3, 3], 1)
