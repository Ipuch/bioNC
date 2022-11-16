from typing import Union
from casadi import MX, vertcat, horzcat, transpose
import numpy as np


class HomogeneousTransform(MX):
    """
    Homogenous transform class
    """

    def __new__(cls, input_array: MX):
        """
        Create a new instance of the class.
        """

        if input_array.shape != (4, 4):
            raise ValueError("input_array must be a 4x4 array")

        obj = MX.__new__(cls)

        return obj

    @classmethod
    def from_components(cls, x: MX, y: MX, z: MX, t: MX):
        """
        Constructor of the class from the components of the homogenous transform

        Parameters
        ----------
        x: MX
            The x axis of the homogenous transform, a 3x1 array
        y: MX
            The y axis of the homogenous transform, a 3x1 array
        z: MX
            The z axis of the homogenous transform, a 3x1 array
        t: MX
            translation vector, a 3x1 array
        """

        if x is None:
            raise ValueError("u must be a numpy array (3x1) or a list of 3 elements")
        if y is None:
            raise ValueError("rp must be a numpy array (3x1) or a list of 3 elements")
        if z is None:
            raise ValueError("rd must be a numpy array (3x1) or a list of 3 elements")
        if t is None:
            raise ValueError("w must be a numpy array (3x1) or a list of 3 elements")

        if not isinstance(x, MX):
            x = MX(x)
        if not isinstance(y, MX):
            y = MX(y)
        if not isinstance(z, MX):
            z = MX(z)
        if not isinstance(t, MX):
            t = MX(t)

        if x.shape != (3, 1):
            raise ValueError("x must be a 3x1 array")
        if y.shape != (3, 1):
            raise ValueError("y must be a 3x1 array")
        if z.shape != (3, 1):
            raise ValueError("z must be a 3x1 array")
        if t.shape != (3, 1):
            raise ValueError("t must be a 3x1 array")

        input_array = horzcat(*(x, y, z, t))
        input_array = vertcat(*(input_array, np.array([[0, 0, 0, 1]])))
        return cls(input_array)

    @classmethod
    def from_rt(cls, rotation: MX, translation: MX):
        """
        Constructor of the class from a rotation matrix and a translation vector

        Parameters
        ----------
        rotation: MX
            A 3x3 rotation matrix
        translation: MX
            A 3x1 translation vector
        """
        if rotation is None:
            raise ValueError("rotation must be a 3x3 array")
        if translation is None:
            raise ValueError("translation must be a 3x1 array")

        if not isinstance(rotation, MX):
            MX(rotation)
        if not isinstance(translation, MX):
            MX(translation)

        if rotation.shape != (3, 3):
            raise ValueError("r must be a 3x3 array")
        if translation.shape != (3, 1):
            raise ValueError("t must be a 3x1 array")

        input_array = horzcat(*(rotation, translation))
        input_array = vertcat(*(input_array, np.array([[0, 0, 0, 1]])))

        return cls(input_array)

    @classmethod
    def eye(cls):
        """
        Returns the identity homogenous transform
        """
        return cls(np.eye(4))

    @property
    def rot(self):
        return self[:3, :3]

    @property
    def translation(self):
        return self[3, 0:3]

    def inv(self):
        """
        Returns the inverse of the homogenous transform
        """
        inv_mat = MX.zeros((4, 4))
        inv_mat[:3, :3] = transpose(self[:3, :3])
        inv_mat[:3, 3] = -inv_mat[:3, :3] @ self[:3, 3]
        inv_mat[3, :] = MX([0, 0, 0, 1])

        return HomogeneousTransform(inv_mat)

    def to_array(self):
        return self
