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

        if not isinstance(input_array, MX):
            raise TypeError("input_array must be a MX")

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
        if not isinstance(x, MX):
            raise TypeError("x must be a array")
        if not isinstance(y, MX):
            raise TypeError("y must be a array")
        if not isinstance(z, MX):
            raise TypeError("z must be a array")
        if not isinstance(t, MX):
            raise TypeError("t must be a array")

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
        if not isinstance(rotation, MX):
            raise TypeError("r must be a array")
        if not isinstance(translation, MX):
            raise TypeError("t must be a array")

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
        return self[:3, :3].to_array()

    @property
    def translation(self):
        return self[3, 0:3].to_array()

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
