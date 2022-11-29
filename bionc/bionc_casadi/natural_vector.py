from typing import Union
from casadi import MX, vertcat
import numpy as np
from ..protocols.natural_vector import AbstractNaturalVector


class NaturalVector(AbstractNaturalVector, MX):
    """
    Class used to create a natural vector, a vector that is expressed in the natural coordinate system of a segment
    """

    def __new__(cls, input_array: Union[MX, np.ndarray, list, tuple]):
        """
        Create a new instance of the class.
        """

        if not isinstance(input_array, (MX, np.ndarray)):
            input_array = np.array(input_array)

        obj = MX.__new__(cls)

        if isinstance(input_array, MX):
            size1 = input_array.shape[0]
            size2 = input_array.shape[1]
        else:
            size1 = input_array.shape[0]
            size2 = input_array.shape[1] if input_array.shape.__len__() == 2 else 1

        if size1 != 3:
            raise ValueError("The input array must have 3 elements")

        if size2 != 1:
            raise ValueError("The position must be a 3d vector with only one column")

        return obj

    @classmethod
    def proximal(cls):
        """This function returns the vector of the proximal point, denoted rp"""
        return cls(np.array([0, 0, 0]))

    @classmethod
    def distal(cls):
        """This function returns the vector of the distal point, denoted rd"""
        return cls(np.array([0, -1, 0]))

    @classmethod
    def u_axis(cls):
        """This function returns the vector of the u axis"""
        return cls(np.array([1, 0, 0]))

    @classmethod
    def w_axis(cls):
        """This function returns the vector of the w axis"""
        return cls(np.array([0, 0, 1]))

    def interpolate(self) -> MX:
        """This function converts the natural vector into the interpolation matrix"""
        interpolation_matrix = MX.zeros((3, 12))
        interpolation_matrix[0:3, 0:3] = self[0] * MX.eye(3)
        interpolation_matrix[0:3, 3:6] = (1 + self[1]) * MX.eye(3)
        interpolation_matrix[0:3, 6:9] = -self[1] * MX.eye(3)
        interpolation_matrix[0:3, 9:12] = self[2] * MX.eye(3)
        return interpolation_matrix
