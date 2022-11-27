from typing import Union
import numpy as np
from numpy import eye
from ..protocols.natural_vector import AbstractNaturalVector


class NaturalVector(AbstractNaturalVector, np.ndarray):
    """
    Class used to create a natural vector, a vector that is expressed in the natural coordinate system of a segment
    """

    def __new__(cls, input_array: Union[np.ndarray, list, tuple]):
        """
        Create a new instance of the class.
        """

        if len(input_array) != 3:
            raise ValueError("The input array must have 3 elements")

        obj = np.asarray(input_array).view(cls)

        # if obj.shape.__len__() == 1:
        #     obj = obj[:, np.newaxis]

        return obj

    @classmethod
    def proximal(cls):
        """ This function returns the vector of the proximal point, denoted rp"""
        return cls(np.array([0, 0, 0]))

    @classmethod
    def distal(cls):
        """ This function returns the vector of the distal point, denoted rd"""
        return cls(np.array([0, -1, 0]))

    @classmethod
    def u_axis(cls):
        """ This function returns the vector of the u axis"""
        return cls(np.array([1, 0, 0]))

    @classmethod
    def w_axis(cls):
        """ This function returns the vector of the w axis"""
        return cls(np.array([0, 0, 1]))

    def interpolation_matrix(self):
        """ This function converts the natural vector into the interpolation matrix """
        interpolation_matrix = np.zeros((3, 12))
        interpolation_matrix[0:3, 0:3] = self[0] * eye(3)
        interpolation_matrix[0:3, 3:6] = (1 + self[1]) * eye(3)
        interpolation_matrix[0:3, 6:9] = -self[1] * eye(3)
        interpolation_matrix[0:3, 9:12] = self[2] * eye(3)
        return interpolation_matrix
