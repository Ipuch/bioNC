from typing import Union
import numpy as np
from numpy import eye
from ..protocols.natural_vector import AbstractNaturalVector
from ..protocols.interpolation_matrix import AbstractInterpolationMatrix
from ..utils.enums import NaturalAxis


class NaturalVector(AbstractNaturalVector, np.ndarray):
    """
    Class used to create a natural vector, a vector that is expressed in the natural coordinate system of a segment
    """

    def __new__(cls, input_array: Union[np.ndarray, list, tuple]):
        """
        Create a new instance of the class.
        """

        if not isinstance(input_array, np.ndarray):
            input_array = np.array(input_array)

        size1 = input_array.shape[0]
        size2 = input_array.shape[1] if input_array.shape.__len__() == 2 else 1

        if size1 != 3:
            raise ValueError("The input array must have 3 elements")

        if size2 != 1:
            raise ValueError("The position must be a 3d vector with only one column")

        obj = np.asarray(input_array).view(cls)

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
    def axis(cls, axis: NaturalAxis):
        if axis == NaturalAxis.U:
            return cls(np.array([1, 0, 0]))
        elif axis == NaturalAxis.V:
            return cls(np.array([0, 1, 0]))
        elif axis == NaturalAxis.W:
            return cls(np.array([0, 0, 1]))

    def interpolate(self) -> np.ndarray:
        """This function converts the natural vector into the interpolation matrix"""
        interpolation_matrix = np.zeros((3, 12))
        interpolation_matrix[0:3, 0:3] = self[0] * eye(3)
        interpolation_matrix[0:3, 3:6] = (1 + self[1]) * eye(3)
        interpolation_matrix[0:3, 6:9] = -self[1] * eye(3)
        interpolation_matrix[0:3, 9:12] = self[2] * eye(3)
        return InterpolationMatrix(interpolation_matrix)


class InterpolationMatrix(AbstractInterpolationMatrix, np.ndarray):
    """
    Class used to create an interpolation matrix, a matrix that is used to interpolate the position of a point in a segment
    such that: point = interpolation_matrix * Q, where Q is the vector of [u, rp, rd, w].

    As for homogenous matrix, there is translation and rotation.
    The translation is called through the .trans method, while the rotation is called through the .rot method.

    Given a interpolation matrix N and generalized coordinates Q,
    the position of a point P is given by: P = N * Q

    if one want to rotate frames for example to turn a vector A one can use:
    A = N.rot * Q

    if one want to translate frames for example, one can use:
    A = N.trans * Q

    Attributes
    ----------
    input_array : Union[np.ndarray, list, tuple]
        The input array must be a 3x12 matrix

    Methods
    -------
    from_natural_vector(natural_vector: NaturalVector)
        This function creates an interpolation matrix
        from a natural vector (i.e: a vector expressed in the natural coordinate system of a segment (rp, u, v ,w)
    rot
        Returns the rotation matrix of the interpolation matrix
    trans
        Returns the translation matrix of the interpolation matrix
    """

    def __new__(cls, input_array: Union[np.ndarray, list, tuple]):
        """
        Create a new instance of the class.
        """

        if not isinstance(input_array, np.ndarray):
            input_array = np.array(input_array)

        size1 = input_array.shape[0]
        size2 = input_array.shape[1] if input_array.shape.__len__() == 2 else 1

        if size1 != 3:
            raise ValueError("The input must be 3x12")

        if size2 != 12:
            raise ValueError("The input must be 3x12")

        obj = np.asarray(input_array).view(cls)

        return obj

    @classmethod
    def from_natural_vector(cls, natural_vector: NaturalVector):
        """This function converts the natural vector into the interpolation matrix"""
        interpolation_matrix = np.zeros((3, 12))
        interpolation_matrix[0:3, 0:3] = natural_vector[0] * eye(3)
        interpolation_matrix[0:3, 3:6] = (1 + natural_vector[1]) * eye(3)
        interpolation_matrix[0:3, 6:9] = -natural_vector[1] * eye(3)
        interpolation_matrix[0:3, 9:12] = natural_vector[2] * eye(3)
        return cls(interpolation_matrix)

    @property
    def rot(self) -> np.ndarray:
        """This function returns the rotation matrix part of the interpolation matrix"""
        rotation_interpolation_matrix = np.zeros((3, 12))
        rotation_interpolation_matrix[0:3, 0:3] = self[0:3, 0:3]
        rotation_interpolation_matrix[0:3, 3:6] = self[0:3, 3:6] - eye(3)
        rotation_interpolation_matrix[0:3, 6:9] = self[0:3, 6:9]
        rotation_interpolation_matrix[0:3, 9:12] = self[0:3, 9:12]

        return np.array(rotation_interpolation_matrix)

    @property
    def trans(self) -> np.ndarray:
        """This function returns the translation matrix part of the interpolation matrix"""
        translation_interpolation_matrix = np.zeros((3, 12))
        translation_interpolation_matrix[0:3, 0:3] = 0
        translation_interpolation_matrix[0:3, 3:6] = eye(3)
        translation_interpolation_matrix[0:3, 6:9] = 0
        translation_interpolation_matrix[0:3, 9:12] = 0

        return np.array(translation_interpolation_matrix)

    def to_array(self):
        return np.array(self)
