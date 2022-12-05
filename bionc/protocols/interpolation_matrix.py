from abc import ABC, abstractmethod
from .natural_vector import AbstractNaturalVector


class AbstractInterpolationMatrix(ABC):
    """
    Abstract class used to create an interpolation matrix, a matrix that is used to interpolate the position of a point in a segment
    such that: point = interpolation_matrix * Q, where Q is the vector of [u, rp, rd, w].

    As for homogenous matrix, there is translation and rotation.
    The translation is called through the .trans method, while the rotation is called through the .rot method.

    Given a interpolation matrix N and generalized coordinates Q,
    the position of a point P is given by: P = N * Q

    if one want to rotate frames for example to turn a vector A one can use:
    A = N.rot * Q

    if one want to translate frames for example, one can use:
    A = N.trans * Q

    Methods
    -------
    from_natural_vector(natural_vector: AbstractNaturalVector)
        This function creates an interpolation matrix
        from a natural vector (i.e: a vector expressed in the natural coordinate system of a segment (rp, u, v ,w)
    rot
        Returns the rotation matrix of the interpolation matrix
    trans
        Returns the translation matrix of the interpolation matrix
    """

    @abstractmethod
    def from_natural_vector(cls, natural_vector: AbstractNaturalVector):
        """This function converts the natural vector into the interpolation matrix"""

    @abstractmethod
    def rot(self):
        """This function returns the rotation matrix part of the interpolation matrix"""

    @abstractmethod
    def trans(self):
        """This function returns the translation matrix part of the interpolation matrix"""
