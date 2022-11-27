from abc import ABC, abstractmethod


class AbstractNaturalVector(ABC):
    """
    Class used to create a natural vector, a vector that is expressed in the natural coordinate system of a segment
    """
    @abstractmethod
    def interpolation_matrix(self):
        """ This function converts the natural vector into the interpolation matrix """

    def proximal(self):
        """ This function returns the vector of the proximal point, denoted rp"""

    @abstractmethod
    def distal(self):
        """ This function returns the vector of the distal point, denoted rd"""

    @abstractmethod
    def u_axis(self):
        """ This function returns the vector of the u axis"""

    @abstractmethod
    def w_axis(self):
        """ This function returns the vector of the w axis"""


