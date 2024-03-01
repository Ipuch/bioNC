from typing import Protocol
import numpy as np


class HomogeneousTransform(Protocol):
    """
    Homogenous transform class
    """

    def __new__(cls, input_array):
        """
        Create a new instance of the class.
        """
        ...

    @classmethod
    def from_components(cls, x, y, z, t):
        """
        Constructor of the class from the components of the homogenous transform
        """
        ...

    @classmethod
    def from_rt(cls, rotation, translation):
        """
        Constructor of the class from a rotation matrix and a translation vector
        """
        ...

    @classmethod
    def eye(cls):
        """
        Returns the identity homogenous transform
        """
        ...

    def rot(self): ...

    def translation(self): ...

    def inv(self):
        """
        Returns the inverse of the homogenous transform
        """
        ...

    def to_array(self): ...
