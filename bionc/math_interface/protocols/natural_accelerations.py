from typing import Protocol


class SegmentNaturalAccelerations(Protocol):
    """
    This class is made to handle Generalized Coordinates of a Segment
    """

    def __new__(cls, input_array):
        """Create a new instance of the class."""
        ...

    @classmethod
    def from_components(cls, uddot, rpddot, rdddot, wddot):
        """Constructor of the class from the components of the natural coordinates"""
        ...

    def to_array(self):
        """This function returns the array of the natural coordinates"""
        ...

    def uddot(self):
        """This property returns the u vector of the natural coordinates"""
        ...

    def rpddot(self):
        """This property returns the rp vector of the natural coordinates"""
        ...

    def rdddot(self):
        """This property returns the rd vector of the natural coordinates"""
        ...

    def wddot(self):
        """This property returns the w vector of the natural coordinates"""
        ...

    def vddot(self):
        """This property returns the v vector of the natural coordinates"""
        ...

    def vector(self):
        """This property returns the vector of the natural coordinates"""
        ...

    def to_components(self):
        """This function returns the components of the natural coordinates"""
        ...


class NaturalAccelerations(Protocol):
    """
    This class is made to handle Natural coordinates of several segments
    """

    def __new__(cls, input_array):
        """Create a new instance of the class."""
        ...

    @classmethod
    def from_qi(cls, tuple_of_Q):
        """Constructor of the class from the components of the natural coordinates"""
        ...

    def to_array(self):
        """This function returns the array of the natural coordinates"""
        ...

    def nb_qddoti(self):
        """This function returns the number of qi"""
    ...

    def uddot(self, segment_index):
        """This property returns the u vector of the natural coordinates"""
    ...

    def rpddot(self, segment_index):
        """This property returns the rp vector of the natural coordinates"""
    ...

    def rdddot(self, segment_index):
        """This property returns the rd vector of the natural coordinates"""
    ...

    def wddot(self, segment_index):
        """This property returns the w vector of the natural coordinates"""
    ...

    def vddot(self, segment_index):
        """This property returns the v vector of the natural coordinates"""
    ...

    def vector(self):
        """This property returns the vector of the natural coordinates"""
    ...
