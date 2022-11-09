from typing import Protocol


class SegmentNaturalVelocities(Protocol):
    """
    This class is made to handle Generalized Coordinates of a Segment
    """

    def __new__(cls, input_array):
        """Create a new instance of the class."""
        ...

    @classmethod
    def from_components(cls, udot, rpdot, rddot, wdot):
        """Constructor of the class from the components of the natural coordinates"""
        ...

    def to_array(self):
        """This function returns the array of the natural coordinates"""
        ...

    def udot(self):
        """This property returns the u vector of the natural coordinates"""
        ...

    def rpdot(self):
        """This property returns the rp vector of the natural coordinates"""
        ...

    def rddot(self):
        """This property returns the rd vector of the natural coordinates"""
        ...

    def wdot(self):
        """This property returns the w vector of the natural coordinates"""
        ...

    def vdot(self):
        """This property returns the v vector of the natural coordinates"""
        ...

    def vector(self):
        """This property returns the vector of the natural coordinates"""
        ...

    def to_components(self):
        """This function returns the components of the natural coordinates"""
        ...


class NaturalVelocities(Protocol):
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

    def nb_qdoti(self):
        """This function returns the number of qi"""
    ...

    def udot(self, segment_index):
        """This property returns the u vector of the natural coordinates"""
    ...

    def rpdot(self, segment_index):
        """This property returns the rp vector of the natural coordinates"""
    ...

    def rddot(self, segment_index):
        """This property returns the rd vector of the natural coordinates"""
    ...

    def wdot(self, segment_index):
        """This property returns the w vector of the natural coordinates"""
    ...

    def vdot(self, segment_index):
        """This property returns the v vector of the natural coordinates"""
    ...

    def vector(self):
        """This property returns the vector of the natural coordinates"""
    ...
