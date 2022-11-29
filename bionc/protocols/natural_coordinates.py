from typing import Protocol


class SegmentNaturalCoordinates(Protocol):
    """
    This class is made to handle Generalized Coordinates of a Segment
    """

    def __new__(cls, input_array):
        """Create a new instance of the class."""
        ...

    @classmethod
    def from_components(cls, u, rp, rd, w):
        """Constructor of the class from the components of the natural coordinates"""
        ...

    def to_array(self):
        """This function returns the array of the natural coordinates"""
        ...

    def u(self):
        """This property returns the u vector of the natural coordinates"""
        ...

    def rp(self):
        """This property returns the rp vector of the natural coordinates"""
        ...

    def rd(self):
        """This property returns the rd vector of the natural coordinates"""
        ...

    def w(self):
        """This property returns the w vector of the natural coordinates"""
        ...

    def v(self):
        """This property returns the v vector of the natural coordinates"""
        ...

    def vector(self):
        """This property returns the vector of the natural coordinates"""
        ...

    def to_components(self):
        """This function returns the components of the natural coordinates"""
        ...

    def to_uvw(self):
        """This function returns the uvw vector of the natural coordinates"""
        ...

    def to_natural_vector(self, vector):
        """
        This function converts a vector expressed in the global coordinate system
        to a vector expressed in a non-orthogonal coordinate system (rp, u, v, w) associated to the segment coordinates.
        """
        ...


class NaturalCoordinates(Protocol):
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

    def nb_qi(self):
        """This function returns the number of qi"""

    ...

    def u(self, segment_index):
        """This property returns the u vector of the natural coordinates"""

    ...

    def rp(self, segment_index):
        """This property returns the rp vector of the natural coordinates"""

    ...

    def rd(self, segment_index):
        """This property returns the rd vector of the natural coordinates"""

    ...

    def w(self, segment_index):
        """This property returns the w vector of the natural coordinates"""

    ...

    def v(self, segment_index):
        """This property returns the v vector of the natural coordinates"""

    ...

    def vector(self, segment_idx):
        """This property returns the vector of the natural coordinates"""

    ...
