import numpy as np
from casadi import MX, vertcat
from typing import Union


class SegmentNaturalAccelerations(MX):
    """
    This class is made to handle Generalized Coordinates of a Segment
    """

    def __new__(cls, input_array: Union[MX, list, tuple]):
        """
        Create a new instance of the class.
        """

        obj = np.asarray(input_array).view(cls)

        return obj

    @classmethod
    def from_components(
        cls,
        uddot: Union[MX, list] = None,
        rpddot: Union[MX, list] = None,
        rdddot: Union[MX, list] = None,
        wddot: Union[MX, list] = None,
    ):
        if uddot is None:
            raise ValueError("uddot must be a numpy array (3x1) or a list of 3 elements")
        if rpddot is None:
            raise ValueError("rpddot must be a numpy array (3x1) or a list of 3 elements")
        if rdddot is None:
            raise ValueError("rdddot must be a numpy array (3x1) or a list of 3 elements")
        if wddot is None:
            raise ValueError("wddot must be a numpy array (3x1) or a list of 3 elements")

        if not isinstance(uddot, MX):
            uddot = np.array(uddot)
        if not isinstance(rpddot, MX):
            rpddot = np.array(rpddot)
        if not isinstance(rdddot, MX):
            rdddot = np.array(rdddot)
        if not isinstance(wddot, MX):
            wddot = np.array(wddot)

        if uddot.shape[0] != 3:
            raise ValueError("uddot must be a 3x1 array")
        if rpddot.shape[0] != 3:
            raise ValueError("rpddot must be a 3x1 array")
        if rdddot.shape[0] != 3:
            raise ValueError("rdddot must be a 3x1 array")
        if wddot.shape[0] != 3:
            raise ValueError("vddot must be a 3x1 array")

        input_array = vertcat(*(uddot, rpddot, rdddot, wddot))

        return cls(input_array)

    def to_array(self):
        return self

    @property
    def uddot(self):
        return self[0:3]

    @property
    def rpddot(self):
        return self[3:6]

    @property
    def rdddot(self):
        return self[6:9]

    @property
    def wddot(self):
        return self[9:12]

    @property
    def vddot(self):
        return self.rpddot - self.rdddot

    @property
    def vector(self):
        return self.to_array()

    @property
    def to_components(self):
        return self.uddot, self.rpddot, self.rdddot, self.wddot


class NaturalAccelerations(MX):
    def __new__(cls, input_array: MX):
        """
        Create a new instance of the class.
        """

        if input_array.shape[0] % 12 != 0:
            raise ValueError("input_array must be a column vector of size 12 x n elements")

        obj = MX.__new__(cls)

        return obj

    @classmethod
    def from_qddoti(cls, tuple_of_Q: tuple):
        """
        Constructor of the class
        """
        if not isinstance(tuple_of_Q, tuple):
            raise ValueError("tuple_of_Q must be a tuple of SegmentGeneralizedCoordinates")

        for Q in tuple_of_Q:
            if not isinstance(Q, SegmentNaturalAccelerations):
                raise ValueError("tuple_of_Q must be a tuple of SegmentGeneralizedCoordinates")

        input_array = vertcat(*tuple_of_Q)
        return cls(input_array)

    def to_array(self):
        return self

    def nb_qddoti(self):
        return self.shape[0] // 12

    def uddot(self, segment_idx: int):
        array_idx = np.arange(segment_idx * 12, (segment_idx + 1) * 12)[0:3]
        return self[array_idx].to_array()

    def rpddot(self, segment_idx: int):
        array_idx = np.arange(segment_idx * 12, (segment_idx + 1) * 12)[3:6]
        return self[array_idx].to_array()

    def rdddot(self, segment_idx: int):
        array_idx = np.arange(segment_idx * 12, (segment_idx + 1) * 12)[6:9]
        return self[array_idx].to_array()

    def wddot(self, segment_idx: int):
        array_idx = np.arange(segment_idx * 12, (segment_idx + 1) * 12)[9:12]
        return self[array_idx].to_array()

    def vddot(self, segment_idx: int):
        return self.rpddot(segment_idx) - self.rdddot(segment_idx)

    def vector(self, segment_idx: int):
        array_idx = np.arange(segment_idx * 12, (segment_idx + 1) * 12)
        return SegmentNaturalAccelerations(self[array_idx])
