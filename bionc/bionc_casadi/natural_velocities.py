import numpy as np
from casadi import MX, vertcat
from typing import Union


class SegmentNaturalVelocities(MX):
    """
    This class is made to handle Generalized Coordinates of a Segment
    """

    def __new__(cls, input_array: MX):
        """
        Create a new instance of the class.
        """

        obj = MX.__new__(cls)

        return obj

    @classmethod
    def from_components(
        cls,
        udot: Union[np.ndarray, MX, list] = None,
        rpdot: Union[np.ndarray, MX, list] = None,
        rddot: Union[np.ndarray, MX, list] = None,
        wdot: Union[np.ndarray, MX, list] = None,
    ):
        if udot is None:
            raise ValueError("u must be a array (3x1) or a list of 3 elements")
        if rpdot is None:
            raise ValueError("rp must be a array (3x1) or a list of 3 elements")
        if rddot is None:
            raise ValueError("rd must be a array (3x1) or a list of 3 elements")
        if wdot is None:
            raise ValueError("w must be a array (3x1) or a list of 3 elements")

        if not isinstance(udot, MX):
            udot = MX(udot)
        if not isinstance(rpdot, MX):
            rpdot = MX(rpdot)
        if not isinstance(rddot, MX):
            rddot = MX(rddot)
        if not isinstance(wdot, MX):
            wdot = MX(wdot)

        if udot.shape[0] != 3:
            raise ValueError("u must be a 3x1 array")
        if rpdot.shape[0] != 3:
            raise ValueError("rp must be a 3x1 array")
        if rddot.shape[0] != 3:
            raise ValueError("rd must be a 3x1 array")
        if wdot.shape[0] != 3:
            raise ValueError("v must be a 3x1 array")

        input_array = vertcat(*(udot, rpdot, rddot, wdot))
        return cls(input_array)

    def to_array(self):
        return self

    @property
    def udot(self):
        return self[0:3]

    @property
    def rpdot(self):
        return self[3:6]

    @property
    def rddot(self):
        return self[6:9]

    @property
    def wdot(self):
        return self[9:12]

    @property
    def vdot(self):
        return self.rpdot - self.rddot

    @property
    def vector(self):
        return self.to_array()

    @property
    def to_components(self):
        return self.udot, self.rpdot, self.rddot, self.wdot


class NaturalVelocities(MX):
    def __new__(cls, input_array: MX):
        """
        Create a new instance of the class.
        """

        if input_array.shape[0] % 12 != 0:
            raise ValueError("input_array must be a column vector of size 12 x n elements")

        obj = MX.__new__(cls)

        return obj

    @classmethod
    def from_qdoti(cls, tuple_of_Q: tuple):
        """
        Create a new instance of the class.
        """
        if not isinstance(tuple_of_Q, tuple):
            raise ValueError("tuple_of_Q must be a tuple of SegmentNaturalVelocities")

        for Q in tuple_of_Q:
            if not isinstance(Q, SegmentNaturalVelocities):
                raise ValueError("tuple_of_Q must be a tuple of SegmentNaturalVelocities")

        input_array = vertcat(*tuple_of_Q)
        return cls(input_array)

    def to_array(self):
        return self

    def nb_qdoti(self):
        return self.shape[0] // 12

    def udot(self, segment_idx: int):
        array_idx = np.arange(segment_idx * 12, (segment_idx + 1) * 12)[0:3]
        return self[array_idx]

    def rpdot(self, segment_idx: int):
        array_idx = np.arange(segment_idx * 12, (segment_idx + 1) * 12)[3:6]
        return self[array_idx]

    def rddot(self, segment_idx: int):
        array_idx = np.arange(segment_idx * 12, (segment_idx + 1) * 12)[6:9]
        return self[array_idx]

    def wdot(self, segment_idx: int):
        array_idx = np.arange(segment_idx * 12, (segment_idx + 1) * 12)[9:12]
        return self[array_idx]

    def vdot(self, segment_idx: int):
        return self.rpdot(segment_idx) - self.rddot(segment_idx)

    def vector(self, segment_idx: int):
        array_idx = np.arange(segment_idx * 12, (segment_idx + 1) * 12)
        return SegmentNaturalVelocities(self[array_idx])
