import numpy as np
from typing import Union


class SegmentNaturalVelocities(np.ndarray):
    """
    This class is made to handle Generalized Coordinates of a Segment
    """

    def __new__(cls, input_array: Union[np.ndarray, list, tuple]):
        """
        Create a new instance of the class.
        """

        obj = np.asarray(input_array).view(cls)

        return obj

    @classmethod
    def from_components(
        cls,
        udot: Union[np.ndarray, list] = None,
        rpdot: Union[np.ndarray, list] = None,
        rddot: Union[np.ndarray, list] = None,
        wdot: Union[np.ndarray, list] = None,
    ):
        if udot is None:
            raise ValueError("u must be a numpy array (3x1) or a list of 3 elements")
        if rpdot is None:
            raise ValueError("rp must be a numpy array (3x1) or a list of 3 elements")
        if rddot is None:
            raise ValueError("rd must be a numpy array (3x1) or a list of 3 elements")
        if wdot is None:
            raise ValueError("w must be a numpy array (3x1) or a list of 3 elements")

        if not isinstance(udot, np.ndarray):
            udot = np.array(udot)
        if not isinstance(rpdot, np.ndarray):
            rpdot = np.array(rpdot)
        if not isinstance(rddot, np.ndarray):
            rddot = np.array(rddot)
        if not isinstance(wdot, np.ndarray):
            wdot = np.array(wdot)

        if udot.shape[0] != 3:
            raise ValueError("u must be a 3x1 numpy array")
        if rpdot.shape[0] != 3:
            raise ValueError("rp must be a 3x1 numpy array")
        if rddot.shape[0] != 3:
            raise ValueError("rd must be a 3x1 numpy array")
        if wdot.shape[0] != 3:
            raise ValueError("v must be a 3x1 numpy array")

        input_array = np.concatenate((udot, rpdot, rddot, wdot), axis=0)
        return cls(input_array)

    def to_array(self):
        return np.array(self)

    @property
    def udot(self):
        return self[0:3].to_array()

    @property
    def rpdot(self):
        return self[3:6].to_array()

    @property
    def rddot(self):
        return self[6:9].to_array()

    @property
    def wdot(self):
        return self[9:12].to_array()

    @property
    def vdot(self):
        return self.rpdot - self.rddot

    @property
    def vector(self):
        return self.to_array()

    @property
    def to_components(self):
        return self.udot, self.rpdot, self.rddot, self.wdot


class NaturalVelocities(np.ndarray):
    def __new__(cls, input_array: np.ndarray):
        """
        Create a new instance of the class.
        """

        return np.asarray(input_array).view(cls)

    @classmethod
    def from_qdoti(cls, tuple_of_Q: tuple):
        """
        Create a new instance of the class.
        """
        if not isinstance(tuple_of_Q, tuple):
            raise ValueError("tuple_of_Q must be a tuple of SegmentGeneralizedCoordinates")

        for Q in tuple_of_Q:
            if not isinstance(Q, SegmentNaturalVelocities):
                raise ValueError("tuple_of_Q must be a tuple of SegmentGeneralizedCoordinates")

        input_array = np.concatenate(tuple_of_Q, axis=0)
        return cls(input_array)

    def to_array(self):
        return np.array(self)

    def nb_qdoti(self):
        return self.shape[0] // 12

    def udot(self, segment_idx: int):
        array_idx = np.arange(segment_idx * 12, (segment_idx + 1) * 12)[0:3]
        return self[array_idx].to_array()

    def rpdot(self, segment_idx: int):
        array_idx = np.arange(segment_idx * 12, (segment_idx + 1) * 12)[3:6]
        return self[array_idx].to_array()

    def rddot(self, segment_idx: int):
        array_idx = np.arange(segment_idx * 12, (segment_idx + 1) * 12)[6:9]
        return self[array_idx].to_array()

    def wdot(self, segment_idx: int):
        array_idx = np.arange(segment_idx * 12, (segment_idx + 1) * 12)[9:12]
        return self[array_idx].to_array()

    def vdot(self, segment_idx: int):
        return self.rpdot(segment_idx) - self.rddot(segment_idx)

    def vector(self, segment_idx: int):
        array_idx = np.arange(segment_idx * 12, (segment_idx + 1) * 12)
        return SegmentNaturalVelocities(self[array_idx].to_array())
