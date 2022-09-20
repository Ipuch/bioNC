import numpy as np
from typing import Union


class SegmentNaturalAccelerationsCreator:
    def __new__(
        cls,
        uddot: Union[np.ndarray, list] = None,
        rpddot: Union[np.ndarray, list] = None,
        rdddot: Union[np.ndarray, list] = None,
        wddot: Union[np.ndarray, list] = None,
    ):
        if uddot is None:
            raise ValueError("u must be a numpy array (3x1) or a list of 3 elements")
        if rpddot is None:
            raise ValueError("rp must be a numpy array (3x1) or a list of 3 elements")
        if rdddot is None:
            raise ValueError("rd must be a numpy array (3x1) or a list of 3 elements")
        if wddot is None:
            raise ValueError("w must be a numpy array (3x1) or a list of 3 elements")

        if not isinstance(uddot, np.ndarray):
            uddot = np.array(uddot)
        if not isinstance(rpddot, np.ndarray):
            rpddot = np.array(rpddot)
        if not isinstance(rdddot, np.ndarray):
            rdddot = np.array(rdddot)
        if not isinstance(wddot, np.ndarray):
            wddot = np.array(wddot)

        if uddot.shape[0] != 3:
            raise ValueError("u must be a 3x1 numpy array")
        if rpddot.shape[0] != 3:
            raise ValueError("rp must be a 3x1 numpy array")
        if rdddot.shape[0] != 3:
            raise ValueError("rd must be a 3x1 numpy array")
        if wddot.shape[0] != 3:
            raise ValueError("v must be a 3x1 numpy array")

        input_array = np.concatenate((uddot, rpddot, rdddot, wddot), axis=0)
        return SegmentNaturalAccelerations(input_array)


class SegmentNaturalAccelerations(np.ndarray):
    """
    This class is made to handle Generalized Coordinates of a Segment
    """

    def __new__(cls, input_array: Union[np.ndarray, list, tuple]):
        """
        Create a new instance of the class.
        """

        obj = np.asarray(input_array).view(cls)

        return obj

    def to_array(self):
        return np.array(self)

    @property
    def uddot(self):
        return self[0:3].to_array()

    @property
    def rpddot(self):
        return self[3:6].to_array()

    @property
    def rdddot(self):
        return self[6:9].to_array()

    @property
    def wddot(self):
        return self[9:12].to_array()

    @property
    def vddot(self):
        return self.rd.to_array() - self.rp.to_array()

    @property
    def vector(self):
        return self.to_array()


class NaturalAccelerationsCreator:
    def __new__(cls, tuple_of_Q: tuple):
        """
        Create a new instance of the class.
        """
        if not isinstance(tuple_of_Q, tuple):
            raise ValueError("tuple_of_Q must be a tuple of SegmentGeneralizedCoordinates")

        for Q in tuple_of_Q:
            if not isinstance(Q, SegmentNaturalAccelerations):
                raise ValueError("tuple_of_Q must be a tuple of SegmentGeneralizedCoordinates")

        input_array = np.concatenate(tuple_of_Q, axis=0)
        return NaturalAccelerations(input_array)


class NaturalAccelerations(np.ndarray):
    def __new__(cls, input_array: np.ndarray):
        """
        Create a new instance of the class.
        """

        return np.asarray(input_array).view(cls)

    def nb_Qddoti(self):
        return self.shape[0] // 12

    def uddot(self, segment_idx: int):
        array_idx = np.arange(segment_idx * 12, (segment_idx + 1) * 12)[0:3]
        return self[array_idx]

    def rpddot(self, segment_idx: int):
        array_idx = np.arange(segment_idx * 12, (segment_idx + 1) * 12)[3:6]
        return self[array_idx]

    def rdddot(self, segment_idx: int):
        array_idx = np.arange(segment_idx * 12, (segment_idx + 1) * 12)[6:9]
        return self[array_idx]

    def wddot(self, segment_idx: int):
        array_idx = np.arange(segment_idx * 12, (segment_idx + 1) * 12)[9:12]
        return self[array_idx]

    def vddot(self, segment_idx: int):
        return self.rdddot(segment_idx) - self.rpddot(segment_idx)

    def vector(self, segment_idx: int):
        array_idx = np.arange(segment_idx * 12, (segment_idx + 1) * 12)
        return SegmentNaturalAccelerations(self[array_idx].vector)


