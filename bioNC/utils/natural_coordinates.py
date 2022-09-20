import numpy as np
from typing import Union


class SegmentNaturalCoordinatesCreator:
    def __new__(
        cls,
        u: Union[np.ndarray, list] = None,
        rp: Union[np.ndarray, list] = None,
        rd: Union[np.ndarray, list] = None,
        w: Union[np.ndarray, list] = None,
    ):
        if u is None:
            raise ValueError("u must be a numpy array (3x1) or a list of 3 elements")
        if rp is None:
            raise ValueError("rp must be a numpy array (3x1) or a list of 3 elements")
        if rd is None:
            raise ValueError("rd must be a numpy array (3x1) or a list of 3 elements")
        if w is None:
            raise ValueError("w must be a numpy array (3x1) or a list of 3 elements")

        if not isinstance(u, np.ndarray):
            u = np.array(u)
        if not isinstance(rp, np.ndarray):
            rp = np.array(rp)
        if not isinstance(rd, np.ndarray):
            rd = np.array(rd)
        if not isinstance(w, np.ndarray):
            w = np.array(w)

        if u.shape[0] != 3:
            raise ValueError("u must be a 3x1 numpy array")
        if rp.shape[0] != 3:
            raise ValueError("rp must be a 3x1 numpy array")
        if rd.shape[0] != 3:
            raise ValueError("rd must be a 3x1 numpy array")
        if w.shape[0] != 3:
            raise ValueError("v must be a 3x1 numpy array")

        input_array = np.concatenate((u, rp, rd, w), axis=0)
        return SegmentNaturalCoordinates(input_array)


class SegmentNaturalCoordinates(np.ndarray):
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
    def u(self):
        return self[0:3].to_array()

    @property
    def rp(self):
        return self[3:6].to_array()

    @property
    def rd(self):
        return self[6:9].to_array()

    @property
    def w(self):
        return self[9:12].to_array()

    @property
    def v(self):
        return self.rd.to_array() - self.rp.to_array()

    @property
    def vector(self):
        return self.to_array()


class NaturalCoordinatesCreator:
    def __new__(cls, tuple_of_Q: tuple):
        """
        Create a new instance of the class.
        """
        if not isinstance(tuple_of_Q, tuple):
            raise ValueError("tuple_of_Q must be a tuple of SegmentGeneralizedCoordinates")

        for Q in tuple_of_Q:
            if not isinstance(Q, SegmentNaturalCoordinates):
                raise ValueError("tuple_of_Q must be a tuple of SegmentGeneralizedCoordinates")

        input_array = np.concatenate(tuple_of_Q, axis=0)
        return NaturalCoordinates(input_array)


class NaturalCoordinates(np.ndarray):
    def __new__(cls, input_array: np.ndarray):
        """
        Create a new instance of the class.
        """

        return np.asarray(input_array).view(cls)

    def nb_Qi(self):
        return self.shape[0] // 12

    def u(self, segment_idx: int):
        array_idx = np.arange(segment_idx * 12, (segment_idx + 1) * 12)[0:3]
        return self[array_idx]

    def rp(self, segment_idx: int):
        array_idx = np.arange(segment_idx * 12, (segment_idx + 1) * 12)[3:6]
        return self[array_idx]

    def rd(self, segment_idx: int):
        array_idx = np.arange(segment_idx * 12, (segment_idx + 1) * 12)[6:9]
        return self[array_idx]

    def w(self, segment_idx: int):
        array_idx = np.arange(segment_idx * 12, (segment_idx + 1) * 12)[9:12]
        return self[array_idx]

    def v(self, segment_idx: int):
        return self.rd(segment_idx) - self.rp(segment_idx)

    def vector(self, segment_idx: int):
        array_idx = np.arange(segment_idx * 12, (segment_idx + 1) * 12)
        return SegmentNaturalCoordinates(self[array_idx].vector)


