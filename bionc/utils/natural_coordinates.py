import numpy as np
from typing import Union
from bionc.utils.vnop_array import vnop_array
from bionc.utils.interpolation_matrix import interpolation_matrix


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

    @classmethod
    def from_components(
        cls,
        u: Union[np.ndarray, list] = None,
        rp: Union[np.ndarray, list] = None,
        rd: Union[np.ndarray, list] = None,
        w: Union[np.ndarray, list] = None,
    ):
        """
        Constructor of the class from the components of the natural coordinates
        """

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

        return cls(input_array)

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
        return self.rp - self.rd

    @property
    def vector(self):
        return self.to_array()

    @property
    def to_components(self):
        return self.u, self.rp, self.rd, self.w

    def to_non_orthogonal_basis(self, vector: np.ndarray) -> np.ndarray:
        """
        This function converts a vector expressed in the global coordinate system
        to a vector expressed in a non-orthogonal coordinate system associated to the segment coordinates.

        Parameters
        ----------
        vector: np.ndarray
            The vector expressed in the global coordinate system

        Returns
        -------
        np.ndarray
            The vector expressed in the non-orthogonal coordinate system

        """
        return vnop_array(vector - self.rp, self.u, self.v, self.w)

    def to_interpolation_matrix(self, vector: np.ndarray) -> np.ndarray:
        """
        This function converts a vector expressed in the global coordinate system
        to a vector expressed in a non-orthogonal coordinate system associated to the segment coordinates.

        Parameters
        ----------
        vector: np.ndarray
            The vector expressed in the global coordinate system

        Returns
        -------
        np.ndarray
            The vector expressed in the non-orthogonal coordinate system

        """
        return interpolation_matrix(vnop_array(vector - self.rp, self.u, self.v, self.w))


class NaturalCoordinates(np.ndarray):
    def __new__(cls, input_array: np.ndarray):
        """
        Create a new instance of the class.
        """

        return np.asarray(input_array).view(cls)

    @classmethod
    def from_Qi(cls, tuple_of_Q: tuple):
        """
        Constructor of the class.
        """
        if not isinstance(tuple_of_Q, tuple):
            raise ValueError("tuple_of_Q must be a tuple of SegmentGeneralizedCoordinates")

        for Q in tuple_of_Q:
            if not isinstance(Q, SegmentNaturalCoordinates):
                raise ValueError("tuple_of_Q must be a tuple of SegmentGeneralizedCoordinates")

        input_array = np.concatenate(tuple_of_Q, axis=0)
        return cls(input_array)

    def to_array(self):
        return np.array(self)

    def nb_Qi(self):
        return self.shape[0] // 12

    def u(self, segment_idx: int):
        array_idx = np.arange(segment_idx * 12, (segment_idx + 1) * 12)[0:3]
        return self[array_idx].to_array()

    def rp(self, segment_idx: int):
        array_idx = np.arange(segment_idx * 12, (segment_idx + 1) * 12)[3:6]
        return self[array_idx].to_array()

    def rd(self, segment_idx: int):
        array_idx = np.arange(segment_idx * 12, (segment_idx + 1) * 12)[6:9]
        return self[array_idx].to_array()

    def w(self, segment_idx: int):
        array_idx = np.arange(segment_idx * 12, (segment_idx + 1) * 12)[9:12]
        return self[array_idx].to_array()

    def v(self, segment_idx: int):
        return self.rp(segment_idx) - self.rd(segment_idx)

    def vector(self, segment_idx: int):
        array_idx = np.arange(segment_idx * 12, (segment_idx + 1) * 12)
        return SegmentNaturalCoordinates(self[array_idx].to_array())
