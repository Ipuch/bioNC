import numpy as np
from typing import Union
from .natural_vector import NaturalVector
from ..utils.enums import NaturalAxis


class SegmentNaturalCoordinates(np.ndarray):
    """
    This class is made to handle Generalized Coordinates of a Segment
    """

    def __new__(cls, input_array: Union[np.ndarray, list, tuple]):
        """
        Create a new instance of the class.
        """

        obj = np.asarray(input_array).view(cls)

        if obj.shape.__len__() == 1:
            obj = obj[:, np.newaxis]

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

        if input_array.shape.__len__() == 1:
            input_array = input_array[:, np.newaxis]

        return cls(input_array)

    def to_array(self):
        return np.array(self).squeeze()

    @property
    def u(self):
        return self[0:3, :].to_array()

    @property
    def rp(self):
        return self[3:6, :].to_array()

    @property
    def rd(self):
        return self[6:9, :].to_array()

    @property
    def w(self):
        return self[9:12, :].to_array()

    @property
    def v(self):
        return self.rp - self.rd

    @property
    def vector(self):
        return self.to_array()

    def to_components(self):
        return self.u, self.rp, self.rd, self.w

    def to_uvw(self):
        return self.u, self.v, self.w

    def to_natural_vector(self, vector: np.ndarray) -> NaturalVector | np.ndarray:
        """
        This function converts a vector expressed in the global coordinate system
        to a vector expressed in a non-orthogonal coordinate system associated to the segment coordinates.

        Parameters
        ----------
        vector: np.ndarray
            The vector expressed in the global coordinate system (3x1) or (3xN)

        Returns
        -------
        np.ndarray
            The vector expressed in the non-orthogonal coordinate system (rp, u, v, w)

        """
        if self.shape[1] > 1:
            return self.vnop_array(vector - self.rp, self.u, self.v, self.w)  # not satisfied yet of this
        else:
            return NaturalVector(self.vnop_array(vector - self.rp, self.u, self.v, self.w))

    @staticmethod
    def vnop_array(V: np.ndarray, e1: np.ndarray, e2: np.ndarray, e3: np.ndarray) -> np.ndarray:
        """
        This function converts a vector expressed in the global coordinate system
        to a vector expressed in a non-orthogonal coordinate system.

        Parameters
        ----------
        V: np.ndarray
            The vector expressed in the global coordinate system
        e1: np.ndarray
            The first vector of the non-orthogonal coordinate system, usually the u-axis
        e2: np.ndarray
            The second vector of the non-orthogonal coordinate system, usually the v-axis
        e3: np.ndarray
            The third vector of the non-orthogonal coordinate system, usually the w-axis

        Returns
        -------
        vnop: np.ndarray
            The vector expressed in the non-orthogonal coordinate system

        """

        if V.shape[0] != 3:
            raise ValueError("The vector must be expressed in 3D.")
        if len(V.shape) == 1:
            V = V[:, np.newaxis]

        if e1.shape[0] != 3:
            raise ValueError("The first vector of the non-orthogonal coordinate system must be expressed in 3D.")
        if len(e1.shape) == 1:
            e1 = e1[:, np.newaxis]
        if e2.shape[0] != 3:
            raise ValueError("The second vector of the non-orthogonal coordinate system must be expressed in 3D.")
        if len(e2.shape) == 1:
            e2 = e2[:, np.newaxis]
        if e3.shape[0] != 3:
            raise ValueError("The third vector of the non-orthogonal coordinate system must be expressed in 3D.")
        if len(e3.shape) == 1:
            e3 = e3[:, np.newaxis]

        vnop = np.zeros(V.shape)

        vnop[0, :] = np.sum(np.cross(e2, e3, axis=0) * V, 0) / np.sum(np.cross(e1, e2, axis=0) * e3, 0)
        vnop[1, :] = np.sum(np.cross(e3, e1, axis=0) * V, 0) / np.sum(np.cross(e1, e2, axis=0) * e3, 0)
        vnop[2, :] = np.sum(np.cross(e1, e2, axis=0) * V, 0) / np.sum(np.cross(e1, e2, axis=0) * e3, 0)

        return vnop

    def axis(self, axis: Union[str, NaturalAxis]) -> np.ndarray:
        """
        This function returns the axis of the segment.

        Parameters
        ----------
        axis: str
            The axis to return (u, v, w)

        Returns
        -------
        np.ndarray
            The axis of the segment

        """
        if axis == "u" or axis == NaturalAxis.U:
            return self.u
        elif axis == "v" or axis == NaturalAxis.V:
            return self.v
        elif axis == "w" or axis == NaturalAxis.W:
            return self.w
        else:
            raise ValueError("The axis must be u, v or w")


class NaturalCoordinates(np.ndarray):
    def __new__(cls, input_array: np.ndarray):
        """
        Create a new instance of the class.
        """

        return np.asarray(input_array).view(cls)

    @classmethod
    def from_qi(cls, tuple_of_Q: tuple):
        """
        Constructor of the class.
        """
        if not isinstance(tuple_of_Q, tuple):
            raise ValueError("tuple_of_Q must be a tuple of SegmentNaturalCoordinates")

        for Q in tuple_of_Q:
            if not isinstance(Q, SegmentNaturalCoordinates):
                raise ValueError("tuple_of_Q must be a tuple of SegmentNaturalCoordinates")

        input_array = np.concatenate(tuple_of_Q, axis=0)
        return cls(input_array)

    def to_array(self):
        return np.array(self).squeeze()

    def nb_qi(self):
        return self.shape[0] // 12

    def u(self, segment_idx: int):
        array_idx = np.arange(segment_idx * 12, (segment_idx + 1) * 12)[0:3]
        return self[array_idx, :].to_array()

    def rp(self, segment_idx: int):
        array_idx = np.arange(segment_idx * 12, (segment_idx + 1) * 12)[3:6]
        return self[array_idx, :].to_array()

    def rd(self, segment_idx: int):
        array_idx = np.arange(segment_idx * 12, (segment_idx + 1) * 12)[6:9]
        return self[array_idx, :].to_array()

    def w(self, segment_idx: int):
        array_idx = np.arange(segment_idx * 12, (segment_idx + 1) * 12)[9:12]
        return self[array_idx, :].to_array()

    def v(self, segment_idx: int):
        return self.rp(segment_idx) - self.rd(segment_idx)

    def vector(self, segment_idx: int):
        array_idx = np.arange(segment_idx * 12, (segment_idx + 1) * 12)
        return SegmentNaturalCoordinates(self[array_idx, :].to_array())
