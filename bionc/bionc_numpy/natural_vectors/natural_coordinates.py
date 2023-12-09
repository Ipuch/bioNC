import numpy as np
from typing import Union

from ..mecamaths.cartesian_vector import vector_projection_in_non_orthogonal_basis
from ..natural_vectors.natural_vector import NaturalVector
from ...utils.enums import NaturalAxis


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
            return vector_projection_in_non_orthogonal_basis(
                vector - self.rp, self.u, self.v, self.w
            )  # not satisfied yet of this
        else:
            return NaturalVector(vector_projection_in_non_orthogonal_basis(vector - self.rp, self.u, self.v, self.w))

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

    def compute_pseudo_interpolation_matrix(self) -> np.ndarray:
        """
        Return the force moment transformation matrix

        Returns
        -------
        np.ndarray
            The force moment transformation matrix
        """
        # default we apply force at the proximal point

        left_interpolation_matrix = np.zeros((12, 3))

        left_interpolation_matrix[9:12, 0] = self.u
        left_interpolation_matrix[0:3, 1] = self.v
        left_interpolation_matrix[3:6, 2] = -self.w
        left_interpolation_matrix[6:9, 2] = self.w

        # Matrix of lever arms for forces equivalent to moment at proximal endpoint, denoted Bstar
        lever_arm_force_matrix = np.zeros((3, 3))

        lever_arm_force_matrix[:, 0] = np.cross(self.w, self.u)
        lever_arm_force_matrix[:, 1] = np.cross(self.u, self.v)
        lever_arm_force_matrix[:, 2] = np.cross(-self.v, self.w)

        return (left_interpolation_matrix @ np.linalg.inv(lever_arm_force_matrix)).T


class NaturalCoordinates(np.ndarray):
    def __new__(cls, input_array: np.ndarray):
        """
        Create a new instance of the class.
        """
        if input_array.shape.__len__() == 1:
            input_array = input_array[:, np.newaxis]
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
