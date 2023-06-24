import numpy as np
from casadi import MX, vertcat, inv, cross, sum1
from typing import Union

from .natural_vector import NaturalVector
from ..utils.enums import NaturalAxis


class SegmentNaturalCoordinates(MX):
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
    def sym(cls, suffix: str = ""):
        """
        Constructor of the class with symbolic variables

        Parameters
        ----------
        suffix : str
            Suffix to add to the variable names
        """

        u = MX.sym(f"u{suffix}", 3, 1)
        rp = MX.sym(f"rp{suffix}", 3, 1)
        rd = MX.sym(f"rd{suffix}", 3, 1)
        w = MX.sym(f"w{suffix}", 3, 1)

        input_array = vertcat(u, rp, rd, w)

        return cls(input_array)

    @classmethod
    def from_components(
        cls,
        u: Union[np.ndarray, MX, list] = None,
        rp: Union[np.ndarray, MX, list] = None,
        rd: Union[np.ndarray, MX, list] = None,
        w: Union[np.ndarray, MX, list] = None,
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

        if not isinstance(u, MX):
            u = MX(u)
        if not isinstance(rp, MX):
            rp = MX(rp)
        if not isinstance(rd, MX):
            rd = MX(rd)
        if not isinstance(w, MX):
            w = MX(w)

        if u.shape[0] != 3:
            raise ValueError("u must be a 3x1 numpy array")
        if rp.shape[0] != 3:
            raise ValueError("rp must be a 3x1 numpy array")
        if rd.shape[0] != 3:
            raise ValueError("rd must be a 3x1 numpy array")
        if w.shape[0] != 3:
            raise ValueError("v must be a 3x1 numpy array")

        input_array = vertcat(u, rp, rd, w)

        return cls(input_array)

    def to_array(self):
        return MX(self)

    @property
    def u(self):
        return self[0:3, :]

    @property
    def rp(self):
        return self[3:6, :]

    @property
    def rd(self):
        return self[6:9, :]

    @property
    def w(self):
        return self[9:12, :]

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

    def to_natural_vector(self, vector: MX | np.ndarray) -> NaturalVector:
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
        return NaturalVector(self.vector_projection_in_non_orthogonal_basis(vector - self.rp, self.u, self.v, self.w))

    @staticmethod
    def vector_projection_in_non_orthogonal_basis(vector: np.ndarray | MX, e1: MX, e2: MX, e3: MX) -> MX:
        """
        This function converts a vector expressed in the global coordinate system
        to a vector expressed in a non-orthogonal coordinate system.

        Parameters
        ----------
        vector: np.ndarray | MX
            The vector expressed in the global coordinate system
        e1: MX
            The first vector of the non-orthogonal coordinate system, usually the u-axis
        e2: MX
            The second vector of the non-orthogonal coordinate system, usually the v-axis
        e3: MX
            The third vector of the non-orthogonal coordinate system, usually the w-axis

        Returns
        -------
        vnop: MX
            The vector expressed in the non-orthogonal coordinate system

        Source
        ------
        Desroches, G., Chèze, L., & Dumas, R. (2010).
        Expression of joint moment in the joint coordinate system. Journal of biomechanical engineering, 132(11).
        https://doi.org/10.1115/1.4002537

        2.1 Expression of a 3D Vector in a Nonorthogonal Coordinate Base.
        """

        if vector.shape[0] != 3:
            raise ValueError("The vector must be expressed in 3D.")
        if isinstance(vector, np.ndarray):
            vector = MX(vector)

        if e1.shape[0] != 3:
            raise ValueError("The first vector of the non-orthogonal coordinate system must be expressed in 3D.")

        if e2.shape[0] != 3:
            raise ValueError("The second vector of the non-orthogonal coordinate system must be expressed in 3D.")

        if e3.shape[0] != 3:
            raise ValueError("The third vector of the non-orthogonal coordinate system must be expressed in 3D.")

        vnop = MX.zeros(vector.shape)

        vnop[0, 0] = sum1(np.cross(e2, e3) * vector) / sum1(cross(e1, e2) * e3)
        vnop[1, 0] = sum1(cross(e3, e1) * vector) / sum1(cross(e1, e2) * e3)
        vnop[2, 0] = sum1(cross(e1, e2) * vector) / sum1(cross(e1, e2) * e3)

        return vnop

    def axis(self, axis: Union[str, NaturalAxis]) -> MX:
        """
        This function returns the axis of the segment.

        Parameters
        ----------
        axis: str
            The axis to return (u, v, w)

        Returns
        -------
        MX
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

    def compute_pseudo_interpolation_matrix(self) -> MX:
        """
        Return the force moment transformation matrix

        Returns
        -------
        MX
            The force moment transformation matrix
        """
        # default we apply force at the proximal point

        left_interpolation_matrix = MX.zeros((12, 3))

        left_interpolation_matrix[9:12, 0] = self.u
        left_interpolation_matrix[0:3, 1] = self.v
        left_interpolation_matrix[3:6, 2] = -self.w
        left_interpolation_matrix[6:9, 2] = self.w

        # Matrix of lever arms for forces equivalent to moment at proximal endpoint, denoted Bstar
        lever_arm_force_matrix = MX.zeros((3, 3))

        lever_arm_force_matrix[:, 0] = cross(self.w, self.u)
        lever_arm_force_matrix[:, 1] = cross(self.u, self.v)
        lever_arm_force_matrix[:, 2] = cross(-self.v, self.w)

        return (left_interpolation_matrix @ inv(lever_arm_force_matrix)).T  # NOTE: inv may induce symbolic error.


class NaturalCoordinates(MX):
    def __new__(cls, input_array: MX):
        """
        Create a new instance of the class.
        """

        if input_array.shape[0] % 12 != 0:
            raise ValueError("input_array must be a column vector of size 12 x n elements")

        obj = MX.__new__(cls)

        return obj

    @classmethod
    def sym(cls, nb_segments: int):
        """
        Constructor of the class with symbolic variables

        Parameters
        ----------
        nb_segments : int
            Number of segments
        """

        input_array = vertcat(*[SegmentNaturalCoordinates.sym(f"_{i}") for i in range(nb_segments)])

        return cls(input_array)

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

        input_array = vertcat(*tuple_of_Q)
        return cls(input_array)

    def to_array(self):
        return self

    def nb_qi(self):
        return self.shape[0] // 12

    def u(self, segment_idx: int):
        array_idx = np.arange(segment_idx * 12, (segment_idx + 1) * 12)[0:3]
        return self[array_idx, :]

    def rp(self, segment_idx: int):
        array_idx = np.arange(segment_idx * 12, (segment_idx + 1) * 12)[3:6]
        return self[array_idx, :]

    def rd(self, segment_idx: int):
        array_idx = np.arange(segment_idx * 12, (segment_idx + 1) * 12)[6:9]
        return self[array_idx, :]

    def w(self, segment_idx: int):
        array_idx = np.arange(segment_idx * 12, (segment_idx + 1) * 12)[9:12]
        return self[array_idx, :]

    def v(self, segment_idx: int):
        return self.rp(segment_idx) - self.rd(segment_idx)

    def vector(self, segment_idx: int = None):
        if segment_idx is None:
            return self[:]
        else:
            array_idx = np.arange(segment_idx * 12, (segment_idx + 1) * 12).tolist()
            return SegmentNaturalCoordinates(self[array_idx])
