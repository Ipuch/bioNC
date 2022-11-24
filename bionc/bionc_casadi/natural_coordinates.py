import numpy as np
from casadi import MX, vertcat
from typing import Union

# from bionc.utils.vnop_array import vnop_array
# from bionc.utils.interpolation_matrix import interpolate_natural_vector


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

    # def to_non_orthogonal_basis(self, vector: MX) -> MX:
    #     """
    #     This function converts a vector expressed in the global coordinate system
    #     to a vector expressed in a non-orthogonal coordinate system associated to the segment coordinates.
    #
    #     Parameters
    #     ----------
    #     vector: MX
    #         The vector expressed in the global coordinate system
    #
    #     Returns
    #     -------
    #     MX
    #         The vector expressed in the non-orthogonal coordinate system
    #
    #     """
    #     return vnop_array(vector - self.rp, self.u, self.v, self.w)
    #
    # def to_interpolation_matrix(self, vector: MX) -> MX:
    #     """
    #     This function converts a vector expressed in the global coordinate system
    #     to a vector expressed in a non-orthogonal coordinate system associated to the segment coordinates.
    #
    #     Parameters
    #     ----------
    #     vector: MX
    #         The vector expressed in the global coordinate system
    #
    #     Returns
    #     -------
    #     MX
    #         The vector expressed in the non-orthogonal coordinate system
    #
    #     """
    #     return interpolate_natural_vector(vnop_array(vector - self.rp, self.u, self.v, self.w))


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
