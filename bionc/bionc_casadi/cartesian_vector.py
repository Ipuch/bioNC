from casadi import MX, sum1, cross
import numpy as np
from ..utils.enums import CartesianAxis


class CartesianVector(MX):
    """
    Class used to create a natural vector, a vector that is expressed in the natural coordinate system of a segment
    """

    def __new__(cls, input_array: MX | np.ndarray | list | tuple):
        """
        Create a new instance of the class.
        """

        if not isinstance(input_array, (MX, np.ndarray)):
            input_array = np.array(input_array)

        obj = MX.__new__(cls)

        if isinstance(input_array, MX):
            size1 = input_array.shape[0]
            size2 = input_array.shape[1]
        else:
            size1 = input_array.shape[0]
            size2 = input_array.shape[1] if input_array.shape.__len__() == 2 else 1

        if size1 != 3:
            raise ValueError("The input array must have 3 elements")

        if size2 != 1:
            raise ValueError("The position must be a 3d vector with only one column")

        return obj

    @classmethod
    def axis(cls, axis: CartesianAxis):
        if axis == CartesianAxis.X:
            return cls(np.array([1, 0, 0]))
        elif axis == CartesianAxis.Y:
            return cls(np.array([0, 1, 0]))
        elif axis == CartesianAxis.Z:
            return cls(np.array([0, 0, 1]))


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

    vnop[0, 0] = sum1(cross(e2, e3) * vector) / sum1(cross(e1, e2) * e3)
    vnop[1, 0] = sum1(cross(e3, e1) * vector) / sum1(cross(e1, e2) * e3)
    vnop[2, 0] = sum1(cross(e1, e2) * vector) / sum1(cross(e1, e2) * e3)

    return vnop
