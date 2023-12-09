import numpy as np

from ...utils.enums import CartesianAxis


class CartesianVector(np.ndarray):
    """
    Class used to create a natural vector, a vector that is expressed in the natural coordinate system of a segment
    """

    def __new__(cls, input_array: np.ndarray | list | tuple):
        """
        Create a new instance of the class.
        """

        if not isinstance(input_array, np.ndarray):
            input_array = np.array(input_array)

        size1 = input_array.shape[0]
        size2 = input_array.shape[1] if input_array.shape.__len__() == 2 else 1

        if size1 != 3:
            raise ValueError("The input array must have 3 elements")

        if size2 != 1:
            raise ValueError("The position must be a 3d vector with only one column")

        obj = np.asarray(input_array).view(cls)

        return obj

    @classmethod
    def axis(cls, axis: CartesianAxis):
        if axis == CartesianAxis.X:
            return cls(np.array([1, 0, 0]))
        elif axis == CartesianAxis.Y:
            return cls(np.array([0, 1, 0]))
        elif axis == CartesianAxis.Z:
            return cls(np.array([0, 0, 1]))


def vector_projection_in_non_orthogonal_basis(
    vector: np.ndarray, e1: np.ndarray, e2: np.ndarray, e3: np.ndarray
) -> np.ndarray:
    """
    This function converts a vector expressed in the global coordinate system
    to a vector expressed in a non-orthogonal coordinate system.

    Parameters
    ----------
    vector: np.ndarray
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

    Source
    ------
    Desroches, G., Chèze, L., & Dumas, R. (2010).
    Expression of joint moment in the joint coordinate system. Journal of biomechanical engineering, 132(11).
    https://doi.org/10.1115/1.4002537

    2.1 Expression of a 3D Vector in a Nonorthogonal Coordinate Base.

    """

    if vector.shape[0] != 3:
        raise ValueError("The vector must be expressed in 3D.")
    if len(vector.shape) == 1:
        vector = vector[:, np.newaxis]

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

    vnop = np.zeros(vector.shape)

    vnop[0, :] = np.sum(np.cross(e2, e3, axis=0) * vector, 0) / np.sum(np.cross(e1, e2, axis=0) * e3, 0)
    vnop[1, :] = np.sum(np.cross(e3, e1, axis=0) * vector, 0) / np.sum(np.cross(e1, e2, axis=0) * e3, 0)
    vnop[2, :] = np.sum(np.cross(e1, e2, axis=0) * vector, 0) / np.sum(np.cross(e1, e2, axis=0) * e3, 0)

    return vnop
