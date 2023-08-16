from casadi import cos, sin, MX, sqrt

from ..utils.enums import NaturalAxis, TransformationMatrixType


def transformation_matrix(
    matrix_type: TransformationMatrixType, length: float, alpha: float, beta: float, gamma: float
):
    """
    Create a transformation matrix from a TransformationMatrixType

    Parameters
    ----------
    matrix_type: TransformationMatrixType
        The type of transformation matrix to create, such as TransformationMatrixType.Buv, TransformationMatrixType.Bvw, etc.
    length: float
        The length of the segment
    alpha: float
        The alpha angle
    beta: float
        The beta angle
    gamma: float
        The gamma angle

    Returns
    -------
    numpy.ndarray
        The transformation matrix
    """
    if matrix_type == TransformationMatrixType.Buv:
        return _transformation_matrix_Buv(length, alpha, beta, gamma)
    elif matrix_type == TransformationMatrixType.Bvu:
        return _transformation_matrix_Bvu(length, alpha, beta, gamma)
    elif matrix_type == TransformationMatrixType.Bwu:
        return _transformation_matrix_Bwu(length, alpha, beta, gamma)
    elif matrix_type == TransformationMatrixType.Buw:
        return _transformation_matrix_Buw(length, alpha, beta, gamma)
    elif matrix_type == TransformationMatrixType.Bvw:
        return _transformation_matrix_Bvw(length, alpha, beta, gamma)
    elif matrix_type == TransformationMatrixType.Bwv:
        return _transformation_matrix_Bwv(length, alpha, beta, gamma)
    else:
        raise ValueError(f"Unknown TransformationMatrixType: {matrix_type}")


def _transformation_matrix_Buv(length: float, alpha: float, beta: float, gamma: float) -> MX:
    """
    Create a transformation matrix of type Buv

    Parameters
    ----------
    length: float
        The length of the segment
    alpha: float
        The alpha angle
    beta: float
        The beta angle
    gamma: float
        The gamma angle

    Returns
    -------
    numpy.ndarray
        The transformation matrix
    """
    B = MX.zeros(3, 3)
    B[:, 0] = MX([1, 0, 0])
    B[0, 1] = length * cos(gamma)
    B[1, 1] = length * sin(gamma)
    B[0, 2] = cos(beta)
    B[1, 2] = (cos(alpha) - cos(beta) * cos(gamma)) / sin(gamma)
    B[2, 2] = sqrt(1 - cos(beta) ** 2 - ((cos(alpha) - cos(beta) * cos(gamma)) / sin(gamma)) ** 2)
    return B


def _transformation_matrix_Bvu(length: float, alpha: float, beta: float, gamma: float) -> MX:
    """
    Create a transformation matrix of type Bvu

    Parameters
    ----------
    length: float
        The length of the segment
    alpha: float
        The alpha angle
    beta: float
        The beta angle
    gamma: float
        The gamma angle

    Returns
    -------
    numpy.ndarray
        The transformation matrix
    """
    B = MX.zeros(3, 3)
    B[0, 0] = sin(gamma)
    B[1, 0] = cos(gamma)
    B[1, 1] = length
    B[0, 2] = (cos(beta) - cos(alpha) * cos(gamma)) / sin(gamma)
    B[1, 2] = cos(alpha)
    B[2, 2] = sqrt(1 - cos(alpha) ** 2 - ((cos(beta) - cos(alpha) * cos(gamma)) / sin(gamma)) ** 2)
    return B


def _transformation_matrix_Bwu(length: float, alpha: float, beta: float, gamma: float) -> MX:
    """
    Create a transformation matrix of type Bwu

    Parameters
    ----------
    length: float
        The length of the segment
    alpha: float
        The alpha angle
    beta: float
        The beta angle
    gamma: float
        The gamma angle

    Returns
    -------
    numpy.ndarray
        The transformation matrix
    """
    B = MX.zeros(3, 3)
    B[0, 0] = sin(beta)
    B[2, 0] = cos(beta)
    B[0, 1] = length * (cos(gamma) - cos(alpha) * cos(beta) / sin(beta))
    B[1, 1] = length * sqrt(1 - cos(alpha) ** 2 - ((cos(gamma) - cos(alpha) * cos(beta)) / sin(beta)) ** 2)
    B[2, 1] = length * cos(alpha)
    B[2, 2] = 1
    return B


def _transformation_matrix_Buw(length: float, alpha: float, beta: float, gamma: float) -> MX:
    raise NotImplementedError("The transformation matrix Buw is not implemented yet.")


def _transformation_matrix_Bvw(length: float, alpha: float, beta: float, gamma: float) -> MX:
    raise NotImplementedError("The transformation matrix Bvw is not implemented yet.")


def _transformation_matrix_Bwv(length: float, alpha: float, beta: float, gamma: float) -> MX:
    raise NotImplementedError("The transformation matrix Bwv is not implemented yet.")


def from_plane_and_axis_to_keep(plane: tuple[NaturalAxis, NaturalAxis], axis_to_keep: NaturalAxis):
    """
    Create a transformation matrix from a plane and an axis to keep

    Parameters
    ----------
    plane: tuple[NaturalAxis, NaturalAxis]
        The plane to define the cross product of the orthogonal axis (axis[0] x axis[1])
    axis_to_keep:
        The axis to keep in the plane

    Returns
    -------

    """
    check_plane(plane)
    check_axis_to_keep(axis_to_keep)

    if NaturalAxis.U in plane and NaturalAxis.V in plane:
        if axis_to_keep == NaturalAxis.U:
            return transformation_matrix(TransformationMatrixType.Buv)
        elif axis_to_keep == NaturalAxis.V:
            return transformation_matrix(TransformationMatrixType.Bvu)

    elif NaturalAxis.U in plane and NaturalAxis.W in plane:
        if axis_to_keep == NaturalAxis.U:
            raise NotImplementedError("The transformation matrix Buw is not implemented yet.")
        elif axis_to_keep == NaturalAxis.W:
            return transformation_matrix(TransformationMatrixType.Bwu)

    elif NaturalAxis.V in plane and NaturalAxis.W in plane:
        if axis_to_keep == NaturalAxis.V:
            raise NotImplementedError("The transformation matrix Bvw is not implemented yet.")
        elif axis_to_keep == NaturalAxis.W:
            raise NotImplementedError("The transformation matrix Bwv is not implemented yet.")


def check_plane(plane: tuple[NaturalAxis, NaturalAxis]):
    """Check if the plane is valid"""
    if len(plane) != 2:
        raise ValueError(f"Plane must be a tuple of length 2, got {len(plane)}")
    if not all(isinstance(axis, NaturalAxis) for axis in plane):
        raise ValueError(f"Plane must be a tuple of NaturalAxis, got {plane}")
    if plane[0] == plane[1]:
        raise ValueError(f"Plane must be a tuple of different axis, got {plane}")
    if (
        (plane[0] == NaturalAxis.V and plane[1] == NaturalAxis.U)
        or (plane[0] == NaturalAxis.U and plane[1] == NaturalAxis.W)
        or (plane[0] == NaturalAxis.W and plane[1] == NaturalAxis.V)
    ):
        raise ValueError(f"Invert Axis in plane, because it would lead to an indirect frame, got {plane}")


def check_axis_to_keep(axis_to_keep: NaturalAxis):
    """Check if the axis to keep is valid"""
    if not isinstance(axis_to_keep, NaturalAxis):
        raise ValueError(f"Axis to keep must be of type NaturalAxis, got {axis_to_keep}")
