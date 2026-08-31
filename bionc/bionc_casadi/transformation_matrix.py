"""
Transformation matrices B, from the natural coordinate system (ui, rpi-rdi, wi) to an
orthonormal segment coordinate system, and their analytical inverses.

See bionc.bionc_numpy.transformation_matrix for the geometric definition, the Gram identity the
implementations satisfy, and the sympy-derived closed forms of the inverses.
"""

from casadi import cos, sin, MX, sqrt

from ..utils.enums import NaturalAxis, TransformationMatrixType


def compute_transformation_matrix(
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
    B[0, 1] = length * (cos(gamma) - cos(alpha) * cos(beta)) / sin(beta)
    B[1, 1] = length * sqrt(1 - cos(alpha) ** 2 - ((cos(gamma) - cos(alpha) * cos(beta)) / sin(beta)) ** 2)
    B[2, 1] = length * cos(alpha)
    B[2, 2] = 1
    return B


def _transformation_matrix_Buw(length: float, alpha: float, beta: float, gamma: float) -> MX:
    """
    Create a transformation matrix of type Buw

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
    B[0, 0] = 1
    B[0, 1] = length * cos(gamma)
    B[0, 2] = cos(beta)
    B[1, 1] = length * sqrt(1 - cos(gamma) ** 2 - ((cos(alpha) - cos(gamma) * cos(beta)) / sin(beta)) ** 2)
    B[2, 1] = length * (cos(alpha) - cos(gamma) * cos(beta)) / sin(beta)
    B[2, 2] = sin(beta)
    return B


def _transformation_matrix_Bvw(length: float, alpha: float, beta: float, gamma: float) -> MX:
    raise NotImplementedError("The transformation matrix Bvw is not implemented yet.")


def _transformation_matrix_Bwv(length: float, alpha: float, beta: float, gamma: float) -> MX:
    raise NotImplementedError("The transformation matrix Bwv is not implemented yet.")


def compute_transformation_matrix_inverse(
    matrix_type: TransformationMatrixType, length: float, alpha: float, beta: float, gamma: float
):
    """
    Create the analytical inverse of a transformation matrix from a TransformationMatrixType

    This is the exact inverse of compute_transformation_matrix called with the same arguments,
    computed in closed form rather than by numerical inversion.

    Parameters
    ----------
    matrix_type: TransformationMatrixType
        The type of transformation matrix to invert, such as TransformationMatrixType.Buv, TransformationMatrixType.Bvw, etc.
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
    MX
        The inverse of the transformation matrix
    """
    if matrix_type == TransformationMatrixType.Buv:
        return _transformation_matrix_Buv_inverse(length, alpha, beta, gamma)
    elif matrix_type == TransformationMatrixType.Bvu:
        return _transformation_matrix_Bvu_inverse(length, alpha, beta, gamma)
    elif matrix_type == TransformationMatrixType.Bwu:
        return _transformation_matrix_Bwu_inverse(length, alpha, beta, gamma)
    elif matrix_type == TransformationMatrixType.Buw:
        return _transformation_matrix_Buw_inverse(length, alpha, beta, gamma)
    elif matrix_type == TransformationMatrixType.Bvw:
        return _transformation_matrix_Bvw_inverse(length, alpha, beta, gamma)
    elif matrix_type == TransformationMatrixType.Bwv:
        return _transformation_matrix_Bwv_inverse(length, alpha, beta, gamma)
    else:
        raise ValueError(f"Unknown TransformationMatrixType: {matrix_type}")


def gram_determinant_sqrt(alpha: float, beta: float, gamma: float) -> MX:
    """
    Square root of the Gram determinant of (u, v/length, w), the quantity every transformation
    matrix and every analytical inverse below is built on.

    It relates to the determinant of any transformation matrix B through det(B) = length * this,
    and it is real if and only if the three angles describe a segment that exists in 3D.

    Parameters
    ----------
    alpha: float
        The alpha angle
    beta: float
        The beta angle
    gamma: float
        The gamma angle

    Returns
    -------
    MX
        The square root of the Gram determinant
    """
    return sqrt(1 - cos(alpha) ** 2 - cos(beta) ** 2 - cos(gamma) ** 2 + 2 * cos(alpha) * cos(beta) * cos(gamma))


def _transformation_matrix_Buv_inverse(length: float, alpha: float, beta: float, gamma: float) -> MX:
    """
    Create the analytical inverse of a transformation matrix of type Buv

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
    MX
        The inverse of the transformation matrix
    """
    determinant = gram_determinant_sqrt(alpha, beta, gamma)
    B = MX.zeros(3, 3)
    B[0, 0] = 1
    B[0, 1] = -cos(gamma) / sin(gamma)
    B[0, 2] = (cos(alpha) * cos(gamma) - cos(beta)) / (sin(gamma) * determinant)
    B[1, 1] = 1 / (length * sin(gamma))
    B[1, 2] = (cos(beta) * cos(gamma) - cos(alpha)) / (length * sin(gamma) * determinant)
    B[2, 2] = sin(gamma) / determinant
    return B


def _transformation_matrix_Bvu_inverse(length: float, alpha: float, beta: float, gamma: float) -> MX:
    """
    Create the analytical inverse of a transformation matrix of type Bvu

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
    MX
        The inverse of the transformation matrix
    """
    determinant = gram_determinant_sqrt(alpha, beta, gamma)
    B = MX.zeros(3, 3)
    B[0, 0] = 1 / sin(gamma)
    B[0, 2] = (cos(alpha) * cos(gamma) - cos(beta)) / (sin(gamma) * determinant)
    B[1, 0] = -cos(gamma) / (length * sin(gamma))
    B[1, 1] = 1 / length
    B[1, 2] = (cos(beta) * cos(gamma) - cos(alpha)) / (length * sin(gamma) * determinant)
    B[2, 2] = sin(gamma) / determinant
    return B


def _transformation_matrix_Bwu_inverse(length: float, alpha: float, beta: float, gamma: float) -> MX:
    """
    Create the analytical inverse of a transformation matrix of type Bwu

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
    MX
        The inverse of the transformation matrix
    """
    determinant = gram_determinant_sqrt(alpha, beta, gamma)
    B = MX.zeros(3, 3)
    B[0, 0] = 1 / sin(beta)
    B[0, 1] = (cos(alpha) * cos(beta) - cos(gamma)) / (sin(beta) * determinant)
    B[1, 1] = sin(beta) / (length * determinant)
    B[2, 0] = -cos(beta) / sin(beta)
    B[2, 1] = (cos(beta) * cos(gamma) - cos(alpha)) / (sin(beta) * determinant)
    B[2, 2] = 1
    return B


def _transformation_matrix_Buw_inverse(length: float, alpha: float, beta: float, gamma: float) -> MX:
    """
    Create the analytical inverse of a transformation matrix of type Buw

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
    MX
        The inverse of the transformation matrix
    """
    determinant = gram_determinant_sqrt(alpha, beta, gamma)
    B = MX.zeros(3, 3)
    B[0, 0] = 1
    B[0, 1] = (cos(alpha) * cos(beta) - cos(gamma)) / (sin(beta) * determinant)
    B[0, 2] = -cos(beta) / sin(beta)
    B[1, 1] = sin(beta) / (length * determinant)
    B[2, 1] = (cos(beta) * cos(gamma) - cos(alpha)) / (sin(beta) * determinant)
    B[2, 2] = 1 / sin(beta)
    return B


def _transformation_matrix_Bvw_inverse(length: float, alpha: float, beta: float, gamma: float) -> MX:
    raise NotImplementedError("The transformation matrix Bvw is not implemented yet.")


def _transformation_matrix_Bwv_inverse(length: float, alpha: float, beta: float, gamma: float) -> MX:
    raise NotImplementedError("The transformation matrix Bwv is not implemented yet.")
