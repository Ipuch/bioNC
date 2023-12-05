import numpy as np
from numpy import cos, sin

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


def _transformation_matrix_Buv(length: float, alpha: float, beta: float, gamma: float) -> np.ndarray:
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
    return np.array(
        [
            [1, length * cos(gamma), cos(beta)],
            [0, length * sin(gamma), (cos(alpha) - cos(beta) * cos(gamma)) / sin(gamma)],
            [0, 0, np.sqrt(1 - cos(beta) ** 2 - ((cos(alpha) - cos(beta) * cos(gamma)) / sin(gamma)) ** 2)],
        ]
    )


def _transformation_matrix_Bvu(length: float, alpha: float, beta: float, gamma: float) -> np.ndarray:
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
    return np.array(
        [
            [sin(gamma), 0, (cos(beta) - cos(alpha) * cos(gamma)) / sin(gamma)],
            [cos(gamma), length, cos(alpha)],
            [0, 0, np.sqrt(1 - cos(alpha) ** 2 - ((cos(beta) - cos(alpha) * cos(gamma)) / sin(gamma)) ** 2)],
        ]
    )


def _transformation_matrix_Bwu(length: float, alpha: float, beta: float, gamma: float) -> np.ndarray:
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
    return np.array(
        [
            [sin(beta), length * (cos(gamma) - cos(alpha) * cos(beta) / sin(beta)), 0],
            [0, length * np.sqrt(1 - cos(alpha) ** 2 - ((cos(gamma) - cos(alpha) * cos(beta)) / sin(beta)) ** 2), 0],
            [cos(beta), length * cos(alpha), 1],
        ]
    )


def _transformation_matrix_Buw(length: float, alpha: float, beta: float, gamma: float) -> np.ndarray:
    # raise NotImplementedError("The transformation matrix Buw is not implemented yet.")
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
    return np.array(
        [
            [1, length * cos(gamma), cos(beta)],
            [0, length * np.sqrt(1 - cos(gamma) ** 2 - ((cos(alpha) - cos(gamma) * cos(beta)) / sin(beta)) ** 2), 0], 
            [0, length*(cos(alpha) - cos(gamma) * cos(beta) / sin(beta)), sin(beta)],
        ]
    )


def _transformation_matrix_Bvw(length: float, alpha: float, beta: float, gamma: float) -> np.ndarray:
    raise NotImplementedError("The transformation matrix Bvw is not implemented yet.")


def _transformation_matrix_Bwv(length: float, alpha: float, beta: float, gamma: float) -> np.ndarray:
    raise NotImplementedError("The transformation matrix Bwv is not implemented yet.")
