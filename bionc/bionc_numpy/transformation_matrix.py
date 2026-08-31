"""
Transformation matrices B, from the natural coordinate system (ui, rpi-rdi, wi) to an
orthonormal segment coordinate system, and their analytical inverses.

The columns of B are the natural vectors (u, rp-rd, w) expressed in the orthonormal frame, so B
carries exactly the segment geometry: with ca, cb, cg = cos(alpha), cos(beta), cos(gamma),

    B.T @ B == [[1, length * cg, cb], [length * cg, length**2, length * ca], [cb, length * ca, 1]]

holds for every matrix type. That Gram identity is the reference the implementations below are
checked against in tests/test_transformation_matrix.py.

The inverses are the closed forms obtained with sympy from that same definition. They all share

    delta = 1 - ca**2 - cb**2 - cg**2 + 2 * ca * cb * cg

the Gram determinant of (u, v/length, w), which gives det(B) == length * sqrt(delta) whatever the
matrix type. delta > 0 is therefore the exact condition for the three angles to describe a
non-degenerate segment, see NaturalSegment._angle_sanity_check.
"""

import numpy as np
from numpy import cos, sin

from ..utils.enums import TransformationMatrixType


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
            [sin(beta), length * (cos(gamma) - cos(alpha) * cos(beta)) / sin(beta), 0],
            [0, length * np.sqrt(1 - cos(alpha) ** 2 - ((cos(gamma) - cos(alpha) * cos(beta)) / sin(beta)) ** 2), 0],
            [cos(beta), length * cos(alpha), 1],
        ]
    )


def _transformation_matrix_Buw(length: float, alpha: float, beta: float, gamma: float) -> np.ndarray:
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
    return np.array(
        [
            [1, length * cos(gamma), cos(beta)],
            [0, length * np.sqrt(1 - cos(gamma) ** 2 - ((cos(alpha) - cos(gamma) * cos(beta)) / sin(beta)) ** 2), 0],
            [0, length * (cos(alpha) - cos(gamma) * cos(beta)) / sin(beta), sin(beta)],
        ]
    )


def _transformation_matrix_Bvw(length: float, alpha: float, beta: float, gamma: float) -> np.ndarray:
    raise NotImplementedError("The transformation matrix Bvw is not implemented yet.")


def _transformation_matrix_Bwv(length: float, alpha: float, beta: float, gamma: float) -> np.ndarray:
    raise NotImplementedError("The transformation matrix Bwv is not implemented yet.")


def gram_determinant_sqrt(alpha: float, beta: float, gamma: float) -> float:
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
    float
        The square root of the Gram determinant
    """
    return np.sqrt(1 - cos(alpha) ** 2 - cos(beta) ** 2 - cos(gamma) ** 2 + 2 * cos(alpha) * cos(beta) * cos(gamma))


def _transformation_matrix_Buv_inverse(length: float, alpha: float, beta: float, gamma: float) -> np.ndarray:
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
    numpy.ndarray
        The inverse of the transformation matrix
    """
    determinant = gram_determinant_sqrt(alpha, beta, gamma)
    return np.array(
        [
            [1, -cos(gamma) / sin(gamma), (cos(alpha) * cos(gamma) - cos(beta)) / (sin(gamma) * determinant)],
            [
                0,
                1 / (length * sin(gamma)),
                (cos(beta) * cos(gamma) - cos(alpha)) / (length * sin(gamma) * determinant),
            ],
            [0, 0, sin(gamma) / determinant],
        ]
    )


def _transformation_matrix_Bvu_inverse(length: float, alpha: float, beta: float, gamma: float) -> np.ndarray:
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
    numpy.ndarray
        The inverse of the transformation matrix
    """
    determinant = gram_determinant_sqrt(alpha, beta, gamma)
    return np.array(
        [
            [1 / sin(gamma), 0, (cos(alpha) * cos(gamma) - cos(beta)) / (sin(gamma) * determinant)],
            [
                -cos(gamma) / (length * sin(gamma)),
                1 / length,
                (cos(beta) * cos(gamma) - cos(alpha)) / (length * sin(gamma) * determinant),
            ],
            [0, 0, sin(gamma) / determinant],
        ]
    )


def _transformation_matrix_Bwu_inverse(length: float, alpha: float, beta: float, gamma: float) -> np.ndarray:
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
    numpy.ndarray
        The inverse of the transformation matrix
    """
    determinant = gram_determinant_sqrt(alpha, beta, gamma)
    return np.array(
        [
            [1 / sin(beta), (cos(alpha) * cos(beta) - cos(gamma)) / (sin(beta) * determinant), 0],
            [0, sin(beta) / (length * determinant), 0],
            [-cos(beta) / sin(beta), (cos(beta) * cos(gamma) - cos(alpha)) / (sin(beta) * determinant), 1],
        ]
    )


def _transformation_matrix_Buw_inverse(length: float, alpha: float, beta: float, gamma: float) -> np.ndarray:
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
    numpy.ndarray
        The inverse of the transformation matrix
    """
    determinant = gram_determinant_sqrt(alpha, beta, gamma)
    return np.array(
        [
            [
                1,
                (cos(alpha) * cos(beta) - cos(gamma)) / (sin(beta) * determinant),
                -cos(beta) / sin(beta),
            ],
            [0, sin(beta) / (length * determinant), 0],
            [0, (cos(beta) * cos(gamma) - cos(alpha)) / (sin(beta) * determinant), 1 / sin(beta)],
        ]
    )


def _transformation_matrix_Bvw_inverse(length: float, alpha: float, beta: float, gamma: float) -> np.ndarray:
    raise NotImplementedError("The transformation matrix Bvw is not implemented yet.")


def _transformation_matrix_Bwv_inverse(length: float, alpha: float, beta: float, gamma: float) -> np.ndarray:
    raise NotImplementedError("The transformation matrix Bwv is not implemented yet.")


TRANSFORMATION_MAP = {
    TransformationMatrixType.Buv: _transformation_matrix_Buv,
    TransformationMatrixType.Bvu: _transformation_matrix_Bvu,
    TransformationMatrixType.Bwu: _transformation_matrix_Bwu,
    TransformationMatrixType.Buw: _transformation_matrix_Buw,
    TransformationMatrixType.Bvw: _transformation_matrix_Bvw,
    TransformationMatrixType.Bwv: _transformation_matrix_Bwv,
}

INVERSE_TRANSFORMATION_MAP = {
    TransformationMatrixType.Buv: _transformation_matrix_Buv_inverse,
    TransformationMatrixType.Bvu: _transformation_matrix_Bvu_inverse,
    TransformationMatrixType.Bwu: _transformation_matrix_Bwu_inverse,
    TransformationMatrixType.Buw: _transformation_matrix_Buw_inverse,
    TransformationMatrixType.Bvw: _transformation_matrix_Bvw_inverse,
    TransformationMatrixType.Bwv: _transformation_matrix_Bwv_inverse,
}


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

    if matrix_type not in TRANSFORMATION_MAP:
        raise ValueError(f"Unknown TransformationMatrixType: {matrix_type}")

    return TRANSFORMATION_MAP[matrix_type](length, alpha, beta, gamma)


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
    numpy.ndarray
        The inverse of the transformation matrix
    """

    if matrix_type not in INVERSE_TRANSFORMATION_MAP:
        raise ValueError(f"Unknown TransformationMatrixType: {matrix_type}")

    return INVERSE_TRANSFORMATION_MAP[matrix_type](length, alpha, beta, gamma)
