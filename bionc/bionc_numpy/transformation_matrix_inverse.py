import numpy as np
from numpy import cos, sin, sqrt

from ..utils.enums import TransformationMatrixType


def compute_transformation_matrix_inverse(
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
    transformation_matrix_functions = {
        TransformationMatrixType.Buv: _transformation_matrix_inverse_Buv,
        TransformationMatrixType.Bvu: _transformation_matrix_inverse_Bvu,
        TransformationMatrixType.Bwu: _transformation_matrix_inverse_Bwu,
        TransformationMatrixType.Buw: _transformation_matrix_inverse_Buw,
        TransformationMatrixType.Bvw: _transformation_matrix_inverse_Bvw,
        TransformationMatrixType.Bwv: _transformation_matrix_inverse_Bwv,
    }
    transformation_matrix_callable = transformation_matrix_functions.get(matrix_type)

    if transformation_matrix_callable is not None:
        return transformation_matrix_callable(length, alpha, beta, gamma)
    else:
        raise ValueError(f"Unknown TransformationMatrixType: {matrix_type}")


def _transformation_matrix_inverse_Buv(length: float, alpha: float, beta: float, gamma: float) -> np.ndarray:
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
        The transformation matrix inverse, denoted Buv^-1
    """
    # denominator = sqrt(-csc(gamma) ** 2 * (cos(alpha) - cos(beta) * csc(gamma)) ** 2 - cos(beta) ** 2 + 1)
    # return np.array(
    #     [
    #         [1, -cot(gamma), csc(gamma) * (cos(alpha) * cot(gamma) - cos(beta) * csc(gamma)) / denominator],
    #         [0, csc(gamma) / length,
    #          csc(gamma) * (cos(beta) * cot(gamma) - cos(alpha) * csc(gamma)) / (length * denominator)],
    #         [0, 0, 1 / denominator],
    #     ]
    # )

    # Calculate sin and cos values
    sin_g = np.sin(gamma)
    cos_g = np.cos(gamma)
    sin_a = np.sin(alpha)
    cos_a = np.cos(alpha)
    sin_b = np.sin(beta)
    cos_b = np.cos(beta)

    # Calculate the common denominator
    common_denominator = np.sqrt(
        -cos_b ** 2 + 1
        - cos_a ** 2 / sin_g ** 2
        + 2 * cos_a * cos_b * cos_g / sin_g ** 2
        - cos_b ** 2 * cos_g ** 2 / sin_g ** 2
    ) * sin_g ** 2

    # Calculate the simplified elements by dividing by the common denominator
    element_11 = 1 / common_denominator
    element_12 = -cos_g / (sin_g * common_denominator)
    element_13 = (-sin_g ** 2 * cos_b + cos_a * cos_g - cos_b * cos_g ** 2) / common_denominator

    element_21 = 0
    element_22 = 1 / (length * sin_g * common_denominator)
    element_23 = (-cos_a + cos_b * cos_g) / (length * sin_g * common_denominator)

    element_31 = 0
    element_32 = 0
    element_33 = 1 / common_denominator

    # Create the factored matrix
    factored_matrix = np.array([[element_11, element_12, element_13],
                                [element_21, element_22, element_23],
                                [element_31, element_32, element_33]])

    # return factored_matrix
    a = alpha
    b = beta
    g = gamma
    L = length
    return np.array(
        [
            [1, -1 / np.tan(g), (cos(a) * cos(g) - cos(b)) / (sqrt(
                cos(a) ** 2 / sin(g) ** 2 - 2 * cos(a) * cos(b) * cos(g) / sin(g) ** 2 + cos(b) ** 2 * cos(
                    g) ** 2 / sin(g) ** 2 + 1 - cos(b) ** 2) * sin(g) ** 2)],
            [0, 1 / (L * sin(g)), (-cos(a) + cos(b) * cos(g)) / (L * sqrt(
                cos(a) ** 2 / sin(g) ** 2 - 2 * cos(a) * cos(b) * cos(g) / sin(g) ** 2 + cos(b) ** 2 * cos(
                    g) ** 2 / sin(g) ** 2 + 1 - cos(b) ** 2) * sin(g) ** 2)],
            [0, 0, 1 / sqrt(cos(a) ** 2 / sin(g) ** 2 - 2 * cos(a) * cos(b) * cos(g) / sin(g) ** 2 + cos(b) ** 2 * cos(
                g) ** 2 / sin(g) ** 2 + 1 - cos(b) ** 2)]
        ])


# >>> pprint(s_im)
# ⎡     -1                           cos(a)⋅cos(g) - cos(b)                     ⎤
# ⎢1   ──────    ────────────────────────────────────────────────────────────── ⎥
# ⎢    tan(g)          ________________________________________________         ⎥
# ⎢                   ╱        2                                  2             ⎥
# ⎢                  ╱      cos (a)   2⋅cos(a)⋅cos(b)⋅cos(g)   cos (b)     2    ⎥
# ⎢                 ╱   1 - ─────── + ────────────────────── - ─────── ⋅sin (g) ⎥
# ⎢                ╱           2                2                 2             ⎥
# ⎢              ╲╱         sin (g)          sin (g)           sin (g)          ⎥
# ⎢                                                                             ⎥
# ⎢      1                          -cos(a) + cos(b)⋅cos(g)                     ⎥
# ⎢0  ────────  ────────────────────────────────────────────────────────────────⎥
# ⎢   L⋅sin(g)          ________________________________________________        ⎥
# ⎢                    ╱        2                                  2            ⎥
# ⎢                   ╱      cos (a)   2⋅cos(a)⋅cos(b)⋅cos(g)   cos (b)     2   ⎥
# ⎢             L⋅   ╱   1 - ─────── + ────────────────────── - ─────── ⋅sin (g)⎥
# ⎢                 ╱           2                2                 2            ⎥
# ⎢               ╲╱         sin (g)          sin (g)           sin (g)         ⎥
# ⎢                                                                             ⎥
# ⎢                                            1                                ⎥
# ⎢0     0           ──────────────────────────────────────────────────────     ⎥
# ⎢                        ________________________________________________     ⎥
# ⎢                       ╱        2                                  2         ⎥
# ⎢                      ╱      cos (a)   2⋅cos(a)⋅cos(b)⋅cos(g)   cos (b)      ⎥
# ⎢                     ╱   1 - ─────── + ────────────────────── - ───────      ⎥
# ⎢                    ╱           2                2                 2         ⎥
# ⎣                  ╲╱         sin (g)          sin (g)           sin (g)      ⎦
# >>> inverse_matrix = matrix.inv('LU')
# >>> s_im = simplify(inverse_matrix)
# >>> pprint(s_im)
# ⎡                   _______________________________________________                         ⎤
# ⎢                  ╱                           2      2       2                             ⎥
# ⎢                 ╱  - (cos(a) - cos(b)⋅cos(g))  + sin (b)⋅sin (g)                          ⎥
# ⎢                ╱   ───────────────────────────────────────────── ⋅(cos(a)⋅cos(g) - cos(b))⎥
# ⎢               ╱                          2                                                ⎥
# ⎢     -1      ╲╱                        sin (g)                                             ⎥
# ⎢1   ──────   ──────────────────────────────────────────────────────────────────────────────⎥
# ⎢    tan(g)                                             2      2       2                    ⎥
# ⎢                             - (cos(a) - cos(b)⋅cos(g))  + sin (b)⋅sin (g)                 ⎥
# ⎢                                                                                           ⎥
# ⎢      1                                 -cos(a) + cos(b)⋅cos(g)                            ⎥
# ⎢0  ────────         ───────────────────────────────────────────────────────────────        ⎥
# ⎢   L⋅sin(g)                 _______________________________________________                ⎥
# ⎢                           ╱                           2      2       2                    ⎥
# ⎢                          ╱  - (cos(a) - cos(b)⋅cos(g))  + sin (b)⋅sin (g)     2           ⎥
# ⎢                    L⋅   ╱   ───────────────────────────────────────────── ⋅sin (g)        ⎥
# ⎢                        ╱                          2                                       ⎥
# ⎢                      ╲╱                        sin (g)                                    ⎥
# ⎢                                                                                           ⎥
# ⎢                                                   1                                       ⎥
# ⎢0     0                      ─────────────────────────────────────────────                 ⎥
# ⎢                                   _______________________________________                 ⎥
# ⎢                                  ╱                           2                            ⎥
# ⎢                                 ╱    (cos(a) - cos(b)⋅cos(g))       2                     ⎥
# ⎢                                ╱   - ───────────────────────── + sin (b)                  ⎥
# ⎢                               ╱                  2                                        ⎥
# ⎣                             ╲╱                sin (g)                                     ⎦
# >>> inverse_matrix = matrix.inv('ADJ')
# >>> s_im = simplify(inverse_matrix)
# >>> pprint(s_im)
# ⎡     -1                           cos(a)⋅cos(g) - cos(b)                     ⎤
# ⎢1   ──────    ────────────────────────────────────────────────────────────── ⎥
# ⎢    tan(g)          ________________________________________________         ⎥
# ⎢                   ╱        2                                  2             ⎥
# ⎢                  ╱      cos (a)   2⋅cos(a)⋅cos(b)⋅cos(g)   cos (b)     2    ⎥
# ⎢                 ╱   1 - ─────── + ────────────────────── - ─────── ⋅sin (g) ⎥
# ⎢                ╱           2                2                 2             ⎥
# ⎢              ╲╱         sin (g)          sin (g)           sin (g)          ⎥
# ⎢                                                                             ⎥
# ⎢      1                          -cos(a) + cos(b)⋅cos(g)                     ⎥
# ⎢0  ────────  ────────────────────────────────────────────────────────────────⎥
# ⎢   L⋅sin(g)          ________________________________________________        ⎥
# ⎢                    ╱        2                                  2            ⎥
# ⎢                   ╱      cos (a)   2⋅cos(a)⋅cos(b)⋅cos(g)   cos (b)     2   ⎥
# ⎢             L⋅   ╱   1 - ─────── + ────────────────────── - ─────── ⋅sin (g)⎥
# ⎢                 ╱           2                2                 2            ⎥
# ⎢               ╲╱         sin (g)          sin (g)           sin (g)         ⎥
# ⎢                                                                             ⎥
# ⎢                                            1                                ⎥
# ⎢0     0           ──────────────────────────────────────────────────────     ⎥
# ⎢                        ________________________________________________     ⎥
# ⎢                       ╱        2                                  2         ⎥
# ⎢                      ╱      cos (a)   2⋅cos(a)⋅cos(b)⋅cos(g)   cos (b)      ⎥
# ⎢                     ╱   1 - ─────── + ────────────────────── - ───────      ⎥
# ⎢                    ╱           2                2                 2         ⎥
# ⎣                  ╲╱         sin (g)          sin (g)           sin (g)      ⎦


def _transformation_matrix_inverse_Bvu(length: float, alpha: float, beta: float, gamma: float) -> np.ndarray:
    raise NotImplementedError("Bvu^-1 is not implemented yet")


def _transformation_matrix_inverse_Bwu(length: float, alpha: float, beta: float, gamma: float) -> np.ndarray:
    raise NotImplementedError("Bwu^-1 is not implemented yet")


def _transformation_matrix_inverse_Buw(length: float, alpha: float, beta: float, gamma: float) -> np.ndarray:
    raise NotImplementedError("Buw^-1 is not implemented yet")


def _transformation_matrix_inverse_Bvw(length: float, alpha: float, beta: float, gamma: float) -> np.ndarray:
    raise NotImplementedError("Bvw^-1 is not implemented yet")


def _transformation_matrix_inverse_Bwv(length: float, alpha: float, beta: float, gamma: float) -> np.ndarray:
    raise NotImplementedError("Bwv^-1 is not implemented yet")


def cot(x):
    return 1 / np.tan(x)


def csc(x):
    return 1 / np.sin(x)
