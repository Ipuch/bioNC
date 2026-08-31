"""
The Gram determinant of the natural basis (u, v/length, w).

Every transformation matrix B and every analytical inverse is built on this single scalar, and the
angle sanity check of a natural segment is a test on its sign. It lives here so that the numpy
backend, the protocols and the tests all state it once.

bionc_casadi keeps its own casadi-native copy, because casadi needs casadi.cos rather than np.cos.
"""

import numpy as np


def gram_determinant(alpha, beta, gamma):
    """
    Gram determinant of (u, v/length, w), denoted delta.

    This is the determinant of the matrix of pairwise dot products of the three unit natural
    vectors, so det(B) == length * sqrt(delta) whatever the TransformationMatrixType.

    delta > 0 is the exact condition for the three angles to describe a segment that exists in 3D;
    delta <= 0 means no three vectors can hold those pairwise angles. Note that delta > 0 also
    implies sin(alpha), sin(beta) and sin(gamma) are all non-zero, which is what makes the divisions
    in the analytical inverses safe.

    Parameters
    ----------
    alpha
        The alpha angle, between v and w
    beta
        The beta angle, between u and w
    gamma
        The gamma angle, between u and v

    Returns
    -------
        The Gram determinant, of the same type as the angles
    """
    return (
        1
        - np.cos(alpha) ** 2
        - np.cos(beta) ** 2
        - np.cos(gamma) ** 2
        + 2 * np.cos(alpha) * np.cos(beta) * np.cos(gamma)
    )
