import numpy as np
from numpy import eye


def interpolation_matrix(vector: np.ndarray) -> np.ndarray:
    """
    This function converts a vector expressed in a non-orthogonal coordinate system
    to an interpolation matrix, denoted Ni, such as:
    Ni * Qi -> location in the global frame

    Parameters
    ----------
    vector : np.ndarray
        Vector in the natural coordinate system to interpolate (Pi, ui, vi, wi)

    Returns
    -------
    interpolation_matrix: np.ndarray
        Interpolation  matrix [3 x 12], denoted Ni to get the location of the vector as linear combination of Q.
        vector in global frame = Ni * Qi
    """

    interpolation_matrix = np.zeros((3, 12))
    interpolation_matrix[0:3, 0:3] = vector[0] * eye(3)
    interpolation_matrix[0:3, 3:6] = (1 + vector[1]) * eye(3)
    interpolation_matrix[0:3, 6:9] = -vector[1] * eye(3)
    interpolation_matrix[0:3, 9:12] = vector[2] * eye(3)

    return interpolation_matrix