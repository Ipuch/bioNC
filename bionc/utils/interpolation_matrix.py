import numpy as np
from numpy import eye


def interpolate_natural_vector(vector: np.ndarray) -> np.ndarray:
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
    interpolate_natural_vector: np.ndarray
        Interpolation  matrix [3 x 12], denoted Ni to get the location of the vector as linear combination of Q.
        vector in global frame = Ni * Qi
    """

    if vector.shape[0] != 3:
        raise ValueError("Vector must be 3x1")

    interpolation_matrix = np.zeros((3, 12))
    interpolation_matrix[0:3, 0:3] = vector[0] * eye(3)
    interpolation_matrix[0:3, 3:6] = (1 + vector[1]) * eye(3)
    interpolation_matrix[0:3, 6:9] = -vector[1] * eye(3)
    interpolation_matrix[0:3, 9:12] = vector[2] * eye(3)

    return interpolation_matrix


def to_natural_vector(interpolation_matrix: np.ndarray) -> np.ndarray:
    """
    This function converts an interpolation matrix, denoted Ni, such as:
    Ni * Qi -> location in the global frame
    to a vector expressed in a non-orthogonal coordinate system associated to the segment coordinates.

    Parameters
    ----------
    interpolation_matrix: np.ndarray
        Interpolation  matrix [3 x 12], denoted Ni to get the location of the vector as linear combination of Q.
        vector in global frame = Ni * Qi

    Returns
    -------
    np.ndarray
        Vector in the natural coordinate system to interpolate (Pi, ui, vi, wi)
    """

    if interpolation_matrix.shape != (3, 12):
        raise ValueError("Interpolation matrix must be 3x12")

    vector = np.zeros(3)
    vector[0] = interpolation_matrix[0, 0]
    vector[1] = interpolation_matrix[0, 3] - 1
    vector[2] = interpolation_matrix[0, 9]

    return vector


# test these two functions with pytest
