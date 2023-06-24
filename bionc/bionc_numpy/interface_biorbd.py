from biorbd import Rotation
# from scipy.spatial.transform import Rotation
import numpy as np


def rotation_matrix_from_numpy_to_biorbd(R: np.ndarray) -> Rotation:
    """
    This function returns the rotation matrix in biorbd formalism

    Parameters
    ---------
    R : np.ndarray
        Rotation matrix (3x3)

    Returns
    ---------
    biorbd.Rotation
        The rotation matrix object
    """

    return Rotation(
        R[0, 0],
        R[0, 1],
        R[0, 2],
        R[1, 0],
        R[1, 1],
        R[1, 2],
        R[2, 0],
        R[2, 1],
        R[2, 2],
    )


def rotation_matrix_to_euler_angles(rotation_matrix: np.ndarray, seq: str = "xyz") -> np.ndarray:
    """
    This function returns the rotation matrix in euler angles vector

    Parameters
    ---------
    rotation_matrix : np.ndarray
        Rotation matrix (3x3)
    seq: str = "xyz"
        order of the coordinates in the returned vector
    Returns
    ---------
    Rotation.toEulerAngles(rotation_matrix_biorbd, seq).to_array()
        The Euler vector in radiant as an array
    """

    rotation_matrix_biorbd = rotation_matrix_from_numpy_to_biorbd(rotation_matrix)
    return Rotation.toEulerAngles(rotation_matrix_biorbd, seq).to_array()

# not possible to use scipy with casadi ...
# do the same function but with scipy rotation matrix
# def rotation_matrix_to_euler_angles_scipy(rotation_matrix: np.ndarray, seq: str = "xyz") -> np.ndarray:
#     """
#     This function returns the rotation matrix in euler angles vector
#
#     Parameters
#     ---------
#     rotation_matrix : np.ndarray
#         Rotation matrix (3x3)
#     seq: str = "xyz"
#         order of the coordinates in the returned vector
#     Returns
#     ---------
#     Rotation.toEulerAngles(rotation_matrix_biorbd, seq).to_array()
#         The Euler vector in radiant as an array
#     """
#     rotation_matrix_scipy = Rotation.from_matrix(rotation_matrix)
#     return rotation_matrix_scipy.as_euler(seq.upper())   # upper because we want intrinsic euler rotations
