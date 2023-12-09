from biorbd_casadi import Rotation
from casadi import MX
import numpy as np


def rotation_matrix_from_mx_to_biorbd(R: np.ndarray | MX) -> Rotation:
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


def rotation_matrix_to_euler_angles(rotation_matrix: MX, seq: str = "xyz") -> MX:
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
    MX
        The Euler vector in radian as an MX
    """

    rotation_matrix_biorbd = rotation_matrix_from_mx_to_biorbd(rotation_matrix)
    return Rotation.toEulerAngles(rotation_matrix_biorbd, seq).to_mx()


# not possible to use scipy with casadi ...
