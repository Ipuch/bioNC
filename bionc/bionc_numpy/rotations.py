import numpy as np

from .interface_biorbd import rotation_matrix_to_euler_angles
from ..utils.enums import CartesianAxis, EulerSequence

# todo: test the whole file


def rotation_x(angle) -> np.ndarray:
    """This function returns the rotation matrix around the x axis by the angle given in argument"""
    return np.array([[1, 0, 0], [0, np.cos(angle), -np.sin(angle)], [0, np.sin(angle), np.cos(angle)]])


def rotation_y(angle) -> np.ndarray:
    """This function returns the rotation matrix around the y axis by the angle given in argument"""
    return np.array([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]])


def rotation_z(angle) -> np.ndarray:
    """This function returns the rotation matrix around the z axis by the angle given in argument"""
    return np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])


def rotation_matrices_from_rotation_matrix(rotation_matrix, sequence: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    This function converts a rotation matrix to individual rotation matrices

    Parameters
    ----------
    rotation_matrix : np.ndarray
        Rotation matrix
    sequence : str
        Sequence of rotations, e.g. 'xyz'

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        Individual rotation matrices
    """

    angles = rotation_matrix_to_euler_angles(rotation_matrix, sequence)

    R0 = rotation_matrix_from_angle_and_axis(angles[0], sequence[0])
    R1 = rotation_matrix_from_angle_and_axis(angles[1], sequence[1])
    R2 = rotation_matrix_from_angle_and_axis(angles[2], sequence[2])

    return R0, R1, R2


def rotation_matrix_from_angle_and_axis(angle: float, axis: str | CartesianAxis) -> np.ndarray:
    """
    This function returns a rotation matrix from an angle and an axis

    Parameters
    ----------
    angle : float
        Angle of rotation
    axis : str or CartesianAxis
        Axis of rotation such as 'x', 'y' or 'z'

    Returns
    -------
    np.ndarray
        Rotation matrix
    """

    if axis == "x" or axis == CartesianAxis.X:
        return rotation_x(angle)
    elif axis == "y" or axis == CartesianAxis.Y:
        return rotation_y(angle)
    elif axis == "z" or axis == CartesianAxis.Z:
        return rotation_z(angle)
    else:
        raise ValueError("The axis must be 'x', 'y' or 'z'.")


def euler_axes_from_rotation_matrices(
    R_0_parent: np.ndarray,
    R_0_child: np.ndarray,
    sequence: EulerSequence,
    axes_source_frame: str = "mixed",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    This function returns the euler axes from the rotation matrices in the global frame

    Parameters
    ----------
    R_0_parent : np.ndarray
        Rotation matrix of the parent ^0R_parent
    R_0_child : np.ndarray
        Rotation matrix of the child ^0R_child
    sequence : EulerSequence
        Sequence of rotations, e.g. 'xyz'
    axes_source_frame : str
        Frame from which the axes get computed from, e.g. 'parent', 'child', 'mixed'.
            - parent: the axes are computed from the parent rotation matrix
            - child: the axes are computed from the child rotation matrix
            - mixed (default): the first and second axes are computed from the parent rotation matrix,
            the third from the child, limitating non-linearities

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        Euler rotation axes in the global frame
    """

    R = R_0_parent.T @ R_0_child
    if isinstance(sequence, EulerSequence):
        sequence = sequence.value

    individual_rotation_matrices = rotation_matrices_from_rotation_matrix(R, sequence)

    euler_axes = [None for _ in range(3)]

    cumulated_rotation_matrix = np.eye(3)

    if axes_source_frame == "parent":
        for i, (axis, rotation) in enumerate(zip(sequence, individual_rotation_matrices)):
            cumulated_rotation_matrix = cumulated_rotation_matrix @ rotation

            v = vector_from_axis(axis)
            # e.g. for xyz, x: R_0_parent @ rotx @ [1, 0, 0]; y: R_0_parent @ rotx @ roty @ [0, 1, 0]; ...
            euler_axes[i] = R_0_parent @ cumulated_rotation_matrix @ v

        return tuple(euler_axes)

    elif axes_source_frame == "child":
        # flip the sequence and the rotation matrices
        sequence = sequence[::-1]
        individual_rotation_matrices = individual_rotation_matrices[::-1]

        for i, (axis, rotation) in enumerate(zip(sequence, individual_rotation_matrices)):
            cumulated_rotation_matrix = cumulated_rotation_matrix @ rotation.T

            v = vector_from_axis(axis)
            # e.g. for xyz, z: R_0_child @ rotz.T @ [0, 0, 1]; y: R_0_child @ rotz.T @ roty.T @ [0, 1, 0]; ...
            euler_axes[i] = R_0_child @ cumulated_rotation_matrix @ v

        # flip the euler axes
        euler_axes = euler_axes[::-1]
        return tuple(euler_axes)

    elif axes_source_frame == "mixed":
        # this method should be better as it relies less on the transformations of the rotation matrices
        # only the second axis depends on the first angle, the third relies on the child matrix

        parent_euler_axes = euler_axes_from_rotation_matrices(
            R_0_parent, R_0_child, sequence, axes_source_frame="parent"
        )
        child_euler_axes = euler_axes_from_rotation_matrices(R_0_parent, R_0_child, sequence, axes_source_frame="child")

        return parent_euler_axes[0], parent_euler_axes[1], child_euler_axes[2]


def vector_from_axis(axis: str | CartesianAxis) -> np.ndarray:
    """
    This function returns the vector associated with an axis

    Parameters
    ----------
    axis : str or CartesianAxis
        Axis of rotation such as 'x', 'y' or 'z'

    Returns
    -------
    np.ndarray
        Vector associated with the axis
    """

    if axis == "x" or axis == CartesianAxis.X:
        return np.array([1, 0, 0])
    elif axis == "y" or axis == CartesianAxis.Y:
        return np.array([0, 1, 0])
    elif axis == "z" or axis == CartesianAxis.Z:
        return np.array([0, 0, 1])
    else:
        raise ValueError("The axis must be 'x', 'y' or 'z'.")


def euler_angles_from_rotation_matrix(
    parent_matrix: np.ndarray, child_matrix: np.ndarray, joint_sequence: EulerSequence
) -> np.ndarray:
    """
    This function returns the euler angles from two rotation matrices

    Parameters
    ----------
    parent_matrix: np.ndarray
        The parent rotation matrix
    child_matrix: np.ndarray
        The child rotation matrix
    joint_sequence: EulerSequence
        The sequence of rotations

    Returns
    -------
    np.ndarray
        The euler angles
    """

    rot = parent_matrix.T @ child_matrix
    euler_angles = rotation_matrix_to_euler_angles(rot, joint_sequence.value)

    return euler_angles
