import numpy as np

from .natural_coordinates import NaturalCoordinates


def total_constraints(Q: np.ndarray, constraint_function: callable):
    """Compute the total constraints for a given model and natural coordinates."""
    total_residual = np.zeros(Q.shape[1])
    for i, Q_frame in enumerate(Q.T):
        total_residual[i] = np.sqrt(np.sum(constraint_function(Q=NaturalCoordinates(Q_frame)) ** 2))

    return total_residual


def constraints(Q: np.ndarray, vertical_size: int, constraint_function: callable):
    """Compute the constraints for a given model and natural coordinates."""
    total_residual = np.zeros((vertical_size, Q.shape[1]))
    for i, Q_frame in enumerate(Q.T):
        total_residual[:, i] = constraint_function(Q=NaturalCoordinates(Q_frame))

    return total_residual


def total_rigid_body_constraints(model, Q: np.ndarray):
    """Compute the total rigid body constraints for a given model and natural coordinates."""
    return total_constraints(Q, constraint_function=model.rigid_body_constraints)


def rigid_body_constraints(model, Q: np.ndarray):
    """Compute the rigid body constraints for a given model and natural coordinates."""
    return constraints(Q, model.nb_rigid_body_constraints, constraint_function=model.rigid_body_constraints)


def total_joint_constraints(model, Q: np.ndarray):
    """Compute the total joint constraints for a given model and natural coordinates."""
    return total_constraints(Q, constraint_function=model.joint_constraints)


def joint_constraints(model, Q: np.ndarray):
    """Compute the joint constraints for a given model and natural coordinates."""
    return constraints(Q, model.nb_joint_constraints, constraint_function=model.joint_constraints)


def total_marker_constraints(model, Q: np.ndarray, markers: np.ndarray):
    """Compute the total marker constraints for a given model and natural coordinates."""
    total_marker_residuals = np.zeros(Q.shape[1])
    for i, Q_frame in enumerate(Q.T):
        total_marker_residuals[i] = np.sqrt(
            np.sum(model.markers_constraints(markers[:, :, i], NaturalCoordinates(Q_frame), only_technical=True) ** 2)
        )
    return total_marker_residuals


def marker_constraints_xyz(model, Q: np.ndarray, markers: np.ndarray):
    """Compute the marker constraints for a given model and natural coordinates."""
    marker_residuals_xyz = np.zeros((3, markers.shape[1], Q.shape[1]))
    for i, Q_frame in enumerate(Q.T):
        marker_residuals_xyz[:, :, i] = model.markers_constraints_xyz(
            markers[:, :, i], NaturalCoordinates(Q_frame), only_technical=True
        )
    return marker_residuals_xyz

def total_euler_angles(model, Q: np.ndarray):

    Qi_temp = NaturalCoordinates(Q[:, 0])
    euler_temp = model.natural_coordinates_to_joint_angles(Qi_temp)
    total_euler_angles = np.zeros((euler_temp.shape[0],euler_temp.shape[1],Q.shape[1]))

    for ind_frame in range(Q.shape[1]):
        Qi = NaturalCoordinates(Q[:,ind_frame])
        total_euler_angles[:,:,ind_frame] = model.natural_coordinates_to_joint_angles(Qi)

    return total_euler_angles

class TimeSeriesUtils:
    """
    This class contains utility functions to compute the constraints of a biomechanical model for
    a given time series of natural coordinates and markers.
    - Q [12 x n, nb_frame] : The natural coordinates of the model.
    - markers [3, nb_markers x nb_frame] : The position of the markers in the global coordinate system.
    """

    total_rigid_body_constraints = total_rigid_body_constraints
    rigid_body_constraints = rigid_body_constraints
    total_joint_constraints = total_joint_constraints
    joint_constraints = joint_constraints
    total_marker_constraints = total_marker_constraints
    marker_constraints_xyz = marker_constraints_xyz
    total_euler_angles = total_euler_angles
