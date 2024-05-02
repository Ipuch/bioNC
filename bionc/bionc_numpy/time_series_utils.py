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
