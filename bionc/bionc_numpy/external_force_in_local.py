import numpy as np

from .external_force_global_on_proximal import ExternalForceInGlobalOnProximal
from .natural_coordinates import SegmentNaturalCoordinates
from .natural_vector import NaturalVector


class ExternalForceInLocal:
    """
    This class represents an external force applied to a segment.

    Attributes
    ----------
    application_point_in_local : np.ndarray
        The application point of the force in the natural coordinate system of the segment
    external_forces : np.ndarray
        The external force vector in the global coordinate system (torque, force), in local frame too
    transformation_matrix : np.ndarray
         The transformation matrix of the segment

    Methods
    -------
    from_components(application_point_in_local, force, torque)
        This function creates an external force from its components.
    force
        This function returns the force vector of the external force.
    torque
        This function returns the torque vector of the external force.
    compute_pseudo_interpolation_matrix()
        This function computes the pseudo interpolation matrix of the external force.
    to_natural_force
        This function returns the external force in the natural coordinate format.
    """

    def __init__(
        self,
        application_point_in_local: np.ndarray,
        external_forces: np.ndarray,
        transformation_matrix: np.ndarray,
    ):
        self.application_point_in_local = application_point_in_local
        self.external_forces = external_forces
        self.transformation_matrix = transformation_matrix
        self.transformation_matrix_inv = np.linalg.inv(self.transformation_matrix.T)

    @classmethod
    def from_components(
        cls,
        application_point_in_local: np.ndarray,
        force: np.ndarray,
        torque: np.ndarray,
        transformation_matrix: np.ndarray,
    ):
        """
        This function creates an external force from its components.

        Parameters
        ----------
        application_point_in_local : np.ndarray
            The application point of the force in the natural coordinate system of the segment
        force
            The force vector in the global coordinate system
        torque
            The torque vector in the global coordinate system
        transformation_matrix : np.ndarray
            The transformation matrix of the segment

        Returns
        -------
        ExternalForce
        """

        return cls(application_point_in_local, np.concatenate((torque, force)), transformation_matrix)

    @property
    def force(self) -> np.ndarray:
        """The force vector in the global coordinate system"""
        return self.external_forces[3:6]

    @property
    def torque(self) -> np.ndarray:
        """The torque vector in the global coordinate system"""
        return self.external_forces[0:3]

    def forces_in_global(self, Qi: SegmentNaturalCoordinates):
        rotation_matrix = Qi.to_uvw_matrix() @ self.transformation_matrix_inv

        force_in_global = rotation_matrix @ self.force
        torque_in_global = rotation_matrix @ self.torque

        return np.concatenate((torque_in_global, force_in_global))

    def transport_on_proximal(
        self,
        Qi: SegmentNaturalCoordinates,
    ):
        """
        Transport the external force to another segment and another application point in cartesian coordinates

        Parameters
        ----------
        Qi: SegmentNaturalCoordinates
            The natural coordinates of the system

        Returns
        -------
        ExternalForceInGlobalOnProximal
            The external force on the proximal point of the segment
        """
        qi_array = np.array(Qi).squeeze()

        old_point_interpolation_matrix = NaturalVector(self.application_point_in_local).interpolate()
        new_point_interpolation_matrix = NaturalVector.proximal().interpolate()

        old_application_point_in_global = np.array(old_point_interpolation_matrix @ qi_array).squeeze()
        new_application_point_in_global = np.array(new_point_interpolation_matrix @ qi_array).squeeze()

        # Bour's formula to transport the moment from the application point to the new application point
        lever_arm = new_application_point_in_global - old_application_point_in_global

        new_external_forces = self.forces_in_global(Qi)
        additional_torque = np.cross(lever_arm, new_external_forces[3:6])

        # Sum the additional torque to the existing torque
        new_external_forces[0:3] += additional_torque

        return ExternalForceInGlobalOnProximal(external_forces=new_external_forces)

    def to_generalized_natural_forces(self, Qi: SegmentNaturalCoordinates) -> np.ndarray:
        """This function returns the external force in the generalized natural forces [12x1] format."""
        return self.transport_on_proximal(Qi).to_generalized_natural_forces(Qi)
