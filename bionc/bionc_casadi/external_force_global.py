from casadi import MX, vertcat, cross

from .external_force_global_on_proximal import ExternalForceInGlobalOnProximal
from .natural_coordinates import SegmentNaturalCoordinates


class ExternalForceInGlobal:
    """
    This class represents an external force applied to a segment.

    Attributes
    ----------
    application_point_in_local : MX
        The application point of the force in the natural coordinate system of the segment
    external_forces : MX
        The external force vector in the global coordinate system (torque, force)

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

    def __init__(self, application_point_in_global: MX, external_forces: MX):
        self.application_point_in_global = MX(application_point_in_global)
        self.external_forces = MX(external_forces)

    @classmethod
    def from_components(cls, application_point_in_global: MX, force: MX, torque: MX):
        """
        This function creates an external force from its components.

        Parameters
        ----------
        application_point_in_global : MX
            The application point of the force in the natural coordinate system of the segment
        force
            The force vector in the global coordinate system
        torque
            The torque vector in the global coordinate system

        Returns
        -------
        ExternalForce
        """

        return cls(application_point_in_global, vertcat(torque, force))

    @property
    def force(self) -> MX:
        """The force vector in the global coordinate system"""
        return self.external_forces[3:6]

    @property
    def torque(self) -> MX:
        """The torque vector in the global coordinate system"""
        return self.external_forces[0:3]

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

        old_application_point_in_global = self.application_point_in_global
        new_application_point_in_global = Qi.rp

        # Bour's formula to transport the moment from the application point to the new application point
        lever_arm = new_application_point_in_global - old_application_point_in_global
        additional_torque = cross(lever_arm, self.force)

        # Sum the additional torque to the existing torque
        new_external_forces = self.external_forces.copy()
        new_external_forces[0:3] += additional_torque

        return ExternalForceInGlobalOnProximal(external_forces=new_external_forces)

    def to_generalized_natural_forces(self, Qi: SegmentNaturalCoordinates):
        """This function returns the external force in the generalized natural forces [12x1] format."""
        return self.transport_on_proximal(Qi).to_generalized_natural_forces(Qi)
