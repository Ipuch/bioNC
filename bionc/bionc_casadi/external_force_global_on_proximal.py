from casadi import MX, vertcat, cross

from .natural_coordinates import SegmentNaturalCoordinates
from .natural_vector import NaturalVector


class ExternalForceInGlobalOnProximal:
    """
    This class represents an external force applied to a segment. This is the standard representation
    of an external force this is why there is no application point as it is assumed to be at the proximal point (rp).

    Attributes
    ----------
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
    natural_forces(Qi)
        Format external forces to the segment
    natural_moments(Qi)
        Format external moments to the segment
    to_generalized_natural_force(Qi)
        Format external moments and forces to the generalized external force in the natural coordinate format.
    """

    def __init__(self, external_forces: MX):
        self.external_forces = MX(external_forces)

    @classmethod
    def from_components(cls, force: MX, torque: MX):
        """
        This function creates an external force from its components.

        Parameters
        ----------
        force
            The force vector in the global coordinate system
        torque
            The torque vector in the global coordinate system

        Returns
        -------
        ExternalForce
        """

        return cls(vertcat(torque, force))

    @property
    def force(self) -> MX:
        """The cartesian force vector in the global coordinate system"""
        return self.external_forces[3:6]

    @property
    def torque(self) -> MX:
        """The cartesian torque vector in the global coordinate system"""
        return self.external_forces[0:3]

    def natural_forces(self) -> MX:
        """
        Apply external forces to the segment

        Parameters
        ----------
        Qi: SegmentNaturalCoordinates
            Segment natural coordinates

        Returns
        -------
        MX
            The external forces adequately transformed for the equation of motion in natural coordinates [12 x 1]
        """
        point_interpolation_matrix = NaturalVector.proximal().interpolate()

        return point_interpolation_matrix.T @ self.force

    def natural_moments(self, Qi: SegmentNaturalCoordinates) -> MX:
        """
        Apply external moments to the segment

        Parameters
        ----------
        Qi: SegmentNaturalCoordinates
            Segment natural coordinates

        Returns
        -------
        MX
            The external forces adequately transformed for the equation of motion in natural coordinates [12 x 1]
        """
        pseudo_interpolation_matrix = Qi.compute_pseudo_interpolation_matrix()

        return pseudo_interpolation_matrix.T @ self.torque

    def to_generalized_natural_forces(self, Qi: SegmentNaturalCoordinates) -> MX:
        """
        Format external moments and forces to the generalized external force in the natural coordinate format.

        Parameters
        ----------
        Qi: SegmentNaturalCoordinates
            Segment natural coordinates

        Returns
        -------
        MX
            The external forces adequately transformed for the equation of motion in natural coordinates [12 x 1]
        """
        return self.natural_forces() + self.natural_moments(Qi)

    def transport_to_another_segment(self, Qfrom: SegmentNaturalCoordinates, Qto: SegmentNaturalCoordinates) -> MX:
        """
        Transport the external force to another segment and another application point in cartesian coordinates

        Parameters
        ----------
        Qfrom: SegmentNaturalCoordinates
            The natural coordinates of the segment from which the force is transported
        Qto: SegmentNaturalCoordinates
            The natural coordinates of the segment to which the force is transported

        Returns
        -------
        ExternalForceInGlobalOnProximal
            The external force on the proximal point of the segment
        """
        qi_array = Qfrom
        qj_array = Qto

        proximal_interpolation_matrix = NaturalVector.proximal().interpolate()

        old_application_point_in_global = proximal_interpolation_matrix @ qi_array
        new_application_point_in_global = proximal_interpolation_matrix @ qj_array

        lever_arm = new_application_point_in_global - old_application_point_in_global

        return ExternalForceInGlobalOnProximal.from_components(self.force, self.torque + cross(lever_arm, self.force))
