import numpy as np

from .natural_coordinates import SegmentNaturalCoordinates
from .natural_vector import NaturalVector


class ExternalForceInGlobalOnProximal:
    """
    This class represents an external force applied to a segment. This is the standard representation
    of an external force this is why there is no application point as it is assumed to be at the proximal point (rp).

    Attributes
    ----------
    external_forces : np.ndarray
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

    def __init__(self, external_forces: np.ndarray):
        self.external_forces = external_forces

    @property
    def force(self) -> np.ndarray:
        """The cartesian force vector in the global coordinate system"""
        return self.external_forces[3:6]

    @property
    def torque(self) -> np.ndarray:
        """The cartesian torque vector in the global coordinate system"""
        return self.external_forces[0:3]

    def natural_forces(self) -> np.ndarray:
        """
        Apply external forces to the segment

        Parameters
        ----------
        Qi: SegmentNaturalCoordinates
            Segment natural coordinates

        Returns
        -------
        np.ndarray
            The external forces adequately transformed for the equation of motion in natural coordinates [12 x 1]
        """
        point_interpolation_matrix = NaturalVector.proximal().interpolate()

        return np.array(point_interpolation_matrix.T @ self.force)

    def natural_moments(self, Qi: SegmentNaturalCoordinates) -> np.ndarray:
        """
        Apply external moments to the segment

        Parameters
        ----------
        Qi: SegmentNaturalCoordinates
            Segment natural coordinates

        Returns
        -------
        np.ndarray
            The external forces adequately transformed for the equation of motion in natural coordinates [12 x 1]
        """
        pseudo_interpolation_matrix = Qi.compute_pseudo_interpolation_matrix()

        return pseudo_interpolation_matrix.T @ self.torque

    def to_generalized_natural_forces(self, Qi: SegmentNaturalCoordinates) -> np.ndarray:
        """
        Format external moments and forces to the generalized external force in the natural coordinate format.

        Parameters
        ----------
        Qi: SegmentNaturalCoordinates
            Segment natural coordinates

        Returns
        -------
        np.ndarray
            The external forces adequately transformed for the equation of motion in natural coordinates [12 x 1]
        """
        return self.natural_forces() + self.natural_moments(Qi)
