import numpy as np

from abc import ABC, abstractmethod
from .natural_coordinates import SegmentNaturalCoordinates, NaturalCoordinates


class ExternalForce(ABC):
    """
    This class represents an external force applied to a segment.

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

    @classmethod
    def from_components(cls, application_point_in_local, force, torque):
        """
        This function creates an external force from its components.

        Parameters
        ----------
        application_point_in_local :
            The application point of the force in the natural coordinate system of the segment
        force
            The force vector in the global coordinate system
        torque
            The torque vector in the global coordinate system

        Returns
        -------
        ExternalForce
        """
        pass

    @abstractmethod
    def force(self):
        """The force vector in the global coordinate system"""

    @abstractmethod
    def torque(self):
        """The torque vector in the global coordinate system"""

    @abstractmethod
    def to_natural_force(self, Qi: SegmentNaturalCoordinates):
        """
        Apply external forces to the segment

        Parameters
        ----------
        Qi: SegmentNaturalCoordinates
            Segment natural coordinates

        Returns
        -------
            The external forces adequately transformed for the equation of motion in natural coordinates [12 x 1]
        """

    @abstractmethod
    def transport_to(
        self,
        to_segment_index: int,
        new_application_point_in_local,
        Q: NaturalCoordinates,
        from_segment_index: int,
    ):
        """
        Transport the external force to another segment and another application point

        Parameters
        ----------
        to_segment_index: int
            The index of the new segment
        new_application_point_in_local
            The application point of the force in the natural coordinate system of the new segment
        Q: NaturalCoordinates
            The natural coordinates of the system
        from_segment_index: int
            The index of the current segment the force is applied on

        Returns
        -------
            The external forces adequately transformed for the equation of motion in natural coordinates [12 x 1]
        """


class ExternalForceList(ABC):
    """
    This class is made to handle all the external forces of each segment, if none are provided, it will be an empty list.
    All segment forces are expressed in natural coordinates to be added to the equation of motion as:

    Q @ Qddot + K^T @ lambda = Weight + f_ext

    Methods
    -------
    add_external_force(segment_index, external_force)
        This function adds an external force to the list of external forces.
    empty_from_nb_segment(nb_segment)
        This function creates an empty ExternalForceList from the number of segments.
    to_natural_external_forces(Q)
        This function returns the external forces in the natural coordinate format.
    segment_external_forces(segment_index)
        This function returns the external forces of a segment.
    nb_segments
        This function returns the number of segments.
    """

    @abstractmethod
    def nb_segments(self) -> int:
        """Returns the number of segments"""

    @abstractmethod
    def empty_from_nb_segment(cls, nb_segment: int):
        """
        Create an empty NaturalExternalForceList from the model size
        """

    @abstractmethod
    def segment_external_forces(self, segment_index: int) -> list[ExternalForce]:
        """Returns the external forces of the segment"""

    def add_external_force(self, segment_index: int, external_force: ExternalForce):
        """
        Add an external force to the segment

        Parameters
        ----------
        segment_index: int
            The index of the segment
        external_force:
            The external force to add
        """

    @abstractmethod
    def to_natural_external_forces(self, Q: NaturalCoordinates):
        """
        Converts and sums all the segment natural external forces to the full vector of natural external forces

        Parameters
        ----------
        Q : NaturalCoordinates
            The natural coordinates of the model
        """

    @abstractmethod
    def to_segment_natural_external_forces(self, Q: NaturalCoordinates, segment_index: int) -> np.ndarray:
        """
        Converts and sums all the segment natural external forces to the full vector of natural external forces
        for one segment

        Parameters
        ----------
        Q : NaturalCoordinates
            The natural coordinates of the model
        segment_index: int
            The index of the segment

        Returns
        -------
        segment_natural_external_forces:

        """