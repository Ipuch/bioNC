from casadi import MX, inv, cross, vertcat
import numpy as np

from .natural_vector import NaturalVector
from .natural_coordinates import SegmentNaturalCoordinates, NaturalCoordinates


class ExternalForce:
    """
    This class represents an external force applied to a segment.

    Attributes
    ----------
    application_point_in_local : np.ndarray
        The application point of the force in the natural coordinate system of the segment
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
    compute_pseudo_interpolation_matrix()
        This function computes the pseudo interpolation matrix of the external force.
    to_natural_force
        This function returns the external force in the natural coordinate format.
    """

    def __init__(self, application_point_in_local: np.ndarray | MX, external_forces: np.ndarray | MX):
        self.application_point_in_local = MX(application_point_in_local)
        self.external_forces = MX(external_forces)

    @classmethod
    def from_components(
        cls, application_point_in_local: np.ndarray | MX, force: np.ndarray | MX, torque: np.ndarray | MX
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

        Returns
        -------
        ExternalForce
        """

        return cls(application_point_in_local, vertcat(torque, force))

    @property
    def force(self) -> MX:
        """The force vector in the global coordinate system"""
        return self.external_forces[3:6]

    @property
    def torque(self) -> MX:
        """The torque vector in the global coordinate system"""
        return self.external_forces[0:3]

    def to_natural_force(self, Qi: SegmentNaturalCoordinates) -> MX:
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

        pseudo_interpolation_matrix = Qi.compute_pseudo_interpolation_matrix()
        point_interpolation_matrix = NaturalVector(self.application_point_in_local).interpolate()

        fext = point_interpolation_matrix.T @ self.force
        fext += pseudo_interpolation_matrix.T @ self.torque

        return fext

    def transport_to(
        self,
        to_segment_index: int,
        new_application_point_in_local: np.ndarray,
        Q: NaturalCoordinates,
        from_segment_index: int,
    ) -> MX:
        """
        Transport the external force to another segment and another application point

        Parameters
        ----------
        to_segment_index: int
            The index of the new segment
        new_application_point_in_local: np.ndarray
            The application point of the force in the natural coordinate system of the new segment
        Q: NaturalCoordinates
            The natural coordinates of the system
        from_segment_index: int
            The index of the current segment the force is applied on

        Returns
        -------
        np.ndarray
            The external forces adequately transformed for the equation of motion in natural coordinates [12 x 1]
        """

        Qi_old = Q.vector(from_segment_index)
        Qi_new = Q.vector(to_segment_index)

        old_point_interpolation_matrix = NaturalVector(self.application_point_in_local).interpolate()
        new_point_interpolation_matrix = NaturalVector(new_application_point_in_local).interpolate()

        old_application_point_in_global = old_point_interpolation_matrix @ Qi_old
        new_application_point_in_global = new_point_interpolation_matrix @ Qi_new

        new_pseudo_interpolation_matrix = Qi_new.compute_pseudo_interpolation_matrix()

        # Bour's formula to transport the moment from the application point to the new application point
        lever_arm = new_application_point_in_global - old_application_point_in_global
        additional_torque = new_pseudo_interpolation_matrix.T @ cross(lever_arm, self.force)
        fext = self.to_natural_force(Qi_new)
        fext += additional_torque

        return fext


class ExternalForceSet:
    """
    This class is made to handle all the external forces of each segment, if none are provided, it will be an empty list.
    All segment forces are expressed in natural coordinates to be added to the equation of motion as:

    Q @ Qddot + K^T @ lambda = Weight + f_ext

    Attributes
    ----------
    external_forces : list
        List of ExternalForces for each segment

    Methods
    -------
    add_external_force(segment_index, external_force)
        This function adds an external force to the list of external forces.
    empty_from_nb_segment(nb_segment)
        This function creates an empty ExternalForceSet from the number of segments.
    to_natural_external_forces(Q)
        This function returns the external forces in the natural coordinate format.
    segment_external_forces(segment_index)
        This function returns the external forces of a segment.
    nb_segments
        This function returns the number of segments.

    Examples
    --------
    >>> from bionc import ExternalForceSet, ExternalForce
    >>> import numpy as np
    >>> f_ext = ExternalForceSet.empty_from_nb_segment(2)
    >>> segment_force = ExternalForce(force=np.array([0,1,1.1]), torque=np.zeros(3), application_point_in_local=np.array([0,0.5,0]))
    >>> f_ext.add_external_force(segment_index=0, external_force=segment_force)
    """

    def __init__(self, external_forces: list[list[ExternalForce, ...]] = None):
        if external_forces is None:
            raise ValueError(
                "f_ext must be a list of ExternalForces, or use the classmethod"
                "NaturalExternalForceSet.empty_from_nb_segment(nb_segment)"
            )
        self.external_forces = external_forces

    @property
    def nb_segments(self) -> int:
        """Returns the number of segments"""
        return len(self.external_forces)

    @classmethod
    def empty_from_nb_segment(cls, nb_segment: int):
        """
        Create an empty NaturalExternalForceSet from the model size
        """
        return cls(external_forces=[[] for _ in range(nb_segment)])

    def segment_external_forces(self, segment_index: int) -> list[ExternalForce]:
        """Returns the external forces of the segment"""
        return self.external_forces[segment_index]

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
        self.external_forces[segment_index].append(external_force)

    def to_natural_external_forces(self, Q: NaturalCoordinates) -> MX:
        """
        Converts and sums all the segment natural external forces to the full vector of natural external forces

        Parameters
        ----------
        Q : NaturalCoordinates
            The natural coordinates of the model
        """

        if len(self.external_forces) != Q.nb_qi():
            raise ValueError(
                "The number of segment in the model and the number of segment in the external forces must be the same"
            )

        natural_external_forces = MX.zeros((12 * Q.nb_qi(), 1))
        for segment_index, segment_external_forces in enumerate(self.external_forces):
            segment_natural_external_forces = MX.zeros((12, 1))
            slice_index = slice(segment_index * 12, (segment_index + 1) * 12)
            for external_force in segment_external_forces:
                segment_natural_external_forces += external_force.to_natural_force(Q.vector(segment_index))
            natural_external_forces[slice_index, 0:1] = segment_natural_external_forces

        return natural_external_forces

    def to_segment_natural_external_forces(self, Q: NaturalCoordinates, segment_index: int) -> MX:
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
        segment_natural_external_forces: MX
        """

        if len(self.external_forces) != Q.nb_qi():
            raise ValueError(
                "The number of segment in the model and the number of segment in the external forces must be the same"
            )

        if segment_index >= len(self.external_forces):
            raise ValueError("The segment index is out of range")

        segment_natural_external_forces = np.zeros((12, 1))
        for external_force in self.external_forces[segment_index]:
            segment_natural_external_forces += external_force.to_natural_force(Q.vector(segment_index))

        return segment_natural_external_forces

    def __iter__(self):
        return iter(self.external_forces)

    def __len__(self):
        return len(self.external_forces)
