import numpy as np

from .natural_vector import NaturalVector
from .natural_coordinates import SegmentNaturalCoordinates, NaturalCoordinates
from ..utils.enums import CartesianAxis, EulerSequence
from .rotations import euler_axes_from_rotation_matrices
from ..protocols.joint import JointBase as Joint


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

    def __init__(self, application_point_in_local: np.ndarray, external_forces: np.ndarray):
        self.application_point_in_local = application_point_in_local
        self.external_forces = external_forces

    @classmethod
    def from_components(cls, application_point_in_local: np.ndarray, force: np.ndarray, torque: np.ndarray):
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

        return cls(application_point_in_local, np.concatenate((torque, force)))

    @property
    def force(self) -> np.ndarray:
        """The force vector in the global coordinate system"""
        return self.external_forces[3:6]

    @property
    def torque(self) -> np.ndarray:
        """The torque vector in the global coordinate system"""
        return self.external_forces[0:3]

    def to_natural_force(self, Qi: SegmentNaturalCoordinates) -> np.ndarray:
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

        pseudo_interpolation_matrix = Qi.compute_pseudo_interpolation_matrix()
        point_interpolation_matrix = NaturalVector(self.application_point_in_local).interpolate()
        application_point_in_global = np.array(point_interpolation_matrix @ Qi).squeeze()

        fext = point_interpolation_matrix.T @ self.force
        fext += pseudo_interpolation_matrix.T @ self.torque

        return np.array(fext)

    def transport_to(
        self,
        to_segment_index: int,
        new_application_point_in_local: np.ndarray,
        Q: NaturalCoordinates,
        from_segment_index: int,
    ):
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

        old_application_point_in_global = np.array(old_point_interpolation_matrix @ Qi_old).squeeze()
        new_application_point_in_global = np.array(new_point_interpolation_matrix @ Qi_new).squeeze()

        new_pseudo_interpolation_matrix = Qi_new.compute_pseudo_interpolation_matrix()

        # Bour's formula to transport the moment from the application point to the new application point
        lever_arm = new_application_point_in_global - old_application_point_in_global
        additional_torque = new_pseudo_interpolation_matrix.T @ np.cross(lever_arm, self.force)
        fext = self.to_natural_force(Qi_new)
        fext += additional_torque

        return fext


class ExternalForceList:
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
        This function creates an empty ExternalForceList from the number of segments.
    to_natural_external_forces(Q)
        This function returns the external forces in the natural coordinate format.
    segment_external_forces(segment_index)
        This function returns the external forces of a segment.
    nb_segments
        This function returns the number of segments.

    Examples
    --------
    >>> from bionc import ExternalForceList, ExternalForce
    >>> import numpy as np
    >>> f_ext = ExternalForceList.empty_from_nb_segment(2)
    >>> segment_force = ExternalForce(force=np.array([0,1,1.1]), torque=np.zeros(3), application_point_in_local=np.array([0,0.5,0]))
    >>> f_ext.add_external_force(segment_index=0, external_force=segment_force)
    """

    def __init__(self, external_forces: list[list[ExternalForce, ...]] = None):
        if external_forces is None:
            raise ValueError(
                "f_ext must be a list of ExternalForces, or use the classmethod"
                "NaturalExternalForceList.empty_from_nb_segment(nb_segment)"
            )
        self.external_forces = external_forces

    @property
    def nb_segments(self) -> int:
        """Returns the number of segments"""
        return len(self.external_forces)

    @classmethod
    def empty_from_nb_segment(cls, nb_segment: int):
        """
        Create an empty NaturalExternalForceList from the model size
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

    def to_natural_external_forces(self, Q: NaturalCoordinates) -> np.ndarray:
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

        natural_external_forces = np.zeros((12 * Q.nb_qi(), 1))
        for segment_index, segment_external_forces in enumerate(self.external_forces):
            segment_natural_external_forces = np.zeros((12, 1))
            slice_index = slice(segment_index * 12, (segment_index + 1) * 12)
            for external_force in segment_external_forces:
                segment_natural_external_forces += external_force.to_natural_force(Q.vector(segment_index))[
                    :, np.newaxis
                ]
            natural_external_forces[slice_index, 0:1] = segment_natural_external_forces

        return natural_external_forces

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
        segment_natural_external_forces: np.ndarray
        """

        if len(self.external_forces) != Q.nb_qi():
            raise ValueError(
                "The number of segment in the model and the number of segment in the external forces must be the same"
            )

        if segment_index >= len(self.external_forces):
            raise ValueError("The segment index is out of range")

        segment_natural_external_forces = np.zeros((12, 1))
        for external_force in self.external_forces[segment_index]:
            segment_natural_external_forces += external_force.to_natural_force(Q.vector(segment_index))[:, np.newaxis]

        return segment_natural_external_forces

    def __iter__(self):
        return iter(self.external_forces)

    def __len__(self):
        return len(self.external_forces)


class JointGeneralizedForces(ExternalForce):
    """
    Made to handle joint generalized forces, it inherits from ExternalForce

    Attributes
    ----------
    external_forces : np.ndarray
        The external forces
    application_point_in_local : np.ndarray
        The application point in local coordinates

    Methods
    -------
    from_joint_generalized_forces(forces, torques, translation_dof, rotation_dof, joint, Q_parent, Q_child)
        This function creates a JointGeneralizedForces from the forces and torques

    Notes
    -----
    The application point of torques is set to the proximal point of the child.
    """

    def __init__(
        self,
        external_forces: np.ndarray,
        application_point_in_local: np.ndarray,
    ):
        super().__init__(external_forces=external_forces, application_point_in_local=application_point_in_local)

    @classmethod
    def from_joint_generalized_forces(
        cls,
        forces: np.ndarray,
        torques: np.ndarray,
        translation_dof: tuple[CartesianAxis, ...] = None,
        rotation_dof: EulerSequence = None,
        joint: Joint = None,
        Q_parent: SegmentNaturalCoordinates = None,
        Q_child: SegmentNaturalCoordinates = None,
    ):
        """
        Create a JointGeneralizedForces from the forces and torques

        Parameters
        ----------
        forces: np.ndarray
            The forces
        torques: np.ndarray
            The torques
        translation_dof: tuple[CartesianAxis, ...]
            The translation degrees of freedom
        rotation_dof: EulerSequence
            The rotation degrees of freedom
        joint: Joint
            The joint
        Q_parent: SegmentNaturalCoordinates
            The natural coordinates of the proximal segment
        Q_child: SegmentNaturalCoordinates
            The natural coordinates of the distal segment
        """

        if forces.shape[0] != len(translation_dof):
            raise ValueError("The number of forces must be equal to the number of translation degrees of freedom")
        if len(translation_dof) != len(set(translation_dof)):
            raise ValueError(f"The translation degrees of freedom must be unique. Got {translation_dof}")

        if torques.shape[0] != len(rotation_dof.value):
            raise ValueError("The number of torques must be equal to the number of rotation degrees of freedom")

        parent_segment = joint.parent
        child_segment = joint.child

        # compute rotation matrix from Qi
        R_parent = (
            np.eye(3)
            if parent_segment is None
            else parent_segment.segment_coordinates_system(Q_parent, joint.parent_basis).rot
        )
        R_child = child_segment.segment_coordinates_system(Q_child, joint.child_basis).rot

        filled_forces = np.zeros((3, 1))
        for rot_dof in rotation_dof.value:
            if rot_dof == CartesianAxis.X:
                filled_forces[0, 0] = forces[0]
            elif rot_dof == CartesianAxis.Y:
                filled_forces[1, 0] = forces[1]
            elif rot_dof == CartesianAxis.Z:
                filled_forces[2, 0] = forces[2]

        # force automatically in proximal segment coordinates system
        force_in_global = project_vector_into_frame(
            filled_forces, (R_parent[:, 0:1], R_parent[:, 1:2], R_parent[:, 2:3])
        )

        filled_torques = np.zeros((3, 1))
        for rot_dof in rotation_dof.value:
            if rot_dof == "x":
                filled_torques[0, 0] = torques[0]
            elif rot_dof == "y":
                filled_torques[1, 0] = torques[1]
            elif rot_dof == "z":
                filled_torques[2, 0] = torques[2]

        euler_axes_in_global = euler_axes_from_rotation_matrices(R_parent, R_child, sequence=joint.projection_basis)

        tau_in_global = project_vector_into_frame(filled_torques, euler_axes_in_global)

        return cls(
            external_forces=np.vstack((tau_in_global, force_in_global)),
            application_point_in_local=np.zeros(3),
        )

    def to_natural_joint_forces(
        self, parent_segment_index: int, child_segment_index: int, Q: NaturalCoordinates
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Converts the generalized forces to natural forces

        Parameters
        ----------
        parent_segment_index: int
            The index of the parent segment
        child_segment_index: int
            The index of the child segment
        Q: NaturalCoordinates
            The natural coordinates of the model

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            The generalized forces in the natural coordinates of the parent and child segments [12x1], [12x1]
        """
        f_child = self.to_natural_force(Q.vector(child_segment_index))
        f_parent = self.transport_to(
            to_segment_index=parent_segment_index,
            new_application_point_in_local=[0, 0, 0],
            Q=Q,
            from_segment_index=child_segment_index,
        )

        return f_child, -f_parent


def project_vector_into_frame(
    vector_in_initial_basis: np.ndarray, basis_axes_in_new_frame: tuple[np.ndarray, np.ndarray, np.ndarray]
) -> np.ndarray:
    """
    Project a vector into a frame

    Parameters
    ----------
    vector_in_initial_basis: np.ndarray
        The vector to project [3x1]
    basis_axes_in_new_frame: tuple[np.ndarray, np.ndarray, np.ndarray]
        The basis axes of the new frame [3x1], [3x1], [3x1]

    Returns
    -------
    np.ndarray
        The projected vector in the new frame [3x1]
    """
    vector_in_new_frame = np.zeros((3, 1))
    for v, ei in zip(vector_in_initial_basis, basis_axes_in_new_frame):
        vector_in_new_frame += v * ei
    return vector_in_new_frame


class JointGeneralizedForcesList:
    """
    This class is made to handle all the external forces of each joint, if none are provided, it will be an empty list.
    All segment forces are expressed in natural coordinates to be added to the equation of motion as:

    Q @ Qddot + K^T @ lambda = gravity_forces + f_ext + joint_forces

    joint_forces being [nb_segments x 12, 1]

    Attributes
    ----------
    joint_generalized_forces : list
        List of JointGeneralizedForces for each joint.

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

    def __init__(self, joint_generalized_forces: list[list[JointGeneralizedForces, ...]] = None):
        if joint_generalized_forces is None:
            raise ValueError(
                "joint_generalized_forces must be a list of JointGeneralizedForces, or use the classmethod"
                "NaturalExternalForceList.empty_from_nb_joint(nb_joint)"
            )
        self.joint_generalized_forces = joint_generalized_forces

    @property
    def nb_joint(self) -> int:
        """Returns the number of segments"""
        return len(self.joint_generalized_forces)

    @classmethod
    def empty_from_nb_joint(cls, nb_joint: int):
        """
        Create an empty NaturalExternalForceList from the model size
        """
        return cls(joint_generalized_forces=[[] for _ in range(nb_joint)])

    def joint_generalized_force(self, joint_index: int) -> list[ExternalForce]:
        """Returns the external forces of the segment"""
        return self.joint_generalized_forces[joint_index]

    def add_generalized_force(self, joint_index: int, joint_generalized_force: JointGeneralizedForces):
        """
        Add an external force to the segment

        Parameters
        ----------
        joint_index: int
            The index of the segment
        joint_generalized_force:
            The joint_generalized_force to add
        """
        self.joint_generalized_forces[joint_index].append(joint_generalized_force)

    def add_all_joint_generalized_forces(self, model:"BiomechanicalModel", joint_generalized_forces: np.ndarray, Q: NaturalCoordinates):
        """
        Add all the generalized forces to the object. It separates the generalized forces into sub generalized forces
        corresponding to each individual joint.

        Parameters
        ----------
        model: BiomechanicalModel
            The model of the system
        joint_generalized_forces: np.ndarray
            The generalized forces to add [nb_joint_dof, 1]
        Q: NaturalCoordinates
            The natural coordinates of the model
        """

        for joint_index, joint in enumerate(model.joints.values()):
            parent_index = None if joint.parent is None else joint.parent.index
            child_index = joint.child.index
            # assuming they are sorted e.g., [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] and not [1,3,5,2]
            joint_slice = slice(model.joint_dof_indexes(joint_index)[0], model.joint_dof_indexes(joint_index)[-1] + 1)
            joint_generalized_force_array = joint_generalized_forces[joint_slice]
            joint_generalized_force = JointGeneralizedForces.from_joint_generalized_forces(
                forces=joint_generalized_force_array,
                torques=joint_generalized_force_array,
                translation_dof=joint.projection_direction,
                rotation_dof=joint.projection_basis,
                joint=joint,
                Q_parent=None if joint.parent is None else Q.vector(parent_index),
                Q_child=Q.vector(child_index),
            )
            self.add_generalized_force(joint_index, joint_generalized_force)

    def to_natural_joint_forces(self, model: "BiomechanicalModel", Q: NaturalCoordinates) -> np.ndarray:
        """
        Converts and sums all the segment natural external forces to the full vector of natural external forces

        Parameters
        ----------
        model: BiomechanicalModel
            The biomechanical model
        Q : NaturalCoordinates
            The natural coordinates of the model
        """

        if len(self.joint_generalized_forces) != Q.nb_qi() or len(self.joint_generalized_forces) != model.nb_joints:
            raise ValueError(
                "The number of joint in the model and the number of segment in the joint forces must be the same."
                f"Got {len(self.joint_generalized_forces)} joint forces and {Q.nb_qi()} segments "
                f"in NaturalCoordinate vector Q."
                f"Got {model.nb_joints} joints in the model."
            )

        natural_joint_forces = np.zeros((12 * Q.nb_qi(), 1))
        for joint_index, joint_generalized_force in enumerate(self.joint_generalized_forces):

            joint = model.joint_from_index(joint_index)

            parent_natural_joint_force = np.zeros((12, 1))
            child_natural_joint_force = np.zeros((12, 1))

            parent_index = None if joint.parent is None else joint.parent.index
            child_index = joint.child.index
            parent_slice_index = None if joint.parent is None else slice(parent_index * 12, (parent_index + 1) * 12)
            child_slice_index = slice(child_index * 12, (child_index + 1) * 12)

            for force in joint_generalized_force:
                a, b = force.to_natural_joint_forces(
                    parent_segment_index=parent_index,
                    child_segment_index=child_index,
                    Q=Q,
                    )[:, np.newaxis]
                parent_natural_joint_force += a
                child_natural_joint_force += b
            if joint.parent is not None:
                natural_joint_forces[parent_slice_index, 0:1] = parent_natural_joint_force
            natural_joint_forces[child_slice_index, 0:1] = child_natural_joint_force

        return natural_joint_forces
