import numpy as np

from .external_force import ExternalForce
from .natural_coordinates import SegmentNaturalCoordinates, NaturalCoordinates
from .rotations import euler_axes_from_rotation_matrices
from ..protocols.joint import JointBase as Joint
from ..utils.enums import CartesianAxis, EulerSequence


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
        check_translation_dof_format(translation_dof, forces)
        check_rotation_dof_format(rotation_dof, torques)

        parent_segment = joint.parent
        child_segment = joint.child

        # compute rotation matrix from Qi
        R_parent = (
            np.eye(3)
            if parent_segment is None
            else parent_segment.segment_coordinates_system(Q_parent, joint.parent_basis).rot
        )
        R_child = child_segment.segment_coordinates_system(Q_child, joint.child_basis).rot

        force_in_global = fill_forces_and_project(forces, rotation_dof, R_parent)
        filled_torques = fill_torques(torques, rotation_dof)

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

        if parent_segment_index is not None:
            f_parent = self.transport_to(
                to_segment_index=parent_segment_index,
                new_application_point_in_local=[0, 0, 0],
                Q=Q,
                from_segment_index=child_segment_index,
            )
            return -f_parent, f_child
        else:
            return None, f_child


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
        if ei.shape != (3, 1):
            ei = ei[:, np.newaxis]
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
        This function creates an empty ExternalForceSet from the number of segments.
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
                "NaturalExternalForceSet.empty_from_nb_joint(nb_joint)"
            )
        self.joint_generalized_forces = joint_generalized_forces

    @property
    def nb_joint(self) -> int:
        """Returns the number of segments"""
        return len(self.joint_generalized_forces)

    @classmethod
    def empty_from_nb_joint(cls, nb_joint: int):
        """
        Create an empty NaturalExternalForceSet from the model size
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

    def add_all_joint_generalized_forces(
        self, model: "BiomechanicalModel", joint_generalized_forces: np.ndarray, Q: NaturalCoordinates
    ):
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
            # splitting forces and torques because they are stacked in the same array
            force_ending_index = 0 if joint.translation_coordinates is None else len(joint.translation_coordinates)
            forces = (
                None if joint.translation_coordinates is None else joint_generalized_force_array[:force_ending_index]
            )
            torques = None if joint.projection_basis is None else joint_generalized_force_array[force_ending_index:]
            # finally computing the joint generalized force to be formatted in natural coordinates
            joint_generalized_force = JointGeneralizedForces.from_joint_generalized_forces(
                forces=forces,
                torques=torques,
                translation_dof=joint.translation_coordinates,
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

        if len(self.joint_generalized_forces) != Q.nb_qi() or len(self.joint_generalized_forces) != model.nb_segments:
            raise ValueError(
                "The number of joint in the model and the number of segment in the joint forces must be the same."
                f"Got {len(self.joint_generalized_forces)} joint forces and {Q.nb_qi()} segments "
                f"in NaturalCoordinate vector Q."
                f"Got {model.nb_segments} joints in the model."
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
                )
                child_natural_joint_force += b
            if joint.parent is not None:
                parent_natural_joint_force += a
                natural_joint_forces[parent_slice_index, 0:1] = parent_natural_joint_force
            natural_joint_forces[child_slice_index, 0:1] = child_natural_joint_force

        return natural_joint_forces


def check_translation_dof_format(translation_dof, forces):
    if translation_dof is None and forces is not None:
        raise ValueError("The translation degrees of freedom must be specified")

    if translation_dof is not None and forces is not None:

        shape_is_not_consistent = forces.shape[0] != len(translation_dof)
        if shape_is_not_consistent:
            raise ValueError("The number of forces must be equal to the number of translation degrees of freedom")

        dof_are_not_unique = len(translation_dof) != len(set(translation_dof))
        if dof_are_not_unique:
            raise ValueError(f"The translation degrees of freedom must be unique. Got {translation_dof}")


def check_rotation_dof_format(rotation_dof, torques):
    if rotation_dof is None and torques is not None:
        raise ValueError("The rotation degrees of freedom must be specified")
    if rotation_dof is not None and torques is None:
        raise ValueError("The torques must be specified")
    if rotation_dof is not None and torques is not None:

        shape_is_not_consistent = torques.shape[0] != len(rotation_dof.value)
        if shape_is_not_consistent:
            raise ValueError("The number of torques must be equal to the number of rotation degrees of freedom")


def fill_forces_and_project(forces: np.ndarray, rotation_dof: EulerSequence, R_parent: np.ndarray) -> np.ndarray:
    """Fill the forces and project them in the global frame"""
    filled_forces = np.zeros((3, 1))

    rot_dof_map = {
        CartesianAxis.X: 0,
        CartesianAxis.Y: 1,
        CartesianAxis.Z: 2,
    }

    for rot_dof in rotation_dof.value:
        index = rot_dof_map[rot_dof]
        filled_forces[index, 0] = forces[index]

    # force automatically in proximal segment coordinates system
    force_in_global = project_vector_into_frame(filled_forces, (R_parent[:, 0:1], R_parent[:, 1:2], R_parent[:, 2:3]))

    return force_in_global


def fill_torques(torques: np.ndarray, rotation_dof: EulerSequence) -> np.ndarray:
    filled_torques = np.zeros((3, 1))
    for rot_dof in rotation_dof.value:
        if rot_dof == "x":
            filled_torques[0, 0] = torques[0]
        elif rot_dof == "y":
            filled_torques[1, 0] = torques[1]
        elif rot_dof == "z":
            filled_torques[2, 0] = torques[2]

    return filled_torques
