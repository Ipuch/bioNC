import dill as pickle
import numpy as np
from abc import ABC, abstractmethod
from casadi import MX
from typing import Union, Any

from .biomechanical_model_joints import GenericBiomechanicalModelJoints
from .biomechanical_model_markers import GenericBiomechanicalModelMarkers
from .biomechanical_model_segments import GenericBiomechanicalModelSegments
from .biomechanical_model_tree import GenericBiomechanicalModelTree
from ..external_force import ExternalForceSet
from ..natural_accelerations import NaturalAccelerations
from ..natural_coordinates import NaturalCoordinates, SegmentNaturalCoordinates
from ..natural_velocities import NaturalVelocities
from ...utils.enums import EulerSequence


class GenericBiomechanicalModel(ABC):
    """
    This is an abstract base class that provides the basic structure and methods for all biomechanical models.
    It contains the segments and the joints of the model. The implemented methods are not specific to numpy or casadi.

    Attributes
    ----------
    segments : dict
        A dictionary containing the segments of the model. The keys are the names of the segments and the values are the corresponding segment objects.
    joints : dict
        A dictionary containing the joints of the model. The keys are the names of the joints and the values are the corresponding joint objects.
    _mass_matrix : np.ndarray
        The generalized mass matrix of the system.
    _markers : GenericBiomechanicalModelMarkers
        The markers of the model handled in a specific class.

    Methods
    -------
    __getitem__(self, name: str)
        Returns the segment with the given name.
    __setitem__(self, name: str, segment: Any)
        Adds a segment to the model.
    save(self, filename: str)
        Saves the model to a file.
    load(self, filename: str)
        Loads the model from a file.
    set_ground_segment(self, name: str)
        Sets the ground segment of the model.
    has_ground_segment(self) -> bool
        Returns true if the model has a ground segment.
    segments_no_ground(self)
        Returns the dictionary of all the segments except the ground segment.
    _add_joint(self, joint: dict)
        Adds a joint to the biomechanical model. It is not recommended to use this function directly.
    joints_with_constraints(self) -> dict
        Returns the dictionary of all the joints with constraints.
    has_free_joint(self, segment_idx: int) -> bool
        Returns true if the segment has a free joint with the ground.
    _remove_free_joint(self, segment_idx: int)
        Removes the free joint of the segment.
    children(self, segment: str | int) -> list[int]
        Returns the children of the given segment.
    parents(self, segment: str | int) -> list[int]
        Returns the parents of the given segment.
    segment_subtrees(self) -> list[list[int]]
        Returns the subtrees of the segments.
    segment_subtree(self, segment: str | int) -> list[int]
        Returns the subtree of the given segment.
    nb_segments(self) -> int
        Returns the number of segments in the model.
    nb_markers(self) -> int
        Returns the number of markers in the model.
    nb_markers_technical(self) -> int
        Returns the number of technical markers in the model.
    segment_names(self) -> list[str]
        Returns the names of the segments in the model.
    marker_names(self) -> list[str]
        Returns the names of the markers in the model.
    marker_names_technical(self) -> list[str]
        Returns the names of the technical markers in the model.
    dof_names(self) -> list[str]
        Returns the names of the degrees of freedom in the model.
    nb_joints(self) -> int
        Returns the number of joints in the model.
    nb_joints_with_constraints(self) -> int
        Returns the number of joints with constraints in the model.
    remove_joint(self, name: str)
        Removes a joint from the model.
    nb_joint_constraints(self) -> int
        Returns the number of joint constraints in the model.
    nb_joint_dof(self) -> int
        Returns the number of joint degrees of freedom in the model.
    joint_names(self) -> list[str]
        Returns the names of the joints in the model.
    nb_rigid_body_constraints(self) -> int
        Returns the number of rigid body constraints in the model.
    nb_holonomic_constraints(self) -> int
        Returns the number of holonomic constraints in the model.
    nb_Q(self) -> int
        Returns the number of generalized coordinates in the model.
    nb_Qdot(self) -> int
        Returns the number of generalized velocities in the model.
    nb_Qddot(self) -> int
        Returns the number of generalized accelerations in the model.
    joint_from_index(self, index: int)
        Returns the joint with the given index.
    joint_dof_indexes(self, joint_id: int) -> tuple[int, ...]
        Returns the index of a given joint.
    joint_constraints_index(self, joint_id: int | str) -> slice
        Returns the slice of constrain of a given joint.
    joints_from_child_index(self, child_index: int, remove_free_joints: bool = False) -> list
        Returns the joints that have the given child index.
    segment_from_index(self, index: int)
        Returns the segment with the given index.
    normalized_coordinates(self) -> tuple[tuple[int, ...]]
        Returns the normalized coordinates.
    mass_matrix(self)
        Returns the generalized mass matrix of the system.
    rigid_body_constraints(self, Q: NaturalCoordinates)
        Returns the rigid body constraints of all segments.
    rigid_body_constraints_derivative(self, Q: NaturalCoordinates, Qdot: NaturalCoordinates)
        Returns the derivative of the rigid body constraints.
    rigid_body_constraints_jacobian(self, Q: NaturalCoordinates)
        Returns the rigid body constraints of all segments.
    rigid_body_constraint_jacobian_derivative(self, Qdot: NaturalVelocities) -> np.ndarray
        Returns the derivative of the Jacobian matrix of the rigid body constraints.
    joint_constraints(self, Q: NaturalCoordinates)
        Returns the joint constraints of all joints.
    joint_constraints_jacobian(self, Q: NaturalCoordinates)
        Returns the joint constraints of all joints.
    joint_constraints_jacobian_derivative(self, Qdot: NaturalVelocities)
        Returns the derivative of the Jacobian matrix of the joint constraints.
    _update_mass_matrix(self)
        Computes the generalized mass matrix of the system.
    kinetic_energy(self, Qdot: NaturalVelocities) -> Union[np.ndarray, MX]
        Computes the kinetic energy of the system.
    potential_energy(self, Q: NaturalCoordinates) -> Union[np.ndarray, MX]
        Computes the potential energy of the system.
    lagrangian(self, Q: NaturalCoordinates, Qdot: NaturalVelocities)
        Returns the lagrangian of the system as a function of the natural coordinates Q and Qdot.
    energy(self, Q: NaturalCoordinates, Qdot: NaturalVelocities)
        Returns the total energy of the model from current Q and Qdot.
    markers(self, Q: NaturalCoordinates)
        Returns the position of the markers of the system as a function of the natural coordinates Q.
    markers_constraints(self, markers: np.ndarray | MX, Q: NaturalCoordinates)
        Returns the marker constraints of all segments.
    markers_constraints_jacobian(self)
        Returns the Jacobian matrix the markers constraints.
    holonomic_constraints(self, Q: NaturalCoordinates) -> MX | np.ndarray
        Returns the holonomic constraints of the system.
    holonomic_constraints_jacobian(self, Q: NaturalCoordinates)
        Returns the Jacobian matrix the holonomic constraints.
    gravity_forces(self)
        Returns the weights caused by the gravity forces on each segment.
    forward_dynamics(self, Q: NaturalCoordinates, Qdot: NaturalCoordinates)
        Computes the forward dynamics of the system, i.e. the acceleration of the segments.
    center_of_mass_position(self, Q: NaturalCoordinates)
        Returns the position of the center of mass of each segment as a function of the natural coordinates Q.
    _depth_first_search(self, segment_index, visited_segments=None) -> list[bool]
        Returns the segments in a depth first search order.
    external_force_set(self) -> ExternalForceSet
        Creates an empty ExternalForceSet object with the number of segments in the current biomechanical model.
    inverse_dynamics(self, Q: NaturalCoordinates, Qddot: NaturalAccelerations, external_forces: ExternalForceSet = None) -> tuple[np.ndarray | MX, np.ndarray | MX, np.ndarray | MX]
        Returns the forces, torques and lambdas computes through recursive Newton-Euler algorithm.
    natural_coordinates_to_joint_angles(self, Q: NaturalCoordinates)
        Converts the natural coordinates to joint angles with Euler Sequences defined for each joint.
    marker_technical_index
        This function returns the index of the marker with the given name
    """

    def __init__(
        self,
        segments: GenericBiomechanicalModelSegments = None,
        joints: GenericBiomechanicalModelJoints = None,
        markers: GenericBiomechanicalModelMarkers = None,
    ):
        self.segments = segments
        self.joints = joints
        self._markers = markers
        self._tree = GenericBiomechanicalModelTree(segments, joints)
        # From Pythom 3.7 the insertion order in a dict is preserved. This is important because when writing a new
        # the order of the segment matters
        self._mass_matrix = self._update_mass_matrix()

    def __getitem__(self, name: str):
        """
        This function returns the segment with the given name

        Parameters
        ----------
        name : str
            Name of the segment

        Returns
        -------
        NaturalSegment
            The segment with the given name
        """
        return self.segments[name]

    def __setitem__(self, name: str, segment: Any):
        """
        This function adds a segment to the model

        Parameters
        ----------
        name : str
            Name of the segment
        segment : NaturalSegment
            The segment to add
        """
        self.segments[name] = segment
        self._update_mass_matrix()  # Update the generalized mass matrix
        if name in ("ground", "GROUND", "Ground"):
            self.set_ground_segment(name)

        # adding a default joint with the world frame to defined standard transformations.
        from ...bionc_numpy.enums import JointType  # prevent circular import
        from ...bionc_casadi.enums import JointType as CasadiJointType  # prevent circular import

        self._add_joint(
            dict(
                name=f"free_joint_{name}",
                joint_type=CasadiJointType.GROUND_FREE
                if hasattr(self, "numpy_model")
                else JointType.GROUND_FREE,  # not satisfying
                parent="GROUND",  # to be popped out
                child=name,
                parent_point=None,
                child_point=None,
                length=None,
                theta=None,
                projection_basis=EulerSequence.XYZ,
                parent_basis=None,
                child_basis=None,
            ),
        )

    def save(self, filename: str):
        """
        This function saves the model to a file

        Parameters
        ----------
        filename : str
            The path to the file
        """
        with open(filename, "wb") as file:
            pickle.dump(self, file)

    @staticmethod
    def load(filename: str):
        """
        This function loads a model from a file

        Parameters
        ----------
        filename : str
            The path to the file

        Returns
        -------
        GenericBiomechanicalModel
            The loaded model
        """
        with open(filename, "rb") as file:
            model = pickle.load(file)

        return model

    def set_ground_segment(self, name: str):
        """
        This function sets the ground segment of the model

        Parameters
        ----------
        name : str
            The name of the ground segment
        """
        # remove the ground segment from the previous ground segment
        for segment in self.segments.values():
            segment._set_is_ground(False)

        # set the new ground segment
        self[name]._set_is_ground(True)

        # reindex the segments
        count = 0
        for i, segment in enumerate(self.segments.values()):
            if segment._is_ground:
                count -= 1
            segment.set_index(i + count)

        # reindex the joints
        for i, joint in enumerate(self.joints.values()):
            if (joint.parent is not None and joint.parent.name == name) or joint.child.name == name:
                raise ValueError("The ground segment cannot be a parent or a child of a joint")

        self._update_mass_matrix()

    @property
    def has_ground_segment(self) -> bool:
        return self.segments.has_ground_segment

    @property
    def segments_no_ground(self):
        return self.segments.segments_no_ground

    def _add_joint(self, joint: dict):
        """todo: we may select the two segments of interest before calling the function"""
        return self.joints._add_joint(joint, self.segments)

    @property
    def joints_with_constraints(self) -> dict:
        return self.joints.joints_with_constraints

    def has_free_joint(self, segment_idx: int) -> bool:
        return self.joints.has_free_joint(segment_idx)

    def _remove_free_joint(self, segment_idx: int):
        return self.joints._remove_free_joint(segment_idx)

    def children(self, segment: str | int) -> list[int]:
        return self._tree.children(segment)

    def parents(self, segment: str | int) -> list[int]:
        return self._tree.parents(segment)

    def segment_subtrees(self) -> list[list[int]]:
        return self._tree.segment_subtrees()

    def segment_subtree(self, segment: str | int) -> list[int]:
        return self._tree.segment_subtree(segment)

    @property
    def nb_segments(self) -> int:
        return self.segments.nb_segments

    @property
    def nb_markers(self) -> int:
        return self._markers.nb_markers

    @property
    def nb_markers_technical(self) -> int:
        return self._markers.nb_markers_technical

    @property
    def segment_names(self) -> list[str]:
        return self.segments.segment_names

    @property
    def marker_names(self) -> list[str]:
        return self._markers.names

    @property
    def marker_names_technical(self) -> list[str]:
        return self._markers.names_technical

    def marker_technical_index(self, name: str) -> int:
        return self.marker_names_technical.index(name)

    @property
    def dof_names(self) -> list[str]:
        """
        This function returns the names of the degrees of freedom in the model,
        namely the names of each decision variable in the model, i.e. the natural coordinates
        """
        dof_names = []
        for key in self.segments_no_ground:
            dof_names += [f"{key}_{dof}" for dof in SegmentNaturalCoordinates.name_dofs]

        return dof_names

    @property
    def nb_joints(self) -> int:
        return self.joints.nb_joints

    @property
    def nb_joints_with_constraints(self) -> int:
        return self.joints.nb_joints_with_constraints

    def remove_joint(self, name: str):
        return self.joints.remove_joint(name)

    @property
    def nb_joint_constraints(self) -> int:
        return self.joints.nb_constraints

    @property
    def nb_joint_dof(self) -> int:
        return self.joints.nb_joint_dof

    @property
    def joint_names(self) -> list[str]:
        return self.joints.joint_names

    @property
    def nb_rigid_body_constraints(self) -> int:
        return self.segments.nb_rigid_body_constraints

    @property
    def nb_holonomic_constraints(self) -> int:
        """
        This function returns the number of holonomic constraints in the model
        """
        return self.nb_joint_constraints + self.nb_rigid_body_constraints

    @property
    def nb_Q(self) -> int:
        return self.segments.nb_Q

    @property
    def nb_Qdot(self) -> int:
        return self.segments.nb_Qdot

    @property
    def nb_Qddot(self) -> int:
        return self.segments.nb_Qddot

    def joint_from_index(self, index: int):
        return self.joints.joint_from_index(index)

    def joint_dof_indexes(self, joint_id: int) -> tuple[int, ...]:
        return self.joints.dof_indexes(joint_id)

    def joint_constraints_index(self, joint_id: int | str) -> slice:
        return self.joints.constraints_index(joint_id)

    def joints_from_child_index(self, child_index: int, remove_free_joints: bool = False) -> list:
        return self.joints.joints_from_child_index(child_index, remove_free_joints)

    def segment_from_index(self, index: int):
        return self.segments.segment_from_index(index)

    @property
    def normalized_coordinates(self) -> tuple[tuple[int, ...]]:
        return self.segments.normalized_coordinates

    @property
    def mass_matrix(self):
        """
        This function returns the generalized mass matrix of the system, denoted G

        Returns
        -------
        np.ndarray
            generalized mass matrix of the segment [12 * nb_segments x 12 * * nb_segment]

        """
        return self._mass_matrix

    def rigid_body_constraints(self, Q: NaturalCoordinates):
        return self.segments.rigid_body_constraints(Q)

    def rigid_body_constraints_derivative(self, Q: NaturalCoordinates, Qdot: NaturalCoordinates):
        return self.segments.rigid_body_constraints_derivative(Q, Qdot)

    def rigid_body_constraints_jacobian(self, Q: NaturalCoordinates):
        return self.segments.rigid_body_constraints_jacobian(Q)

    def rigid_body_constraint_jacobian_derivative(self, Qdot: NaturalVelocities) -> np.ndarray:
        return self.segments.rigid_body_constraint_jacobian_derivative(Qdot)

    def joint_constraints(self, Q: NaturalCoordinates) -> MX:
        return self.joints.constraints(Q)

    def joint_constraints_jacobian(self, Q: NaturalCoordinates) -> np.ndarray:
        return self.joints.constraints_jacobian(Q)

    def joint_constraints_jacobian_derivative(self, Qdot: NaturalVelocities) -> MX:
        return self.joints.constraints_jacobian_derivative(Qdot)

    @abstractmethod
    def _update_mass_matrix(self):
        """
        This function computes the generalized mass matrix of the system, denoted G

        Returns
        -------
        np.ndarray
            generalized mass matrix of the segment [12 * nb_segments, 12 * nb_segment]
        """

    @abstractmethod
    def kinetic_energy(self, Qdot: NaturalVelocities) -> Union[np.ndarray, MX]:
        """
        This function computes the kinetic energy of the system

        Parameters
        ----------
        Qdot : NaturalVelocities
            The natural velocities of the segment [12 * nb_segments, 1]

        Returns
        -------
        Union[np.ndarray, MX]
            The kinetic energy of the system
        """

    @abstractmethod
    def potential_energy(self, Q: NaturalCoordinates) -> Union[np.ndarray, MX]:
        """
        This function computes the potential energy of the system

        Parameters
        ----------
        Q : NaturalCoordinates
            The natural coordinates of the segment [12 * nb_segments, 1]

        Returns
        -------
        Union[np.ndarray, MX]
            The potential energy of the system
        """

    def lagrangian(self, Q: NaturalCoordinates, Qdot: NaturalVelocities):
        """
        This function returns the lagrangian of the system as a function of the natural coordinates Q and Qdot

        Parameters
        ----------
        Q : NaturalCoordinates
            The natural coordinates of the segment [12 * nb_segments, 1]
        Qdot : NaturalVelocities
            The natural velocities of the segment [12 * nb_segments, 1]

        Returns
        -------
            The lagrangian of the system
        """

        return self.kinetic_energy(Qdot) - self.potential_energy(Q)

    def energy(self, Q: NaturalCoordinates, Qdot: NaturalVelocities):
        """
        This function returns the total energy of the model from current Q and Qdot

        Parameters
        ----------
        Q : NaturalCoordinates
            The natural coordinates of the segment [12 * nb_segments, 1]
        Qdot : NaturalVelocities
            The natural velocities of the segment [12 * nb_segments, 1]

        Returns
        -------
            The energy of the system
        """

        return self.kinetic_energy(Qdot) + self.potential_energy(Q)

    def Q_from_markers(self, markers: np.ndarray) -> NaturalCoordinates:
        return self._markers.Q_from_markers(markers)

    def markers(self, Q: NaturalCoordinates):
        return self._markers.markers(Q)

    def markers_constraints(self, markers: np.ndarray | MX, Q: NaturalCoordinates, only_technical: bool = True):
        return self._markers.constraints(markers, Q, only_technical)

    def markers_constraints_jacobian(self, only_technical: bool = True):
        return self._markers.constraints_jacobian(only_technical)

    def center_of_mass_position(self, Q: NaturalCoordinates):
        return self._markers.center_of_mass_position(Q)

    @abstractmethod
    def holonomic_constraints(self, Q: NaturalCoordinates) -> MX | np.ndarray:
        """
        This function returns the holonomic constraints of the system, denoted Phi_h
        as a function of the natural coordinates Q. They are organized as follow, for each segment:
            [Phi_k_0, Phi_r_0, Phi_k_1, Phi_r_1, ..., Phi_k_n, Phi_r_n]

        Parameters
        ----------
        Q : NaturalCoordinates
            The natural coordinates of the segment [12 * nb_segments, 1]

        Returns
        -------
            Holonomic constraints of the segment [nb_holonomic_constraints, 1]
        """

    @abstractmethod
    def holonomic_constraints_jacobian(self, Q: NaturalCoordinates):
        """
        This function returns the Jacobian matrix the holonomic constraints, denoted k_h.
        They are organized as follow, for each segment, the rows of the matrix are:
        [Phi_k_0, Phi_r_0, Phi_k_1, Phi_r_1, ..., Phi_k_n, Phi_r_n]

        Parameters
        ----------
        Q : NaturalCoordinates
            The natural coordinates of the segment [12 * nb_segments, 1]

        Returns
        -------
            Joint constraints of the holonomic constraints [nb_holonomic_constraints, 12 * nb_segments]
        """

    @abstractmethod
    def gravity_forces(self):
        """
        This function returns the weights caused by the gravity forces on each segment

        Returns
        -------
            The gravity_force of each segment [12 * nb_segments, 1]
        """

    @abstractmethod
    def forward_dynamics(
        self,
        Q: NaturalCoordinates,
        Qdot: NaturalCoordinates,
        # external_forces: ExternalForces
    ):
        """
        This function computes the forward dynamics of the system, i.e. the acceleration of the segments

        Parameters
        ----------
        Q : NaturalCoordinates
            The natural coordinates of the segment [12 * nb_segments, 1]
        Qdot : NaturalCoordinates
            The natural coordinates time derivative of the segment [12 * nb_segments, 1]

        Returns
        -------
            Qddot : NaturalAccelerations
                The natural accelerations [12 * nb_segments, 1]
        """

    def _depth_first_search(self, segment_index) -> list[bool]:
        return self._tree.starting_depth_first_search(segment_index)

    @abstractmethod
    def external_force_set(self) -> ExternalForceSet:
        """
        This method creates an empty ExternalForceSet object with the number of segments in the current biomechanical model.
        The ExternalForceSet object is used to store and manage the external forces applied to each segment in the model.

        Returns
        -------
        ExternalForceSet
            An empty ExternalForceSet object with the same number of segment as the biomechanical model.
        """

    @abstractmethod
    def inverse_dynamics(
        self, Q: NaturalCoordinates, Qddot: NaturalAccelerations, external_forces: ExternalForceSet = None
    ) -> tuple[np.ndarray | MX, np.ndarray | MX, np.ndarray | MX]:
        """
        This function returns the forces, torques and lambdas computes through recursive Newton-Euler algorithm

        Source
        ------
        Dumas. R and Chèze. L (2006).
        3D inverse dynamics in non-orthonormal segment coordinate system. Med Bio Eng Comput.
        DOI 10.1007/s11517-006-0156-8

        Parameters
        ----------
        Q: NaturalCoordinates
           The generalized coordinates of the model
        Qddot: NaturalAccelerations
           The generalized accelerations of the model
        external_forces: ExternalForceSet
           The external forces applied to the model

        Returns
        -------
        tuple[Any, Any, Any]
            The forces, torques and lambdas, all expressed in the global coordinate system
            It may be a good idea to express them in the local or euler basis coordinate system
        """

    @abstractmethod
    def natural_coordinates_to_joint_angles(self, Q: NaturalCoordinates):
        """
        This function converts the natural coordinates to joint angles with Euler Sequences defined for each joint

        Parameters
        ----------
        Q: NaturalCoordinates
            The natural coordinates of the model

        Returns
        -------
            The joint angles [3 x nb_joints]
        """
