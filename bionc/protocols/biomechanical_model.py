import numpy as np
from casadi import MX

from typing import Union, Any
from abc import ABC, abstractmethod
import dill as pickle

from bionc.protocols.natural_coordinates import NaturalCoordinates
from bionc.protocols.natural_velocities import NaturalVelocities
from bionc.protocols.natural_accelerations import NaturalAccelerations
from bionc.protocols.external_force import ExternalForceList


class GenericBiomechanicalModel(ABC):
    """
    This class is the base with simple methods for all biomechanical models.
    It contains the segments and the joints of the model.

    The implemented method are not specific to numpy or casadi.

    Methods
    ----------


    """

    def __init__(
        self,
        segments: dict[str:Any, ...] = None,
        joints: dict[str:Any, ...] = None,
    ):
        from .natural_segment import AbstractNaturalSegment  # Imported here to prevent from circular imports
        from .joint import JointBase  # Imported here to prevent from circular imports

        self.segments: dict[str:AbstractNaturalSegment, ...] = {} if segments is None else segments
        self.joints: dict[str:JointBase, ...] = {} if joints is None else joints
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
        if name == "ground":
            for segment in self.segments.values():
                if segment.is_ground:
                    return segment
            else:
                raise ValueError("No ground segment found")
        else:
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
        if segment.name == name:  # Make sure the name of the segment fits the internal one
            segment.set_index(len(self.segments))
            self.segments[name] = segment
            self._update_mass_matrix()  # Update the generalized mass matrix
            if name in ("ground", "GROUND", "Ground"):
                self.set_ground_segment(name)
        else:
            raise ValueError("The name of the segment does not match the name of the segment")

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
        """
        This function returns true if the model has a ground segment

        Returns
        -------
        bool
            True if the model has a ground segment
        """
        for segment in self.segments.values():
            if segment._is_ground:
                return True
        else:
            return False

    @property
    def segments_no_ground(self):
        """
        This function returns the dictionary of all the segments except the ground segment

        Returns
        -------
        dict[str: NaturalSegment, ...]
            The dictionary of all the segments except the ground segment
        """
        return {name: segment for name, segment in self.segments.items() if not segment._is_ground}

    def _add_joint(self, joint: dict):
        """
        This function adds a joint to the biomechanical model. It is not recommended to use this function directly.

        Parameters
        ----------
        joint : dict
            A dictionary containing the joints to be added to the biomechanical model:
            {name: str, joint: Joint, parent: str, child: str}
        """
        if joint["parent"] is not None and joint["parent"] != "GROUND" and joint["parent"] not in self.segments.keys():
            raise ValueError("The parent segment does not exist")
        if joint["child"] not in self.segments.keys():
            raise ValueError("The child segment does not exist")
        if joint["name"] in self.joints.keys():
            raise ValueError("The joint name already exists")

        # remove name of the joint_type from the dictionary
        joint_type = joint.pop("joint_type")
        # remove None values from the dictionary
        joint = {key: value for key, value in joint.items() if value is not None}
        # replace parent field by the parent segment
        if joint["parent"] == "GROUND":
            joint.pop("parent")
        else:
            joint["parent"] = self.segments[joint["parent"]]

        # replace child field by the child segment
        joint["child"] = self.segments[joint["child"]]
        joint["index"] = self.nb_joints

        self.joints[joint["name"]] = joint_type.value(**joint)

    def children(self, segment: str | int) -> list[int]:
        """
        This function returns the children of the given segment

        Parameters
        ----------
        segment : str | int
            The segment for which the children are returned

        Returns
        -------
        list[int]
            The children of the given segment
        """
        children = []
        if isinstance(segment, str):
            segment = self.segments[segment]
        elif isinstance(segment, int):
            segment = self.segment_from_index(segment)
        for joint in self.joints.values():
            if joint.parent is not None and joint.parent.name == segment.name:
                children.append(joint.child.index)
        return children

    def parents(self, segment: str | int) -> list[int]:
        """
        This function returns the parents of the given segment

        Parameters
        ----------
        segment : str | int
            The segment for which the parents are returned

        Returns
        -------
        list[int]
            The parents of the given segment
        """
        parents = []
        if isinstance(segment, str):
            segment = self.segments[segment]
        elif isinstance(segment, int):
            segment = self.segment_from_index(segment)
        for joint in self.joints.values():
            if joint.child == segment:
                parents.append(joint.parent.index)
        return parents

    def segment_subtrees(self) -> list[list[int]]:
        """
        This function returns the subtrees of the segments

        Returns
        -------
        list[list[int]]
            The subtrees of the segments
        """
        subtrees = []
        for segment in self.segments.values():
            subtrees.append(self.segment_subtree(segment))
        return subtrees

    def segment_subtree(self, segment: str | int) -> list[int]:
        """
        This function returns the subtree of the given segment

        Parameters
        ----------
        segment : str | int
            The segment for which the subtree is returned

        Returns
        -------
        list[int]
            The subtree of the given segment
        """
        if isinstance(segment, str):
            segment = self.segments[segment]
        elif isinstance(segment, int):
            segment = self.segment_from_index(segment)
        subtree = [segment.index]
        for child in self.children(segment):
            subtree += self.segment_subtree(child)
        return subtree

    @property
    def nb_segments(self) -> int:
        """
        This function returns the number of segments in the model
        """
        return len(self.segments) - 1 if self.has_ground_segment else len(self.segments)

    @property
    def nb_markers(self) -> int:
        """
        This function returns the number of markers in the model
        """
        nb_markers = 0
        for key in self.segments_no_ground:
            nb_markers += self.segments[key].nb_markers
        return nb_markers

    @property
    def nb_markers_technical(self) -> int:
        """
        This function returns the number of technical markers in the model
        """
        nb_markers = 0
        for key in self.segments_no_ground:
            nb_markers += self.segments[key].nb_markers_technical
        return nb_markers

    @property
    def segment_names(self) -> list[str]:
        """
        This function returns the names of the segments in the model
        """
        return list(self.segments.keys())

    @property
    def marker_names(self) -> list[str]:
        """
        This function returns the names of the markers in the model
        """
        marker_names = []
        for key in self.segments_no_ground:
            marker_names += self.segments[key].marker_names
        return marker_names

    @property
    def marker_names_technical(self) -> list[str]:
        """
        This function returns the names of the technical markers in the model
        """
        marker_names = []
        for key in self.segments_no_ground:
            marker_names += self.segments[key].marker_names_technical
        return marker_names

    @property
    def nb_joints(self) -> int:
        """
        This function returns the number of joints in the model
        """
        return len(self.joints)

    def remove_joint(self, name: str):
        """
        This function removes a joint from the model

        Parameters
        ----------
        name : str
            The name of the joint to be removed
        """
        if name not in self.joints.keys():
            raise ValueError("The joint does not exist")
        joint_index_to_remove = self.joints[name].index
        self.joints.pop(name)
        for joint in self.joints.values():
            if joint.index > joint_index_to_remove:
                joint.index -= 1

    @property
    def nb_joint_constraints(self) -> int:
        """
        This function returns the number of joint constraints in the model
        """
        nb_joint_constraints = 0
        for _, joint in self.joints.items():
            nb_joint_constraints += joint.nb_constraints
        return nb_joint_constraints

    @property
    def nb_joint_dof(self) -> int:
        """
        This function returns the number of joint degrees of freedom in the model
        """
        nb_joint_dof = 6 * self.nb_segments
        for _, joint in self.joints.items():
            nb_joint_dof -= joint.nb_constraints
        return nb_joint_dof

    @property
    def joint_names(self) -> list[str]:
        """
        This function returns the names of the joints in the model
        """
        return list(self.joints.keys())

    @property
    def nb_rigid_body_constraints(self) -> int:
        """
        This function returns the number of rigid body constraints in the model
        """
        return 6 * self.nb_segments

    @property
    def nb_holonomic_constraints(self) -> int:
        """
        This function returns the number of holonomic constraints in the model
        """
        return self.nb_joint_constraints + self.nb_rigid_body_constraints

    @property
    def nb_Q(self) -> int:
        """
        This function returns the number of generalized coordinates in the model
        """
        return 12 * self.nb_segments

    @property
    def nb_Qdot(self) -> int:
        """
        This function returns the number of generalized velocities in the model
        """
        return 12 * self.nb_segments

    @property
    def nb_Qddot(self) -> int:
        """
        This function returns the number of generalized accelerations in the model
        """
        return 12 * self.nb_segments

    def joint_from_index(self, index: int):
        """
        This function returns the joint with the given index

        Parameters
        ----------
        index : int
            The index of the joint

        Returns
        -------
        Joint
            The joint with the given index
        """
        for joint in self.joints.values():
            if joint.index == index:
                return joint
        raise ValueError("No joint with index " + str(index))

    def joints_from_child_index(self, child_index: int) -> list:
        """
        This function returns the joints that have the given child index

        Parameters
        ----------
        child_index : int
            The child index

        Returns
        -------
        list[JointBase]
            The joints that have the given child index
        """
        joints = []
        for joint in self.joints.values():
            if joint.child.index == child_index:
                joints.append(joint)
        return joints

    def segment_from_index(self, index: int):
        """
        This function returns the segment with the given index

        Parameters
        ----------
        index : int
            The index of the segment

        Returns
        -------
        Segment
            The segment with the given index
        """
        for segment in self.segments.values():
            if segment.index == index:
                return segment
        raise ValueError(f"The segment index does not exist, the model has only {self.nb_segments} segments")

    @property
    def normalized_coordinates(self) -> tuple[tuple[int, ...]]:
        idx = []
        for i in range(self.nb_segments):
            # create list from i* 12 to (i+1) * 12
            segment_idx = [i for i in range(i * 12, (i + 1) * 12)]
            idx.append(segment_idx[0:3])
            idx.append(segment_idx[9:12])

        return idx

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

    @abstractmethod
    def rigid_body_constraints(self, Q: NaturalCoordinates):
        """
        This function returns the rigid body constraints of all segments, denoted Phi_r
        as a function of the natural coordinates Q.

        Returns
        -------
        np.ndarray
            Rigid body constraints of the segment [6 * nb_segments, 1]
        """

    @abstractmethod
    def rigid_body_constraints_derivative(self, Q: NaturalCoordinates, Qdot: NaturalCoordinates):
        """
        This function returns the derivative of the rigid body constraints denoted Phi_r_dot

        Parameters
        ----------
        Q : NaturalCoordinates
            The natural coordinates of the model
        Qdot : NaturalVelocities
            The natural velocities of the model

        Returns
        -------
            Derivative of the rigid body constraints
        """

    @abstractmethod
    def rigid_body_constraints_jacobian(self, Q: NaturalCoordinates):
        """
        This function returns the rigid body constraints of all segments, denoted K_r
        as a function of the natural coordinates Q.

        Returns
        -------
        np.ndarray
            Rigid body constraints of the segment [6 * nb_segments, nbQ]
        """

    @abstractmethod
    def rigid_body_constraint_jacobian_derivative(self, Qdot: NaturalVelocities) -> np.ndarray:
        """
        This function returns the derivative of the Jacobian matrix of the rigid body constraints denoted Kr_dot

        Parameters
        ----------
        Qdot : NaturalVelocities
            The natural velocities of the segment [12 * nb_segments, 1]

        Returns
        -------
            The derivative of the Jacobian matrix of the rigid body constraints [6 * nb_segments, 12 * nb_segments]
        """

    @abstractmethod
    def joint_constraints(self, Q: NaturalCoordinates):
        """
        This function returns the joint constraints of all joints, denoted Phi_k
        as a function of the natural coordinates Q.

        Returns
        -------
            Joint constraints of the segment [nb_joint_constraints, 1]
        """

    @abstractmethod
    def joint_constraints_jacobian(self, Q: NaturalCoordinates):
        """
        This function returns the joint constraints of all joints, denoted K_k

        Returns
        -------
            Joint constraints of the segment [nb_joint_constraints, 1]
        """

    @abstractmethod
    def joint_constraints_jacobian_derivative(self, Qdot: NaturalVelocities):
        """
        This function returns the derivative of the Jacobian matrix of the joint constraints denoted Kk_dot

        Parameters
        ----------
        Qdot : NaturalVelocities
            The natural velocities of the segment [12 * nb_segments, 1]

        Returns
        -------
            The derivative of the Jacobian matrix of the joint constraints [nb_joint_constraints, 12 * nb_segments]
        """

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

    @abstractmethod
    def markers(self, Q: NaturalCoordinates):
        """
        This function returns the position of the markers of the system as a function of the natural coordinates Q
        also referred as forward kinematics

        Parameters
        ----------
        Q : NaturalCoordinates
            The natural coordinates of the segment [12 x n, 1]

        Returns
        -------
            The position of the markers [3, nbMarkers, nbFrames]
            in the global coordinate system/ inertial coordinate system
        """

    @abstractmethod
    def markers_constraints(self, markers: np.ndarray | MX, Q: NaturalCoordinates):
        """
        This function returns the marker constraints of all segments, denoted Phi_r
        as a function of the natural coordinates Q.

        markers : np.ndarray | MX
            The markers positions [3,nb_markers]

        Q : NaturalCoordinates
            The natural coordinates of the segment [12 * nb_segments, 1]

        Returns
        -------
            Rigid body constraints of the segment [nb_markers x 3, 1]
        """

    @abstractmethod
    def markers_constraints_jacobian(self):
        """
        This function returns the Jacobian matrix the markers constraints, denoted k_m.

        Returns
        -------
            Joint constraints of the marker [nb_markers x 3, nb_Q]
        """

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

    @abstractmethod
    def center_of_mass_position(self, Q: NaturalCoordinates):
        """
        This function returns the position of the center of mass of each segment as a function of the natural coordinates Q

        Parameters
        ----------
        Q : NaturalCoordinates
            The natural coordinates of the segment [12 x n, 1]

        Returns
        -------
            The position of the center of mass [3, nbSegments]
            in the global coordinate system/ inertial coordinate system
        """

    def _depth_first_search(self, segment_index, visited_segments=None) -> list[bool]:
        """
        This function returns the segments in a depth first search order.

        todo: generalize to any number of segments with no parent.

        Parameters
        ----------
        segment_index: int
            The index of the segment to start the search from
        visited_segments: list[Segment]
            The segments already visited

        Returns
        -------
        list[bool, ...
            The segments in a depth first search order
        """
        if visited_segments is None:
            visited_segments = [False for _ in range(self.nb_segments)]

        visited_segments[segment_index] = True
        for child_index in self.children(segment_index):
            if visited_segments[child_index]:
                raise RuntimeError("The model contain closed loops, we cannot use this algorithm")
            if not visited_segments[child_index]:
                visited_segments = self._depth_first_search(child_index, visited_segments)

        return visited_segments

    @abstractmethod
    def inverse_dynamics(
        self, Q: NaturalCoordinates, Qddot: NaturalAccelerations, external_forces: ExternalForceList = None
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
        external_forces: ExternalForceList
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
