from abc import ABC, abstractmethod
from typing import Any

from .biomechanical_model_segments import GenericBiomechanicalModelSegments
from .natural_coordinates import NaturalCoordinates
from .natural_velocities import NaturalVelocities


class GenericBiomechanicalModelJoints(ABC):
    """
    This is an abstract base class that provides the basic structure and methods for all joints in biomechanical models.
    It contains the segments and the joints of the model. The implemented methods are not specific to numpy or casadi.

    Attributes
    ----------
    joints : dict
        A dictionary containing the joints of the model.
        The keys are the names of the joints and the values are the corresponding joint objects.

    Methods
    -------
    _add_joint(self, joint: dict)
        Adds a joint to the biomechanical model. It is not recommended to use this function directly.
    joints_with_constraints(self) -> dict
        Returns the dictionary of all the joints with constraints.
    has_free_joint(self, segment_idx: int) -> bool
        Returns true if the segment has a free joint with the ground.
    _remove_free_joint(self, segment_idx: int)
        Removes the free joint of the segment.
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
    joint_from_index(self, index: int)
        Returns the joint with the given index.
    joint_dof_indexes(self, joint_id: int) -> tuple[int, ...]
        Returns the index of a given joint.
    joint_constraints_index(self, joint_id: int | str) -> slice
        Returns the slice of constrain of a given joint.
    joints_from_child_index(self, child_index: int, remove_free_joints: bool = False) -> list
        Returns the joints that have the given child index.
    joint_constraints(self, Q: NaturalCoordinates)
        Returns the joint constraints of all joints.
    joint_constraints_jacobian(self, Q: NaturalCoordinates)
        Returns the joint constraints of all joints.
    joint_constraints_jacobian_derivative(self, Qdot: NaturalVelocities)
        Returns the derivative of the Jacobian matrix of the joint constraints.
    """

    def __init__(
        self,
        joints: dict[str:Any, ...] = None,
    ):
        from .joint import JointBase  # Imported here to prevent from circular imports

        self.joints: dict[str:JointBase, ...] = {} if joints is None else joints
        # From Pythom 3.7 the insertion order in a dict is preserved. This is important because when writing a new
        # the order of the segment matters

    def __getitem__(self, key):
        return self.joints[key]

    def values(self):
        return self.joints.values()

    def items(self):
        return self.joints.items()

    def _add_joint(self, joint: dict, segments: GenericBiomechanicalModelSegments):
        """
        This function adds a joint to the biomechanical model. It is not recommended to use this function directly.

        Parameters
        ----------
        joint : dict
            A dictionary containing the joints to be added to the biomechanical model:
            {name: str, joint: Joint, parent: str, child: str}
        segments : GenericBiomechanicalModelSegments
            The segments of the biomechanical model
        """
        if joint["parent"] is not None and joint["parent"] != "GROUND" and joint["parent"] not in segments.keys():
            raise ValueError("The parent segment does not exist")
        if joint["child"] not in segments.keys():
            raise ValueError("The child segment does not exist")
        if joint["name"] in self.joints.keys():
            raise ValueError("The joint name already exists")
        # remove the default joint GROUND_FREE if it still exists
        # There is automatically a free joint for each segment when created. This joint is not needed anymore when
        # adding a new joint to the segment
        if self.has_free_joint(segments[joint["child"]].index):
            self._remove_free_joint(segments[joint["child"]].index)

        # remove name of the joint_type from the dictionary
        joint_type = joint.pop("joint_type")
        # remove None values from the dictionary
        joint = {key: value for key, value in joint.items() if value is not None}
        # replace parent field by the parent segment
        if joint["parent"] == "GROUND":
            joint.pop("parent")
        else:
            joint["parent"] = segments[joint["parent"]]

        # replace child field by the child segment
        joint["child"] = segments[joint["child"]]
        joint["index"] = self.nb_joints

        self.joints[joint["name"]] = joint_type.value(**joint)

    @property
    def joints_with_constraints(self) -> dict:
        """
        This function returns the dictionary of all the joints with constraints
        It removes the joints with no constraints from self.joints

        Returns
        -------
        dict[str: Joint, ...]
            The dictionary of all the joints with constraints
        """
        return {name: joint for name, joint in self.joints.items() if joint.nb_constraints > 0}

    def has_free_joint(self, segment_idx: int) -> bool:
        """
        This function returns true if the segment has a free joint with the ground

        Parameters
        ----------
        segment_idx : int
            The index of the segment

        Returns
        -------
        bool
            True if the segment has a free joint with the ground
        """
        from ..bionc_numpy.enums import JointType  # prevent circular import
        from ..bionc_casadi.enums import JointType as CasadiJointType  # prevent circular import

        joints = self.joints_from_child_index(segment_idx, remove_free_joints=False)
        return any(
            isinstance(joint, (JointType.GROUND_FREE.value, CasadiJointType.GROUND_FREE.value)) for joint in joints
        )

    def _remove_free_joint(self, segment_idx: int):
        """
        This function removes the free joint of the segment

        Notes
        -----
        Don't use this function if you don't know what you are doing

        Parameters
        ----------
        segment_idx : int
            The index of the segment
        """
        from ..bionc_numpy.enums import JointType  # prevent circular import
        from ..bionc_casadi.enums import JointType as CasadiJointType  # prevent circular import

        joints = self.joints_from_child_index(segment_idx, remove_free_joints=False)

        free_joints = [
            joint
            for joint in joints
            if isinstance(joint, (JointType.GROUND_FREE.value, CasadiJointType.GROUND_FREE.value))
        ]

        if not free_joints:
            raise ValueError("The segment does not have a free joint")

        for joint in free_joints:
            self.remove_joint(joint.name)

    @property
    def nb_joints(self) -> int:
        """
        This function returns the number of joints in the model
        """
        return len(self.joints)

    @property
    def nb_joints_with_constraints(self) -> int:
        """
        This function returns the number of joints with constraints in the model
        """
        return len(self.joints_with_constraints)

    def remove_joint(self, name: str):
        """
        This function removes a joint from the model

        Parameters
        ----------
        name : str
            The name of the joint to be removed
        """
        # if name not in self.joints.keys():
        #     raise ValueError("The joint does not exist")
        # joint_index_to_remove = self.joints[name].index
        # self.joints.pop(name)
        # for joint in self.joints.values():
        #     if joint.index > joint_index_to_remove:
        #         joint.index -= 1

        joint_to_remove = self.joints.pop(name, None)
        if joint_to_remove is None:
            raise ValueError("The joint does not exist")

        for joint in self.joints.values():
            if joint.index > joint_to_remove.index:
                joint.index -= 1

    @property
    def nb_constraints(self) -> int:
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
        nb_joint_dof = 0
        for _, joint in self.joints.items():
            nb_joint_dof += joint.nb_joint_dof
        return nb_joint_dof

    @property
    def joint_names(self) -> list[str]:
        """
        This function returns the names of the joints in the model
        """
        return list(self.joints.keys())

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

    def dof_indexes(self, joint_id: int) -> tuple[int, ...]:
        """
        This function returns the index of a given joint.

        Parameters
        ----------
        joint_id : int
            The index of the joint for which the joint dof indexes are returned

        Returns
        -------
        tuple[int, ...]
            The indexes of the joint dof
        """
        joint = self.joint_from_index(joint_id)
        joint_dof_inx = [joint.index + i for i in range(joint.nb_joint_dof)]
        return tuple(joint_dof_inx)

    def constraints_index(self, joint_id: int | str) -> slice:
        """
        This function returns the slice of constrain of a given joint.

        Parameters
        ----------
        joint_id : int | str
            The index or the name of the joint for which the joint constraint indexes are returned

        Returns
        -------
        slice_joint_constraint: slice
            The slice of the given constraint
        """
        if isinstance(joint_id, str):
            if joint_id not in self.joint_names:
                raise ValueError("The joint name " + joint_id + " does not exist")
            joint_id = self.joint_names.index(joint_id)

        if isinstance(joint_id, int):
            if joint_id > self.nb_joints:
                raise ValueError("The joint index " + str(joint_id) + " does not exist")

        nb_constraint_before_joint = 0
        for ind_joint in range(joint_id):
            nb_constraint_before_joint += self.joints[self.joint_names[ind_joint]].nb_constraints

        begin_slice = nb_constraint_before_joint
        nb_joint_constraints = self.joints[self.joint_names[joint_id]].nb_constraints
        end_slice = nb_constraint_before_joint + nb_joint_constraints

        slice_joint_constraint = slice(begin_slice, end_slice)

        return slice_joint_constraint

    def joints_from_child_index(self, child_index: int, remove_free_joints: bool = False) -> list:
        """
        This function returns the joints that have the given child index

        Parameters
        ----------
        child_index : int
            The child index
        remove_free_joints : bool
            If True, the free joints are not returned

        Returns
        -------
        list[JointBase]
            The joints that have the given child index
        """
        from ..bionc_numpy.enums import JointType  # prevent circular import
        from ..bionc_casadi.enums import JointType as CasadiJointType  # prevent circular import

        joints = []
        for joint in self.joints.values():
            is_free_joint = isinstance(joint, (JointType.GROUND_FREE.value, CasadiJointType.GROUND_FREE.value))
            if remove_free_joints and is_free_joint:
                continue
            if joint.child.index == child_index:
                joints.append(joint)
        return joints

    @abstractmethod
    def constraints(self, Q: NaturalCoordinates, segments):
        """
        This function returns the joint constraints of all joints, denoted Phi_k
        as a function of the natural coordinates Q.

        Parameters
        ----------
        Q : NaturalCoordinates
            The natural coordinates of the segment [12 * nb_segments, 1]
        segments : GenericBiomechanicalModelSegments
            The segments of the biomechanical model

        Returns
        -------
            Joint constraints of the segment [nb_joint_constraints, 1]
        """

    @abstractmethod
    def constraints_jacobian(self, Q: NaturalCoordinates, segments):
        """
        This function returns the joint constraints of all joints, denoted K_k

        Parameters
        ----------
        Q : NaturalCoordinates
            The natural coordinates of the segment [12 * nb_segments, 1]
        segments : GenericBiomechanicalModelSegments
            The segments of the biomechanical model


        Returns
        -------
            Joint constraints of the segment [nb_joint_constraints, 1]
        """

    @abstractmethod
    def constraints_jacobian_derivative(self, Qdot: NaturalVelocities, segments):
        """
        This function returns the derivative of the Jacobian matrix of the joint constraints denoted Kk_dot

        Parameters
        ----------
        Qdot : NaturalVelocities
            The natural velocities of the segment [12 * nb_segments, 1]
        segments : BiomechanicalModelSegments
            The segments of the model

        Returns
        -------
            The derivative of the Jacobian matrix of the joint constraints [nb_joint_constraints, 12 * nb_segments]
        """
