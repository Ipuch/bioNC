import numpy as np
from casadi import MX

from typing import Union, Any
from abc import ABC, abstractmethod
import dill as pickle

from bionc.protocols.natural_coordinates import NaturalCoordinates
from bionc.protocols.natural_velocities import NaturalVelocities


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

        self.joints[joint["name"]] = joint_type.value(**joint)

    def nb_segments(self) -> int:
        """
        This function returns the number of segments in the model
        """
        return len(self.segments)

    def nb_markers(self) -> int:
        """
        This function returns the number of markers in the model
        """
        nb_markers = 0
        for key in self.segments:
            nb_markers += self.segments[key].nb_markers()
        return nb_markers

    def nb_markers_technical(self) -> int:
        """
        This function returns the number of technical markers in the model
        """
        nb_markers = 0
        for key in self.segments:
            nb_markers += self.segments[key].nb_markers_technical()
        return nb_markers

    def marker_names(self) -> list[str]:
        """
        This function returns the names of the markers in the model
        """
        marker_names = []
        for key in self.segments:
            marker_names += self.segments[key].marker_names()
        return marker_names

    def marker_names_technical(self) -> list[str]:
        """
        This function returns the names of the technical markers in the model
        """
        marker_names = []
        for key in self.segments:
            marker_names += self.segments[key].marker_names_technical()
        return marker_names

    def nb_joints(self) -> int:
        """
        This function returns the number of joints in the model
        """
        return len(self.joints)

    def nb_joint_constraints(self) -> int:
        """
        This function returns the number of joint constraints in the model
        """
        nb_joint_constraints = 0
        for joint_name, joint in self.joints.items():
            nb_joint_constraints += joint.nb_constraints
        return nb_joint_constraints

    def nb_rigid_body_constraints(self) -> int:
        """
        This function returns the number of rigid body constraints in the model
        """
        return 6 * self.nb_segments()

    def nb_holonomic_constraints(self) -> int:
        """
        This function returns the number of holonomic constraints in the model
        """
        return self.nb_joint_constraints() + self.nb_rigid_body_constraints()

    def nb_Q(self) -> int:
        """
        This function returns the number of generalized coordinates in the model
        """
        return 12 * self.nb_segments()

    def nb_Qdot(self) -> int:
        """
        This function returns the number of generalized velocities in the model
        """
        return 12 * self.nb_segments()

    def nb_Qddot(self) -> int:
        """
        This function returns the number of generalized accelerations in the model
        """
        return 12 * self.nb_segments()

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

        pass

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
        pass

    @abstractmethod
    def rigid_body_constraint_jacobian_derivative(self, Qdot: NaturalVelocities) -> np.ndarray:
        """
        This function returns the derivative of the Jacobian matrix of the rigid body constraints denoted Kr_dot

        Parameters
        ----------
        Qdot : NaturalVelocities
            The natural velocities of the segment [nb_segments * 12, 1]

        Returns
        -------
        np.ndarray
            The derivative of the Jacobian matrix of the rigid body constraints [6, 12]
        """

        pass

    @abstractmethod
    def joint_constraints(self, Q: NaturalCoordinates):
        """
        This function returns the joint constraints of all joints, denoted Phi_k
        as a function of the natural coordinates Q.

        Returns
        -------
            Joint constraints of the segment [nb_joint_constraints, 1]
        """

        pass

    @abstractmethod
    def joint_constraints_jacobian(self, Q: NaturalCoordinates):
        """
        This function returns the joint constraints of all joints, denoted K_k

        Returns
        -------
            Joint constraints of the segment [nb_joint_constraints, 1]
        """

        pass

    @abstractmethod
    def _update_mass_matrix(self):
        """
        This function computes the generalized mass matrix of the system, denoted G

        Returns
        -------
        np.ndarray
            generalized mass matrix of the segment [12 * nb_segments, 12 * nb_segment]
        """
        pass

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
        pass

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
        pass

    def lagrangian(self, Q: NaturalCoordinates, Qdot: NaturalVelocities) -> np.ndarray:
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
        pass

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
        pass

    @abstractmethod
    def markers_constraints_jacobian(self):
        """
        This function returns the Jacobian matrix the markers constraints, denoted k_m.

        Returns
        -------
            Joint constraints of the marker [nb_markers x 3, nb_Q]
        """
        pass

    # @abstractmethod
    # def holononmic_constraints(self, Q: NaturalCoordinates):
    #     """
    #     This function returns the holonomic constraints of the system, denoted Phi_h
    #     as a function of the natural coordinates Q. They are organized as follow, for each segment:
    #         [Phi_k_0, Phi_r_0, Phi_k_1, Phi_r_1, ..., Phi_k_n, Phi_r_n]
    #
    #     Parameters
    #     ----------
    #     Q : NaturalCoordinates
    #         The natural coordinates of the segment [12 * nb_segments, 1]
    #
    #     Returns
    #     -------
    #         Holonomic constraints of the segment [nb_holonomic_constraints, 1]
    #     """
    #     pass
    #
    # @abstractmethod
    # def holonomic_constraints_jacobian(self, Q: NaturalCoordinates):
    #     """
    #     This function returns the Jacobian matrix the holonomic constraints, denoted k_h.
    #     They are organized as follow, for each segmen, the rows of the matrix are:
    #     [Phi_k_0, Phi_r_0, Phi_k_1, Phi_r_1, ..., Phi_k_n, Phi_r_n]
    #
    #     Parameters
    #     ----------
    #     Q : NaturalCoordinates
    #         The natural coordinates of the segment [12 * nb_segments, 1]
    #
    #     Returns
    #     -------
    #         Joint constraints of the holonomic constraints [nb_holonomic_constraints, 12 * nb_segments]
    #     """
    #     pass
