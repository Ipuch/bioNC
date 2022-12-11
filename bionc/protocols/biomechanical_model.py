import numpy as np
from casadi import MX

from typing import Union, Any
from abc import ABC, abstractmethod
import dill as pickle

from bionc.protocols.natural_coordinates import NaturalCoordinates
from bionc.protocols.natural_velocities import NaturalVelocities


class AbstractBiomechanicalModel(ABC):
    """
    This class is the base class for all biomechanical models. It contains the segments and the joints of the model.

    Methods
    ----------


    """

    @abstractmethod
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

    @abstractmethod
    def __setitem__(self, name: str, segment: "NaturalSegment"):
        """
        This function adds a segment to the model

        Parameters
        ----------
        name : str
            Name of the segment
        segment : NaturalSegment
            The segment to add
        """

    @abstractmethod
    def __str__(self):
        """
        This function returns a string representation of the model
        """

    @abstractmethod
    def save(self, filename: str):
        """
        This function saves the model to a file

        Parameters
        ----------
        filename : str
            The path to the file
        """

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
        AbstractBiomechanicalModel
            The loaded model
        """

    @abstractmethod
    def nb_segments(self):
        """
        This function returns the number of segments in the model
        """

    @abstractmethod
    def nb_markers(self):
        """
        This function returns the number of markers in the model

        Returns
        -------
        int
            number of markers in the model
        """

    @abstractmethod
    def nb_markers_technical(self):
        """
        This function returns the number of technical markers in the model

        Returns
        -------
        int
            number of technical markers in the model
        """

    @abstractmethod
    def nb_joints(self):
        """
        This function returns the number of joints in the model

        Returns
        -------
        int
            number of joints in the model
        """

    @abstractmethod
    def nb_joint_constraints(self):
        """
        This function returns the number of joint constraints in the model

        Returns
        -------
        int
            number of joint constraints in the model
        """

    @abstractmethod
    def nb_Q(self):
        """
        This function returns the number of natural coordinates of the model
        """

    @abstractmethod
    def nb_Qdot(self):
        """
        This function returns the number of natural velocities of the model
        """

    @abstractmethod
    def nb_Qddot(self):
        """
        This function returns the number of natural accelerations of the model
        """

    @abstractmethod
    def rigid_body_constraints(self, Q: NaturalCoordinates):
        """
        This function returns the rigid body constraints of all segments, denoted Phi_r
        as a function of the natural coordinates Q.

        Returns
        -------
        Rigid body constraints of the segment [6 * nb_segments, 1]
        """

    @abstractmethod
    def rigid_body_constraints_jacobian(self, Q: NaturalCoordinates):
        """
        This function returns the rigid body constraints of all segments, denoted K_r
        as a function of the natural coordinates Q.

        Returns
        -------
        Rigid body constraints of the segment [6 * nb_segments, nbQ]
        """

    @abstractmethod
    def rigid_body_constraint_jacobian_derivative(self, Qdot: NaturalVelocities):
        """
        This function returns the derivative of the Jacobian matrix of the rigid body constraints denoted Kr_dot

        Parameters
        ----------
        Qdot : NaturalVelocities
            The natural velocities of the segment [12, 1]

        Returns
        -------
            The derivative of the Jacobian matrix of the rigid body constraints [6, 12]
        """

    @abstractmethod
    def joint_constraints(self, Q: NaturalCoordinates):
        """
        This function returns the kinematic constraints of all joints, denoted Phi_k
        as a function of the natural coordinates Q.

        Returns
        -------
        Kinematic constraints of the joints [nb_joints_constraints, 1]
        """
        ...

    @abstractmethod
    def _update_mass_matrix(self):
        """
        This function computes the generalized mass matrix of the system, denoted G

        Returns
        -------
        np.ndarray
            generalized mass matrix of the segment [12 * nbSegment x 12 * * nbSegment]
        """

    @abstractmethod
    def mass_matrix(self):
        """
        This function returns the generalized mass matrix of the system, denoted G

        Returns
        -------
        np.ndarray
            generalized mass matrix of the segment [12 * nbSegment x 12 * * nbSegment]

        """

    def markers_constraints(self, markers: np.ndarray | MX, Q: NaturalCoordinates):
        """
        This function returns the marker constraints of all segments, denoted Phi_r
        as a function of the natural coordinates Q.

        markers : np.ndarray | MX
            The markers positions [3,nb_markers]

        Q : NaturalCoordinates
            The natural coordinates of the segment [12 x n, 1]

        Returns
        -------
            Rigid body constraints of the segment [nb_markers x 3, 1]
        """

    def markers_constraints_jacobian(self):
        """
        This function returns the Jacobian matrix the markers constraints, denoted k_m.

        Returns
        -------
            Joint constraints of the marker [nb_markers x 3, nb_Q]
        """


class GenericBiomechanicalModel(AbstractBiomechanicalModel):
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
        from .joint import AbstractJoint  # Imported here to prevent from circular imports

        self.segments: dict[str:AbstractNaturalSegment, ...] = {} if segments is None else segments
        self.joints: dict[str:AbstractJoint, ...] = {} if joints is None else joints
        # From Pythom 3.7 the insertion order in a dict is preserved. This is important because when writing a new
        # the order of the segment matters
        self._mass_matrix = self._update_mass_matrix()

    def __getitem__(self, name: str):
        return self.segments[name]

    def __setitem__(self, name: str, segment: Any):
        if segment.name == name:  # Make sure the name of the segment fits the internal one
            segment.set_index(len(self.segments))
            self.segments[name] = segment
            self._update_mass_matrix()  # Update the generalized mass matrix
        else:
            raise ValueError("The name of the segment does not match the name of the segment")

    def __str__(self):
        out_string = "version 4\n\n"
        for name in self.segments:
            out_string += str(self.segments[name])
            out_string += "\n\n\n"  # Give some space between segments
        return out_string

    def save(self, filename: str):
        with open(filename, "wb") as file:
            pickle.dump(self, file)

    @staticmethod
    def load(filename: str):
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
        if joint["parent"] not in self.segments.keys():
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
        joint["parent"] = self.segments[joint["parent"]]
        # replace child field by the child segment
        joint["child"] = self.segments[joint["child"]]

        self.joints[joint["name"]] = joint_type.value(**joint)

    def nb_segments(self):
        return len(self.segments)

    def nb_markers(self):
        nb_markers = 0
        for key in self.segments:
            nb_markers += self.segments[key].nb_markers()
        return nb_markers

    def nb_markers_technical(self):
        nb_markers = 0
        for key in self.segments:
            nb_markers += self.segments[key].nb_markers_technical()
        return nb_markers

    def marker_names(self):
        marker_names = []
        for key in self.segments:
            marker_names += self.segments[key].marker_names()
        return marker_names

    def marker_names_technical(self):
        marker_names = []
        for key in self.segments:
            marker_names += self.segments[key].marker_names_technical()
        return marker_names

    def nb_joints(self):
        return len(self.joints)

    def nb_joint_constraints(self):
        nb_joint_constraints = 0
        for joint_name, joint in self.joints.items():
            nb_joint_constraints += joint.nb_constraints
        return nb_joint_constraints

    def nb_rigid_body_constraints(self):
        return 6 * self.nb_segments()

    def nb_holonomic_constraints(self):
        return self.nb_joint_constraints() + self.nb_rigid_body_constraints()

    def nb_Q(self):
        return 12 * self.nb_segments()

    def nb_Qdot(self):
        return 12 * self.nb_segments()

    def nb_Qddot(self):
        return 12 * self.nb_segments()

    @property
    def mass_matrix(self):
        """
        This function returns the generalized mass matrix of the system, denoted G

        Returns
        -------
        np.ndarray
            generalized mass matrix of the segment [12 * nbSegment x 12 * * nbSegment]

        """
        return self._mass_matrix

    def rigid_body_constraints(self, Q: NaturalCoordinates) -> np.ndarray:
        """
        This function returns the rigid body constraints of all segments, denoted Phi_r
        as a function of the natural coordinates Q.

        Returns
        -------
        np.ndarray
            Rigid body constraints of the segment [6 * nb_segments, 1]
        """

        pass

    def rigid_body_constraints_jacobian(self, Q: NaturalCoordinates) -> np.ndarray:
        """
        This function returns the rigid body constraints of all segments, denoted K_r
        as a function of the natural coordinates Q.

        Returns
        -------
        np.ndarray
            Rigid body constraints of the segment [6 * nb_segments, nbQ]
        """

        pass

    def rigid_body_constraint_jacobian_derivative(self, Qdot: NaturalVelocities) -> np.ndarray:
        """
        This function returns the derivative of the Jacobian matrix of the rigid body constraints denoted Kr_dot

        Parameters
        ----------
        Qdot : NaturalVelocities
            The natural velocities of the segment [12, 1]

        Returns
        -------
        np.ndarray
            The derivative of the Jacobian matrix of the rigid body constraints [6, 12]
        """

        pass

    def joint_constraints(self, Q: NaturalCoordinates):
        """
        This function returns the joint constraints of all joints, denoted Phi_k
        as a function of the natural coordinates Q.

        Returns
        -------
            Joint constraints of the segment [nb_joint_constraints, 1]
        """

        pass

    def joint_constraints_jacobian(self, Q: NaturalCoordinates):
        """
        This function returns the joint constraints of all joints, denoted K_k

        Returns
        -------
            Joint constraints of the segment [nb_joint_constraints, 1]
        """

        pass

    def _update_mass_matrix(self):
        """
        This function computes the generalized mass matrix of the system, denoted G

        Returns
        -------
        np.ndarray
            generalized mass matrix of the segment [12 * nbSegment x 12 * * nbSegment]
        """
        pass

    def kinetic_energy(self, Qdot: NaturalVelocities) -> Union[np.ndarray, MX]:
        """
        This function computes the kinetic energy of the system

        Parameters
        ----------
        Qdot : NaturalVelocities
            The natural velocities of the segment [12 * n, 1]

        Returns
        -------
        Union[np.ndarray, MX]
            The kinetic energy of the system
        """
        pass

    def potential_energy(self, Q: NaturalCoordinates) -> Union[np.ndarray, MX]:
        """
        This function computes the potential energy of the system

        Parameters
        ----------
        Q : NaturalCoordinates
            The natural coordinates of the segment [12 * n, 1]

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
            The natural coordinates of the segment [12, 1]
        Qdot : NaturalVelocities
            The natural velocities of the segment [12, 1]

        Returns
        -------
        MX
            The lagrangian of the system
        """

        return self.kinetic_energy(Qdot) - self.potential_energy(Q)

    def markers_constraints(self, markers: np.ndarray | MX, Q: NaturalCoordinates):
        pass

    def markers_constraints_jacobian(self):
        pass
