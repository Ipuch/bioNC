import numpy as np


from abc import ABC, abstractmethod
from bionc.protocols.natural_coordinates import NaturalCoordinates
from bionc.protocols.natural_velocities import NaturalVelocities


class AbstractBiomechanicalModel:
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
    def nb_joints(self):
        """
        This function returns the number of joints in the model

        Returns
        -------
        int
            number of joints in the model
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


# def kinematicConstraints(self, Q):
#     # Method to calculate the kinematic constraints

# def forwardDynamics(self, Q, Qdot):
#
#     return Qddot, lambdas

# def inverseDynamics(self):
