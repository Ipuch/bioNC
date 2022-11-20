import numpy as np

from bionc.protocols.natural_coordinates import NaturalCoordinates
from bionc.bionc_numpy.natural_velocities import NaturalVelocities
from ..protocols.biomechanical_model import GenericBiomechanicalModel


class BiomechanicalModel(GenericBiomechanicalModel):
    def __init__(self):
        super().__init__()

    def rigid_body_constraints(self, Q: NaturalCoordinates) -> np.ndarray:
        """
        This function returns the rigid body constraints of all segments, denoted Phi_r
        as a function of the natural coordinates Q.

        Returns
        -------
        np.ndarray
            Rigid body constraints of the segment [6 * nb_segments, 1]
        """

        Phi_r = np.zeros(6 * self.nb_segments())
        for i, segment_name in enumerate(self.segments):
            idx = slice(6 * i, 6 * (i + 1))
            Phi_r[idx] = self.segments[segment_name].rigid_body_constraint(Q.vector(i))

        return Phi_r

    def rigid_body_constraints_jacobian(self, Q: NaturalCoordinates) -> np.ndarray:
        """
        This function returns the rigid body constraints of all segments, denoted K_r
        as a function of the natural coordinates Q.

        Returns
        -------
        np.ndarray
            Rigid body constraints of the segment [6 * nb_segments, nbQ]
        """

        K_r = np.zeros((6 * self.nb_segments(), Q.shape[0]))
        for i, segment_name in enumerate(self.segments):
            idx_row = slice(6 * i, 6 * (i + 1))
            idx_col = slice(12 * i, 12 * (i + 1))
            K_r[idx_row, idx_col] = self.segments[segment_name].rigid_body_constraint_jacobian(Q.vector(i))

        return K_r

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

        Kr_dot = np.zeros((6 * self.nb_segments(), Qdot.shape[0]))
        for i, segment_name in enumerate(self.segments):
            idx_row = slice(6 * i, 6 * (i + 1))
            idx_col = slice(12 * i, 12 * (i + 1))
            Kr_dot[idx_row, idx_col] = self.segments[segment_name].rigid_body_constraint_jacobian_derivative(
                Qdot.vector(i)
            )

        return Kr_dot

    def joint_constraints(self, Q: NaturalCoordinates) -> np.ndarray:
        """
        This function returns the joint constraints of all joints, denoted Phi_k
        as a function of the natural coordinates Q.

        Returns
        -------
        np.ndarray
            Joint constraints of the segment [nb_joint_constraints, 1]
        """

        Phi_k = np.zeros(self.nb_joint_constraints())
        nb_constraints = 0
        for joint_name, joint in self.joints.items():
            idx = slice(nb_constraints, nb_constraints + joint.nb_constraints)

            Q_parent = Q.vector(self.segments[joint.parent.name].index)
            Q_child = Q.vector(self.segments[joint.child.name].index)
            Phi_k[idx] = joint.constraint(Q_parent, Q_child)

            nb_constraints += self.joints[joint_name].nb_constraints

        return Phi_k

    def _update_mass_matrix(self):
        """
        This function computes the generalized mass matrix of the system, denoted G

        Returns
        -------
        np.ndarray
            generalized mass matrix of the segment [12 * nbSegment x 12 * * nbSegment]
        """
        G = np.zeros((12 * self.nb_segments(), 12 * self.nb_segments()))
        for i, segment_name in enumerate(self.segments):
            Gi = self.segments[segment_name].mass_matrix
            if Gi is None:
                # mass matrix is None if one the segment doesn't have any inertial properties
                self._mass_matrix = None
                return
            idx = slice(12 * i, 12 * (i + 1))
            G[idx, idx] = self.segments[segment_name].mass_matrix

        self._mass_matrix = G

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


# def kinematicConstraints(self, Q):
#     # Method to calculate the kinematic constraints

# def forwardDynamics(self, Q, Qdot):
#
#     return Qddot, lambdas

# def inverseDynamics(self):
