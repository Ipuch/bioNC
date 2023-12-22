from casadi import MX
from typing import Any

from .natural_coordinates import NaturalCoordinates
from .natural_velocities import NaturalVelocities
from ..protocols import GenericBiomechanicalModelJoints


class BiomechanicalModelJoints(GenericBiomechanicalModelJoints):
    def __init__(self, joints: dict[str, Any] = None):
        super().__init__(joints=joints)

    def constraints(self, Q: NaturalCoordinates) -> MX:
        """
        This function returns the joint constraints of all joints, denoted Phi_k
        as a function of the natural coordinates Q.

        Parameters
        ----------
        Q : NaturalCoordinates
            The natural coordinates of the segment [12 * nb_segments, 1]

        Returns
        -------
        MX
            Joint constraints of the segment [nb_joint_constraints, 1]
        """

        Phi_k = MX.zeros(self.nb_constraints)

        for joint_name, joint in self.joints_with_constraints.items():
            constraint_slice = self.constraints_index(joint.index)

            Q_parent = (
                None if joint.parent is None else Q.vector(joint.parent.index)
            )  # if the joint is a joint with the ground, the parent is None
            Q_child = Q.vector(joint.child.index)

            Phi_k[constraint_slice] = joint.constraint(Q_parent, Q_child)

        return Phi_k

    def constraints_jacobian(self, Q: NaturalCoordinates) -> MX:
        """
        This function returns the joint constraints of all joints, denoted K_k
        as a function of the natural coordinates Q.

        Parameters
        ----------
        Q : NaturalCoordinates
            The natural coordinates of the segment [12 * nb_segments, 1]

        Returns
        -------
        MX
            Joint constraints of the segment [nb_joint_constraints, nbQ]
        """

        K_k = MX.zeros((self.nb_constraints, Q.shape[0]))

        for joint_name, joint in self.joints_with_constraints.items():
            idx_row = self.constraints_index(joint.index)

            idx_col_child = joint.child.coordinates_slice
            idx_col_parent = joint.parent.coordinates_slice if joint.parent is not None else None

            Q_parent = (
                None if joint.parent is None else Q.vector(joint.parent.index)
            )  # if the joint is a joint with the ground, the parent is None
            Q_child = Q.vector(joint.child.index)

            if joint.parent is not None:  # If the joint is not a ground joint
                K_k[idx_row, idx_col_parent] = joint.parent_constraint_jacobian(Q_parent, Q_child)

            K_k[idx_row, idx_col_child] = joint.child_constraint_jacobian(Q_parent, Q_child)

        return K_k

    def constraints_jacobian_derivative(self, Qdot: NaturalVelocities) -> MX:
        """
        This function returns the derivative of the Jacobian matrix of the joint constraints denoted K_k_dot

        Parameters
        ----------
        Qdot : NaturalVelocities
            The natural velocities of the segment [12 * nb_segments, 1]

        Returns
        -------
        MX
            The derivative of the Jacobian matrix of the joint constraints [nb_joint_constraints, 12 * nb_segments]
        """

        K_k_dot = MX.zeros((self.nb_constraints, Qdot.shape[0]))

        for joint_name, joint in self.joints_with_constraints.items():
            idx_row = self.constraints_index(joint.index)

            idx_col_child = joint.child.coordinates_slice
            idx_col_parent = joint.parent.coordinates_slice if joint.parent is not None else None

            Qdot_parent = (
                None if joint.parent is None else Qdot.vector(joint.parent.index)
            )  # if the joint is a joint with the ground, the parent is None
            Qdot_child = Qdot.vector(joint.child.index)

            if joint.parent is not None:  # If the joint is not a ground joint
                K_k_dot[idx_row, idx_col_parent] = joint.parent_constraint_jacobian_derivative(Qdot_parent, Qdot_child)

            K_k_dot[idx_row, idx_col_child] = joint.child_constraint_jacobian_derivative(Qdot_parent, Qdot_child)

        return K_k_dot
