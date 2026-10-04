import numpy as np

from .cartesian_vector import CartesianVector
from .natural_coordinates import SegmentNaturalCoordinates
from .natural_segment import NaturalSegment
from .natural_vector import NaturalVector
from .natural_velocities import SegmentNaturalVelocities
from .rotations import euler_axes_matrix
from ..protocols.joint import JointBase
from ..utils.enums import NaturalAxis, CartesianAxis, EulerSequence, TransformationMatrixType


class GroundJoint:
    """
    The public interface to joints with the ground as parent segment.
    """

    class Free(JointBase):
        """
        This joint is defined by 0 constraints to let the joint to be free with the world.
        """

        def __init__(
            self,
            name: str,
            child: NaturalSegment,
            index: int = None,
            projection_basis: EulerSequence = None,
            child_basis: TransformationMatrixType = None,
        ):
            super(GroundJoint.Free, self).__init__(
                name,
                None,
                child,
                index,
                projection_basis,
                None,
                child_basis,
                (CartesianAxis.X, CartesianAxis.Y, CartesianAxis.Z),
            )

        def constraint(self, Q_parent: SegmentNaturalCoordinates, Q_child: SegmentNaturalCoordinates) -> np.ndarray:
            return None

        def parent_constraint_jacobian(
            self, Q_parent: SegmentNaturalCoordinates, Q_child: SegmentNaturalCoordinates
        ) -> np.ndarray:
            return None

        def child_constraint_jacobian(
            self, Q_parent: SegmentNaturalCoordinates, Q_child: SegmentNaturalCoordinates
        ) -> np.ndarray:
            return None

        def constraint_jacobian(
            self, Q_parent: SegmentNaturalCoordinates, Q_child: SegmentNaturalCoordinates
        ) -> np.ndarray:
            return None

        def constraint_acceleration_bias(
            self, Qdot_parent: SegmentNaturalVelocities, Qdot_child: SegmentNaturalVelocities
        ) -> np.ndarray:
            """
            Ground Free joint has no constraints, so no acceleration bias.

            Returns
            -------
            None
            """
            return None

        def dof_axes(self, Q_parent: SegmentNaturalCoordinates, Q_child: SegmentNaturalCoordinates):
            """3 translations along the global axes, then 3 rotations about the projection_basis Euler axes"""
            R_child = self.child.segment_coordinates_system(Q_child, self.child_basis).rot
            return np.eye(3), euler_axes_matrix(np.eye(3), R_child, self.projection_basis)

        def to_mx(self):
            """
            This function returns the joint as a mx joint

            Returns
            -------
            JointBase
                The joint as a mx joint
            """
            from ..bionc_casadi.joints_with_ground import GroundJoint as CasadiGroundJoint

            return CasadiGroundJoint.Free(
                name=self.name,
                child=self.child.to_mx(),
                index=self.index,
                projection_basis=self.projection_basis,
                child_basis=self.child_basis,
            )

    class Hinge(JointBase):
        """
        This joint is defined by 3 constraints to pivot around an axis of the inertial coordinate system
        defined by two angles.
        """

        def __init__(
            self,
            name: str,
            child: NaturalSegment,
            parent_axis: tuple[CartesianAxis] | list[CartesianAxis],
            child_axis: tuple[NaturalAxis] | list[NaturalAxis],
            theta: tuple[float] | list[float] | np.ndarray = None,
            index: int = None,
            projection_basis: EulerSequence = None,
            child_basis: TransformationMatrixType = None,
        ):
            super(GroundJoint.Hinge, self).__init__(name, None, child, index, projection_basis, None, child_basis, None)

            # check size and type of parent axis
            if not isinstance(parent_axis, (tuple, list)) or len(parent_axis) != 2:
                raise TypeError("parent_axis should be a tuple or list with 2 CartesianAxis")
            if not all(isinstance(axis, CartesianAxis) for axis in parent_axis):
                raise TypeError("parent_axis should be a tuple or list with 2 CartesianAxis")

            # check size and type of child axis
            if not isinstance(child_axis, (tuple, list)) or len(child_axis) != 2:
                raise TypeError("child_axis should be a tuple or list with 2 NaturalAxis")
            if not all(isinstance(axis, NaturalAxis) for axis in child_axis):
                raise TypeError("child_axis should be a tuple or list with 2 NaturalAxis")

            # check size and type of theta
            if theta is None:
                theta = np.ones(2) * np.pi / 2
            if not isinstance(theta, (tuple, list, np.ndarray)) or len(theta) != 2:
                raise TypeError("theta should be a tuple or list with 2 float")

            # todo: there should be a check on the euler sequence and transformation matrix type here
            #   with respected to the chosen parent and child axis

            self.parent_axis = parent_axis

            self.parent_vector = [CartesianVector.axis(axis) for axis in parent_axis]

            self.child_axis = child_axis

            self.child_vector = [NaturalVector.axis(axis) for axis in child_axis]

            self.theta = theta

            self.nb_constraints = 5

        def constraint(self, Q_parent: SegmentNaturalCoordinates, Q_child: SegmentNaturalCoordinates) -> np.ndarray:
            """
            This function returns the kinematic constraints of the joint, denoted Phi_k
            as a function of the natural coordinates Q_parent and Q_child.

            Returns
            -------
            np.ndarray
                Kinematic constraints of the joint [5, 1]
            """
            constraint = np.zeros(self.nb_constraints)
            constraint[:3] = -Q_child.rp  # NOTE: only fixed with the origin of the inertial coordinate system
            # todo: extend to any point of the inertial coordinate system

            for i in range(2):
                constraint[i + 3] = np.dot(
                    self.parent_vector[i],
                    Q_child.axis(self.child_axis[i]),
                ) - np.cos(self.theta[i])

            return constraint

        def parent_constraint_jacobian(
            self, Q_parent: SegmentNaturalCoordinates, Q_child: SegmentNaturalCoordinates
        ) -> np.ndarray:
            return None

        def child_constraint_jacobian(
            self, Q_parent: SegmentNaturalCoordinates, Q_child: SegmentNaturalCoordinates
        ) -> np.ndarray:
            K_k_child = np.zeros((self.nb_constraints, 12))
            K_k_child[:3, 3:6] = -np.eye(3)

            for i in range(2):
                K_k_child[i + 3, :] = np.squeeze((self.parent_vector[i]).T @ self.child_vector[i].interpolate().rot)

            return K_k_child

        def constraint_jacobian(
            self, Q_parent: SegmentNaturalCoordinates, Q_child: SegmentNaturalCoordinates
        ) -> np.ndarray:
            """
            This function returns the kinematic constraints of the joint, denoted K_k
            as a function of the natural coordinates Q_parent and Q_child.

            Returns
            -------
            np.ndarray
                joint constraints jacobian of the child segment [5, 12]
            """

            return self.child_constraint_jacobian(Q_parent, Q_child)

        def constraint_acceleration_bias(
            self, Qdot_parent: SegmentNaturalVelocities, Qdot_child: SegmentNaturalVelocities
        ) -> np.ndarray:
            """
            Compute the acceleration bias (quadratic velocity terms) for this ground Hinge joint.

            All constraints have constant Jacobians w.r.t. the child coordinates
            (the parent is the ground, a constant). Therefore the Hessian is zero
            and the bias = qdot^T H qdot = 0.

            Returns
            -------
            np.ndarray
                Acceleration bias vector [5, 1]. All zeros.
            """
            return np.zeros((self.nb_constraints, 1))

        def dof_axes(self, Q_parent: SegmentNaturalCoordinates, Q_child: SegmentNaturalCoordinates):
            """1 rotation about the global axis shared by the two constraints"""
            if self.parent_axis[0] != self.parent_axis[1]:
                raise NotImplementedError(f"The hinge {self.name} must use the same parent axis in both constraints")
            return None, np.asarray(self.parent_vector[0], dtype=float).reshape(3, 1)

        def to_mx(self):
            """
            This function returns the joint as a mx joint

            Returns
            -------
            JointBase
                The joint as a mx joint
            """
            from ..bionc_casadi.joints_with_ground import GroundJoint as CasadiGroundJoint

            return CasadiGroundJoint.Hinge(
                name=self.name,
                child=self.child.to_mx(),
                index=self.index,
                parent_axis=self.parent_axis,
                child_axis=self.child_axis,
                theta=self.theta,
                projection_basis=self.projection_basis,
                child_basis=self.child_basis,
            )

    class Universal(JointBase):
        """
        This class is to define a Universal joint between two segments.

        Methods
        -------
        constraint(Q_parent, Q_child)
            This function returns the kinematic constraints of the joint, denoted Phi_k
            as a function of the natural coordinates Q_parent and Q_child.
        constraint_jacobian(Q_parent, Q_child)
            This function returns the jacobian of the kinematic constraints of the joint, denoted Phi_k
            as a function of the natural coordinates Q_parent and Q_child.
        to_mx()
            This function returns the joint as a mx joint to be used with the bionc_casadi package.

        Attributes
        ----------
        name : str
            Name of the joint
        parent : NaturalSegment
            Parent segment of the joint
        child : NaturalSegment
            Child segment of the joint
        parent_axis : NaturalAxis
            Axis of the parent segment
        child_axis : NaturalAxis
            Axis of the child segment
        theta : float
            Angle between the two axes
        """

        def __init__(
            self,
            name: str,
            child: NaturalSegment,
            parent_axis: CartesianAxis,
            child_axis: NaturalAxis,
            theta: float,
            index: int,
            projection_basis: EulerSequence = None,
            child_basis: TransformationMatrixType = None,
        ):
            super(GroundJoint.Universal, self).__init__(
                name, None, child, index, projection_basis, None, child_basis, None
            )

            # todo: there should be a check on the euler sequence and transformation matrix type here
            #   with respected to the chosen parent and child axis

            self.parent_axis = parent_axis
            self.parent_vector = CartesianVector.axis(self.parent_axis)

            self.child_axis = child_axis
            self.child_vector = NaturalVector.axis(self.child_axis)

            self.theta = theta

            self.nb_constraints = 4

        def constraint(self, Q_parent: SegmentNaturalCoordinates, Q_child: SegmentNaturalCoordinates) -> np.ndarray:
            """
            This function returns the kinematic constraints of the joint, denoted Phi_k
            as a function of the natural coordinates Q_parent and Q_child.

            Returns
            -------
            np.ndarray
                Kinematic constraints of the joint [4, 1]
            """
            constraint = np.zeros(self.nb_constraints)
            constraint[:3] = -Q_child.rp
            constraint[3] = np.dot(self.parent_vector, Q_child.axis(self.child_axis)) - np.cos(self.theta)

            return constraint

        def parent_constraint_jacobian(
            self, Q_parent: SegmentNaturalCoordinates, Q_child: SegmentNaturalCoordinates
        ) -> np.ndarray:
            return None

        def child_constraint_jacobian(
            self, Q_parent: SegmentNaturalCoordinates, Q_child: SegmentNaturalCoordinates
        ) -> np.ndarray:
            K_k_child = np.zeros((self.nb_constraints, 12))
            K_k_child[:3, 3:6] = -np.eye(3)

            K_k_child[3, :] = np.squeeze(self.parent_vector.T @ self.child_vector.interpolate().rot)

            return K_k_child

        def constraint_jacobian(
            self, Q_parent: SegmentNaturalCoordinates, Q_child: SegmentNaturalCoordinates
        ) -> tuple[np.ndarray, np.ndarray]:
            """
            This function returns the kinematic constraints of the joint, denoted K_k
            as a function of the natural coordinates Q_parent and Q_child.

            Returns
            -------
            tuple[np.ndarray, np.ndarray]
                joint constraints jacobian of the parent and child segment [4, 12] and [4, 12]
            """

            return self.child_constraint_jacobian(Q_parent, Q_child)

        def constraint_acceleration_bias(
            self, Qdot_parent: SegmentNaturalVelocities, Qdot_child: SegmentNaturalVelocities
        ) -> np.ndarray:
            """
            Compute the acceleration bias (quadratic velocity terms) for this ground Universal joint.

            All constraints have constant Jacobians w.r.t. the child coordinates
            (the parent is the ground, a constant). Therefore the Hessian is zero
            and the bias = qdot^T H qdot = 0.

            Returns
            -------
            np.ndarray
                Acceleration bias vector [4, 1]. All zeros.
            """
            return np.zeros((self.nb_constraints, 1))

        def dof_axes(self, Q_parent: SegmentNaturalCoordinates, Q_child: SegmentNaturalCoordinates):
            """2 rotations, about the global axis then the child axis"""
            child_axis = np.asarray(Q_child.axis(self.child_axis)).reshape(3)
            return None, np.column_stack(
                (np.asarray(self.parent_vector, dtype=float).reshape(3), child_axis / np.linalg.norm(child_axis))
            )

        def to_mx(self):
            """
            This function returns the joint as a mx joint

            Returns
            -------
            JointBase
                The joint as a mx joint
            """
            from ..bionc_casadi.joints_with_ground import GroundJoint as CasadiJoint

            return CasadiJoint.Universal(
                name=self.name,
                child=self.child.to_mx(),
                index=self.index,
                parent_axis=self.parent_axis,
                child_axis=self.child_axis,
                theta=self.theta,
                projection_basis=self.projection_basis,
                child_basis=self.child_basis,
            )

    class Spherical(JointBase):
        """
        This joint is defined by 3 constraints to pivot around an axis of the inertial coordinate system
        defined by two angles.
        """

        def __init__(
            self,
            name: str,
            child: NaturalSegment,
            ground_application_point: np.ndarray = None,
            index: int = None,
            projection_basis: EulerSequence = None,
            child_basis: TransformationMatrixType = None,
        ):
            super(GroundJoint.Spherical, self).__init__(
                name, None, child, index, projection_basis, None, child_basis, None
            )
            self.nb_constraints = 3
            self.ground_application_point = (
                ground_application_point if ground_application_point is not None else np.zeros(3)
            )

        def constraint(self, Q_parent: SegmentNaturalCoordinates, Q_child: SegmentNaturalCoordinates) -> np.ndarray:
            """
            This function returns the kinematic constraints of the joint, denoted Phi_k
            as a function of the natural coordinates Q_parent and Q_child.

            Returns
            -------
            np.ndarray
                Kinematic constraints of the joint [3, 1]
            """
            constraint = self.ground_application_point - Q_child.rp

            return constraint

        def parent_constraint_jacobian(
            self, Q_parent: SegmentNaturalCoordinates, Q_child: SegmentNaturalCoordinates
        ) -> np.ndarray:
            return None

        def child_constraint_jacobian(
            self, Q_parent: SegmentNaturalCoordinates, Q_child: SegmentNaturalCoordinates
        ) -> np.ndarray:
            K_k_child = np.zeros((self.nb_constraints, 12))
            K_k_child[:3, 3:6] = -np.eye(3)

            return K_k_child

        def constraint_jacobian(
            self, Q_parent: SegmentNaturalCoordinates, Q_child: SegmentNaturalCoordinates
        ) -> np.ndarray:
            """
            This function returns the kinematic constraints of the joint, denoted K_k
            as a function of the natural coordinates Q_parent and Q_child.

            Returns
            -------
            np.ndarray
                joint constraints jacobian of the child segment [3, 12]
            """

            return self.child_constraint_jacobian(Q_parent, Q_child)

        def constraint_acceleration_bias(
            self, Qdot_parent: SegmentNaturalVelocities, Qdot_child: SegmentNaturalVelocities
        ) -> np.ndarray:
            """
            Compute the acceleration bias (quadratic velocity terms) for this ground Spherical joint.

            The constraint is:
              phi = ground_point - rp_child  (linear in Q_child, constant parent)

            Since the Jacobian is constant, the Hessian is zero.
            Therefore: bias = qdot^T H qdot = 0

            Returns
            -------
            np.ndarray
                Acceleration bias vector [3, 1]. All zeros.
            """
            return np.zeros((self.nb_constraints, 1))

        def dof_axes(self, Q_parent: SegmentNaturalCoordinates, Q_child: SegmentNaturalCoordinates):
            """3 rotations about the projection_basis Euler axes"""
            R_child = self.child.segment_coordinates_system(Q_child, self.child_basis).rot
            return None, euler_axes_matrix(np.eye(3), R_child, self.projection_basis)

        def to_mx(self):
            """
            This function returns the joint as a mx joint

            Returns
            -------
            JointBase
                The joint as a mx joint
            """
            from ..bionc_casadi.joints_with_ground import GroundJoint as CasadiGroundJoint

            return CasadiGroundJoint.Spherical(
                name=self.name,
                child=self.child.to_mx(),
                ground_application_point=self.ground_application_point,
                index=self.index,
                projection_basis=self.projection_basis,
                child_basis=self.child_basis,
            )

    class Weld(JointBase):
        """
        This joint welds the child segment to the ground with 6 linear constraints S (Q_ref - Q_child) = 0,
        see weld_selection_matrix. Give Q_child_ref to fully weld the segment, or only rp_child_ref and
        rd_child_ref to fix rp and rd (the rotation about rp - rd then stays free).
        """

        def __init__(
            self,
            name: str,
            child: NaturalSegment,
            rp_child_ref: SegmentNaturalCoordinates | np.ndarray = None,
            rd_child_ref: SegmentNaturalCoordinates | np.ndarray = None,
            Q_child_ref: SegmentNaturalCoordinates | np.ndarray = None,
            index: int = None,
            projection_basis: EulerSequence = None,
            child_basis: TransformationMatrixType = None,
        ):
            super(GroundJoint.Weld, self).__init__(name, None, child, index, projection_basis, None, child_basis, None)

            if Q_child_ref is not None:
                Q_child_ref = np.asarray(Q_child_ref, dtype=float).reshape(12)
                rp_child_ref, rd_child_ref = Q_child_ref[3:6], Q_child_ref[6:9]
            elif rp_child_ref is None or rd_child_ref is None:
                raise ValueError("Q_child_ref, or rp_child_ref and rd_child_ref, must be given for a Weld joint")

            self.rp_child_ref = rp_child_ref
            self.rd_child_ref = rd_child_ref
            self.Q_child_ref = Q_child_ref
            self.selection = weld_selection_matrix(Q_child_ref)
            self._q_ref = np.zeros(12) if Q_child_ref is None else Q_child_ref.copy()
            self._q_ref[3:6], self._q_ref[6:9] = np.asarray(rp_child_ref).reshape(3), np.asarray(rd_child_ref).reshape(
                3
            )
            self.nb_constraints = 6

        def constraint(self, Q_parent: SegmentNaturalCoordinates, Q_child: SegmentNaturalCoordinates) -> np.ndarray:
            """
            This function returns the kinematic constraints of the joint, denoted Phi_k
            as a function of the natural coordinates Q_parent and Q_child.

            Returns
            -------
            np.ndarray
                Kinematic constraints of the joint [12, 1]
            """

            return self.selection @ (self._q_ref - np.asarray(Q_child).reshape(12))

        def parent_constraint_jacobian(
            self, Q_parent: SegmentNaturalCoordinates, Q_child: SegmentNaturalCoordinates
        ) -> np.ndarray:
            return None

        def child_constraint_jacobian(
            self, Q_parent: SegmentNaturalCoordinates, Q_child: SegmentNaturalCoordinates
        ) -> np.ndarray:
            K_k_child = -self.selection

            return K_k_child

        def constraint_jacobian(
            self, Q_parent: SegmentNaturalCoordinates, Q_child: SegmentNaturalCoordinates
        ) -> np.ndarray:
            """
            This function returns the kinematic constraints of the joint, denoted K_k
            as a function of the natural coordinates Q_parent and Q_child.

            Returns
            -------
            np.ndarray
                joint constraints jacobian of the child segment [12, 12]
            """

            return self.child_constraint_jacobian(Q_parent, Q_child)

        def constraint_acceleration_bias(
            self, Qdot_parent: SegmentNaturalVelocities, Qdot_child: SegmentNaturalVelocities
        ) -> np.ndarray:
            """
            Compute the acceleration bias (quadratic velocity terms) for this ground Weld joint.

            The constraint is:
              phi = S (Q_ref - Q_child)  (linear in Q_child, constant parent)

            Since the Jacobian is constant, the Hessian is zero.
            Therefore: bias = qdot^T H qdot = 0

            Returns
            -------
            np.ndarray
                Acceleration bias vector [6, 1]. All zeros.
            """
            return np.zeros((self.nb_constraints, 1))

        def to_mx(self):
            """
            This function returns the joint as a mx joint

            Returns
            -------
            JointBase
                The joint as a mx joint
            """
            from ..bionc_casadi.joints_with_ground import GroundJoint as CasadiGroundJoint

            return CasadiGroundJoint.Weld(
                name=self.name,
                child=self.child.to_mx(),
                index=self.index,
                rp_child_ref=self.rp_child_ref,
                rd_child_ref=self.rd_child_ref,
                Q_child_ref=self.Q_child_ref,
                projection_basis=self.projection_basis,
                child_basis=self.child_basis,
            )


def weld_selection_matrix(Q_child_ref: np.ndarray = None) -> np.ndarray:
    """
    Rows S of the linear weld constraints S (Q_ref - Q_child) = 0 [6 x 12].

    Without Q_child_ref, rp and rd are fixed: the rotation about rp - rd stays free, and the constraint
    |rp - rd| = L is redundant with the rigid body constraints (singular in forward dynamics).
    With Q_child_ref, rp is fixed, rd is fixed in the two directions orthogonal to v = rp - rd, and u is fixed in
    the direction v x u: six constraints independent of the rigid body constraints, the segment is fully welded.
    """
    selection = np.zeros((6, 12))
    selection[0:3, 3:6] = np.eye(3)
    if Q_child_ref is None:
        selection[3:6, 6:9] = np.eye(3)
        return selection

    Q_child_ref = np.asarray(Q_child_ref, dtype=float).reshape(12)
    u, v = Q_child_ref[0:3], Q_child_ref[3:6] - Q_child_ref[6:9]
    v = v / np.linalg.norm(v)
    e_u = np.cross(v, u)  # direction of u when the segment rotates about v
    e_u = e_u / np.linalg.norm(e_u)
    e_2 = np.cross(v, e_u)  # (e_u, e_2) span the plane orthogonal to v
    selection[3, 6:9] = e_u
    selection[4, 6:9] = e_2
    selection[5, 0:3] = e_u
    return selection
