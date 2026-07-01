import numpy as np

from .natural_coordinates import SegmentNaturalCoordinates
from .natural_marker import NaturalMarker
from .natural_segment import NaturalSegment
from .natural_vector import NaturalVector
from .natural_velocities import SegmentNaturalVelocities
from ..protocols.joint import JointBaseWithTwoSegments as JointBase
from ..utils.enums import NaturalAxis, CartesianAxis, EulerSequence, TransformationMatrixType


def _point_on_ellipsoid_terms(q_parent, q_child, N_axes, N_C, N_P, semi_squared):
    """
    Constraint and jacobian rows for a single scapula contact point lying on the thorax ellipsoid.

    phi = sum_i (a_i . (P - C))^2 / s_i - 1, with a_i = N_i q_parent the principal axes,
    C = N_C q_parent the centre, P = N_P q_child the contact point and s_i = semi_i^2.
    """
    C = N_C @ q_parent
    P = N_P @ q_child
    w = P - C
    axes = [N_i @ q_parent for N_i in N_axes]
    d = [axes[i] @ w for i in range(3)]
    phi = sum(d[i] ** 2 / semi_squared[i] for i in range(3)) - 1.0
    K_parent = sum((2 * d[i] / semi_squared[i]) * (w @ N_axes[i] - axes[i] @ N_C) for i in range(3))
    K_child = sum((2 * d[i] / semi_squared[i]) * (axes[i] @ N_P) for i in range(3))
    return phi, K_parent, K_child


class Joint:
    """
    The public interface to the different Joint classes
    """

    class Free(JointBase):
        """
        This joint is defined by 3 constraints to pivot around a given axis defined by two angles.
        """

        def __init__(
            self,
            name: str,
            parent: NaturalSegment,
            child: NaturalSegment,
            index: int,
            projection_basis: EulerSequence = None,
            parent_basis: TransformationMatrixType = None,
            child_basis: TransformationMatrixType = None,
        ):
            super(Joint.Free, self).__init__(
                name, parent, child, index, projection_basis, parent_basis, child_basis, None
            )

        def constraint(self, Q_parent: SegmentNaturalCoordinates, Q_child: SegmentNaturalCoordinates) -> np.ndarray:
            """
            This function returns the kinematic constraints of the joint, denoted Phi_k
            as a function of the natural coordinates Q_parent and Q_child.

            Returns
            -------
            np.ndarray
                Kinematic constraints of the joint [5, 1]
            """
            return None

        def parent_constraint_jacobian(
            self, Q_parent: SegmentNaturalCoordinates, Q_child: SegmentNaturalCoordinates
        ) -> np.ndarray:
            return None

        def child_constraint_jacobian(
            self, Q_parent: SegmentNaturalCoordinates, Q_child: SegmentNaturalCoordinates
        ) -> np.ndarray:
            return None

        def parent_constraint_jacobian_derivative(
            self, Qdot_parent: SegmentNaturalVelocities, Qdot_child: SegmentNaturalVelocities
        ) -> np.ndarray:
            return None

        def child_constraint_jacobian_derivative(
            self, Qdot_parent: SegmentNaturalVelocities, Qdot_child: SegmentNaturalVelocities
        ) -> np.ndarray:

            return None

        def constraint_jacobian(
            self, Q_parent: SegmentNaturalCoordinates, Q_child: SegmentNaturalCoordinates
        ) -> tuple[np.ndarray, np.ndarray]:
            """
            This function returns the kinematic constraints of the joint, denoted K_k
            as a function of the natural coordinates Q_parent and Q_child.

            Returns
            -------
            tuple[np.ndarray, np.ndarray]
                joint constraints jacobian of the parent and child segment [5, 12] and [5, 12]
            """

            return None

        def constraint_jacobian_derivative(
            self, Qdot_parent: SegmentNaturalVelocities, Qdot_child: SegmentNaturalVelocities
        ) -> tuple[np.ndarray, np.ndarray]:
            """
            This function returns the jacobian derivative of the kinematic constraints of the joint, denoted Phi_k
            as a function of the natural coordinates Q_parent and Q_child.

            Returns
            -------
            tuple[np.ndarray, np.ndarray]
                joint constraints jacobian derivative of the parent and child segment [5, 12] and [5, 12]
            """
            return None

        def to_mx(self):
            """
            This function returns the joint as a mx joint

            Returns
            -------
            JointBase
                The joint as a mx joint
            """
            from ..bionc_casadi.joints import Joint as CasadiJoint

            return CasadiJoint.Free(
                name=self.name,
                parent=self.parent.to_mx(),
                child=self.child.to_mx(),
                index=self.index,
                projection_basis=self.projection_basis,
                parent_basis=self.parent_basis,
                child_basis=self.child_basis,
            )

    class Hinge(JointBase):
        """
        This joint is defined by 3 constraints to pivot around a given axis defined by two angles.
        """

        def __init__(
            self,
            name: str,
            parent: NaturalSegment,
            child: NaturalSegment,
            parent_axis: tuple[NaturalAxis] | list[NaturalAxis],
            child_axis: tuple[NaturalAxis] | list[NaturalAxis],
            theta: tuple[float] | list[float] | np.ndarray,
            index: int,
            projection_basis: EulerSequence = None,
            parent_basis: TransformationMatrixType = None,
            child_basis: TransformationMatrixType = None,
        ):
            super(Joint.Hinge, self).__init__(
                name, parent, child, index, projection_basis, parent_basis, child_basis, None
            )

            # check size and type of parent axis
            if not isinstance(parent_axis, (tuple, list)) or len(parent_axis) != 2:
                raise TypeError("parent_axis should be a tuple or list with 2 NaturalAxis")
            if not all(isinstance(axis, NaturalAxis) for axis in parent_axis):
                raise TypeError("parent_axis should be a tuple or list with 2 NaturalAxis")

            # check size and type of child axis
            if not isinstance(child_axis, (tuple, list)) or len(child_axis) != 2:
                raise TypeError("child_axis should be a tuple or list with 2 NaturalAxis")
            if not all(isinstance(axis, NaturalAxis) for axis in child_axis):
                raise TypeError("child_axis should be a tuple or list with 2 NaturalAxis")

            # check size and type of theta
            if not isinstance(theta, (tuple, list, np.ndarray)) or len(theta) != 2:
                raise TypeError("theta should be a tuple or list with 2 float")

            # todo: there should be a check on the euler sequence and transformation matrix type here
            #   with respected to the chosen parent and child axis

            self.parent_axis = parent_axis

            self.parent_vector = [NaturalVector.axis(axis) for axis in parent_axis]

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
            constraint[:3] = Q_parent.rd - Q_child.rp

            for i in range(2):
                constraint[i + 3] = np.dot(
                    Q_parent.axis(self.parent_axis[i]), Q_child.axis(self.child_axis[i])
                ) - np.cos(self.theta[i])

            return constraint

        def parent_constraint_jacobian(
            self, Q_parent: SegmentNaturalCoordinates, Q_child: SegmentNaturalCoordinates
        ) -> np.ndarray:
            K_k_parent = np.zeros((self.nb_constraints, 12))
            K_k_parent[:3, 6:9] = np.eye(3)

            for i in range(2):
                K_k_parent[i + 3, :] = np.squeeze(
                    self.parent_vector[i].interpolate().rot.T
                    @ np.array(self.child_vector[i].interpolate().rot @ Q_child)
                )

            return K_k_parent

        def child_constraint_jacobian(
            self, Q_parent: SegmentNaturalCoordinates, Q_child: SegmentNaturalCoordinates
        ) -> np.ndarray:
            K_k_child = np.zeros((self.nb_constraints, 12))
            K_k_child[:3, 3:6] = -np.eye(3)

            for i in range(2):
                K_k_child[i + 3, :] = np.squeeze(
                    (self.parent_vector[i].interpolate().rot @ Q_parent).T @ self.child_vector[i].interpolate().rot
                )

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
                joint constraints jacobian of the parent and child segment [5, 12] and [5, 12]
            """

            return self.parent_constraint_jacobian(Q_parent, Q_child), self.child_constraint_jacobian(Q_parent, Q_child)

        def constraint_acceleration_bias(
            self, Qdot_parent: SegmentNaturalVelocities, Qdot_child: SegmentNaturalVelocities
        ) -> np.ndarray:
            """
            Compute the acceleration bias (quadratic velocity terms) for this Hinge joint.

            The Hinge joint has 5 constraints:
              phi_0..2 = rp_parent - rd_child  (linear in Q => Hessian = 0 => bias = 0)
              phi_i = u_p^T v_c  (for i=3,4; bilinear in Q_parent, Q_child)

            For the bilinear constraints phi_i = (N_up * q_p)^T (N_vc * q_c):
              The cross-Hessian H_pc = N_up^T N_vc, H_cp = N_vc^T N_up, H_pp = H_cc = 0
              => bias_i = qdot^T H qdot = 2 * (N_up * qdot_p)^T (N_vc * qdot_c)

            Returns
            -------
            np.ndarray
                Acceleration bias vector [5, 1]. Enter the DAE RHS as -bias.
            """
            bias = np.zeros((self.nb_constraints, 1))

            # First 3 constraints are linear => bias = 0
            # Last 2 constraints are bilinear dot products
            for i in range(2):
                N_up = self.parent_vector[i].interpolate().rot  # 3x12
                N_vc = self.child_vector[i].interpolate().rot  # 3x12
                u_p_dot = N_up @ np.array(Qdot_parent)  # 3x1
                v_c_dot = N_vc @ np.array(Qdot_child)  # 3x1
                bias[i + 3] = 2 * u_p_dot.T @ v_c_dot

            return bias

        def to_mx(self):
            """
            This function returns the joint as a mx joint

            Returns
            -------
            JointBase
                The joint as a mx joint
            """
            from ..bionc_casadi.joints import Joint as CasadiJoint

            return CasadiJoint.Hinge(
                name=self.name,
                parent=self.parent.to_mx(),
                child=self.child.to_mx(),
                index=self.index,
                parent_axis=self.parent_axis,
                child_axis=self.child_axis,
                theta=self.theta,
                projection_basis=self.projection_basis,
                parent_basis=self.parent_basis,
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
            parent: NaturalSegment,
            child: NaturalSegment,
            parent_axis: NaturalAxis,
            child_axis: NaturalAxis,
            theta: float,
            index: int,
            projection_basis: EulerSequence = None,
            parent_basis: TransformationMatrixType = None,
            child_basis: TransformationMatrixType = None,
        ):
            super(Joint.Universal, self).__init__(
                name, parent, child, index, projection_basis, parent_basis, child_basis, None
            )

            # todo: there should be a check on the euler sequence and transformation matrix type here
            #   with respected to the chosen parent and child axis

            self.parent_axis = parent_axis
            self.parent_vector = NaturalVector.axis(self.parent_axis)

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
            constraint[:3] = Q_parent.rd - Q_child.rp
            constraint[3] = np.dot(Q_parent.axis(self.parent_axis), Q_child.axis(self.child_axis)) - np.cos(self.theta)

            return constraint

        def parent_constraint_jacobian(
            self, Q_parent: SegmentNaturalCoordinates, Q_child: SegmentNaturalCoordinates
        ) -> np.ndarray:
            K_k_parent = np.zeros((self.nb_constraints, 12))
            K_k_parent[:3, 6:9] = np.eye(3)

            K_k_parent[3, :] = np.squeeze(
                self.parent_vector.interpolate().rot.T @ np.array(self.child_vector.interpolate().rot @ Q_child)
            )

            return K_k_parent

        def child_constraint_jacobian(
            self, Q_parent: SegmentNaturalCoordinates, Q_child: SegmentNaturalCoordinates
        ) -> np.ndarray:
            K_k_child = np.zeros((self.nb_constraints, 12))
            K_k_child[:3, 3:6] = -np.eye(3)

            K_k_child[3, :] = np.squeeze(
                (self.parent_vector.interpolate().rot @ Q_parent).T @ self.child_vector.interpolate().rot
            )

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

            return self.parent_constraint_jacobian(Q_parent, Q_child), self.child_constraint_jacobian(Q_parent, Q_child)

        def constraint_acceleration_bias(
            self, Qdot_parent: SegmentNaturalVelocities, Qdot_child: SegmentNaturalVelocities
        ) -> np.ndarray:
            """
            Compute the acceleration bias (quadratic velocity terms) for this Universal joint.

            The Universal joint has 4 constraints:
              phi_0..2 = rp_parent - rd_child  (linear in Q => Hessian = 0 => bias = 0)
              phi_3 = u_p^T v_c  (bilinear in Q_parent, Q_child)

            For the bilinear constraint phi_3 = (N_up * q_p)^T (N_vc * q_c):
              H_pp = H_cc = 0, H_pc = N_up^T N_vc, H_cp = N_vc^T N_up
              => bias_3 = qdot^T H qdot = 2 * (N_up * qdot_p)^T (N_vc * qdot_c)

            Returns
            -------
            np.ndarray
                Acceleration bias vector [4, 1]. Enter the DAE RHS as -bias.
            """
            bias = np.zeros((self.nb_constraints, 1))

            # First 3 constraints are linear => bias = 0
            # Last constraint is a bilinear dot product
            N_up = self.parent_vector.interpolate().rot  # 3x12
            N_vc = self.child_vector.interpolate().rot  # 3x12
            u_p_dot = N_up @ np.array(Qdot_parent)  # 3x1
            v_c_dot = N_vc @ np.array(Qdot_child)  # 3x1
            bias[3] = 2 * u_p_dot.T @ v_c_dot

            return bias

        def to_mx(self):
            """
            This function returns the joint as a mx joint

            Returns
            -------
            JointBase
                The joint as a mx joint
            """
            from ..bionc_casadi.joints import Joint as CasadiJoint

            return CasadiJoint.Universal(
                name=self.name,
                parent=self.parent.to_mx(),
                child=self.child.to_mx(),
                index=self.index,
                parent_axis=self.parent_axis,
                child_axis=self.child_axis,
                theta=self.theta,
                projection_basis=self.projection_basis,
                parent_basis=self.parent_basis,
                child_basis=self.child_basis,
            )

    class Spherical(JointBase):
        def __init__(
            self,
            name: str,
            parent: NaturalSegment,
            child: NaturalSegment,
            index: int,
            parent_point: str = None,
            child_point: str = None,
            projection_basis: EulerSequence = None,
            parent_basis: TransformationMatrixType = None,
            child_basis: TransformationMatrixType = None,
        ):
            super(Joint.Spherical, self).__init__(
                name, parent, child, index, projection_basis, parent_basis, child_basis, None
            )
            self.nb_constraints = 3
            self.parent_point_str = parent_point
            self.child_point_str = child_point  # to transfer to casadi later on

            self.parent_point = (
                NaturalMarker(
                    name=f"{self.name}_parent_point",
                    parent_name=self.parent.name,
                    position=NaturalVector.distal(),
                    is_technical=False,
                    is_anatomical=True,
                )
                if parent_point is None
                else parent.marker_from_name(parent_point)
            )

            self.child_point = (
                NaturalMarker(
                    name=f"{self.name}_child_point",
                    parent_name=self.child.name,
                    position=NaturalVector.proximal(),
                    is_technical=False,
                    is_anatomical=True,
                )
                if child_point is None
                else child.marker_from_name(child_point)
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
            parent_point_location = self.parent_point.position_in_global(Q_parent)
            child_point_location = self.child_point.position_in_global(Q_child)

            constraint = parent_point_location - child_point_location

            return constraint.squeeze()

        def parent_constraint_jacobian(
            self, Q_parent: SegmentNaturalCoordinates, Q_child: SegmentNaturalCoordinates
        ) -> np.ndarray:
            K_k_parent = np.zeros((self.nb_constraints, 12))
            K_k_parent[:3, :] = self.parent_point.interpolation_matrix

            return K_k_parent

        def child_constraint_jacobian(
            self, Q_parent: SegmentNaturalCoordinates, Q_child: SegmentNaturalCoordinates
        ) -> np.ndarray:
            K_k_child = np.zeros((self.nb_constraints, 12))
            K_k_child[:3, :] = -self.child_point.interpolation_matrix

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
                joint constraints jacobian of the parent and child segment [3, 12] and [3, 12]
            """
            return self.parent_constraint_jacobian(Q_parent, Q_child), self.child_constraint_jacobian(Q_parent, Q_child)

        def constraint_acceleration_bias(
            self, Qdot_parent: SegmentNaturalVelocities, Qdot_child: SegmentNaturalVelocities
        ) -> np.ndarray:
            """
            Compute the acceleration bias (quadratic velocity terms) for this Spherical joint.

            The Spherical joint constraint is:
              phi = N_p * q_p - N_c * q_c  (linear in Q)

            Since all constraints are linear, the Hessian is zero everywhere.
            Therefore: bias = qdot^T H qdot = 0

            Returns
            -------
            np.ndarray
                Acceleration bias vector [3, 1]. All zeros for spherical joints.
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
            from ..bionc_casadi.joints import Joint as CasadiJoint

            return CasadiJoint.Spherical(
                name=self.name,
                parent=self.parent.to_mx(),
                child=self.child.to_mx(),
                index=self.index,
                parent_point=self.parent_point_str,
                child_point=self.child_point_str,
                projection_basis=self.projection_basis,
                parent_basis=self.parent_basis,
                child_basis=self.child_basis,
            )

    class SphereOnPlane(JointBase):
        """
        This class represents a sphere-on-plane joint: parent is the sphere, and child is the plane.
        """

        def __init__(
            self,
            name: str,
            parent: NaturalSegment,
            child: NaturalSegment,
            index: int,
            sphere_radius: float = None,
            sphere_center: str = None,
            plane_point: str = None,
            plane_normal: str = None,
            projection_basis: EulerSequence = None,
            parent_basis: TransformationMatrixType = None,
            child_basis: TransformationMatrixType = None,
        ):
            super(Joint.SphereOnPlane, self).__init__(
                name,
                parent,
                child,
                index,
                projection_basis,
                parent_basis,
                child_basis,
                (CartesianAxis.X, CartesianAxis.Y, CartesianAxis.Z),
            )
            self.nb_constraints = 1

            if sphere_radius is None:
                raise ValueError("sphere_radius must be specified for joint SphereOnPlane")
            if sphere_center is None:
                raise ValueError("sphere_center must be specified for joint SphereOnPlane")
            if plane_point is None:
                raise ValueError("plane_point must be specified for joint SphereOnPlane")
            if plane_normal is None:
                raise ValueError("plane_normal must be specified for joint SphereOnPlane")

            self.sphere_radius = sphere_radius
            self.sphere_center = parent.marker_from_name(sphere_center)
            self.plane_point = child.marker_from_name(plane_point)
            self.plane_normal = child.vector_from_name(plane_normal)

        def constraint(self, Q_parent: SegmentNaturalCoordinates, Q_child: SegmentNaturalCoordinates) -> np.ndarray:
            """
            This function returns the kinematic constraints of the joint, denoted Phi_k
            as a function of the natural coordinates Q_parent and Q_child.

            Returns
            -------
            np.ndarray
                Kinematic constraints of the joint [1, 1]
            """

            parent_point_location = self.sphere_center.interpolation_matrix.to_array() @ Q_parent.to_array()
            child_point_location = self.plane_point.interpolation_matrix.to_array() @ Q_child.to_array()
            normal_orientation = self.plane_normal.interpolation_matrix @ Q_child.to_array()

            constraint = (parent_point_location - child_point_location).T @ normal_orientation - self.sphere_radius

            return constraint

        def parent_constraint_jacobian(
            self,
            Q_parent: SegmentNaturalCoordinates,
            Q_child: SegmentNaturalCoordinates,
        ) -> np.ndarray:
            parent_point_location = self.sphere_center.interpolation_matrix.to_array() @ Q_parent
            child_point_location = self.plane_point.interpolation_matrix.to_array() @ Q_child

            K_k_parent = (
                -(self.plane_normal.interpolation_matrix @ Q_child).T @ self.plane_point.interpolation_matrix
                + (parent_point_location - child_point_location).T @ self.plane_normal.interpolation_matrix
            )

            return K_k_parent

        def child_constraint_jacobian(
            self, Q_parent: SegmentNaturalCoordinates, Q_child: SegmentNaturalCoordinates
        ) -> np.ndarray:
            K_k_child = (
                self.plane_normal.interpolation_matrix @ Q_child
            ).T @ self.sphere_center.interpolation_matrix.to_array()

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
                joint constraints jacobian of the parent and child segment [1, 12] and [1, 12]
            """
            return self.parent_constraint_jacobian(Q_parent, Q_child), self.child_constraint_jacobian(Q_parent, Q_child)

        def constraint_acceleration_bias(
            self, Qdot_parent: SegmentNaturalVelocities, Qdot_child: SegmentNaturalVelocities
        ) -> np.ndarray:
            """
            Compute the acceleration bias (quadratic velocity terms) for this SphereOnPlane joint.

            The constraint is:
              phi = (P - A)^T n - r
            where P = Np*q_p (sphere center), A = Na*q_c (plane point), n = Nn*q_c (plane normal), r = const.

            Writing q = [q_p; q_c], the full Hessian blocks are:
              H_pp = 0  (P is linear in q_p, and n doesn't depend on q_p)
              H_pc = Np^T Nn  (cross term: d/dq_c of dPhi/dq_p = Np^T n => d(Np^T Nn q_c)/dq_c)
              H_cp = Nn^T Np  (transpose of H_pc)
              H_cc = -(Nn^T Na + Na^T Nn)  (from -A^T n term, both A and n depend on q_c)

            bias = qdot^T H qdot
                 = 2*(Np*qdot_p)^T*(Nn*qdot_c) - 2*(Nn*qdot_c)^T*(Na*qdot_c)

            Returns
            -------
            np.ndarray
                Acceleration bias vector [1, 1]. Enter the DAE RHS as -bias.
            """
            Np = self.sphere_center.interpolation_matrix.to_array()  # 3x12
            Na = self.plane_point.interpolation_matrix.to_array()  # 3x12
            Nn = self.plane_normal.interpolation_matrix  # 3x12

            P_dot = Np @ np.array(Qdot_parent)  # 3x1
            A_dot = Na @ np.array(Qdot_child)  # 3x1
            n_dot = Nn @ np.array(Qdot_child)  # 3x1

            bias = 2 * P_dot.T @ n_dot - 2 * n_dot.T @ A_dot

            return np.array(bias).reshape(self.nb_constraints, 1)

        def to_mx(self):
            """
            This function returns the joint as a mx joint

            Returns
            -------
            JointBase
                The joint as a mx joint
            """

            from ..bionc_casadi.joints import Joint as CasadiJoint

            return CasadiJoint.SphereOnPlane(
                name=self.name,
                parent=self.parent.to_mx(),
                child=self.child.to_mx(),
                index=self.index,
                sphere_radius=self.sphere_radius,
                sphere_center=self.sphere_center.name,
                plane_point=self.plane_point.name,
                plane_normal=self.plane_normal.name,
                projection_basis=self.projection_basis,
                parent_basis=self.parent_basis,
                child_basis=self.child_basis,
            )

    class EllipsoidOnPlane(JointBase):
        """
        Plane tangent to an ellipsoid joint (e.g. scapulothoracic joint).

        The parent segment carries the ellipsoid (centre + 3 principal axes + semi-axis
        lengths), the child segment carries the plane (a point and its unit normal).
        It generalizes ``SphereOnPlane``: with ``a == b == c`` and axes aligned with the
        global frame it reduces to the sphere-on-plane constraint (radius ``a``).

        The single scalar constraint (Naaim 2016, plane-tangent-to-ellipsoid appendix) is

        ``phi = sqrt(u^T R^T B R u) + u^T (C - A)``

        with ``u`` the plane normal, ``C`` the ellipsoid centre, ``A`` the plane point,
        ``B = diag(a^2, b^2, c^2)`` and ``R`` the ellipsoid orientation whose rows are the
        three principal axes evaluated at ``Q_parent``.
        """

        def __init__(
            self,
            name: str,
            parent: NaturalSegment,
            child: NaturalSegment,
            index: int,
            semi_axis_lengths: tuple[float, float, float] | np.ndarray = None,
            ellipsoid_center: str = None,
            ellipsoid_axis_a: str = None,
            ellipsoid_axis_b: str = None,
            ellipsoid_axis_c: str = None,
            plane_point: str = None,
            plane_normal: str = None,
            projection_basis: EulerSequence = None,
            parent_basis: TransformationMatrixType = None,
            child_basis: TransformationMatrixType = None,
        ):
            super(Joint.EllipsoidOnPlane, self).__init__(
                name,
                parent,
                child,
                index,
                projection_basis,
                parent_basis,
                child_basis,
                (CartesianAxis.X, CartesianAxis.Y, CartesianAxis.Z),
            )
            self.nb_constraints = 1

            if semi_axis_lengths is None:
                raise ValueError("semi_axis_lengths (a, b, c) must be specified for joint EllipsoidOnPlane")
            if ellipsoid_center is None:
                raise ValueError("ellipsoid_center must be specified for joint EllipsoidOnPlane")
            if ellipsoid_axis_a is None or ellipsoid_axis_b is None or ellipsoid_axis_c is None:
                raise ValueError("ellipsoid_axis_a, _b and _c must be specified for joint EllipsoidOnPlane")
            if plane_point is None:
                raise ValueError("plane_point must be specified for joint EllipsoidOnPlane")
            if plane_normal is None:
                raise ValueError("plane_normal must be specified for joint EllipsoidOnPlane")

            self.semi_axis_lengths = tuple(float(length) for length in semi_axis_lengths)
            if len(self.semi_axis_lengths) != 3 or any(length <= 0 for length in self.semi_axis_lengths):
                raise ValueError("semi_axis_lengths must be 3 strictly positive values (a, b, c)")

            self.ellipsoid_center = parent.marker_from_name(ellipsoid_center)
            self.ellipsoid_axes = [
                parent.vector_from_name(axis) for axis in (ellipsoid_axis_a, ellipsoid_axis_b, ellipsoid_axis_c)
            ]
            self.plane_point = child.marker_from_name(plane_point)
            self.plane_normal = child.vector_from_name(plane_normal)

        def _kinematic_terms(self, Q_parent: SegmentNaturalCoordinates, Q_child: SegmentNaturalCoordinates):
            """Common quantities reused by the constraint, its jacobian and the bias."""
            Q_p = np.array(Q_parent).reshape(-1)
            Q_c = np.array(Q_child).reshape(-1)

            N_axes = [np.array(axis.interpolation_matrix) for axis in self.ellipsoid_axes]  # 3 x (3, 12)
            N_C = self.ellipsoid_center.interpolation_matrix.to_array()  # (3, 12)
            N_A = self.plane_point.interpolation_matrix.to_array()  # (3, 12)
            N_n = np.array(self.plane_normal.interpolation_matrix)  # (3, 12)

            u = N_n @ Q_c  # plane normal in global (3,)
            axes = [N_i @ Q_p for N_i in N_axes]  # ellipsoid principal axes in global (3,)
            projections = [float(axis @ u) for axis in axes]  # a_i^T u
            b = [length**2 for length in self.semi_axis_lengths]
            s = sum(b[i] * projections[i] ** 2 for i in range(3))

            return dict(
                Q_p=Q_p,
                Q_c=Q_c,
                N_axes=N_axes,
                N_C=N_C,
                N_A=N_A,
                N_n=N_n,
                u=u,
                axes=axes,
                p=projections,
                b=b,
                s=s,
                sqrt_s=np.sqrt(s),
                C=N_C @ Q_p,
                A=N_A @ Q_c,
            )

        def constraint(self, Q_parent: SegmentNaturalCoordinates, Q_child: SegmentNaturalCoordinates) -> np.ndarray:
            """
            This function returns the kinematic constraints of the joint, denoted Phi_k
            as a function of the natural coordinates Q_parent and Q_child.

            Returns
            -------
            np.ndarray
                Kinematic constraints of the joint [1, 1]
            """
            t = self._kinematic_terms(Q_parent, Q_child)
            return t["sqrt_s"] + t["u"] @ (t["C"] - t["A"])

        def parent_constraint_jacobian(
            self, Q_parent: SegmentNaturalCoordinates, Q_child: SegmentNaturalCoordinates
        ) -> np.ndarray:
            t = self._kinematic_terms(Q_parent, Q_child)
            b, p, u, N_axes, N_C, sqrt_s = t["b"], t["p"], t["u"], t["N_axes"], t["N_C"], t["sqrt_s"]

            d_sqrt = sum(b[i] * p[i] * (u @ N_axes[i]) for i in range(3)) / sqrt_s
            K_k_parent = d_sqrt + u @ N_C

            return np.array(K_k_parent).reshape(self.nb_constraints, 12)

        def child_constraint_jacobian(
            self, Q_parent: SegmentNaturalCoordinates, Q_child: SegmentNaturalCoordinates
        ) -> np.ndarray:
            t = self._kinematic_terms(Q_parent, Q_child)
            b, p, u, axes, N_n, N_A, sqrt_s = t["b"], t["p"], t["u"], t["axes"], t["N_n"], t["N_A"], t["sqrt_s"]

            d_sqrt = sum(b[i] * p[i] * (axes[i] @ N_n) for i in range(3)) / sqrt_s
            K_k_child = d_sqrt + (t["C"] - t["A"]) @ N_n - u @ N_A

            return np.array(K_k_child).reshape(self.nb_constraints, 12)

        def constraint_jacobian(
            self, Q_parent: SegmentNaturalCoordinates, Q_child: SegmentNaturalCoordinates
        ) -> tuple[np.ndarray, np.ndarray]:
            """
            This function returns the kinematic constraints of the joint, denoted K_k
            as a function of the natural coordinates Q_parent and Q_child.

            Returns
            -------
            tuple[np.ndarray, np.ndarray]
                joint constraints jacobian of the parent and child segment [1, 12] and [1, 12]
            """
            return self.parent_constraint_jacobian(Q_parent, Q_child), self.child_constraint_jacobian(Q_parent, Q_child)

        def constraint_acceleration_bias(
            self, Qdot_parent: SegmentNaturalVelocities, Qdot_child: SegmentNaturalVelocities
        ) -> np.ndarray:
            """
            Not implemented.

            Unlike the other joints, the Hessian of the EllipsoidOnPlane constraint depends on
            the configuration Q (through the ``sqrt(u^T R^T B R u)`` term), so the acceleration
            bias ``qdot^T H qdot`` cannot be computed from velocities alone. The current
            ``BiomechanicalModel`` only forwards velocities to this method, so dynamics with this
            joint would require the framework to also pass the configuration. The analytic bias is

              pdot_i    = (N_i qdot_p)^T u + a_i^T (N_n qdot_c)
              sdot      = sum_i 2 b_i p_i pdot_i
              bias_s    = sum_i 2 b_i [ pdot_i^2 + 2 p_i (N_i qdot_p)^T (N_n qdot_c) ]
              bias_sqrt = bias_s / (2 sqrt(s)) - sdot^2 / (4 s^(3/2))
              bias_h    = 2 (N_n qdot_c)^T (N_C qdot_p) - 2 (N_n qdot_c)^T (N_A qdot_c)
              bias      = bias_sqrt + bias_h
            """
            raise NotImplementedError(
                "constraint_acceleration_bias is configuration-dependent for EllipsoidOnPlane and "
                "is not supported by the velocity-only model interface; use this joint for "
                "kinematics/constraint evaluation only."
            )

        def to_mx(self):
            """
            This function returns the joint as a mx joint

            Returns
            -------
            JointBase
                The joint as a mx joint
            """
            from ..bionc_casadi.joints import Joint as CasadiJoint

            return CasadiJoint.EllipsoidOnPlane(
                name=self.name,
                parent=self.parent.to_mx(),
                child=self.child.to_mx(),
                index=self.index,
                semi_axis_lengths=self.semi_axis_lengths,
                ellipsoid_center=self.ellipsoid_center.name,
                ellipsoid_axis_a=self.ellipsoid_axes[0].name,
                ellipsoid_axis_b=self.ellipsoid_axes[1].name,
                ellipsoid_axis_c=self.ellipsoid_axes[2].name,
                plane_point=self.plane_point.name,
                plane_normal=self.plane_normal.name,
                projection_basis=self.projection_basis,
                parent_basis=self.parent_basis,
                child_basis=self.child_basis,
            )

    class PointOnEllipsoid(JointBase):
        """
        One-contact-point scapulothoracic joint (Naaim 2016/2017).

        A single fixed point of the child segment (the scapula) is constrained to lie on the
        ellipsoid carried by the parent segment (the thorax). The scalar constraint is

            phi = sum_i (a_i . (P - C))^2 / s_i - 1

        with ``P`` the contact point, ``C`` the ellipsoid centre, ``a_i`` its principal axes and
        ``s_i = semi_i^2``. phi = 0 means the point is exactly on the surface, phi < 0 inside,
        phi > 0 outside.
        """

        def __init__(
            self,
            name: str,
            parent: NaturalSegment,
            child: NaturalSegment,
            index: int,
            semi_axis_lengths: tuple[float, float, float] | np.ndarray = None,
            ellipsoid_center: str = None,
            ellipsoid_axis_a: str = None,
            ellipsoid_axis_b: str = None,
            ellipsoid_axis_c: str = None,
            contact_point: str = None,
            projection_basis: EulerSequence = None,
            parent_basis: TransformationMatrixType = None,
            child_basis: TransformationMatrixType = None,
        ):
            super(Joint.PointOnEllipsoid, self).__init__(
                name,
                parent,
                child,
                index,
                projection_basis,
                parent_basis,
                child_basis,
                (CartesianAxis.X, CartesianAxis.Y, CartesianAxis.Z),
            )
            self.nb_constraints = 1

            if semi_axis_lengths is None:
                raise ValueError("semi_axis_lengths (a, b, c) must be specified for joint PointOnEllipsoid")
            if ellipsoid_center is None:
                raise ValueError("ellipsoid_center must be specified for joint PointOnEllipsoid")
            if ellipsoid_axis_a is None or ellipsoid_axis_b is None or ellipsoid_axis_c is None:
                raise ValueError("ellipsoid_axis_a, _b and _c must be specified for joint PointOnEllipsoid")
            if contact_point is None:
                raise ValueError("contact_point must be specified for joint PointOnEllipsoid")

            self.semi_axis_lengths = tuple(float(length) for length in semi_axis_lengths)
            if len(self.semi_axis_lengths) != 3 or any(length <= 0 for length in self.semi_axis_lengths):
                raise ValueError("semi_axis_lengths must be 3 strictly positive values (a, b, c)")

            self.ellipsoid_center = parent.marker_from_name(ellipsoid_center)
            self.ellipsoid_axes = [
                parent.vector_from_name(axis) for axis in (ellipsoid_axis_a, ellipsoid_axis_b, ellipsoid_axis_c)
            ]
            self.contact_point = child.marker_from_name(contact_point)

        def _ellipsoid_matrices(self):
            N_axes = [np.array(axis.interpolation_matrix) for axis in self.ellipsoid_axes]
            N_C = self.ellipsoid_center.interpolation_matrix.to_array()
            s = [length**2 for length in self.semi_axis_lengths]
            return N_axes, N_C, s

        def constraint(self, Q_parent: SegmentNaturalCoordinates, Q_child: SegmentNaturalCoordinates) -> np.ndarray:
            """Kinematic constraint of the joint [1, 1]."""
            q_parent = np.array(Q_parent).reshape(-1)
            q_child = np.array(Q_child).reshape(-1)
            N_axes, N_C, s = self._ellipsoid_matrices()
            N_P = self.contact_point.interpolation_matrix.to_array()
            phi, _, _ = _point_on_ellipsoid_terms(q_parent, q_child, N_axes, N_C, N_P, s)
            return np.array(phi)

        def parent_constraint_jacobian(
            self, Q_parent: SegmentNaturalCoordinates, Q_child: SegmentNaturalCoordinates
        ) -> np.ndarray:
            q_parent = np.array(Q_parent).reshape(-1)
            q_child = np.array(Q_child).reshape(-1)
            N_axes, N_C, s = self._ellipsoid_matrices()
            N_P = self.contact_point.interpolation_matrix.to_array()
            _, K_parent, _ = _point_on_ellipsoid_terms(q_parent, q_child, N_axes, N_C, N_P, s)
            return np.array(K_parent).reshape(self.nb_constraints, 12)

        def child_constraint_jacobian(
            self, Q_parent: SegmentNaturalCoordinates, Q_child: SegmentNaturalCoordinates
        ) -> np.ndarray:
            q_parent = np.array(Q_parent).reshape(-1)
            q_child = np.array(Q_child).reshape(-1)
            N_axes, N_C, s = self._ellipsoid_matrices()
            N_P = self.contact_point.interpolation_matrix.to_array()
            _, _, K_child = _point_on_ellipsoid_terms(q_parent, q_child, N_axes, N_C, N_P, s)
            return np.array(K_child).reshape(self.nb_constraints, 12)

        def constraint_jacobian(
            self, Q_parent: SegmentNaturalCoordinates, Q_child: SegmentNaturalCoordinates
        ) -> tuple[np.ndarray, np.ndarray]:
            """Constraint jacobian of the parent and child segment [1, 12] and [1, 12]."""
            return self.parent_constraint_jacobian(Q_parent, Q_child), self.child_constraint_jacobian(Q_parent, Q_child)

        def constraint_acceleration_bias(
            self, Qdot_parent: SegmentNaturalVelocities, Qdot_child: SegmentNaturalVelocities
        ) -> np.ndarray:
            """
            Not implemented: like EllipsoidOnPlane, the Hessian of the point-on-ellipsoid constraint
            depends on the configuration Q, which the velocity-only model interface cannot supply.
            Use this joint for kinematics / multibody kinematics optimisation only.
            """
            raise NotImplementedError(
                "constraint_acceleration_bias is configuration-dependent for PointOnEllipsoid and is not "
                "supported by the velocity-only model interface; use this joint for kinematics only."
            )

        def to_mx(self):
            """This function returns the joint as a mx joint."""
            from ..bionc_casadi.joints import Joint as CasadiJoint

            return CasadiJoint.PointOnEllipsoid(
                name=self.name,
                parent=self.parent.to_mx(),
                child=self.child.to_mx(),
                index=self.index,
                semi_axis_lengths=self.semi_axis_lengths,
                ellipsoid_center=self.ellipsoid_center.name,
                ellipsoid_axis_a=self.ellipsoid_axes[0].name,
                ellipsoid_axis_b=self.ellipsoid_axes[1].name,
                ellipsoid_axis_c=self.ellipsoid_axes[2].name,
                contact_point=self.contact_point.name,
                projection_basis=self.projection_basis,
                parent_basis=self.parent_basis,
                child_basis=self.child_basis,
            )

    class TwoPointsOnEllipsoid(JointBase):
        """
        Two-contact-point scapulothoracic joint (Naaim 2016/2017).

        Two fixed points of the child segment (the scapula) are constrained to lie on the ellipsoid
        carried by the parent segment (the thorax), giving two scalar constraints, each of the form
        ``phi = sum_i (a_i . (P - C))^2 / s_i - 1`` (see ``PointOnEllipsoid``).
        """

        def __init__(
            self,
            name: str,
            parent: NaturalSegment,
            child: NaturalSegment,
            index: int,
            semi_axis_lengths: tuple[float, float, float] | np.ndarray = None,
            ellipsoid_center: str = None,
            ellipsoid_axis_a: str = None,
            ellipsoid_axis_b: str = None,
            ellipsoid_axis_c: str = None,
            contact_point_1: str = None,
            contact_point_2: str = None,
            projection_basis: EulerSequence = None,
            parent_basis: TransformationMatrixType = None,
            child_basis: TransformationMatrixType = None,
        ):
            super(Joint.TwoPointsOnEllipsoid, self).__init__(
                name,
                parent,
                child,
                index,
                projection_basis,
                parent_basis,
                child_basis,
                (CartesianAxis.X, CartesianAxis.Y, CartesianAxis.Z),
            )
            self.nb_constraints = 2

            if semi_axis_lengths is None:
                raise ValueError("semi_axis_lengths (a, b, c) must be specified for joint TwoPointsOnEllipsoid")
            if ellipsoid_center is None:
                raise ValueError("ellipsoid_center must be specified for joint TwoPointsOnEllipsoid")
            if ellipsoid_axis_a is None or ellipsoid_axis_b is None or ellipsoid_axis_c is None:
                raise ValueError("ellipsoid_axis_a, _b and _c must be specified for joint TwoPointsOnEllipsoid")
            if contact_point_1 is None or contact_point_2 is None:
                raise ValueError("contact_point_1 and contact_point_2 must be specified for joint TwoPointsOnEllipsoid")

            self.semi_axis_lengths = tuple(float(length) for length in semi_axis_lengths)
            if len(self.semi_axis_lengths) != 3 or any(length <= 0 for length in self.semi_axis_lengths):
                raise ValueError("semi_axis_lengths must be 3 strictly positive values (a, b, c)")

            self.ellipsoid_center = parent.marker_from_name(ellipsoid_center)
            self.ellipsoid_axes = [
                parent.vector_from_name(axis) for axis in (ellipsoid_axis_a, ellipsoid_axis_b, ellipsoid_axis_c)
            ]
            self.contact_points = [child.marker_from_name(contact_point_1), child.marker_from_name(contact_point_2)]

        def _ellipsoid_matrices(self):
            N_axes = [np.array(axis.interpolation_matrix) for axis in self.ellipsoid_axes]
            N_C = self.ellipsoid_center.interpolation_matrix.to_array()
            s = [length**2 for length in self.semi_axis_lengths]
            return N_axes, N_C, s

        def constraint(self, Q_parent: SegmentNaturalCoordinates, Q_child: SegmentNaturalCoordinates) -> np.ndarray:
            """Kinematic constraints of the joint [2, 1]."""
            q_parent = np.array(Q_parent).reshape(-1)
            q_child = np.array(Q_child).reshape(-1)
            N_axes, N_C, s = self._ellipsoid_matrices()
            constraint = np.zeros(self.nb_constraints)
            for k, point in enumerate(self.contact_points):
                N_P = point.interpolation_matrix.to_array()
                constraint[k], _, _ = _point_on_ellipsoid_terms(q_parent, q_child, N_axes, N_C, N_P, s)
            return constraint

        def parent_constraint_jacobian(
            self, Q_parent: SegmentNaturalCoordinates, Q_child: SegmentNaturalCoordinates
        ) -> np.ndarray:
            q_parent = np.array(Q_parent).reshape(-1)
            q_child = np.array(Q_child).reshape(-1)
            N_axes, N_C, s = self._ellipsoid_matrices()
            K_parent = np.zeros((self.nb_constraints, 12))
            for k, point in enumerate(self.contact_points):
                N_P = point.interpolation_matrix.to_array()
                _, row, _ = _point_on_ellipsoid_terms(q_parent, q_child, N_axes, N_C, N_P, s)
                K_parent[k, :] = row
            return K_parent

        def child_constraint_jacobian(
            self, Q_parent: SegmentNaturalCoordinates, Q_child: SegmentNaturalCoordinates
        ) -> np.ndarray:
            q_parent = np.array(Q_parent).reshape(-1)
            q_child = np.array(Q_child).reshape(-1)
            N_axes, N_C, s = self._ellipsoid_matrices()
            K_child = np.zeros((self.nb_constraints, 12))
            for k, point in enumerate(self.contact_points):
                N_P = point.interpolation_matrix.to_array()
                _, _, row = _point_on_ellipsoid_terms(q_parent, q_child, N_axes, N_C, N_P, s)
                K_child[k, :] = row
            return K_child

        def constraint_jacobian(
            self, Q_parent: SegmentNaturalCoordinates, Q_child: SegmentNaturalCoordinates
        ) -> tuple[np.ndarray, np.ndarray]:
            """Constraint jacobian of the parent and child segment [2, 12] and [2, 12]."""
            return self.parent_constraint_jacobian(Q_parent, Q_child), self.child_constraint_jacobian(Q_parent, Q_child)

        def constraint_acceleration_bias(
            self, Qdot_parent: SegmentNaturalVelocities, Qdot_child: SegmentNaturalVelocities
        ) -> np.ndarray:
            """
            Not implemented: the Hessian of the point-on-ellipsoid constraints depends on the
            configuration Q, which the velocity-only model interface cannot supply. Use this joint
            for kinematics / multibody kinematics optimisation only.
            """
            raise NotImplementedError(
                "constraint_acceleration_bias is configuration-dependent for TwoPointsOnEllipsoid and is not "
                "supported by the velocity-only model interface; use this joint for kinematics only."
            )

        def to_mx(self):
            """This function returns the joint as a mx joint."""
            from ..bionc_casadi.joints import Joint as CasadiJoint

            return CasadiJoint.TwoPointsOnEllipsoid(
                name=self.name,
                parent=self.parent.to_mx(),
                child=self.child.to_mx(),
                index=self.index,
                semi_axis_lengths=self.semi_axis_lengths,
                ellipsoid_center=self.ellipsoid_center.name,
                ellipsoid_axis_a=self.ellipsoid_axes[0].name,
                ellipsoid_axis_b=self.ellipsoid_axes[1].name,
                ellipsoid_axis_c=self.ellipsoid_axes[2].name,
                contact_point_1=self.contact_points[0].name,
                contact_point_2=self.contact_points[1].name,
                projection_basis=self.projection_basis,
                parent_basis=self.parent_basis,
                child_basis=self.child_basis,
            )

    class ConstantLength(JointBase):
        def __init__(
            self,
            name: str,
            parent: NaturalSegment,
            child: NaturalSegment,
            index: int,
            length: float = None,
            parent_point: str = None,
            child_point: str = None,
            projection_basis: EulerSequence = None,
            parent_basis: TransformationMatrixType = None,
            child_basis: TransformationMatrixType = None,
        ):
            super(Joint.ConstantLength, self).__init__(
                name,
                parent,
                child,
                index,
                projection_basis,
                parent_basis,
                child_basis,
                (CartesianAxis.X, CartesianAxis.Y, CartesianAxis.Z),
            )

            if length is None:
                raise ValueError("length must be provided")
            if parent_point is None:
                raise ValueError("parent_point must be provided")
            if child_point is None:
                raise ValueError("child_point must be provided")

            self.nb_constraints = 1
            self.length = length
            self.parent_point = parent.marker_from_name(parent_point)
            self.child_point = child.marker_from_name(child_point)

        @property
        def nb_joint_dof(self) -> int:
            """
            erase the parent method because remove no proper dof when looking at absolute joint rotation and torques.
            ex : one constant length won't block rotations and translations of the child segment
            """
            return 6

        def constraint(self, Q_parent: SegmentNaturalCoordinates, Q_child: SegmentNaturalCoordinates) -> np.ndarray:
            """
            This function returns the kinematic constraints of the joint, denoted Phi_k
            as a function of the natural coordinates Q_parent and Q_child.

            Returns
            -------
            np.ndarray
                Kinematic constraints of the joint [3, 1]
            """
            parent_point_location = self.parent_point.position_in_global(Q_parent)
            child_point_location = self.child_point.position_in_global(Q_child)

            constraint = np.sum((parent_point_location - child_point_location) ** 2) - self.length**2

            return constraint

        def parent_constraint_jacobian(
            self, Q_parent: SegmentNaturalCoordinates, Q_child: SegmentNaturalCoordinates
        ) -> np.ndarray:
            parent_point_location = self.parent_point.position_in_global(Q_parent)
            child_point_location = self.child_point.position_in_global(Q_child)

            K_k_parent = 2 * (parent_point_location - child_point_location).T @ self.parent_point.interpolation_matrix

            return np.array(K_k_parent)

        def child_constraint_jacobian(
            self, Q_parent: SegmentNaturalCoordinates, Q_child: SegmentNaturalCoordinates
        ) -> np.ndarray:
            parent_point_location = self.parent_point.position_in_global(Q_parent)
            child_point_location = self.child_point.position_in_global(Q_child)

            K_k_child = -2 * (parent_point_location - child_point_location).T @ self.child_point.interpolation_matrix

            return np.array(K_k_child)

        def constraint_jacobian(
            self, Q_parent: SegmentNaturalCoordinates, Q_child: SegmentNaturalCoordinates
        ) -> tuple[np.ndarray, np.ndarray]:
            """
            This function returns the kinematic constraints of the joint, denoted K_k
            as a function of the natural coordinates Q_parent and Q_child.

            Returns
            -------
            tuple[np.ndarray, np.ndarray]
                Kinematic constraints of the joint [1, 12] for parent and child
            """

            K_k_parent = self.parent_constraint_jacobian(Q_parent, Q_child)
            K_k_child = self.child_constraint_jacobian(Q_parent, Q_child)

            return K_k_parent, K_k_child

        def constraint_acceleration_bias(
            self, Qdot_parent: SegmentNaturalVelocities, Qdot_child: SegmentNaturalVelocities
        ) -> np.ndarray:
            """
            Compute the acceleration bias (quadratic velocity terms) for this ConstantLength joint.

            The constraint is:
              phi = ||Np*q_p - Nc*q_c||^2 - L^2

            The Hessian of phi w.r.t. q = [q_p; q_c] is the constant matrix:
              H = 2 * [Np; -Nc]^T [Np; -Nc]
            i.e. H_pp = 2*Np^T*Np, H_cc = 2*Nc^T*Nc, H_pc = -2*Np^T*Nc

            Therefore:
              bias = qdot^T H qdot = 2 * ||Np*qdot_p - Nc*qdot_c||^2

            Returns
            -------
            np.ndarray
                Acceleration bias vector [1, 1]. Enter the DAE RHS as -bias.
            """
            Np = self.parent_point.interpolation_matrix  # 3x12
            Nc = self.child_point.interpolation_matrix  # 3x12

            diff_vel = Np @ np.array(Qdot_parent) - Nc @ np.array(Qdot_child)  # 3x1
            bias = 2 * diff_vel.T @ diff_vel

            return np.array(bias).reshape(self.nb_constraints, 1)

        def to_mx(self):
            """
            This function returns the joint as a mx joint

            Returns
            -------
            JointBase
                The joint as a mx joint
            """
            from ..bionc_casadi.joints import Joint as CasadiJoint

            return CasadiJoint.ConstantLength(
                name=self.name,
                parent=self.parent.to_mx(),
                child=self.child.to_mx(),
                index=self.index,
                length=self.length,
                parent_point=self.parent_point.name,
                child_point=self.child_point.name,
                projection_basis=self.projection_basis,
                parent_basis=self.parent_basis,
                child_basis=self.child_basis,
            )
