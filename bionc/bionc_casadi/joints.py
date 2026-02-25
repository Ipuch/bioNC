import numpy as np
from casadi import MX, dot, cos, transpose, sumsqr

from .natural_coordinates import SegmentNaturalCoordinates
from .natural_marker import NaturalMarker
from .natural_segment import NaturalSegment
from .natural_vector import NaturalVector
from .natural_velocities import SegmentNaturalVelocities
from ..protocols.joint import JointBaseWithTwoSegments as JointBase
from ..utils.enums import NaturalAxis, EulerSequence, TransformationMatrixType


class Joint:
    """
    The public interface to the different Joint classes
    """

    class Free(JointBase):
        """
        This joint is defined by 0 constraints to let the joint be free between parent and child.
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

        def constraint(self, Q_parent: SegmentNaturalCoordinates, Q_child: SegmentNaturalCoordinates) -> MX:
            """
            This function returns the kinematic constraints of the joint, denoted Phi_k
            as a function of the natural coordinates Q_parent and Q_child.

            Returns
            -------
            MX
                Kinematic constraints of the joint (None for free joint - 0 constraints)
            """
            return None

        def parent_constraint_jacobian(
            self, Q_parent: SegmentNaturalCoordinates, Q_child: SegmentNaturalCoordinates
        ) -> MX:
            return None

        def child_constraint_jacobian(
            self, Q_parent: SegmentNaturalCoordinates, Q_child: SegmentNaturalCoordinates
        ) -> MX:

            return None

        def parent_constraint_jacobian_derivative(
            self, Qdot_parent: SegmentNaturalVelocities, Qdot_child: SegmentNaturalVelocities
        ) -> MX:

            return None

        def child_constraint_jacobian_derivative(
            self, Qdot_parent: SegmentNaturalVelocities, Qdot_child: SegmentNaturalVelocities
        ) -> MX:

            return None

        def constraint_jacobian(
            self, Q_parent: SegmentNaturalCoordinates, Q_child: SegmentNaturalCoordinates
        ) -> tuple[MX, MX]:
            """
            This function returns the jacobian of the kinematic constraints of the joint, denoted Phi_k
            as a function of the natural coordinates Q_parent and Q_child.

            Returns
            -------
            tuple[MX, MX]
                joint constraints jacobian of the parent and child segment (None for free joint)
            """
            return None

        def constraint_jacobian_derivative(
            self, Qdot_parent: SegmentNaturalVelocities, Qdot_child: SegmentNaturalVelocities
        ) -> tuple[MX, MX]:
            """
            This function returns the jacobian derivative of the kinematic constraints of the joint, denoted Phi_k
            as a function of the natural coordinates Q_parent and Q_child.

            Returns
            -------
            tuple[MX, MX]
                joint constraints jacobian derivative of the parent and child segment (None for free joint)
            """
            return None

    class Hinge(JointBase):
        """
        This joint is defined by 3 constraints to pivot around a given axis defined by two angles theta_1 and theta_2.

        """

        def __init__(
            self,
            name: str,
            parent: NaturalSegment,
            child: NaturalSegment,
            parent_axis: tuple[NaturalAxis] | list[NaturalAxis],
            child_axis: tuple[NaturalAxis] | list[NaturalAxis],
            theta: tuple[float] | list[float] | np.ndarray | MX,
            index: int,
            projection_basis: EulerSequence = None,
            parent_basis: TransformationMatrixType = None,
            child_basis: TransformationMatrixType = None,
        ):
            super(Joint.Hinge, self).__init__(name, parent, child, index, projection_basis, parent_basis, child_basis)

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
            if not isinstance(theta, (tuple, list, np.ndarray, MX)) or len(theta) != 2:
                raise TypeError("theta should be a tuple or list with 2 float")
            if isinstance(theta, (tuple, list)):
                theta = MX(np.array(theta))
            if isinstance(theta, np.ndarray):
                theta = MX(theta)

            self.parent_axis = parent_axis

            self.parent_vector = [NaturalVector.axis(axis) for axis in parent_axis]

            self.child_axis = child_axis

            self.child_vector = [NaturalVector.axis(axis) for axis in child_axis]

            self.theta = theta

            self.nb_constraints = 5

        def constraint(self, Q_parent: SegmentNaturalCoordinates, Q_child: SegmentNaturalCoordinates) -> MX:
            """
            This function returns the kinematic constraints of the joint, denoted Phi_k
            as a function of the natural coordinates Q_parent and Q_child.

            Returns
            -------
            MX
                Kinematic constraints of the joint [5, 1]
            """
            constraint = MX.zeros(self.nb_constraints)
            constraint[:3, 0] = Q_parent.rd - Q_child.rp

            for i in range(2):
                constraint[i + 3] = dot(Q_parent.axis(self.parent_axis[i]), Q_child.axis(self.child_axis[i])) - cos(
                    self.theta[i]
                )

            return constraint

        def parent_constraint_jacobian(
            self, Q_parent: SegmentNaturalCoordinates, Q_child: SegmentNaturalCoordinates
        ) -> MX:
            K_k_parent = MX.zeros((self.nb_constraints, 12))
            K_k_parent[:3, 6:9] = MX.eye(3)

            for i in range(2):
                K_k_parent[i + 3, :] = (
                    transpose(self.parent_vector[i].interpolate().rot)
                    @ self.child_vector[i].interpolate().rot
                    @ Q_child
                )

            return K_k_parent

        def child_constraint_jacobian(
            self, Q_parent: SegmentNaturalCoordinates, Q_child: SegmentNaturalCoordinates
        ) -> MX:
            K_k_child = MX.zeros((self.nb_constraints, 12))
            K_k_child[:3, 3:6] = -MX.eye(3)

            for i in range(2):
                K_k_child[i + 3, :] = (
                    transpose(self.parent_vector[i].interpolate().rot @ Q_parent)
                    @ self.child_vector[i].interpolate().rot
                )

            return K_k_child

        def constraint_jacobian(
            self, Q_parent: SegmentNaturalCoordinates, Q_child: SegmentNaturalCoordinates
        ) -> tuple[MX, MX]:
            """
            This function returns the jacobian of the kinematic constraints of the joint, denoted Phi_k
            as a function of the natural coordinates Q_parent and Q_child.

            Returns
            -------
            tuple[MX, MX]
                joint constraints jacobian of the parent and child segment [5, 12] and [5, 12]
            """
            return self.parent_constraint_jacobian(Q_parent, Q_child), self.child_constraint_jacobian(Q_parent, Q_child)

        def constraint_acceleration_bias(
            self, Qdot_parent: SegmentNaturalVelocities, Qdot_child: SegmentNaturalVelocities
        ) -> MX:
            """
            Compute the acceleration bias (quadratic velocity terms) for this Hinge joint.

            The Hinge joint has 5 constraints:
              phi_0..2 = rp_parent - rd_child  (linear in Q => Hessian = 0 => bias = 0)
              phi_i = u_p^T v_c  (for i=3,4; bilinear in Q_parent, Q_child)

            For the bilinear constraints phi_i = (N_up * q_p)^T (N_vc * q_c):
              H_pp = H_cc = 0, H_pc = N_up^T N_vc, H_cp = N_vc^T N_up
              => bias_i = qdot^T H qdot = 2 * (N_up * qdot_p)^T (N_vc * qdot_c)

            Returns
            -------
            MX
                Acceleration bias vector [5, 1]. Enter the DAE RHS as -bias.
            """
            bias = MX.zeros(self.nb_constraints, 1)

            for i in range(2):
                N_up = self.parent_vector[i].interpolate().rot  # 3x12
                N_vc = self.child_vector[i].interpolate().rot   # 3x12
                u_p_dot = N_up @ Qdot_parent  # 3x1
                v_c_dot = N_vc @ Qdot_child   # 3x1
                bias[i + 3] = 2 * dot(u_p_dot, v_c_dot)

            return bias

        # Backward-compatibility alias
        def constraint_acceleration_biais(
            self, Qdot_parent: SegmentNaturalVelocities, Qdot_child: SegmentNaturalVelocities
        ) -> MX:
            return self.constraint_acceleration_bias(Qdot_parent, Qdot_child)

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
        theta :
            Angle between the two axes
        """

        def __init__(
            self,
            name: str,
            parent: NaturalSegment,
            child: NaturalSegment,
            parent_axis: NaturalAxis,
            child_axis: NaturalAxis,
            theta: float | np.ndarray | MX,
            index: int,
            projection_basis: EulerSequence = None,
            parent_basis: TransformationMatrixType = None,
            child_basis: TransformationMatrixType = None,
        ):
            super(Joint.Universal, self).__init__(
                name, parent, child, index, projection_basis, parent_basis, child_basis
            )

            self.parent_axis = parent_axis
            self.parent_vector = NaturalVector.axis(self.parent_axis)

            self.child_axis = child_axis
            self.child_vector = NaturalVector.axis(self.child_axis)

            if isinstance(theta, (float, int, np.ndarray)):
                theta = MX(theta)
            if theta.shape[0] != 1:
                raise TypeError("theta should be a float or a MX of shape (1, 1)")
            self.theta = theta

            self.nb_constraints = 4

        def constraint(self, Q_parent: SegmentNaturalCoordinates, Q_child: SegmentNaturalCoordinates) -> MX:
            """
            This function returns the kinematic constraints of the joint, denoted Phi_k
            as a function of the natural coordinates Q_parent and Q_child.

            Returns
            -------
            MX
                Kinematic constraints of the joint [4, 1]
            """

            constraint = MX.zeros(self.nb_constraints)
            constraint[:3, 0] = Q_parent.rd - Q_child.rp
            constraint[3, 0] = dot(Q_parent.axis(self.parent_axis), Q_child.axis(self.child_axis)) - cos(self.theta)

            return constraint

        def parent_constraint_jacobian(
            self, Q_parent: SegmentNaturalCoordinates, Q_child: SegmentNaturalCoordinates
        ) -> MX:
            K_k_parent = MX.zeros((self.nb_constraints, 12))
            K_k_parent[:3, 6:9] = MX.eye(3)

            K_k_parent[3, :] = (
                transpose(self.parent_vector.interpolate().rot) @ self.child_vector.interpolate().rot @ Q_child
            )

            return K_k_parent

        def child_constraint_jacobian(
            self, Q_parent: SegmentNaturalCoordinates, Q_child: SegmentNaturalCoordinates
        ) -> MX:
            K_k_child = MX.zeros((self.nb_constraints, 12))
            K_k_child[:3, 3:6] = -MX.eye(3)
            K_k_child[3, :] = (
                transpose(self.parent_vector.interpolate().rot @ Q_parent) @ self.child_vector.interpolate().rot
            )

            return K_k_child

        def constraint_jacobian(
            self, Q_parent: SegmentNaturalCoordinates, Q_child: SegmentNaturalCoordinates
        ) -> tuple[MX, MX]:
            """
            This function returns the kinematic constraints of the joint, denoted K_k
            as a function of the natural coordinates Q_parent and Q_child.

            Returns
            -------
            tuple[MX, MX]
                joint constraints jacobian of the parent and child segment [4, 12] and [4, 12]
            """
            return self.parent_constraint_jacobian(Q_parent, Q_child), self.child_constraint_jacobian(Q_parent, Q_child)

        def constraint_acceleration_bias(
            self, Qdot_parent: SegmentNaturalVelocities, Qdot_child: SegmentNaturalVelocities
        ) -> MX:
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
            MX
                Acceleration bias vector [4, 1]. Enter the DAE RHS as -bias.
            """
            bias = MX.zeros(self.nb_constraints, 1)

            # First 3 constraints are linear => bias = 0
            # Last constraint is a bilinear dot product
            N_up = self.parent_vector.interpolate().rot  # 3x12
            N_vc = self.child_vector.interpolate().rot   # 3x12
            u_p_dot = N_up @ Qdot_parent  # 3x1
            v_c_dot = N_vc @ Qdot_child   # 3x1
            bias[3] = 2 * dot(u_p_dot, v_c_dot)

            return bias

        # Backward-compatibility alias
        def constraint_acceleration_biais(
            self, Qdot_parent: SegmentNaturalVelocities, Qdot_child: SegmentNaturalVelocities
        ) -> MX:
            return self.constraint_acceleration_bias(Qdot_parent, Qdot_child)

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
                name, parent, child, index, projection_basis, parent_basis, child_basis
            )
            self.nb_constraints = 3
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

        def constraint(self, Q_parent: SegmentNaturalCoordinates, Q_child: SegmentNaturalCoordinates) -> MX:
            """
            This function returns the kinematic constraints of the joint, denoted Phi_k
            as a function of the natural coordinates Q_parent and Q_child.

            Returns
            -------
            MX
                Kinematic constraints of the joint [3, 1]
            """
            parent_point_location = self.parent_point.position_in_global(Q_parent)
            child_point_location = self.child_point.position_in_global(Q_child)

            constraint = parent_point_location - child_point_location

            return constraint

        def parent_constraint_jacobian(
            self, Q_parent: SegmentNaturalCoordinates, Q_child: SegmentNaturalCoordinates
        ) -> MX:
            K_k_parent = MX.zeros((self.nb_constraints, 12))
            K_k_parent[:3, :] = self.parent_point.interpolation_matrix

            return K_k_parent

        def child_constraint_jacobian(
            self, Q_parent: SegmentNaturalCoordinates, Q_child: SegmentNaturalCoordinates
        ) -> MX:
            K_k_child = MX.zeros((self.nb_constraints, 12))
            K_k_child[:3, :] = -self.child_point.interpolation_matrix

            return K_k_child

        def constraint_jacobian(
            self, Q_parent: SegmentNaturalCoordinates, Q_child: SegmentNaturalCoordinates
        ) -> tuple[MX, MX]:
            """
            This function returns the kinematic constraints of the joint, denoted K_k
            as a function of the natural coordinates Q_parent and Q_child.

            Returns
            -------
            tuple[MX, MX]
                joint constraints jacobian of the parent and child segment [3, 12] and [3, 12]
            """
            return self.parent_constraint_jacobian(Q_parent, Q_child), self.child_constraint_jacobian(Q_parent, Q_child)

        def constraint_acceleration_bias(
            self, Qdot_parent: SegmentNaturalVelocities, Qdot_child: SegmentNaturalVelocities
        ) -> MX:
            """
            Compute the acceleration bias (quadratic velocity terms) for this Spherical joint.

            The Spherical joint constraint is:
              phi = N_p * q_p - N_c * q_c  (linear in Q)

            Since all constraints are linear, the Hessian is zero everywhere.
            Therefore: bias = qdot^T H qdot = 0

            Returns
            -------
            MX
                Acceleration bias vector [3, 1]. All zeros for spherical joints.
            """
            return MX.zeros(self.nb_constraints, 1)

        # Backward-compatibility alias
        def constraint_acceleration_biais(
            self, Qdot_parent: SegmentNaturalVelocities, Qdot_child: SegmentNaturalVelocities
        ) -> MX:
            return self.constraint_acceleration_bias(Qdot_parent, Qdot_child)

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
                name, parent, child, index, projection_basis, parent_basis, child_basis
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

        def constraint(self, Q_parent: SegmentNaturalCoordinates, Q_child: SegmentNaturalCoordinates) -> MX:
            """
            This function returns the kinematic constraints of the joint, denoted Phi_k
            as a function of the natural coordinates Q_parent and Q_child.

            Returns
            -------
            MX
                Kinematic constraints of the joint [1, 1]
            """

            parent_point_location = self.sphere_center.interpolation_matrix @ Q_parent
            child_point_location = self.plane_point.interpolation_matrix @ Q_child
            normal_orientation = self.plane_normal.interpolation_matrix @ Q_child

            constraint = (parent_point_location - child_point_location).T @ normal_orientation - self.sphere_radius

            return constraint

        def parent_constraint_jacobian(
            self,
            Q_parent: SegmentNaturalCoordinates,
            Q_child: SegmentNaturalCoordinates,
        ) -> MX:
            parent_point_location = self.sphere_center.interpolation_matrix @ Q_parent
            child_point_location = self.plane_point.interpolation_matrix @ Q_child

            K_k_parent = (
                -(self.plane_normal.interpolation_matrix @ Q_child).T @ self.plane_point.interpolation_matrix
                + (parent_point_location - child_point_location).T @ self.plane_normal.interpolation_matrix
            )

            return K_k_parent

        def child_constraint_jacobian(
            self, Q_parent: SegmentNaturalCoordinates, Q_child: SegmentNaturalCoordinates
        ) -> MX:
            K_k_child = (self.plane_normal.interpolation_matrix @ Q_child).T @ self.sphere_center.interpolation_matrix

            return K_k_child

        def constraint_jacobian(
            self, Q_parent: SegmentNaturalCoordinates, Q_child: SegmentNaturalCoordinates
        ) -> tuple[MX, MX]:
            """
            This function returns the kinematic constraints of the joint, denoted K_k
            as a function of the natural coordinates Q_parent and Q_child.

            Returns
            -------
            tuple[MX, MX]
                joint constraints jacobian of the parent and child segment [1, 12] and [1, 12]
            """
            return self.parent_constraint_jacobian(Q_parent, Q_child), self.child_constraint_jacobian(Q_parent, Q_child)

        def constraint_acceleration_bias(
            self, Qdot_parent: SegmentNaturalVelocities, Qdot_child: SegmentNaturalVelocities
        ) -> MX:
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
            MX
                Acceleration bias vector [1, 1]. Enter the DAE RHS as -bias.
            """
            Np = self.sphere_center.interpolation_matrix  # 3x12
            Na = self.plane_point.interpolation_matrix     # 3x12
            Nn = self.plane_normal.interpolation_matrix    # 3x12

            P_dot = Np @ Qdot_parent  # 3x1
            A_dot = Na @ Qdot_child   # 3x1
            n_dot = Nn @ Qdot_child   # 3x1

            bias = 2 * transpose(P_dot) @ n_dot - 2 * transpose(n_dot) @ A_dot

            return bias

        # Backward-compatibility alias
        def constraint_acceleration_biais(
            self, Qdot_parent: SegmentNaturalVelocities, Qdot_child: SegmentNaturalVelocities
        ) -> MX:
            return self.constraint_acceleration_bias(Qdot_parent, Qdot_child)

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
                name, parent, child, index, projection_basis, parent_basis, child_basis
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

        def constraint(self, Q_parent: SegmentNaturalCoordinates, Q_child: SegmentNaturalCoordinates) -> MX:
            """
            This function returns the kinematic constraints of the joint, denoted Phi_k
            as a function of the natural coordinates Q_parent and Q_child.

            Returns
            -------
            MX
                Kinematic constraints of the joint [3, 1]
            """
            parent_point_location = self.parent_point.position_in_global(Q_parent)
            child_point_location = self.child_point.position_in_global(Q_child)

            constraint = sumsqr(parent_point_location - child_point_location) - self.length**2

            return constraint

        def parent_constraint_jacobian(
            self, Q_parent: SegmentNaturalCoordinates, Q_child: SegmentNaturalCoordinates
        ) -> MX:
            parent_point_location = self.parent_point.position_in_global(Q_parent)
            child_point_location = self.child_point.position_in_global(Q_child)

            K_k_parent = 2 * (parent_point_location - child_point_location).T @ self.parent_point.interpolation_matrix

            return K_k_parent

        def child_constraint_jacobian(
            self, Q_parent: SegmentNaturalCoordinates, Q_child: SegmentNaturalCoordinates
        ) -> MX:
            parent_point_location = self.parent_point.position_in_global(Q_parent)
            child_point_location = self.child_point.position_in_global(Q_child)

            K_k_child = -2 * (parent_point_location - child_point_location).T @ self.child_point.interpolation_matrix

            return K_k_child

        def constraint_jacobian(
            self, Q_parent: SegmentNaturalCoordinates, Q_child: SegmentNaturalCoordinates
        ) -> tuple[MX, MX]:
            """
            This function returns the kinematic constraints of the joint, denoted K_k
            as a function of the natural coordinates Q_parent and Q_child.

            Returns
            -------
            tuple[MX, MX]
                Kinematic constraints of the joint [1, 12] for parent and child
            """

            K_k_parent = self.parent_constraint_jacobian(Q_parent, Q_child)
            K_k_child = self.child_constraint_jacobian(Q_parent, Q_child)

            return K_k_parent, K_k_child

        def constraint_acceleration_bias(
            self, Qdot_parent: SegmentNaturalVelocities, Qdot_child: SegmentNaturalVelocities
        ) -> MX:
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
            MX
                Acceleration bias vector [1, 1]. Enter the DAE RHS as -bias.
            """
            Np = self.parent_point.interpolation_matrix  # 3x12
            Nc = self.child_point.interpolation_matrix   # 3x12

            diff_vel = Np @ Qdot_parent - Nc @ Qdot_child  # 3x1
            bias = 2 * sumsqr(diff_vel)

            return bias

        # Backward-compatibility alias
        def constraint_acceleration_biais(
            self, Qdot_parent: SegmentNaturalVelocities, Qdot_child: SegmentNaturalVelocities
        ) -> MX:
            return self.constraint_acceleration_bias(Qdot_parent, Qdot_child)
