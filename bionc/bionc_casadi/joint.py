from casadi import MX, dot, cos, transpose, sumsqr, vertcat
import numpy as np

from .natural_segment import NaturalSegment
from .natural_coordinates import SegmentNaturalCoordinates
from .natural_velocities import SegmentNaturalVelocities
from .natural_marker import NaturalMarker
from ..protocols.joint import JointBase
from .natural_vector import NaturalVector
from ..utils.enums import NaturalAxis, CartesianAxis, EulerSequence, TransformationMatrixType
from .cartesian_vector import CartesianVector


class Joint:
    """
    The public interface to the different Joint classes
    """

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

        def parent_constraint_jacobian_derivative(
            self, Qdot_parent: SegmentNaturalVelocities, Qdot_child: SegmentNaturalVelocities
        ) -> MX:
            K_k_parent_dot = MX.zeros((self.nb_constraints, 12))
            for i in range(2):
                K_k_parent_dot[i + 3, :] = (
                    transpose(self.parent_vector[i].interpolate().rot)
                    @ self.child_vector[i].interpolate().rot
                    @ Qdot_child
                )

            return K_k_parent_dot

        def child_constraint_jacobian_derivative(
            self, Qdot_parent: SegmentNaturalVelocities, Qdot_child: SegmentNaturalVelocities
        ) -> MX:
            K_k_child_dot = MX.zeros((self.nb_constraints, 12))
            for i in range(2):
                K_k_child_dot[i + 3, :] = (
                    transpose(self.parent_vector[i].interpolate().rot @ Qdot_parent)
                    @ self.child_vector[i].interpolate().rot
                )

            return K_k_child_dot

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

        def constraint_jacobian_derivative(
            self, Qdot_parent: SegmentNaturalVelocities, Qdot_child: SegmentNaturalVelocities
        ) -> tuple[MX, MX]:
            """
            This function returns the jacobian derivative of the kinematic constraints of the joint, denoted Phi_k
            as a function of the natural coordinates Q_parent and Q_child.

            Returns
            -------
            tuple[MX, MX]
                joint constraints jacobian derivative of the parent and child segment [5, 12] and [5, 12]
            """
            return self.parent_constraint_jacobian_derivative(
                Qdot_parent, Qdot_child
            ), self.child_constraint_jacobian_derivative(Qdot_parent, Qdot_child)

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

        def parent_constraint_jacobian_derivative(
            self, Qdot_parent: SegmentNaturalVelocities, Qdot_child: SegmentNaturalVelocities
        ) -> MX:
            K_k_parent = MX.zeros((self.nb_constraints, 12))

            K_k_parent[3, :] = (
                transpose(self.parent_vector.interpolate().rot) @ self.child_vector.interpolate().rot @ Qdot_child
            )

            return K_k_parent

        def child_constraint_jacobian_derivative(
            self, Qdot_parent: SegmentNaturalVelocities, Qdot_child: SegmentNaturalVelocities
        ) -> MX:
            K_k_child = MX.zeros((self.nb_constraints, 12))

            K_k_child[3, :] = (
                transpose(self.parent_vector.interpolate().rot @ Qdot_parent) @ self.child_vector.interpolate().rot
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

        def constraint_jacobian_derivative(
            self, Qdot_parent: SegmentNaturalVelocities, Qdot_child: SegmentNaturalVelocities
        ) -> tuple[MX, MX]:
            """
            This function returns the derivative of kinematic constraints of the joint, denoted K_k
            as a function of the natural velocities Qdot_parent and Qdot_child.

            Returns
            -------
            tuple[MX, MX]
                joint constraints jacobian derivative of the parent and child segment [4, 12] and [4, 12]
            """
            return self.parent_constraint_jacobian_derivative(
                Qdot_parent, Qdot_child
            ), self.child_constraint_jacobian_derivative(Qdot_parent, Qdot_child)

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
                name, parent, child, index, parent_point, child_point, projection_basis, parent_basis, child_basis
            )
            self.nb_constraints = 3
            self.parent_point = (
                NaturalMarker(
                    name=f"{self.name}_parent_point",
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

        def parent_constraint_jacobian_derivative(
            self, Qdot_parent: SegmentNaturalVelocities, Qdot_child: SegmentNaturalVelocities
        ) -> MX:
            K_k_parent = MX.zeros((self.nb_constraints, 12))
            return K_k_parent

        def child_constraint_jacobian_derivative(
            self, Qdot_parent: SegmentNaturalVelocities, Qdot_child: SegmentNaturalVelocities
        ) -> MX:
            K_k_child = MX.zeros((self.nb_constraints, 12))
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

        def constraint_jacobian_derivative(
            self, Qdot_parent: SegmentNaturalVelocities, Qdot_child: SegmentNaturalVelocities
        ) -> tuple[MX, MX]:
            """
            This function returns the kinematic constraints of the joint, denoted K_k
            as a function of the natural coordinates Q_parent and Q_child.

            Returns
            -------
            tuple[MX, MX]
                joint constraints jacobian of the parent and child segment [3, 12] and [3, 12]
            """
            return self.parent_constraint_jacobian_derivative(
                Qdot_parent, Qdot_child
            ), self.child_constraint_jacobian_derivative(Qdot_parent, Qdot_child)

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

        def parent_constraint_jacobian_derivative(
            self, Qdot_parent: SegmentNaturalVelocities, Qdot_child: SegmentNaturalVelocities
        ) -> MX:
            parent_point_velocity = self.sphere_center.interpolation_matrix @ Qdot_parent
            child_point_velocity = self.plane_point.interpolation_matrix @ Qdot_child

            K_k_parent_dot = (
                -(self.plane_normal.interpolation_matrix @ Qdot_child).T @ self.plane_point.interpolation_matrix
                + (parent_point_velocity - child_point_velocity).T @ self.plane_normal.interpolation_matrix
            )

            return K_k_parent_dot

        def child_constraint_jacobian_derivative(
            self, Qdot_parent: SegmentNaturalVelocities, Qdot_child: SegmentNaturalVelocities
        ) -> MX:
            K_k_child_dot = (
                self.plane_normal.interpolation_matrix @ Qdot_child
            ).T @ self.sphere_center.interpolation_matrix

            return K_k_child_dot

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

        def constraint_jacobian_derivative(
            self, Qdot_parent: SegmentNaturalVelocities, Qdot_child: SegmentNaturalVelocities
        ) -> tuple[MX, MX]:
            """
            This function returns the kinematic constraints of the joint, denoted K_k
            as a function of the natural coordinates Q_parent and Q_child.

            Returns
            -------
            tuple[MX, MX]
                joint constraints jacobian of the parent and child segment [1, 12] and [1, 12]
            """
            return self.parent_constraint_jacobian_derivative(
                Qdot_parent, Qdot_child
            ), self.child_constraint_jacobian_derivative(Qdot_parent, Qdot_child)

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

        def parent_constraint_jacobian_derivative(
            self, Qdot_parent: SegmentNaturalVelocities, Qdot_child: SegmentNaturalVelocities
        ) -> MX:
            parent_point_location = self.parent_point.position_in_global(Qdot_parent)
            child_point_location = self.child_point.position_in_global(Qdot_child)

            K_k_child_dot = (
                2 * (parent_point_location - child_point_location).T @ self.parent_point.interpolation_matrix
            )

            return K_k_child_dot

        def child_constraint_jacobian_derivative(
            self, Qdot_parent: SegmentNaturalVelocities, Qdot_child: SegmentNaturalVelocities
        ) -> MX:
            parent_point_location = self.parent_point.position_in_global(Qdot_parent)
            child_point_location = self.child_point.position_in_global(Qdot_child)

            K_k_parent_dot = (
                -2 * (parent_point_location - child_point_location).T @ self.child_point.interpolation_matrix
            )

            return K_k_parent_dot

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

        def constraint_jacobian_derivative(
            self, Qdot_parent: SegmentNaturalVelocities, Qdot_child: SegmentNaturalVelocities
        ) -> tuple[MX, MX]:
            """
            This function returns the kinematic constraints of the joint, denoted K_k
            as a function of the natural coordinates Q_parent and Q_child.

            Returns
            -------
            tuple[MX, MX]
                Kinematic constraints of the joint [1, 12] for parent and child
            """

            K_k_parent_dot = self.parent_constraint_jacobian_derivative(Qdot_parent, Qdot_child)
            K_k_child_dot = self.child_constraint_jacobian_derivative(Qdot_parent, Qdot_child)

            return K_k_parent_dot, K_k_child_dot


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
            super(GroundJoint.Free, self).__init__(name, None, child, index, projection_basis, None, child_basis)

        def constraint(self, Q_parent: SegmentNaturalCoordinates, Q_child: SegmentNaturalCoordinates) -> MX:
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

        def constraint_jacobian(self, Q_parent: SegmentNaturalCoordinates, Q_child: SegmentNaturalCoordinates) -> MX:
            return None

        def constraint_jacobian_derivative(
            self, Qdot_parent: SegmentNaturalVelocities, Qdot_child: SegmentNaturalVelocities
        ) -> MX:
            return None

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
            super(GroundJoint.Hinge, self).__init__(name, None, child, index, projection_basis, None, child_basis)

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
            if not isinstance(theta, (tuple, list, MX)) or len(theta) != 2:
                raise TypeError("theta should be a tuple or list with 2 float")

            self.parent_axis = parent_axis

            self.parent_vector = [CartesianVector.axis(axis) for axis in parent_axis]

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
            constraint[:3] = -Q_child.rp

            for i in range(2):
                constraint[i + 3] = dot(
                    self.parent_vector[i],
                    Q_child.axis(self.child_axis[i]),
                ) - cos(self.theta[i])

            return constraint

        def parent_constraint_jacobian(
            self, Q_parent: SegmentNaturalCoordinates, Q_child: SegmentNaturalCoordinates
        ) -> MX:
            return None

        def child_constraint_jacobian(
            self, Q_parent: SegmentNaturalCoordinates, Q_child: SegmentNaturalCoordinates
        ) -> MX:
            K_k_child = MX.zeros((self.nb_constraints, 12))
            K_k_child[:3, 3:6] = -MX.eye(3)

            for i in range(2):
                K_k_child[i + 3, :] = transpose(self.parent_vector[i]) @ self.child_vector[i].interpolate().rot

            return K_k_child

        def parent_constraint_jacobian_derivative(
            self, Qdot_parent: SegmentNaturalVelocities, Qdot_child: SegmentNaturalVelocities
        ) -> MX:
            return None

        def child_constraint_jacobian_derivative(
            self, Qdot_parent: SegmentNaturalVelocities, Qdot_child: SegmentNaturalVelocities
        ) -> MX:
            K_k_child = MX.zeros((self.nb_constraints, 12))

            return K_k_child

        def constraint_jacobian(self, Q_parent: SegmentNaturalCoordinates, Q_child: SegmentNaturalCoordinates) -> MX:
            """
            This function returns the kinematic constraints of the joint, denoted K_k
            as a function of the natural coordinates Q_parent and Q_child.

            Returns
            -------
            MX
                joint constraints jacobian of the child segment [5, 12]
            """

            return self.child_constraint_jacobian(Q_parent, Q_child)

        def constraint_jacobian_derivative(
            self, Qdot_parent: SegmentNaturalVelocities, Qdot_child: SegmentNaturalVelocities
        ) -> MX:
            """
            This function returns the kinematic constraints of the joint, denoted K_kdot
            as a function of the natural velocities Qdot_parent and Qdot_child.

            Returns
            -------
            MX
                joint constraints jacobian of the child segment [5, 12]
            """

            return self.child_constraint_jacobian_derivative(Qdot_parent, Qdot_child)

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
            constraint[:3] = -Q_child.rp
            constraint[3] = dot(self.parent_vector, Q_child.axis(self.child_axis)) - np.cos(self.theta)

            return constraint

        def parent_constraint_jacobian(
            self, Q_parent: SegmentNaturalCoordinates, Q_child: SegmentNaturalCoordinates
        ) -> MX:
            return None

        def child_constraint_jacobian(
            self, Q_parent: SegmentNaturalCoordinates, Q_child: SegmentNaturalCoordinates
        ) -> MX:
            K_k_child = MX.zeros((self.nb_constraints, 12))
            K_k_child[:3, 3:6] = -MX.eye(3)

            K_k_child[3, :] = self.parent_vector.T @ self.child_vector.interpolate().rot

            return K_k_child

        def parent_constraint_jacobian_derivative(
            self, Qdot_parent: SegmentNaturalVelocities, Qdot_child: SegmentNaturalVelocities
        ) -> MX:
            return None

        def child_constraint_jacobian_derivative(
            self, Qdot_parent: SegmentNaturalVelocities, Qdot_child: SegmentNaturalVelocities
        ) -> MX:
            K_k_child_dot = MX.zeros((self.nb_constraints, 12))

            return K_k_child_dot

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

            return self.child_constraint_jacobian(Q_parent, Q_child)

        def constraint_jacobian_derivative(
            self, Qdot_parent: SegmentNaturalVelocities, Qdot_child: SegmentNaturalVelocities
        ) -> tuple[MX, MX]:
            """
            This function returns the kinematic constraints of the joint, denoted K_k
            as a function of the natural coordinates Q_parent and Q_child.

            Returns
            -------
            tuple[MX, MX]
                joint constraints jacobian of the parent and child segment [4, 12] and [4, 12]
            """

            return self.child_constraint_jacobian_derivative(Qdot_parent, Qdot_child)

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
            super(GroundJoint.Spherical, self).__init__(name, None, child, index, projection_basis, None, child_basis)
            self.nb_constraints = 3
            self.ground_application_point = MX(ground_application_point)

        def constraint(self, Q_parent: SegmentNaturalCoordinates, Q_child: SegmentNaturalCoordinates) -> MX:
            """
            This function returns the kinematic constraints of the joint, denoted Phi_k
            as a function of the natural coordinates Q_parent and Q_child.

            Returns
            -------
            MX
                Kinematic constraints of the joint [3, 1]
            """
            constraint = self.ground_application_point - Q_child.rp

            return constraint

        def parent_constraint_jacobian(
            self, Q_parent: SegmentNaturalCoordinates, Q_child: SegmentNaturalCoordinates
        ) -> MX:
            return None

        def child_constraint_jacobian(
            self, Q_parent: SegmentNaturalCoordinates, Q_child: SegmentNaturalCoordinates
        ) -> MX:
            K_k_child = MX.zeros((self.nb_constraints, 12))
            K_k_child[:3, 3:6] = -MX.eye(3)

            return K_k_child

        def parent_constraint_jacobian_derivative(
            self, Qdot_parent: SegmentNaturalVelocities, Qdot_child: SegmentNaturalVelocities
        ) -> np.ndarray:
            return None

        def child_constraint_jacobian_derivative(
            self, Qdot_parent: SegmentNaturalVelocities, Qdot_child: SegmentNaturalVelocities
        ) -> np.ndarray:
            K_k_child = MX.zeros((self.nb_constraints, 12))

            return K_k_child

        def constraint_jacobian(self, Q_parent: SegmentNaturalCoordinates, Q_child: SegmentNaturalCoordinates) -> MX:
            """
            This function returns the kinematic constraints of the joint, denoted K_k
            as a function of the natural coordinates Q_parent and Q_child.

            Returns
            -------
            np.ndarray
                joint constraints jacobian of the child segment [3, 12]
            """

            return self.child_constraint_jacobian(Q_parent, Q_child)

        def constraint_jacobian_derivative(
            self, Qdot_parent: SegmentNaturalVelocities, Qdot_child: SegmentNaturalVelocities
        ) -> MX:
            """
            This function returns the kinematic constraints of the joint, denoted K_kdot
            as a function of the natural velocities Qdot_parent and Qdot_child.

            Returns
            -------
            MX
                joint constraints jacobian of the child segment [3, 12]
            """

            return self.child_constraint_jacobian_derivative(Qdot_parent, Qdot_child)

    class Weld(JointBase):
        """
        This joint is defined by 3 constraints to pivot around an axis of the inertial coordinate system
        defined by two angles.
        """

        def __init__(
            self,
            name: str,
            child: NaturalSegment,
            rp_child_ref: SegmentNaturalCoordinates = None,
            rd_child_ref: SegmentNaturalCoordinates = None,
            index: int = None,
            projection_basis: EulerSequence = None,
            child_basis: TransformationMatrixType = None,
        ):
            super(GroundJoint.Weld, self).__init__(name, None, child, index, projection_basis, None, child_basis)

            # check size and type of parent axis
            self.rp_child_ref = rp_child_ref
            self.rd_child_ref = rd_child_ref
            self.nb_constraints = 6

        def constraint(self, Q_parent: SegmentNaturalCoordinates, Q_child: SegmentNaturalCoordinates) -> MX:
            """
            This function returns the kinematic constraints of the joint, denoted Phi_k
            as a function of the natural coordinates Q_parent and Q_child.

            Returns
            -------
            np.ndarray
                Kinematic constraints of the joint [12, 1]
            """

            return vertcat(self.rp_child_ref - Q_child.rp, self.rd_child_ref - Q_child.rd)

        def parent_constraint_jacobian(
            self, Q_parent: SegmentNaturalCoordinates, Q_child: SegmentNaturalCoordinates
        ) -> MX:
            return None

        def child_constraint_jacobian(
            self, Q_parent: SegmentNaturalCoordinates, Q_child: SegmentNaturalCoordinates
        ) -> MX:
            K_k_child = -MX.eye(12)[3:9, :]

            return K_k_child

        def parent_constraint_jacobian_derivative(
            self, Qdot_parent: SegmentNaturalVelocities, Qdot_child: SegmentNaturalVelocities
        ) -> MX:
            return None

        def child_constraint_jacobian_derivative(
            self, Qdot_parent: SegmentNaturalVelocities, Qdot_child: SegmentNaturalVelocities
        ) -> MX:
            K_k_child = MX.zeros((self.nb_constraints, 12))

            return K_k_child

        def constraint_jacobian(self, Q_parent: SegmentNaturalCoordinates, Q_child: SegmentNaturalCoordinates) -> MX:
            """
            This function returns the kinematic constraints of the joint, denoted K_k
            as a function of the natural coordinates Q_parent and Q_child.

            Returns
            -------
            MX
                joint constraints jacobian of the child segment [12, 12]
            """

            return self.child_constraint_jacobian(Q_parent, Q_child)

        def constraint_jacobian_derivative(
            self, Qdot_parent: SegmentNaturalVelocities, Qdot_child: SegmentNaturalVelocities
        ) -> MX:
            """
            This function returns the kinematic constraints of the joint, denoted K_kdot
            as a function of the natural velocities Qdot_parent and Qdot_child.

            Returns
            -------
            MX
                joint constraints jacobian of the child segment [12, 12]
            """

            return self.child_constraint_jacobian_derivative(Qdot_parent, Qdot_child)
