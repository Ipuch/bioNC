import numpy as np

from .natural_segment import NaturalSegment
from .natural_coordinates import SegmentNaturalCoordinates
from .natural_velocities import SegmentNaturalVelocities
from ..protocols.joint import JointBase
from ..utils.enums import NaturalAxis, CartesianAxis, EulerSequence
from .natural_vector import NaturalVector
from .cartesian_vector import CartesianVector


class Joint:
    """
    The public interface to the different Joint classes
    """

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
        ):
            super(Joint.Hinge, self).__init__(name, parent, child, index, projection_basis)

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

        def parent_constraint_jacobian_derivative(
            self, Qdot_parent: SegmentNaturalVelocities, Qdot_child: SegmentNaturalVelocities
        ) -> np.ndarray:
            K_k_parent_dot = np.zeros((self.nb_constraints, 12))
            for i in range(2):
                K_k_parent_dot[i + 3, :] = np.squeeze(
                    self.parent_vector[i].interpolate().rot.T @ self.child_vector[i].interpolate().rot @ Qdot_child
                )

            return K_k_parent_dot

        def child_constraint_jacobian_derivative(
            self, Qdot_parent: SegmentNaturalVelocities, Qdot_child: SegmentNaturalVelocities
        ) -> np.ndarray:
            K_k_child_dot = np.zeros((self.nb_constraints, 12))
            for i in range(2):
                K_k_child_dot[i + 3, :] = np.squeeze(
                    (self.parent_vector[i].interpolate().rot @ Qdot_parent).T @ self.child_vector[i].interpolate().rot
                )

            return K_k_child_dot

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
            return self.parent_constraint_jacobian_derivative(
                Qdot_parent, Qdot_child
            ), self.child_constraint_jacobian_derivative(Qdot_parent, Qdot_child)

        def to_mx(self):
            """
            This function returns the joint as a mx joint

            Returns
            -------
            JointBase
                The joint as a mx joint
            """
            from ..bionc_casadi.joint import Joint as CasadiJoint

            return CasadiJoint.Hinge(
                name=self.name,
                parent=self.parent.to_mx(),
                child=self.child.to_mx(),
                index=self.index,
                parent_axis=self.parent_axis,
                child_axis=self.child_axis,
                theta=self.theta,
                projection_basis=self.projection_basis,
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
        ):
            super(Joint.Universal, self).__init__(name, parent, child, index, projection_basis)
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

        def parent_constraint_jacobian_derivative(
            self, Qdot_parent: SegmentNaturalVelocities, Qdot_child: SegmentNaturalVelocities
        ) -> np.ndarray:
            K_k_parent_dot = np.zeros((self.nb_constraints, 12))
            K_k_parent_dot[3, :] = np.squeeze(
                self.parent_vector.interpolate().rot.T @ np.array(self.child_vector.interpolate().rot @ Qdot_child)
            )

            return K_k_parent_dot

        def child_constraint_jacobian_derivative(
            self, Qdot_parent: SegmentNaturalVelocities, Qdot_child: SegmentNaturalVelocities
        ) -> np.ndarray:
            K_k_child_dot = np.zeros((self.nb_constraints, 12))
            K_k_child_dot[3, :] = np.squeeze(
                (self.parent_vector.interpolate().rot @ Qdot_parent).T @ self.child_vector.interpolate().rot
            )

            return K_k_child_dot

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

        def constraint_jacobian_derivative(
            self, Qdot_parent: SegmentNaturalVelocities, Qdot_child: SegmentNaturalVelocities
        ) -> tuple[np.ndarray, np.ndarray]:
            """
            This function returns the kinematic constraints of the joint, denoted K_k
            as a function of the natural coordinates Q_parent and Q_child.

            Returns
            -------
            tuple[np.ndarray, np.ndarray]
                joint constraints jacobian of the parent and child segment [4, 12] and [4, 12]
            """

            return self.parent_constraint_jacobian_derivative(
                Qdot_parent, Qdot_child
            ), self.child_constraint_jacobian_derivative(Qdot_parent, Qdot_child)

        def to_mx(self):
            """
            This function returns the joint as a mx joint

            Returns
            -------
            JointBase
                The joint as a mx joint
            """
            from ..bionc_casadi.joint import Joint as CasadiJoint

            return CasadiJoint.Universal(
                name=self.name,
                parent=self.parent.to_mx(),
                child=self.child.to_mx(),
                index=self.index,
                parent_axis=self.parent_axis,
                child_axis=self.child_axis,
                theta=self.theta,
                projection_basis=self.projection_basis,
            )

    class Spherical(JointBase):
        def __init__(
            self,
            name: str,
            parent: NaturalSegment,
            child: NaturalSegment,
            index: int,
            projection_basis: EulerSequence = None,
        ):
            super(Joint.Spherical, self).__init__(name, parent, child, index, projection_basis)
            self.nb_constraints = 3

        def constraint(self, Q_parent: SegmentNaturalCoordinates, Q_child: SegmentNaturalCoordinates) -> np.ndarray:
            """
            This function returns the kinematic constraints of the joint, denoted Phi_k
            as a function of the natural coordinates Q_parent and Q_child.

            Returns
            -------
            np.ndarray
                Kinematic constraints of the joint [3, 1]
            """
            constraint = Q_parent.rd - Q_child.rp

            return constraint

        def parent_constraint_jacobian(
            self, Q_parent: SegmentNaturalCoordinates, Q_child: SegmentNaturalCoordinates
        ) -> np.ndarray:
            K_k_parent = np.zeros((self.nb_constraints, 12))
            K_k_parent[:3, 6:9] = np.eye(3)

            return K_k_parent

        def child_constraint_jacobian(
            self, Q_parent: SegmentNaturalCoordinates, Q_child: SegmentNaturalCoordinates
        ) -> np.ndarray:
            K_k_child = np.zeros((self.nb_constraints, 12))
            K_k_child[:3, 3:6] = -np.eye(3)

            return K_k_child

        def parent_constraint_jacobian_derivative(
            self, Qdot_parent: SegmentNaturalVelocities, Qdot_child: SegmentNaturalVelocities
        ) -> np.ndarray:
            K_k_parent_dot = np.zeros((self.nb_constraints, 12))

            return K_k_parent_dot

        def child_constraint_jacobian_derivative(
            self, Qdot_parent: SegmentNaturalVelocities, Qdot_child: SegmentNaturalVelocities
        ) -> np.ndarray:
            K_k_child_dot = np.zeros((self.nb_constraints, 12))

            return K_k_child_dot

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

        def constraint_jacobian_derivative(
            self, Qdot_parent: SegmentNaturalVelocities, Qdot_child: SegmentNaturalVelocities
        ) -> tuple[np.ndarray, np.ndarray]:
            """
            This function returns the kinematic constraints of the joint, denoted K_k
            as a function of the natural coordinates Q_parent and Q_child.

            Returns
            -------
            tuple[np.ndarray, np.ndarray]
                joint constraints jacobian of the parent and child segment [3, 12] and [3, 12]
            """
            return self.parent_constraint_jacobian_derivative(
                Qdot_parent, Qdot_child
            ), self.child_constraint_jacobian_derivative(Qdot_parent, Qdot_child)

        def to_mx(self):
            """
            This function returns the joint as a mx joint

            Returns
            -------
            JointBase
                The joint as a mx joint
            """
            from ..bionc_casadi.joint import Joint as CasadiJoint

            return CasadiJoint.Spherical(
                name=self.name,
                parent=self.parent.to_mx(),
                child=self.child.to_mx(),
                index=self.index,
                projection_basis=self.projection_basis,
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
        ):
            super(Joint.SphereOnPlane, self).__init__(name, parent, child, index, projection_basis)
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

        def parent_constraint_jacobian_derivative(
            self, Qdot_parent: SegmentNaturalVelocities, Qdot_child: SegmentNaturalVelocities
        ) -> np.ndarray:
            parent_point_velocity = self.sphere_center.interpolation_matrix.to_array() @ Qdot_parent
            child_point_velocity = self.plane_point.interpolation_matrix.to_array() @ Qdot_child

            K_k_parent_dot = (
                -(self.plane_normal.interpolation_matrix @ Qdot_child).T
                @ self.plane_point.interpolation_matrix.to_array()
                + (parent_point_velocity - child_point_velocity).T @ self.plane_normal.interpolation_matrix
            )

            return K_k_parent_dot

        def child_constraint_jacobian_derivative(
            self, Qdot_parent: SegmentNaturalVelocities, Qdot_child: SegmentNaturalVelocities
        ) -> np.ndarray:
            K_k_child_dot = (
                self.plane_normal.interpolation_matrix @ Qdot_child
            ).T @ self.sphere_center.interpolation_matrix.to_array()

            return K_k_child_dot

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

        def constraint_jacobian_derivative(
            self, Qdot_parent: SegmentNaturalVelocities, Qdot_child: SegmentNaturalVelocities
        ) -> tuple[np.ndarray, np.ndarray]:
            """
            This function returns the kinematic constraints of the joint, denoted K_k
            as a function of the natural coordinates Q_parent and Q_child.

            Returns
            -------
            tuple[np.ndarray, np.ndarray]
                joint constraints jacobian of the parent and child segment [1, 12] and [1, 12]
            """
            return self.parent_constraint_jacobian_derivative(
                Qdot_parent, Qdot_child
            ), self.child_constraint_jacobian_derivative(Qdot_parent, Qdot_child)

        def to_mx(self):
            """
            This function returns the joint as a mx joint

            Returns
            -------
            JointBase
                The joint as a mx joint
            """

            from ..bionc_casadi.joint import Joint as CasadiJoint

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
        ):
            super(Joint.ConstantLength, self).__init__(name, parent, child, index, projection_basis)

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

        def parent_constraint_jacobian_derivative(
            self, Qdot_parent: SegmentNaturalVelocities, Qdot_child: SegmentNaturalVelocities
        ) -> np.ndarray:
            parent_point_location = self.parent_point.position_in_global(Qdot_parent)
            child_point_location = self.child_point.position_in_global(Qdot_child)

            K_k_child_dot = (
                2 * (parent_point_location - child_point_location).T @ self.parent_point.interpolation_matrix
            )

            return np.array(K_k_child_dot)

        def child_constraint_jacobian_derivative(
            self, Qdot_parent: SegmentNaturalVelocities, Qdot_child: SegmentNaturalVelocities
        ) -> np.ndarray:
            parent_point_location = self.parent_point.position_in_global(Qdot_parent)
            child_point_location = self.child_point.position_in_global(Qdot_child)

            K_k_parent_dot = (
                -2 * (parent_point_location - child_point_location).T @ self.child_point.interpolation_matrix
            )

            return np.array(K_k_parent_dot)

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

        def constraint_jacobian_derivative(
            self, Qdot_parent: SegmentNaturalVelocities, Qdot_child: SegmentNaturalVelocities
        ) -> tuple[np.ndarray, np.ndarray]:
            """
            This function returns the kinematic constraints of the joint, denoted K_k
            as a function of the natural coordinates Q_parent and Q_child.

            Returns
            -------
            tuple[np.ndarray, np.ndarray]
                Kinematic constraints of the joint [1, 12] for parent and child
            """

            K_k_parent_dot = self.parent_constraint_jacobian_derivative(Qdot_parent, Qdot_child)
            K_k_child_dot = self.child_constraint_jacobian_derivative(Qdot_parent, Qdot_child)

            return K_k_parent_dot, K_k_child_dot

        def to_mx(self):
            """
            This function returns the joint as a mx joint

            Returns
            -------
            JointBase
                The joint as a mx joint
            """
            from ..bionc_casadi.joint import Joint as CasadiJoint

            return CasadiJoint.ConstantLength(
                name=self.name,
                parent=self.parent.to_mx(),
                child=self.child.to_mx(),
                index=self.index,
                length=self.length,
                parent_point=self.parent_point.name,
                child_point=self.child_point.name,
                projection_basis=self.projection_basis,
            )


class GroundJoint:
    """
    The public interface to joints with the ground as parent segment.
    """

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
        ):
            super(GroundJoint.Hinge, self).__init__(name, None, child, index, projection_basis)

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

        def parent_constraint_jacobian_derivative(
            self, Qdot_parent: SegmentNaturalVelocities, Qdot_child: SegmentNaturalVelocities
        ) -> np.ndarray:
            return None

        def child_constraint_jacobian_derivative(
            self, Qdot_parent: SegmentNaturalVelocities, Qdot_child: SegmentNaturalVelocities
        ) -> np.ndarray:
            K_k_child = np.zeros((self.nb_constraints, 12))

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

        def constraint_jacobian_derivative(
            self, Qdot_parent: SegmentNaturalVelocities, Qdot_child: SegmentNaturalVelocities
        ) -> np.ndarray:
            """
            This function returns the kinematic constraints of the joint, denoted K_kdot
            as a function of the natural velocities Qdot_parent and Qdot_child.

            Returns
            -------
            np.ndarray
                joint constraints jacobian of the child segment [5, 12]
            """

            return self.child_constraint_jacobian_derivative(Qdot_parent, Qdot_child)

        def to_mx(self):
            """
            This function returns the joint as a mx joint

            Returns
            -------
            JointBase
                The joint as a mx joint
            """
            from ..bionc_casadi.joint import GroundJoint as CasadiGroundJoint

            return CasadiGroundJoint.Hinge(
                name=self.name,
                child=self.child.to_mx(),
                index=self.index,
                parent_axis=self.parent_axis,
                child_axis=self.child_axis,
                theta=self.theta,
                projection_basis=self.projection_basis,
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
        ):
            super(GroundJoint.Spherical, self).__init__(name, None, child, index, projection_basis)
            self.nb_constraints = 3
            self.ground_application_point = ground_application_point

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

        def parent_constraint_jacobian_derivative(
            self, Qdot_parent: SegmentNaturalVelocities, Qdot_child: SegmentNaturalVelocities
        ) -> np.ndarray:
            return None

        def child_constraint_jacobian_derivative(
            self, Qdot_parent: SegmentNaturalVelocities, Qdot_child: SegmentNaturalVelocities
        ) -> np.ndarray:
            K_k_child = np.zeros((self.nb_constraints, 12))

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

        def constraint_jacobian_derivative(
            self, Qdot_parent: SegmentNaturalVelocities, Qdot_child: SegmentNaturalVelocities
        ) -> np.ndarray:
            """
            This function returns the kinematic constraints of the joint, denoted K_kdot
            as a function of the natural velocities Qdot_parent and Qdot_child.

            Returns
            -------
            MX
                joint constraints jacobian of the child segment [3, 12]
            """

            return self.child_constraint_jacobian_derivative(Qdot_parent, Qdot_child)

        def to_mx(self):
            """
            This function returns the joint as a mx joint

            Returns
            -------
            JointBase
                The joint as a mx joint
            """
            from ..bionc_casadi.joint import GroundJoint as CasadiGroundJoint

            return CasadiGroundJoint.Spherical(
                name=self.name,
                child=self.child.to_mx(),
                ground_application_point=self.ground_application_point,
                index=self.index,
                projection_basis=self.projection_basis,
            )

    class Weld(JointBase):
        """
        This joint is defined by 3 constraints to pivot around an axis of the inertial coordinate system
        defined by two angles.
        """

        def __init__(
            self,
            name: str,
            child: NaturalSegment,
            rp_child_ref: SegmentNaturalCoordinates | np.ndarray = None,
            rd_child_ref: SegmentNaturalCoordinates | np.ndarray = None,
            index: int = None,
            projection_basis: EulerSequence = None,
        ):
            super(GroundJoint.Weld, self).__init__(name, None, child, index, projection_basis)

            self.rp_child_ref = rp_child_ref
            self.rd_child_ref = rd_child_ref
            # check size and type of parent axis
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

            return np.concatenate((self.rp_child_ref - Q_child.rp, self.rd_child_ref - Q_child.rd), axis=0)

        def parent_constraint_jacobian(
            self, Q_parent: SegmentNaturalCoordinates, Q_child: SegmentNaturalCoordinates
        ) -> np.ndarray:
            return None

        def child_constraint_jacobian(
            self, Q_parent: SegmentNaturalCoordinates, Q_child: SegmentNaturalCoordinates
        ) -> np.ndarray:
            K_k_child = -np.eye(12)[3:9, :]

            return K_k_child

        def parent_constraint_jacobian_derivative(
            self, Qdot_parent: SegmentNaturalVelocities, Qdot_child: SegmentNaturalVelocities
        ) -> np.ndarray:
            return None

        def child_constraint_jacobian_derivative(
            self, Qdot_parent: SegmentNaturalVelocities, Qdot_child: SegmentNaturalVelocities
        ) -> np.ndarray:
            K_k_child = np.zeros((self.nb_constraints, 12))

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

        def constraint_jacobian_derivative(
            self, Qdot_parent: SegmentNaturalVelocities, Qdot_child: SegmentNaturalVelocities
        ) -> np.ndarray:
            """
            This function returns the kinematic constraints of the joint, denoted K_kdot
            as a function of the natural velocities Qdot_parent and Qdot_child.

            Returns
            -------
            MX
                joint constraints jacobian of the child segment [12, 12]
            """

            return self.child_constraint_jacobian_derivative(Qdot_parent, Qdot_child)

        def to_mx(self):
            """
            This function returns the joint as a mx joint

            Returns
            -------
            JointBase
                The joint as a mx joint
            """
            from ..bionc_casadi.joint import GroundJoint as CasadiGroundJoint

            return CasadiGroundJoint.Weld(
                name=self.name,
                child=self.child.to_mx(),
                index=self.index,
                rp_child_ref=self.rp_child_ref,
                rd_child_ref=self.rd_child_ref,
                projection_basis=self.projection_basis,
            )
