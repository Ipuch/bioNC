from casadi import MX, dot, cos, transpose
import numpy as np

from .natural_segment import NaturalSegment
from .natural_coordinates import SegmentNaturalCoordinates
from .natural_velocities import SegmentNaturalVelocities
from ..protocols.joint import JointBase
from .natural_vector import NaturalVector
from ..utils.enums import NaturalAxis, CartesianAxis
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
        ):

            super(Joint.Hinge, self).__init__(name, parent, child)

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

        def parent_constraint_jacobian(self, Q_child: SegmentNaturalCoordinates) -> MX:
            K_k_parent = MX.zeros((self.nb_constraints, 12))
            K_k_parent[:3, 6:9] = MX.eye(3)

            for i in range(2):
                K_k_parent[i + 3, :] = (
                    transpose(self.parent_vector[i].interpolate().rot)
                    @ self.child_vector[i].interpolate().rot
                    @ Q_child
                )

            return K_k_parent

        def child_constraint_jacobian(self, Q_parent: SegmentNaturalCoordinates) -> MX:
            K_k_child = MX.zeros((self.nb_constraints, 12))
            K_k_child[:3, 3:6] = -MX.eye(3)

            for i in range(2):
                K_k_child[i + 3, :] = (
                    transpose(self.parent_vector[i].interpolate().rot @ Q_parent)
                    @ self.child_vector[i].interpolate().rot
                )

            return K_k_child

        def parent_constraint_jacobian_derivative(self, Qdot_child: SegmentNaturalVelocities) -> MX:
            K_k_parent_dot = MX.zeros((self.nb_constraints, 12))
            for i in range(2):
                K_k_parent_dot[i + 3, :] = (
                    transpose(self.parent_vector[i].interpolate().rot)
                    @ self.child_vector[i].interpolate().rot
                    @ Qdot_child
                )

            return K_k_parent_dot

        def child_constraint_jacobian_derivative(self, Qdot_parent: SegmentNaturalVelocities) -> MX:
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
            return self.parent_constraint_jacobian(Q_child), self.child_constraint_jacobian(Q_parent)

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
            return self.parent_constraint_jacobian_derivative(Qdot_child), self.child_constraint_jacobian_derivative(Qdot_parent)

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
        ):
            super(Joint.Universal, self).__init__(name, parent, child)

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

        def parent_constraint_jacobian(self, Q_child: SegmentNaturalCoordinates) -> MX:
            K_k_parent = MX.zeros((self.nb_constraints, 12))
            K_k_parent[:3, 6:9] = MX.eye(3)

            K_k_parent[3, :] = (
                transpose(self.parent_vector.interpolate().rot) @ self.child_vector.interpolate().rot @ Q_child
            )

            return K_k_parent

        def child_constraint_jacobian(self, Q_parent: SegmentNaturalCoordinates) -> MX:
            K_k_child = MX.zeros((self.nb_constraints, 12))
            K_k_child[:3, 3:6] = -MX.eye(3)
            K_k_child[3, :] = (
                transpose(self.parent_vector.interpolate().rot @ Q_parent) @ self.child_vector.interpolate().rot
            )

            return K_k_child

        def parent_constraint_jacobian_derivative(self, Qdot_child: SegmentNaturalVelocities) -> MX:
            K_k_parent = MX.zeros((self.nb_constraints, 12))

            K_k_parent[3, :] = (
                    transpose(self.parent_vector.interpolate().rot) @ self.child_vector.interpolate().rot @ Qdot_child
            )

            return K_k_parent

        def child_constraint_jacobian_derivative(self, Qdot_parent: SegmentNaturalVelocities) -> MX:
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
            return self.parent_constraint_jacobian(Q_child), self.child_constraint_jacobian(Q_parent)

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
            return self.parent_constraint_jacobian_derivative(Qdot_child), self.child_constraint_jacobian_derivative(Qdot_parent)

    class Spherical(JointBase):
        def __init__(
            self,
            name: str,
            parent: NaturalSegment,
            child: NaturalSegment,
        ):

            super(Joint.Spherical, self).__init__(name, parent, child)
            self.nb_constraints = 3

        def constraint(self, Q_parent: SegmentNaturalCoordinates, Q_child: SegmentNaturalCoordinates) -> MX:
            """
            This function returns the kinematic constraints of the joint, denoted Phi_k
            as a function of the natural coordinates Q_parent and Q_child.

            Returns
            -------
            MX
                Kinematic constraints of the joint [3, 1]
            """
            constraint = Q_parent.rd - Q_child.rp

            return constraint

        def parent_constraint_jacobian(self, Q_child: SegmentNaturalCoordinates) -> MX:
            K_k_parent = MX.zeros((self.nb_constraints, 12))
            K_k_parent[:3, 6:9] = MX.eye(3)

            return K_k_parent

        def child_constraint_jacobian(self, Q_parent: SegmentNaturalCoordinates) -> MX:
            K_k_child = MX.zeros((self.nb_constraints, 12))
            K_k_child[:3, 3:6] = -MX.eye(3)

            return K_k_child

        def parent_constraint_jacobian_derivative(self, Qdot_child: SegmentNaturalVelocities) -> MX:
            K_k_parent = MX.zeros((self.nb_constraints, 12))
            return K_k_parent

        def child_constraint_jacobian_derivative(self, Qdot_parent: SegmentNaturalVelocities) -> MX:
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
            return self.parent_constraint_jacobian(Q_child), self.child_constraint_jacobian(Q_parent)

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
            return self.parent_constraint_jacobian_derivative(Qdot_child), self.child_constraint_jacobian_derivative(Qdot_parent)


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
        ):
            super(GroundJoint.Hinge, self).__init__(name, None, child)

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

        def parent_constraint_jacobian(self, Q_child: SegmentNaturalCoordinates) -> MX:
            return None

        def child_constraint_jacobian(self, Q_parent: SegmentNaturalCoordinates) -> MX:
            K_k_child = MX.zeros((self.nb_constraints, 12))
            K_k_child[:3, 3:6] = -MX.eye(3)

            for i in range(2):
                K_k_child[i + 3, :] = transpose(self.parent_vector[i]) @ self.child_vector[i].interpolate().rot

            return K_k_child

        def parent_constraint_jacobian_derivative(self, Qdot_child: SegmentNaturalVelocities) -> MX:
            return None

        def child_constraint_jacobian_derivative(self, Qdot_parent: SegmentNaturalVelocities) -> MX:
            K_k_child = MX.zeros((self.nb_constraints, 12))

            return K_k_child

        def constraint_jacobian(
            self, Q_parent: SegmentNaturalCoordinates, Q_child: SegmentNaturalCoordinates
        ) -> MX:
            """
            This function returns the kinematic constraints of the joint, denoted K_k
            as a function of the natural coordinates Q_parent and Q_child.

            Returns
            -------
            MX
                joint constraints jacobian of the child segment [5, 12]
            """

            return self.child_constraint_jacobian(Q_parent)

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

            return self.child_constraint_jacobian_derivative(Qdot_parent)
