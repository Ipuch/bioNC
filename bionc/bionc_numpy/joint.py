import numpy as np

from .natural_segment import NaturalSegment
from .natural_coordinates import SegmentNaturalCoordinates
from ..protocols.joint import JointBase


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
            joint_name: str,
            parent: NaturalSegment,
            child: NaturalSegment,
            theta_1: float,
            theta_2: float,
        ):

            super(Joint.Hinge, self).__init__(joint_name, parent, child)
            self.theta_1 = theta_1
            self.theta_2 = theta_2
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
            constraint[3] = np.dot(Q_parent.w, Q_child.rp - Q_child.rd) - self.child.length * np.cos(self.theta_1)
            constraint[4] = np.dot(Q_parent.w, Q_child.u) - np.cos(self.theta_2)

            return constraint

        def constraint_jacobian(self, Q_parent: SegmentNaturalCoordinates, Q_child: SegmentNaturalCoordinates) -> np.ndarray:
            """
            This function returns the jacobian of the kinematic constraints of the joint, denoted Phi_k
            as a function of the natural coordinates Q_parent and Q_child.

            Returns
            -------
            MX
                Jacobian of the kinematic constraints of the joint
            """
            raise NotImplementedError("This function is not implemented yet")

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
                joint_name=self.joint_name,
                parent=self.parent.to_mx(),
                child=self.child.to_mx(),
                theta_1=self.theta_1,
                theta_2=self.theta_2,
            )

    class Universal(JointBase):
        def __init__(
            self,
            joint_name: str,
            parent: NaturalSegment,
            child: NaturalSegment,
            theta: float,
        ):

            super(Joint.Universal, self).__init__(joint_name, parent, child)
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
            N = np.zeros((3, 12))
            constraint = np.zeros(self.nb_constraints)
            constraint[:3] = Q_parent.rd - Q_child.rp
            constraint[3] = np.dot(Q_parent.w, N @ Q_child.vector) - np.cos(self.theta)

            return constraint

        def constraint_jacobian(self, Q_parent: SegmentNaturalCoordinates, Q_child: SegmentNaturalCoordinates) -> np.ndarray:
            """
            This function returns the jacobian of the kinematic constraints of the joint, denoted Phi_k
            as a function of the natural coordinates Q_parent and Q_child.

            Returns
            -------
            MX
                Jacobian of the kinematic constraints of the joint [1, 24]
            """
            raise NotImplementedError("This function is not implemented yet")

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
                joint_name=self.joint_name,
                parent=self.parent.to_mx(),
                child=self.child.to_mx(),
                theta=self.theta,
            )

    class Spherical(JointBase):
        def __init__(
            self,
            joint_name: str,
            parent: NaturalSegment,
            child: NaturalSegment,
            point_interpolation_matrix_in_child: float = None,
        ):

            super(Joint.Spherical, self).__init__(joint_name, parent, child)
            self.nb_constraints = 3
            # todo: do something better
            # this thing is not none if the joint is not located at rp nor at rd and it needs to be used
            self.point_interpolation_matrix_in_child = point_interpolation_matrix_in_child

        def constraint(self, Q_parent: SegmentNaturalCoordinates, Q_child: SegmentNaturalCoordinates) -> np.ndarray:
            """
            This function returns the kinematic constraints of the joint, denoted Phi_k
            as a function of the natural coordinates Q_parent and Q_child.

            Returns
            -------
            np.ndarray
                Kinematic constraints of the joint [3, 1]
            """

            # if self.point_interpolation_matrix_in_child is None:
            constraint = Q_parent.rd - Q_child.rp
            # else:
            #     constraint = self.point_interpolation_matrix_in_child @ Q_parent.vector - Q_child.rd

            return constraint

        def constraint_jacobian(
            self, Q_parent: SegmentNaturalCoordinates, Q_child: SegmentNaturalCoordinates
        ) -> np.ndarray:
            """
            This function returns the kinematic constraints of the joint, denoted Phi_k
            as a function of the natural coordinates Q_parent and Q_child.

            Returns
            -------
            np.ndarray
                Kinematic constraints of the joint [2, 1]
            """
            raise (NotImplementedError, "This function is not implemented yet")

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
                joint_name=self.joint_name,
                parent=self.parent.to_mx(),
                child=self.child.to_mx(),
                point_interpolation_matrix_in_child=self.point_interpolation_matrix_in_child,
            )


# todo : more to come
