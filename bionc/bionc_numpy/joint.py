from abc import ABC, abstractmethod

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
            segment_parent: NaturalSegment,
            segment_child: NaturalSegment,
            theta_1: float,
            theta_2: float,
        ):

            super(Joint.Hinge, self).__init__(joint_name, segment_parent, segment_child)
            self.theta_1 = theta_1
            self.theta_2 = theta_2
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
            constraint = np.zeros(3)
            constraint[0] = Q_parent.rd - Q_child.rp
            constraint[1] = np.dot(Q_parent.w, Q_child.rp - Q_child.rd) - self.segment_child.length * np.cos(
                self.theta_1
            )
            constraint[2] = np.dot(Q_parent.w, Q_child.u) - np.cos(self.theta_2)

            return constraint

    class Universal(JointBase, ABC):
        def __init__(
            self,
            joint_name: str,
            segment_parent: NaturalSegment,
            segment_child: NaturalSegment,
            theta: float,
        ):

            super(Joint.Universal, self).__init__(joint_name, segment_parent, segment_child)
            self.theta = theta
            self.nb_constraints = 2

        def constraint(self, Q_parent: SegmentNaturalCoordinates, Q_child: SegmentNaturalCoordinates) -> np.ndarray:
            """
            This function returns the kinematic constraints of the joint, denoted Phi_k
            as a function of the natural coordinates Q_parent and Q_child.

            Returns
            -------
            np.ndarray
                Kinematic constraints of the joint [2, 1]
            """
            N = np.zeros((3, 12))  # interpolation matrix of the given axis

            constraint = np.zeros(2)
            constraint[0] = Q_parent.rd - Q_child.rp
            constraint[1] = np.dot(Q_parent.w, np.matmul(N, Q_child.vector)) - np.cos(self.theta)

            return constraint

    class Spherical(JointBase):
        def __init__(
            self,
            joint_name: str,
            segment_parent: NaturalSegment,
            segment_child: NaturalSegment,
            point_interpolation_matrix_in_child: float = None,
        ):

            super(Joint.Spherical, self).__init__(joint_name, segment_parent, segment_child)
            self.nb_constraints = 1
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
                Kinematic constraints of the joint [2, 1]
            """
            N = np.zeros((3, 12))  # interpolation matrix of the given axis

            constraint = np.zeros(1)
            if self.point_interpolation_matrix_in_child is None:
                constraint[0] = Q_parent.rd - Q_child.rp
            else:
                constraint[0] = np.matmul(self.point_interpolation_matrix_in_child, Q_parent.vector) - Q_child.rd

            return constraint

        def constraint_jacobian(self, Q_parent: SegmentNaturalCoordinates, Q_child: SegmentNaturalCoordinates) -> np.ndarray:
            """
            This function returns the kinematic constraints of the joint, denoted Phi_k
            as a function of the natural coordinates Q_parent and Q_child.

            Returns
            -------
            np.ndarray
                Kinematic constraints of the joint [2, 1]
            """
            raise(NotImplementedError, "This function is not implemented yet")


# todo : more to come
