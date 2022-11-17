from abc import ABC, abstractmethod

import numpy as np

from .natural_segment import AbstractNaturalSegment
from .natural_coordinates import SegmentNaturalCoordinates


class AbstractJoint(ABC):
    """
    This class is made to handle the kinematics of a joint

    Methods
    -------
    constraints(self, Q_parent: NaturalCoordinates, Q_child: NaturalCoordinates) -> np.ndarray
        Returns the constraints of the joint, this defect function should be zero when the joint is in a valid position

    """

    @abstractmethod
    def constraint(self, Q_parent: SegmentNaturalCoordinates, Q_child: SegmentNaturalCoordinates) -> np.ndarray:
        """
        This function returns the constraints of the joint, denoted Phi_k as a function of the natural coordinates Q.

        Returns
        -------
        np.ndarray
            Constraints of the joint
        """

    @abstractmethod
    def constraint_jacobian(
        self, Q_parent: SegmentNaturalCoordinates, Q_child: SegmentNaturalCoordinates
    ) -> np.ndarray:
        """
        This function returns the constraint Jacobians of the joint, denoted K_k
        as a function of the natural coordinates Q_parent and Q_child.

        Returns
        -------
        np.ndarray
            Constraint Jacobians of the joint [3, 2 * nbQ]
        """


class JointBase(AbstractJoint):
    """
    This class is made to handle the kinematics of a joint

    Attributes
    ----------
    joint_name : str
        The name of the joint
    segment_parent : NaturalSegment
        The parent segment of the joint
    segment_child : NaturalSegment
        The child segment of the joint

    Methods
    -------
    constraints(self, Q_parent: NaturalCoordinates, Q_child: NaturalCoordinates) -> np.ndarray
        Returns the constraints of the joint, this defect function should be zero when the joint is in a valid position

    """

    def __init__(
        self,
        joint_name: str,
        segment_parent: AbstractNaturalSegment,
        segment_child: AbstractNaturalSegment,
    ):
        self.joint_name = joint_name
        self.segment_parent = segment_parent
        self.segment_child = segment_child

    @abstractmethod
    def constraint(self, Q_parent: SegmentNaturalCoordinates, Q_child: SegmentNaturalCoordinates) -> np.ndarray:
        """
        This function returns the constraints of the joint, denoted Phi_k as a function of the natural coordinates Q.

        Returns
        -------
        np.ndarray
            Constraints of the joint
        """
        pass

    @abstractmethod
    def constraint_jacobian(
        self, Q_parent: SegmentNaturalCoordinates, Q_child: SegmentNaturalCoordinates
    ) -> np.ndarray:
        """
        This function returns the constraint Jacobians of the joint, denoted K_k
        as a function of the natural coordinates Q_parent and Q_child.

        Returns
        -------
        np.ndarray
            Constraint Jacobians of the joint [3, 2 * nbQ]
        """
        pass