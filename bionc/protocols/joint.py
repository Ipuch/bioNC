from abc import ABC, abstractmethod

from .natural_segment import AbstractNaturalSegment
from .natural_coordinates import SegmentNaturalCoordinates


class AbstractJoint(ABC):
    """
    This class is made to handle the kinematics of a joint

    Methods
    -------
    constraints(self, Q_parent: NaturalCoordinates, Q_child: NaturalCoordinates)
        Returns the constraints of the joint, this defect function should be zero when the joint is in a valid position

    """

    @abstractmethod
    def constraint(self, Q_parent: SegmentNaturalCoordinates, Q_child: SegmentNaturalCoordinates):
        """
        This function returns the constraints of the joint, denoted Phi_k as a function of the natural coordinates Q.

        Returns
        -------
            Constraints of the joint
        """

    @abstractmethod
    def constraint_jacobian(self, Q_parent: SegmentNaturalCoordinates, Q_child: SegmentNaturalCoordinates):
        """
        This function returns the constraint Jacobians of the joint, denoted K_k
        as a function of the natural coordinates Q_parent and Q_child.

        Returns
        -------
            Constraint Jacobians of the joint [3, 2 * nbQ]
        """


class JointBase(AbstractJoint):
    """
    This class is made to handle the kinematics of a joint

    Attributes
    ----------
    name : str
        The name of the joint
    parent : NaturalSegment
        The parent segment of the joint
    child : NaturalSegment
        The child segment of the joint

    Methods
    -------
    constraints(self, Q_parent: NaturalCoordinates, Q_child: NaturalCoordinates) -> np.ndarray
        Returns the constraints of the joint, this defect function should be zero when the joint is in a valid position

    """

    def __init__(
        self,
        name: str,
        parent: AbstractNaturalSegment,
        child: AbstractNaturalSegment,
    ):
        self.name = name
        self.parent = parent
        self.child = child

    @abstractmethod
    def constraint(self, Q_parent: SegmentNaturalCoordinates, Q_child: SegmentNaturalCoordinates):
        """
        This function returns the constraints of the joint, denoted Phi_k as a function of the natural coordinates Q.

        Returns
        -------
            Constraints of the joint
        """
        pass

    @abstractmethod
    def constraint_jacobian(self, Q_parent: SegmentNaturalCoordinates, Q_child: SegmentNaturalCoordinates):
        """
        This function returns the constraint Jacobians of the joint, denoted K_k
        as a function of the natural coordinates Q_parent and Q_child.

        Returns
        -------
            Constraint Jacobians of the joint [3, 2 * nbQ]
        """
        pass
