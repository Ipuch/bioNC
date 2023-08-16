from abc import ABC, abstractmethod

from .natural_segment import AbstractNaturalSegment
from .natural_coordinates import SegmentNaturalCoordinates
from .natural_velocities import SegmentNaturalVelocities
from ..utils.enums import EulerSequence, TransformationMatrixType


class JointBase(ABC):
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
    index : int
        The index of the joint in the model
    projection_basis : EulerSequence
        The euler sequence of the joint, used for post computation, not directly related to natural coordinates
        it can be used to project the joint angles or the joint torques on a specific euler projection_basis

    Methods
    -------
    constraints(self, Q_parent: NaturalCoordinates, Q_child: NaturalCoordinates) -> np.ndarray
        Returns the constraints of the joint, this defect function should be zero when the joint is in a valid position
    constraint_jacobian(self, Q_parent: NaturalCoordinates, Q_child: NaturalCoordinates) -> np.ndarray
        Returns the jacobian of the constraints of the joint
    parent_constraint_jacobian(self, Q_parent: NaturalCoordinates, Q_child: NaturalCoordinates) -> np.ndarray
        Returns the jacobian of the constraints of the joint with respect to the parent segment
    child_constraint_jacobian(self, Q_parent: NaturalCoordinates, Q_child: NaturalCoordinates) -> np.ndarray
        Returns the jacobian of the constraints of the joint with respect to the child segment

    """

    def __init__(
        self,
        name: str,
        parent: AbstractNaturalSegment,
        child: AbstractNaturalSegment,
        index: int,
        projection_basis: EulerSequence = EulerSequence.ZXY,  # biomechanics default isb
        parent_basis: TransformationMatrixType = TransformationMatrixType.Bwu,  # by default as eulersequence starts with Z (~W)
        child_basis: TransformationMatrixType = TransformationMatrixType.Bvu,  # by default as eulersequence ends with Y (~V)
    ):
        self.name = name
        self.parent = parent
        self.child = child
        self.index = index
        self.projection_basis = projection_basis
        self.parent_basis = parent_basis
        self.child_basis = child_basis

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
            Constraint Jacobians of the joint [n, 2 * nbQ]
        """

    @abstractmethod
    def parent_constraint_jacobian(self, Q_parent: SegmentNaturalCoordinates, Q_child: SegmentNaturalCoordinates):
        """
        This function returns the parent constraint Jacobians of the joint, denoted K_k
        as a function of the natural coordinates Q_child.

        Returns
        -------
            Constraint Jacobians of the joint [n, nbQ]
        """

    @abstractmethod
    def child_constraint_jacobian(self, Q_parent: SegmentNaturalCoordinates, Q_child: SegmentNaturalCoordinates):
        """
        This function returns the child constraint Jacobians of the joint, denoted K_k
        as a function of the natural coordinates Q_parent.

        Returns
        -------
            Constraint Jacobians of the joint [n, nbQ]
        """

    @abstractmethod
    def parent_constraint_jacobian_derivative(
        self, Qdot_parent: SegmentNaturalVelocities, Qdot_child: SegmentNaturalVelocities
    ):
        """
        This function returns the derivative of the parent constraint Jacobians of the joint, denoted K_k
        as a function of the natural velocities Qdot_child.

        Returns
        -------
            derivative of Constraint Jacobians of the joint [n, 12]
        """

    @abstractmethod
    def child_constraint_jacobian_derivative(
        self, Qdot_parent: SegmentNaturalVelocities, Qdot_child: SegmentNaturalVelocities
    ):
        """
        This function returns the derivative of the child constraint Jacobians of the joint, denoted K_k
        as a function of the natural velocities Qdot_parent.

        Returns
        -------
            derivative of Constraint Jacobians of the joint [n, 12]
        """
