from abc import ABC, abstractmethod

from .natural_coordinates import SegmentNaturalCoordinates
from .natural_segment import AbstractNaturalSegment
from .natural_velocities import SegmentNaturalVelocities
from ..utils.enums import EulerSequence, TransformationMatrixType, CartesianAxis


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
    parent_basis : TransformationMatrixType
        The transformation to get the orthogonal parent basis of the joint,
        used for post computation, to compute minimal coordinates afterward
    child_basis : TransformationMatrixType
        The transformation to get the orthogonal child basis of the joint,
        used for post computation, to compute minimal coordinates afterward
    nb_constraints : int
        The number of constraints of the joint, e.g. 3 for a spherical joint, 5 for a revolute joint, etc...
    translation_coordinates : tuple[CartesianAxis, CartesianAxis, CartesianAxis]
        The translation basis of the joint, used for post computation, to compute minimal coordinates afterward


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
        projection_basis: EulerSequence = None,
        parent_basis: TransformationMatrixType = None,
        child_basis: TransformationMatrixType = None,
        translation_coordinates: tuple[CartesianAxis, CartesianAxis, CartesianAxis] = None,
    ):
        self.name = name
        self.parent = parent
        self.child = child
        self.index = index
        self.projection_basis = projection_basis or EulerSequence.ZXY  # biomechanics default isb
        self.parent_basis = (
            parent_basis or TransformationMatrixType.Bwu
        )  # by default as eulersequence starts with Z (~W)
        self.child_basis = child_basis or TransformationMatrixType.Bvu  # by default as eulersequence ends with Y (~V)
        self.nb_constraints = 0
        self.translation_coordinates = translation_coordinates or (CartesianAxis.X, CartesianAxis.Y, CartesianAxis.Z)

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

        ```math
        K_k = \frac{\partial \Phi_k}{\partial Q_{parent}}
        ```

        Returns
        -------
            Constraint Jacobians of the joint [n, nbQ]
        """

    @abstractmethod
    def child_constraint_jacobian(self, Q_parent: SegmentNaturalCoordinates, Q_child: SegmentNaturalCoordinates):
        """
        This function returns the child constraint Jacobians of the joint, denoted K_k
        as a function of the natural coordinates Q_parent.

        ```math
        K_k = \frac{\partial \Phi_k}{\partial Q_{child}}
        ```

        Returns
        -------
            Constraint Jacobians of the joint [n, nbQ]
        """

    @property
    def nb_joint_dof(self) -> int:
        """
        Returns the number of degrees of freedom of the joint

        Returns
        -------

        """
        return 6 - self.nb_constraints


class JointBaseWithTwoSegments(JointBase, ABC):
    """
    This class is made to handle the kinematics of a joint with two segments
    """

    def __init__(
        self,
        name: str,
        parent: AbstractNaturalSegment,
        child: AbstractNaturalSegment,
        index: int,
        projection_basis: EulerSequence = None,
        parent_basis: TransformationMatrixType = None,
        child_basis: TransformationMatrixType = None,
        translation_coordinates: tuple[CartesianAxis, CartesianAxis, CartesianAxis] = None,
    ):
        if not isinstance(parent, AbstractNaturalSegment) or parent is None:
            raise ValueError(f"You must provide a parent segment for the joint {name}.")

        super(JointBaseWithTwoSegments, self).__init__(
            name, parent, child, index, projection_basis, parent_basis, child_basis, translation_coordinates
        )
