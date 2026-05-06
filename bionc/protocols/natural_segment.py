import numpy as np
from abc import ABC, abstractmethod
from casadi import MX
from typing import Union

from .homogenous_transform import HomogeneousTransform
from .natural_accelerations import SegmentNaturalAccelerations
from .natural_coordinates import SegmentNaturalCoordinates
from .natural_markers import AbstractNaturalMarker
from .natural_velocities import SegmentNaturalVelocities
from ..utils.enums import TransformationMatrixType


class AbstractNaturalSegment(ABC):
    """
    This class represents a generic natural segment for both MX and numpy

    Attributes
    ----------
    _name : str
        name of the segment
    _index : int
        index of the segment
    _length : float
        length of the segment
    _alpha : float
        angle between u and w
    _beta : float
        angle between w and (rp-rd)
    _gamma : float
        angle between (rp-rd) and u
    _mass : float
        mass of the segment in Segment Coordinate System
    _center_of_mass : np.ndarray
        center of mass of the segment in Segment Coordinate System
    _inertia: np.ndarray
        inertia matrix of the segment in Segment Coordinate System [3x3]
    """

    def __init__(
        self,
        name: str = None,
        alpha: Union[MX, float, np.float64] = np.pi / 2,
        beta: Union[MX, float, np.float64] = np.pi / 2,
        gamma: Union[MX, float, np.float64] = np.pi / 2,
        length: Union[MX, float, np.float64] = None,
        mass: Union[MX, float, np.float64] = None,
        natural_center_of_mass: Union[MX, np.ndarray] = None,
        natural_pseudo_inertia: Union[MX, np.ndarray] = None,
        inertial_transformation_matrix_type: TransformationMatrixType = None,
        index: int = None,
        is_ground: bool = False,
    ):
        self._name = name
        self._index = index

        self._length = length
        self._alpha = alpha
        self._beta = beta
        self._gamma = gamma

        self._mass = mass
        self._natural_center_of_mass = natural_center_of_mass
        self._natural_pseudo_inertia = natural_pseudo_inertia
        self._mass_matrix = None
        self._natural_inertial_parameters = None
        self._inertial_transformation_matrix_type = inertial_transformation_matrix_type

        if mass is not None and natural_center_of_mass is not None and natural_pseudo_inertia is not None:
            self.set_natural_inertial_parameters(mass, natural_center_of_mass, natural_pseudo_inertia)
            self._natural_inertial_parameters._initial_transformation_matrix = self.compute_transformation_matrix(
                inertial_transformation_matrix_type
            )

        # to know if the segment is the ground
        self._is_ground = is_ground

    @staticmethod
    def _angle_sanity_check(alpha: np.ndarray, beta: np.ndarray, gamma: np.ndarray):
        """
        This function checks if angles would produce a singular transformation matrix
        """
        if 1 - np.cos(beta) ** 2 - (np.cos(alpha) - np.cos(beta) * np.cos(gamma)) / np.sin(beta) ** 2 < 0:
            raise ValueError(
                f"The angles alpha, beta, gamma, would produce a singular transformation matrix for the segment"
            )

    def set_name(self, name: str):
        """
        This function sets the name of the segment

        Parameters
        ----------
        name : str
            Name of the segment
        """
        self._name = name

    def set_index(self, index: int):
        """
        This function sets the index of the segment

        Parameters
        ----------
        index : int
            Index of the segment
        """
        self._index = index

    def _set_is_ground(self, is_ground: bool):
        """
        This function sets the segment as the ground
        """
        self._is_ground = is_ground

    @classmethod
    def from_experimental_Q(
        cls,
        Qi: SegmentNaturalCoordinates,
    ) -> "NaturalSegment":
        """
        Parameters
        ----------
        Qi : SegmentNaturalCoordinates
            Experimental segment natural coordinates (12 x n_frames)

        Returns
        -------
        NaturalSegment
        """

        alpha = np.zeros(Qi.shape[1])
        beta = np.zeros(Qi.shape[1])
        gamma = np.zeros(Qi.shape[1])
        length = np.zeros(Qi.shape[1])

        for i, Qif in enumerate(Qi.vector.T):
            alpha[i], beta[i], gamma[i], length[i] = cls.parameters_from_Q(Qif)

        return cls(
            alpha=np.mean(alpha, axis=0),
            beta=np.mean(beta, axis=0),
            gamma=np.mean(gamma, axis=0),
            length=np.mean(length, axis=0),
        )

    @staticmethod
    def parameters_from_Q(Q: SegmentNaturalCoordinates) -> tuple:
        """
        This function computes the parameters of the segment from the natural coordinates

        Parameters
        ----------
        Q: SegmentNaturalCoordinates
            The natural coordinates of the segment

        Returns
        -------
        tuple
            The parameters of the segment (alpha, beta, gamma, length)
        """

    @property
    def name(self):
        return self._name

    @property
    def index(self):
        return self._index

    @property
    def coordinates_slice(self):
        return slice(12 * self._index, 12 * (self._index + 1))

    @property
    def length(self):
        return self._length

    @property
    def alpha(self):
        return self._alpha

    @property
    def beta(self):
        return self._beta

    @property
    def gamma(self):
        return self._gamma

    @property
    def mass(self):
        return self._mass

    @property
    def natural_center_of_mass(self):
        return self._natural_center_of_mass

    @property
    def natural_pseudo_inertia(self):
        return self._natural_pseudo_inertia

    @abstractmethod
    def set_natural_inertial_parameters(self, mass, natural_center_of_mass, natural_pseudo_inertia):
        """
        This function sets the natural inertial parameters of the segment

        Parameters
        ----------
        mass
            The mass of the segment
        natural_center_of_mass
            The center of mass of the segment in the natural coordinate system
        natural_pseudo_inertia
            The pseudo inertia matrix of the segment in the natural coordinate system
        """

    @abstractmethod
    def set_inertial_parameters(self, mass, center_of_mass, inertia_matrix, transformation_matrix_type):
        """
        This function sets the natural inertial parameters of the segment

        Parameters
        ----------
        mass
            The mass of the segment
        center_of_mass
            The center of mass of the segment in the segment coordinate system
        inertia_matrix
            The inertia matrix of the segment in the segment coordinate system
        transformation_matrix_type
            The transformation matrix type
        """

    @abstractmethod
    def center_of_mass(self, transformation_matrix: TransformationMatrixType):
        """
        This function returns the center of mass of the segment in a given coordinate system
        specified by the transformation matrix

        Parameters
        ----------
        transformation_matrix:
            The transformation matrix from the natural coordinate system to the desired coordinate system

        Returns
        -------
            Center of mass of the segment in the desired coordinate system [3x1]
        """

    @property
    def mass_matrix(self):
        """
        This function returns the generalized mass matrix of the segment, denoted G_i.

        Returns
        -------
            mass matrix of the segment [12 x 12]
        """

        return self._mass_matrix

    @abstractmethod
    def compute_transformation_matrix(self, transformation_matrix_type: TransformationMatrixType):
        """
        This function returns the transformation matrix, denoted Bi,
        from Natural Coordinate System to point to the orthogonal Segment Coordinate System.
        Example : if vector a expressed in (Pi, X, Y, Z), inv(B) * a is expressed in (Pi, ui, vi, wi)

        Returns
        -------
            Transformation matrix from natural coordinate to segment coordinate system [3x3]
        """

    @abstractmethod
    def segment_coordinates_system(self, Q: SegmentNaturalCoordinates) -> HomogeneousTransform:
        """
        This function computes the segment coordinates from the natural coordinates

        Parameters
        ----------
        Q: SegmentNaturalCoordinates
            The natural coordinates of the segment
        """

    @abstractmethod
    def location_from_homogenous_transform(
        self,
        T,
    ) -> SegmentNaturalCoordinates:
        """
        This function returns the location of the segment in natural coordinate from its homogenous transform

        Parameters
        ----------
        T: np.ndarray or HomogeneousTransform
            Homogenous transform of the segment Ti which transforms from the local frame (Oi, Xi, Yi, Zi)
            to the global frame (Xi, Yi, Zi)
        """

    @abstractmethod
    def rigid_body_constraint(self, Qi: Union[SegmentNaturalCoordinates, np.ndarray]) -> MX:
        """
        This function returns the rigid body constraints of the segment, denoted phi_r.
        """

    @staticmethod
    def rigid_body_constraint_jacobian(Qi: SegmentNaturalCoordinates):
        """
        This function returns the Jacobian matrix of the rigid body constraints denoted K_r
        """

    @abstractmethod
    def rigid_body_constraint_derivative(
        self,
        Qi: SegmentNaturalCoordinates,
        Qdoti: SegmentNaturalVelocities,
    ):
        """
        This function returns the derivative of the rigid body constraints denoted Phi_r_dot

        Parameters
        ----------
        Qi : SegmentNaturalCoordinates
            The natural coordinates of the segment
        Qdoti : SegmentNaturalVelocities
            The natural velocities of the segment
        """

    @staticmethod
    def compute_pseudo_inertia_matrix(
        mass,
        cartesian_center_of_mass,
        cartesian_inertia,
        transformation_matrix,
    ):
        """
        This function returns the pseudo-inertia matrix of the segment, denoted J_i.
        It transforms the inertia matrix of the segment in the segment coordinate system to the natural coordinate system.
        """

    @abstractmethod
    def gravity_force(self):
        """
        This function returns the gravity_force applied on the segment through gravity force.
        """

    @abstractmethod
    def differential_algebraic_equation(
        self,
        Qi: Union[SegmentNaturalCoordinates, np.ndarray],
        Qdoti: Union[SegmentNaturalVelocities, np.ndarray],
        stabilization: dict = None,
    ) -> tuple:
        """
        This function returns the differential algebraic equation of the segment

        Parameters
        ----------
        Qi: SegmentNaturalCoordinates
            Natural coordinates of the segment
        Qdoti: SegmentNaturalCoordinates
            Derivative of the natural coordinates of the segment
        stabilization: dict
            Dictionary containing the Baumgarte's stabilization parameters:
            * alpha: float
                Stabilization parameter for the constraint
            * beta: float
                Stabilization parameter for the constraint derivative
        """

    @abstractmethod
    def add_natural_marker(self, marker: AbstractNaturalMarker):
        """
        Add a new marker to the segment

        Parameters
        ----------
        marker
            The marker to add
        """

    @abstractmethod
    def marker_from_name(self, marker_name: str) -> AbstractNaturalMarker:
        """
        This function returns the marker with the given name

        Parameters
        ----------
        marker_name: str
            Name of the marker
        """

    @abstractmethod
    def nb_markers(self) -> int:
        """Returns the number of markers of the segment"""

    @abstractmethod
    def nb_markers_technical(self) -> int:
        """Returns the number of technical markers of the segment"""

    @abstractmethod
    def marker_names(self) -> list[str]:
        """Returns the names of the markers of the segment"""

    @abstractmethod
    def marker_names_technical(self) -> list[str]:
        """Returns the names of the technical markers of the segment"""

    @abstractmethod
    def marker_constraints(self, marker_locations: np.ndarray, Qi: SegmentNaturalCoordinates) -> MX:
        """
        This function returns the marker constraints of the segment

        Parameters
        ----------
        marker_locations: np.ndarray
            Marker locations in the global/inertial coordinate system (3 x N_markers)
        Qi: SegmentNaturalCoordinates
            Natural coordinates of the segment
        """

    @abstractmethod
    def markers_jacobian(self):
        """
        This function returns the marker jacobian of the segment
        """

    @abstractmethod
    def potential_energy(self, Qi: SegmentNaturalCoordinates):
        """
        This function returns the potential energy of the segment

        Parameters
        ----------
        Qi: SegmentNaturalCoordinates
            Natural coordinates of the segment

        Returns
        -------
            Potential energy of the segment
        """

    @abstractmethod
    def kinetic_energy(self, Qdoti: SegmentNaturalVelocities) -> float:
        """
        This function returns the kinetic energy of the segment

        Parameters
        ----------
        Qdoti: SegmentNaturalVelocities
            Derivative of the natural coordinates of the segment

        Returns
        -------
        float
            Kinetic energy of the segment
        """

    def to_mx(self):
        """
        This function returns the segment as a MX object
        """
        raise NotImplementedError("This function is only implemented for the bionc_casadi")

    @abstractmethod
    def inverse_dynamics(
        self,
        Qi: SegmentNaturalCoordinates,
        Qddoti: SegmentNaturalAccelerations,
        subtree_intersegmental_generalized_forces: np.ndarray | MX,
        segment_external_forces: np.ndarray | MX,
    ) -> tuple[np.ndarray | MX, np.ndarray | MX, np.ndarray | MX]:
        """
        This function computes the inverse dynamics of a segment.

        Parameters
        ----------
        Qi: SegmentNaturalCoordinates
            The generalized coordinates of the segment
        Qddoti: SegmentNaturalAccelerations
            The generalized accelerations of the segment
        subtree_intersegmental_generalized_forces : np.ndarray | MX
            The generalized forces applied to the segment by its children
        segment_external_forces : np.ndarray | MX
            The generalized forces applied to the segment by the external forces

        Returns
        -------
        force: np.ndarray | MX
            The force generated by the segment
        torque: np.ndarray | MX
            The torque generated by the segment
        lambdas: np.ndarray | MX
            The forces generated by the rigid body constraints
        """
