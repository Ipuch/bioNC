from typing import Union
from abc import ABC, abstractmethod
import numpy as np
from casadi import MX
from .natural_coordinates import SegmentNaturalCoordinates
from .natural_velocities import SegmentNaturalVelocities
from .homogenous_transform import HomogeneousTransform
from .natural_markers import AbstractNaturalMarker


class AbstractNaturalSegment(ABC):
    """
        Class used to define anatomical segment based on natural coordinate.

    Methods
    -------
    set_name
        This method is used to set the name of the segment.
    set_index
        This method is used to set the index of the segment.
    name
        This method is used to get the name of the segment.
    index
        This method is used to get the index of the segment.
    alpha
        This method is used to return the alpha angle of the segment.
    beta
        This method is used to return the beta angle of the segment.
    gamma
        This method is used to return the gamma angle of the segment.
    length
        This method is used to return the length of the segment.
    mass
        This method is used to return the mass of the segment.
    inertia
        This method is used to return the inertia of the segment.
    center_of_mass
        This method is used to return the center of mass of the segment.
    transformation_matrix()
        This function returns the transformation matrix, denoted Bi.
    rigid_body_constraint()
        This function returns the rigid body constraints of the segment, denoted phi_r.
    rigid_body_constraint_jacobian()
        This function returns the jacobian of rigid body constraints of the segment, denoted K_r.


    add_natural_marker()
        This function adds a marker to the segment
    nb_markers
        This function returns the number of markers in the segment
    marker_constraints()
        This function returns the defects of the marker constraints of the segment, denoted Phi_m
    markers_jacobian()
        This function returns the jacobian of the marker constraints of the segment, denoted K_m

    """

    @abstractmethod
    def set_name(self, name: str):
        """
        This function sets the name of the segment

        Parameters
        ----------
        name : str
            Name of the segment
        """

    @abstractmethod
    def set_index(self, index: int):
        """
        This function sets the index of the segment

        Parameters
        ----------
        index : int
            Index of the segment
        """

    @abstractmethod
    def name(self):
        """
        This function returns the name of the segment

        Returns
        -------
        str
            Name of the segment
        """

    @abstractmethod
    def index(self):
        """
        This function returns the index of the segment

        Returns
        -------
        int
            Index of the segment
        """

    @abstractmethod
    def length(self):
        """
        This function returns the length of the segment

        Returns
        -------
        float
            Length of the segment
        """

    @abstractmethod
    def alpha(self):
        """
        This function returns the alpha angle of the segment

        Returns
        -------
        float
            Alpha angle of the segment
        """

    @abstractmethod
    def beta(self):
        """
        This function returns the beta angle of the segment

        Returns
        -------
        float
            Beta angle of the segment
        """

    @abstractmethod
    def gamma(self):
        """
        This function returns the gamma angle of the segment

        Returns
        -------
        float
            Gamma angle of the segment
        """

    @abstractmethod
    def mass(self):
        """
        This function returns the mass of the segment

        Returns
        -------
        float
            Mass of the segment
        """

    @abstractmethod
    def center_of_mass(self):
        """
        This function returns the center of mass of the segment

        Returns
        -------
        np.ndarray
            Center of mass of the segment
        """

    @abstractmethod
    def inertia(self):
        """
        This function returns the inertia matrix of the segment

        Returns
        -------
        np.ndarray
            Inertia matrix of the segment
        """

    @abstractmethod
    def _transformation_matrix(self):
        """
        This function computes the transformation matrix, denoted Bi,
        from Natural Coordinate System to point to the orthogonal Segment Coordinate System.
        Example : if vector a expressed in (Pi, X, Y, Z), inv(B) * a is expressed in (Pi, ui, vi, wi)

        Returns
        -------
        np.ndarray
            Transformation matrix from natural coordinate to segment coordinate system [3x3]
        """

    @abstractmethod
    def transformation_matrix(self):
        """
        This function returns the transformation matrix, denoted Bi,
        from Natural Coordinate System to point to the orthogonal Segment Coordinate System.
        Example : if vector a expressed in (Pi, X, Y, Z), inv(B) * a is expressed in (Pi, ui, vi, wi)

        Returns
        -------
        np.ndarray
            Transformation matrix from natural coordinate to segment coordinate system [3x3]
        """

    @abstractmethod
    def segment_coordinates_system(self, Q):
        """
        This function computes the segment coordinates from the natural coordinates

        Parameters
        ----------
        Q: SegmentNaturalCoordinates
            The natural coordinates of the segment

        Returns
        -------
        SegmentCoordinates
            The segment coordinates
        """

    @abstractmethod
    def location_from_homogenous_transform(self, T):
        """
        This function returns the location of the segment in natural coordinate from its homogenous transform

        Parameters
        ----------
        T: np.ndarray or HomogeneousTransform
            Homogenous transform of the segment Ti which transforms from the local frame (Oi, Xi, Yi, Zi)
            to the global frame (Xi, Yi, Zi)

        Returns
        -------
        np.ndarray
            Location of the segment [3 x 1]
        """

    @abstractmethod
    def rigid_body_constraint(self, Qi):
        """
        This function returns the rigid body constraints of the segment, denoted phi_r.

        Returns
        -------
        np.ndarray
            Rigid body constraints of the segment [6 x 1 x N_frame]
        """

    @abstractmethod
    def rigid_body_constraint_derivative(self, Qi):
        """
        This function returns the rigid body constraints derivative of the segment, denoted dphi_r/dQ.

        Returns
        -------
        np.ndarray
            Rigid body constraints derivative of the segment [6 x 6 x N_frame]
        """

    @abstractmethod
    def _pseudo_inertia_matrix(self):
        """
        This function returns the pseudo-inertia matrix of the segment, denoted J_i.
        It transforms the inertia matrix of the segment in the segment coordinate system to the natural coordinate system.

        Returns
        -------
        np.ndarray
            Pseudo-inertia matrix of the segment in the natural coordinate system [3x3]
        """

    @abstractmethod
    def pseudo_inertia_matrix(self):
        """
        This function returns the pseudo-inertia matrix of the segment, denoted J_i.
        It transforms the inertia matrix of the segment in the segment coordinate system to the natural coordinate system.

        Returns
        -------
        np.ndarray
            Pseudo-inertia matrix of the segment in the natural coordinate system [3x3]
        """

    @abstractmethod
    def _natural_center_of_mass(self):
        """
        This function computes the center of mass of the segment in the natural coordinate system.
        It transforms the center of mass of the segment in the segment coordinate system to the natural coordinate system.

        Returns
        -------
        np.ndarray
            Center of mass of the segment in the natural coordinate system [3x1]
        """

    @abstractmethod
    def natural_center_of_mass(self):
        """
        This function returns the center of mass of the segment in the natural coordinate system.
        It transforms the center of mass of the segment in the segment coordinate system to the natural coordinate system.

        Returns
        -------
        np.ndarray
            Center of mass of the segment in the natural coordinate system [3x1]
        """

    @abstractmethod
    def center_of_mass_position(self, Qi: SegmentNaturalCoordinates):
        """
        This function returns the position of the center of mass of the segment in the global coordinate system.

        Returns
        -------
            Position of the center of mass of the segment in the global coordinate system [3x1]
        """

    @abstractmethod
    def _update_mass_matrix(self):
        """
        This function returns the generalized mass matrix of the segment, denoted G_i.

        Returns
        -------
        np.ndarray
            mass matrix of the segment [12 x 12]
        """

    @abstractmethod
    def mass_matrix(self):
        """
        This function returns the generalized mass matrix of the segment, denoted G_i.

        Returns
        -------
        np.ndarray
            mass matrix of the segment [12 x 12]
        """

    @abstractmethod
    def weight(self):
        """
        This function returns the weight applied on the segment through gravity force.

        Returns
        -------
        np.ndarray
            Weight applied on the segment through gravity force [12 x 1]
        """

    @abstractmethod
    def differential_algebraic_equation(
        self,
        Qi,
        Qdoti,
        stabilization,
    ):
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


        Returns
        -------
        np.ndarray
            Differential algebraic equation of the segment [12 x 1]
        """

    @abstractmethod
    def add_natural_marker(self, marker):
        """
        Add a new marker to the segment

        Parameters
        ----------
        marker
            The marker to add
        """

    @abstractmethod
    def nb_markers(self):
        """
        Returns the number of markers of the natural segment

        Returns
        -------
        int
            Number of markers of the segment
        """

    @abstractmethod
    def nb_markers_technical(self):
        """
        Returns the number of technical markers of the natural segment

        Returns
        -------
        int
            Number of technical markers of the segment
        """

    @abstractmethod
    def marker_names(self) -> list[str]:
        """
        Returns the names of the markers of the natural segment

        Returns
        -------
        list[str]
            Names of the markers of the segment
        """

    @abstractmethod
    def marker_names_technical(self) -> list[str]:
        """
        Returns the names of the technical markers (involved in experimental markers)

        Returns
        -------
        list[str]
            Names of the technical markers of the segment
        """

    @abstractmethod
    def marker_constraints(self, marker_locations, Qi):
        """
        This function returns the marker constraints of the segment

        Parameters
        ----------
        marker_locations: np.ndarray
            Marker locations in the global/inertial coordinate system (3 x N_markers)
        Qi: SegmentNaturalCoordinates
            Natural coordinates of the segment

        Returns
        -------
        np.ndarray
            The defects of the marker constraints of the segment (3 x N_markers)
        """

    @abstractmethod
    def markers_jacobian(self):
        """
        This function returns the marker jacobian of the segment

        Returns
        -------
        np.ndarray
            The jacobian of the marker constraints of the segment (3 x N_markers)
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
    def kinetic_energy(self, Qdoti: SegmentNaturalCoordinates):
        """
        This function returns the kinetic energy of the segment

        Parameters
        ----------
        Qdoti: SegmentNaturalCoordinates
            Derivative of the natural coordinates of the segment

        Returns
        -------
            Kinetic energy of the segment
        """


class GenericNaturalSegment(AbstractNaturalSegment):
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
        inertia matrix of the segment in Segment Coordinate System
    """

    def __init__(
        self,
        name: str = None,
        alpha: Union[MX, float, np.float64] = np.pi / 2,
        beta: Union[MX, float, np.float64] = np.pi / 2,
        gamma: Union[MX, float, np.float64] = np.pi / 2,
        length: Union[MX, float, np.float64] = None,
        mass: Union[MX, float, np.float64] = None,
        center_of_mass: Union[MX, np.ndarray] = None,
        inertia: Union[MX, np.ndarray] = None,
        index: int = None,
    ):

        self._name = name
        self._index = index

        self._length = length
        self._alpha = alpha
        self._beta = beta
        self._gamma = gamma

        # todo: sanity check to make sure u, v or w are not collinear
        # todo: implement all the transformations matrix according the Ph.D thesis of Alexandre Naaim
        self._transformation_matrix = self._transformation_matrix()

        self._mass = mass
        if center_of_mass is None:
            self._center_of_mass = center_of_mass
            self._natural_center_of_mass = None
        else:
            if center_of_mass.shape[0] != 3:
                raise ValueError("Center of mass must be 3x1")
            self._center_of_mass = center_of_mass
            self._natural_center_of_mass = self._natural_center_of_mass()

        if inertia is None:
            self._inertia = inertia
            self._inertia_in_natural_coordinates_system = None
            self._interpolation_matrix_inertia = None
            self._mass_matrix = None
        else:
            if inertia.shape != (3, 3):
                raise ValueError("Inertia matrix must be 3x3")
            self._inertia = inertia
            self._pseudo_inertia_matrix = self._pseudo_inertia_matrix()
            self._mass_matrix = self._update_mass_matrix()

        # list of markers embedded in the segment
        self._markers = []

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
        pass

    @property
    def name(self):
        return self._name

    @property
    def index(self):
        return self._index

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
    def center_of_mass(self):
        return self._center_of_mass

    @property
    def inertia(self):
        return self._inertia

    @property
    def mass_matrix(self):
        """
        This function returns the generalized mass matrix of the segment, denoted G_i.

        Returns
        -------
            mass matrix of the segment [12 x 12]
        """

        return self._mass_matrix

    @property
    def transformation_matrix(self):
        """
        This function returns the transformation matrix, denoted Bi,
        from Natural Coordinate System to point to the orthogonal Segment Coordinate System.
        Example : if vector a expressed in (Pi, X, Y, Z), inv(B) * a is expressed in (Pi, ui, vi, wi)

        Returns
        -------
            Transformation matrix from natural coordinate to segment coordinate system [3x3]
        """
        return self._transformation_matrix

    @property
    def pseudo_inertia_matrix(self):
        """
        This function returns the pseudo-inertia matrix of the segment, denoted J_i.
        It transforms the inertia matrix of the segment in the segment coordinate system to the natural coordinate system.

        Returns
        -------
            Pseudo-inertia matrix of the segment in the natural coordinate system [3x3]
        """
        return self._pseudo_inertia_matrix

    @property
    def natural_center_of_mass(self):
        """
        This function returns the center of mass of the segment in the natural coordinate system.
        It transforms the center of mass of the segment in the segment coordinate system to the natural coordinate system.

        Returns
        -------
            Center of mass of the segment in the natural coordinate system [3x1]
        """
        return self._natural_center_of_mass

    def _transformation_matrix(self) -> MX:
        """
        This function computes the transformation matrix, denoted Bi,
        from Natural Coordinate System to point to the orthogonal Segment Coordinate System.
        Example : if vector a expressed in (Pi, X, Y, Z), inv(B) * a is expressed in (Pi, ui, vi, wi)
        """
        pass

    def segment_coordinates_system(self, Q: SegmentNaturalCoordinates) -> HomogeneousTransform:
        """
        This function computes the segment coordinates from the natural coordinates

        Parameters
        ----------
        Q: SegmentNaturalCoordinates
            The natural coordinates of the segment
        """
        pass

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
        pass

    def rigid_body_constraint(self, Qi: Union[SegmentNaturalCoordinates, np.ndarray]) -> MX:
        """
        This function returns the rigid body constraints of the segment, denoted phi_r.
        """
        pass

    @staticmethod
    def rigid_body_constraint_jacobian(Qi: SegmentNaturalCoordinates):
        """
        This function returns the Jacobian matrix of the rigid body constraints denoted K_r
        """
        pass

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
        pass

    @staticmethod
    def rigid_body_constraint_jacobian_derivative(Qdoti: SegmentNaturalVelocities):
        """
        This function returns the derivative of the Jacobian matrix of the rigid body constraints denoted Kr_dot [6 x 12 x N_frame]
        """
        pass

    def _pseudo_inertia_matrix(self):
        """
        This function returns the pseudo-inertia matrix of the segment, denoted J_i.
        It transforms the inertia matrix of the segment in the segment coordinate system to the natural coordinate system.
        """
        pass

    def _natural_center_of_mass(self):
        """
        This function computes the center of mass of the segment in the natural coordinate system.
        It transforms the center of mass of the segment in the segment coordinate system to the natural coordinate system.
        """
        pass

    def _update_mass_matrix(self):
        """
        This function returns the generalized mass matrix of the segment, denoted G_i.
        """
        pass

    def weight(self):
        """
        This function returns the weight applied on the segment through gravity force.
        """
        pass

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

        pass

    def add_natural_marker(self, marker: AbstractNaturalMarker):
        """
        Add a new marker to the segment

        Parameters
        ----------
        marker
            The marker to add
        """
        if marker.parent_name is not None and marker.parent_name != self.name:
            raise ValueError(
                "The marker name should be the same as the 'key'. Alternatively, marker.name can be left undefined"
            )

        marker.parent_name = self.name
        self._markers.append(marker)

    @property
    def nb_markers(self) -> int:
        return len(self._markers)

    @property
    def nb_markers_technical(self) -> int:
        return len(self.marker_names_technical)

    @property
    def marker_names(self) -> list[str]:
        return [marker.name for marker in self._markers]

    @property
    def marker_names_technical(self) -> list[str]:
        return [marker.name for marker in self._markers if marker.is_technical]

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
        pass

    def markers_jacobian(self):
        """
        This function returns the marker jacobian of the segment
        """
        pass

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
        pass

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
        pass
