from typing import Union, Tuple, Callable

import numpy as np
from numpy import cos, zeros, sum, dot, transpose
from numpy.linalg import inv

from .homogenous_transform import HomogeneousTransform
from .natural_accelerations import SegmentNaturalAccelerations
from .natural_coordinates import SegmentNaturalCoordinates
from .natural_inertial_parameters import NaturalInertialParameters
from .natural_marker import NaturalMarker, SegmentNaturalVector
from .natural_segment_markers import NaturalSegmentMarkers
from .natural_segment_vectors import NaturalSegmentVectors
from .natural_vector import NaturalVector
from .natural_velocities import SegmentNaturalVelocities
from .transformation_matrix import compute_transformation_matrix
from ..model_creation.protocols import Data
from ..protocols.natural_segment import AbstractNaturalSegment
from ..utils.enums import TransformationMatrixType


class NaturalSegment(AbstractNaturalSegment):
    """
        Class used to define anatomical segment based on natural coordinate.

    Methods
    -------
    to_mx()
        This function returns the segment in MX format.
    from_experimental_Q()
        It builds a NaturalSegment from the segment natural coordinates.
    parameters_from_Q()
        It computes the parameters of the segment from SegmentNaturalCoordinates Q.
    set_experimental_Q_function()
        This function sets the experimental Q function that evaluates Q from marker locations.
    _Qi_from_markers()
        This function evaluates segment natural coordinates Q from markers locations.
    compute_transformation_matrix()
        This function returns the transformation matrix, denoted Bi
    segment_coordinates_system()
        This function computes the segment coordinates from the natural coordinates
    location_from_homogenous_transform()
        This function returns the location of the segment in natural coordinate from its homogenous transform

    rigid_body_constraint()
        This function returns the rigid body constraints of the segment, denoted phi_r
    rigid_body_constraint_jacobian()
        This function returns the jacobian of rigid body constraints of the segment, denoted K_r
    rigid_body_constraint_derivative()
        This function returns the derivative of the rigid body constraints denoted Phi_r_dot
    rigid_body_constraint_jacobian_derivative()
        This function returns the derivative of the Jacobian matrix of the rigid body constraints

    _pseudo_inertia_matrix()
        This function returns the pseudo-inertia matrix of the segment
    _natural_center_of_mass()
        This function computes the center of mass of the segment in the natural coordinate system.
    center_of_mass_position()
        This function returns the position of the center of mass of the segment in the global coordinate system.
    _update_mass_matrix()
        This function returns the generalized mass matrix of the segment
    gravity_force()
        This function returns the gravity_force applied on the segment through gravity force
    differential_algebraic_equation()
        This function returns the differential algebraic equation of the segment

    add_natural_marker()
        This function adds a marker to the segment
    add_natural_vector()
        Add a new natural vector to the segment
    add_natural_marker_from_segment_coordinates()
        Add a new marker to the segment
    add_natural_vector_from_segment_coordinates()
        Add a new marker to the segment

    markers()
        This function returns the position of the markers of the system as a function of the natural coordinates Q also referred as forward kinematics
    marker_constraints()
        This function returns the defects of the marker constraints of the segment, denoted Phi_m
    markers_jacobian()
        This function returns the jacobian of the marker constraints of the segment, denoted K_m
    potential_energy()
        This function returns the potential energy of the segment
    kinetic_energy()
        This function returns the kinetic energy of the segment
    inverse_dynamics()
        Computes inverse dynamics for one segment.


    Attributes
    ----------
    _name : str
        name of the segment
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
    _markers : NaturalSegmentMarkers
        markers of the segment
    _vector : NaturalSegmentVectors
        list of vectors in the segment
    _index : int
        index of the segment in the model
    _is_ground : bool
        is_ground to indicate if the segment is the ground segment
    """

    def __init__(
        self,
        name: str = None,
        alpha: Union[np.ndarray, float, np.float64] = np.pi / 2,
        beta: Union[np.ndarray, float, np.float64] = np.pi / 2,
        gamma: Union[np.ndarray, float, np.float64] = np.pi / 2,
        length: Union[np.ndarray, float, np.float64] = None,
        mass: Union[np.ndarray, float, np.float64] = None,
        natural_center_of_mass: np.ndarray = None,
        natural_pseudo_inertia: np.ndarray = None,
        inertial_transformation_matrix_type: TransformationMatrixType = None,
        index: int = None,
        is_ground: bool = False,
    ):
        self._angle_sanity_check(alpha, beta, gamma)

        super().__init__(
            name=name,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            length=length,
            mass=mass,
            natural_center_of_mass=natural_center_of_mass,
            natural_pseudo_inertia=natural_pseudo_inertia,
            inertial_transformation_matrix_type=inertial_transformation_matrix_type,
            index=index,
            is_ground=is_ground,
        )

        self._markers = NaturalSegmentMarkers()  # list of markers embedded in the segment
        self._vectors = NaturalSegmentVectors()  # list of vectors embedded in the segment

    def __repr__(self) -> str:
        return f"NaturalSegment(name={self.name!r}, index={self.index})"

    def __str__(self) -> str:
        mass_str = f"{self.mass:.4f} kg" if self.mass is not None else "not defined"
        out = f"NaturalSegment: {self.name}\n"
        out += f"  index: {self.index}\n"
        out += f"  length: {self.length:.4f} m\n"
        out += f"  alpha: {np.rad2deg(self.alpha):.2f}°, beta: {np.rad2deg(self.beta):.2f}°, gamma: {np.rad2deg(self.gamma):.2f}°\n"
        out += f"  mass: {mass_str}\n"
        out += f"  markers: {self.nb_markers}, vectors: {self._vectors.nb_vectors}\n"
        return out

    def set_natural_inertial_parameters(
        self, mass: float, natural_center_of_mass: np.ndarray, natural_pseudo_inertia: np.ndarray
    ):
        self._mass = mass
        self._natural_center_of_mass = NaturalVector(natural_center_of_mass)
        self._natural_pseudo_inertia = natural_pseudo_inertia
        self._natural_inertial_parameters = NaturalInertialParameters(
            mass=mass,
            natural_center_of_mass=NaturalVector(natural_center_of_mass),
            natural_pseudo_inertia=natural_pseudo_inertia,
        )
        self._mass_matrix = self._natural_inertial_parameters.mass_matrix

    def set_inertial_parameters(
        self,
        mass: float,
        center_of_mass: np.ndarray,
        inertia_matrix: np.ndarray,
        transformation_matrix: TransformationMatrixType,
    ):
        self._natural_inertial_parameters = NaturalInertialParameters.from_cartesian_inertial_parameters(
            mass=mass,
            center_of_mass=center_of_mass,
            inertia_matrix=inertia_matrix,
            inertial_transformation_matrix=self.compute_transformation_matrix(transformation_matrix),
        )
        self._mass = mass
        self._natural_center_of_mass = self._natural_inertial_parameters.natural_center_of_mass
        self._natural_pseudo_inertia = self.natural_pseudo_inertia
        self._mass_matrix = self._natural_inertial_parameters.mass_matrix

    def center_of_mass(self, transformation_matrix: TransformationMatrixType = None):
        return self._natural_inertial_parameters.center_of_mass(transformation_matrix)

    def to_mx(self) -> AbstractNaturalSegment:
        """
        This function returns the segment in MX format
        """
        from ..bionc_casadi.natural_segment import NaturalSegment as NaturalSegmentCasadi

        natural_segment = NaturalSegmentCasadi(
            name=self.name,
            index=self.index,
            alpha=self.alpha,
            beta=self.beta,
            gamma=self.gamma,
            length=self.length,
            mass=self.mass,
            natural_center_of_mass=self.natural_center_of_mass,
            natural_pseudo_inertia=self.natural_pseudo_inertia,
            inertial_transformation_matrix_type=self._inertial_transformation_matrix_type,
        )
        for marker in self._markers:
            natural_segment.add_natural_marker(marker.to_mx())
        for vector in self._vectors:
            natural_segment.add_natural_vector(vector.to_mx())

        return natural_segment

    @classmethod
    def with_cartesian_inertial_parameters(
        cls,
        name: str = None,
        alpha: Union[np.ndarray, float, np.float64] = np.pi / 2,
        beta: Union[np.ndarray, float, np.float64] = np.pi / 2,
        gamma: Union[np.ndarray, float, np.float64] = np.pi / 2,
        length: Union[np.ndarray, float, np.float64] = None,
        mass: Union[np.ndarray, float, np.float64] = None,
        center_of_mass: np.ndarray = None,
        inertia: np.ndarray = None,
        inertial_transformation_matrix: TransformationMatrixType = TransformationMatrixType.Buv,
        index: int = None,
        is_ground: bool = False,
    ):
        cls._angle_sanity_check(alpha, beta, gamma)

        if inertia.shape != (3, 3):
            raise ValueError("Inertia matrix must be 3x3")

        natural_inertial_parameters = NaturalInertialParameters.from_cartesian_inertial_parameters(
            mass=mass,
            center_of_mass=center_of_mass,
            inertia_matrix=inertia,
            inertial_transformation_matrix=compute_transformation_matrix(
                inertial_transformation_matrix, length, alpha, beta, gamma
            ).T,
        )

        return cls(
            name=name,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            length=length,
            mass=mass,
            natural_center_of_mass=natural_inertial_parameters.natural_center_of_mass,
            natural_pseudo_inertia=natural_inertial_parameters.natural_pseudo_inertia,
            inertial_transformation_matrix_type=inertial_transformation_matrix,
            index=index,
            is_ground=is_ground,
        )

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
        from ..bionc_numpy import SegmentNaturalCoordinates

        Q = SegmentNaturalCoordinates(Q)

        u, v, w = Q.to_uvw()

        length = np.linalg.norm(v, axis=0)
        alpha = np.arccos(np.sum(v * w, axis=0) / length)
        beta = np.arccos(np.sum(u * w, axis=0))
        gamma = np.arccos(np.sum(u * v, axis=0) / length)

        return alpha, beta, gamma, length

    def set_experimental_Q_function(self, function: Callable):
        """
        This function sets the experimental Q function

        Parameters
        ----------
        function : Callable
            The function that returns the experimental Q
        """
        self._experimental_Q_function: Callable = function

    def _Qi_from_markers(self, markers: Data, model) -> SegmentNaturalCoordinates:
        """
        This function sets the experimental Q function

        Parameters
        ----------
        markers : Data
            The markers of all the model
        """

        return self._experimental_Q_function(markers, model)

    def compute_transformation_matrix(self, matrix_type: str | TransformationMatrixType = None) -> np.ndarray:
        """
        This function computes the transformation matrix, denoted Bi,
        from Natural Coordinate System to point to the orthogonal Segment Coordinate System.
        Example : if vector a expressed in (Pi, X, Y, Z), inv(B) * a is expressed in (Pi, ui, vi, wi)

        Parameters
        ----------
        matrix_type : str or TransformationMatrixType
            The type of the transformation matrix to compute, either "Buv" or TransformationMatrixType.Buv

        Returns
        -------
        np.ndarray
            Transformation matrix from natural coordinate to segment coordinate system [3x3]
        """
        if isinstance(matrix_type, str):
            matrix_type = TransformationMatrixType.from_string(matrix_type)

        if matrix_type is None:
            matrix_type = TransformationMatrixType.Buv  # NOTE: default value

        return compute_transformation_matrix(
            matrix_type, length=self.length, alpha=self.alpha, beta=self.beta, gamma=self.gamma
        ).T

    def segment_coordinates_system(
        self, Q: SegmentNaturalCoordinates, transformation_matrix_type: TransformationMatrixType | str = None
    ) -> HomogeneousTransform:
        """
        This function computes the segment coordinates from the natural coordinates

        Parameters
        ----------
        Q: SegmentNaturalCoordinates
            The natural coordinates of the segment
        transformation_matrix_type : TransformationMatrixType or str
            The type of the transformation matrix to compute, either "Buv" or TransformationMatrixType.Buv

        Returns
        -------
        SegmentCoordinates
            The segment coordinates
        """
        if not isinstance(Q, SegmentNaturalCoordinates):
            Q = SegmentNaturalCoordinates(Q)

        return HomogeneousTransform.from_rt(
            # rotation=self.compute_transformation_matrix(transformation_matrix_type)
            # @ np.concatenate((Q.u[:, np.newaxis], Q.v[:, np.newaxis], Q.w[:, np.newaxis]), axis=1),
            rotation=Q.to_uvw_matrix() @
            # NOTE: I would like to make numerical inversion disappear and the transpose too x)
            np.linalg.inv(self.compute_transformation_matrix(transformation_matrix_type).T),
            translation=Q.rp[:, np.newaxis],
        )

    def location_from_homogenous_transform(
        self, T: Union[np.ndarray, HomogeneousTransform]
    ) -> SegmentNaturalCoordinates:
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

        u = self.compute_transformation_matrix @ T[0:3, 0]
        w = self.compute_transformation_matrix @ T[0:3, 2]
        rp = self.compute_transformation_matrix @ T[0:3, 4]
        rd = (T @ np.array([0, self.length, 0, 1]))[0:3]  # not sure of this line.

        return SegmentNaturalCoordinates((u, rp, rd, w))

    def rigid_body_constraint(self, Qi: Union[SegmentNaturalCoordinates, np.ndarray]) -> np.ndarray:
        """
        This function returns the rigid body constraints of the segment, denoted phi_r.

        Returns
        -------
        np.ndarray
            Rigid body constraints of the segment [6 x 1 x N_frame]
        """
        phir = zeros(6)
        u, v, w = Qi.to_uvw()

        phir[0] = sum(u**2) - 1
        phir[1] = dot(u, v) - self.length * cos(self.gamma)
        phir[2] = dot(u, w) - cos(self.beta)
        phir[3] = sum(v**2) - self.length**2
        phir[4] = dot(v, w) - self.length * cos(self.alpha)
        phir[5] = sum(w**2) - 1

        return phir

    @staticmethod
    def rigid_body_constraint_jacobian(Qi: SegmentNaturalCoordinates) -> np.ndarray:
        """
        This function returns the Jacobian matrix of the rigid body constraints denoted K_r

        Returns
        -------
        Kr : np.ndarray
            Jacobian matrix of the rigid body constraints denoted Kr [6 x 12 x N_frame]
        """
        # initialisation
        Kr = zeros((6, 12))

        u, v, w = Qi.to_uvw()

        Kr[0, 0:3] = 2 * u

        Kr[1, 0:3] = v
        Kr[1, 3:6] = u
        Kr[1, 6:9] = -u

        Kr[2, 0:3] = w
        Kr[2, 9:12] = u

        Kr[3, 3:6] = 2 * v
        Kr[3, 6:9] = -2 * v

        Kr[4, 3:6] = w
        Kr[4, 6:9] = -w
        Kr[4, 9:12] = v

        Kr[5, 9:12] = 2 * w

        return Kr

    def rigid_body_constraint_derivative(
        self,
        Qi: SegmentNaturalCoordinates,
        Qdoti: SegmentNaturalVelocities,
    ) -> np.ndarray:
        """
        This function returns the derivative of the rigid body constraints denoted Phi_r_dot

        Parameters
        ----------
        Qi : SegmentNaturalCoordinates
            The natural coordinates of the segment
        Qdoti : SegmentNaturalVelocities
            The natural velocities of the segment

        Returns
        -------
        np.ndarray
            Derivative of the rigid body constraints [6 x 1 x N_frame]
        """

        return self.rigid_body_constraint_jacobian(Qi) @ np.array(Qdoti)

    @staticmethod
    def rigid_body_constraint_jacobian_derivative(Qdoti: SegmentNaturalVelocities) -> np.ndarray:
        """
        This function returns the derivative of the Jacobian matrix of the rigid body constraints denoted Kr_dot [6 x 12 x N_frame]

        Returns
        -------
        Kr_dot : np.ndarray
            derivative of the Jacobian matrix of the rigid body constraints denoted Kr_dot [6 x 12 ]
        """
        # initialisation
        Kr_dot = zeros((6, 12))

        Kr_dot[0, 0:3] = 2 * Qdoti.udot

        Kr_dot[1, 0:3] = Qdoti.vdot
        Kr_dot[1, 3:6] = Qdoti.udot
        Kr_dot[1, 6:9] = -Qdoti.udot

        Kr_dot[2, 0:3] = Qdoti.wdot
        Kr_dot[2, 9:12] = Qdoti.udot

        Kr_dot[3, 3:6] = 2 * Qdoti.vdot
        Kr_dot[3, 6:9] = -2 * Qdoti.vdot

        Kr_dot[4, 3:6] = Qdoti.wdot
        Kr_dot[4, 6:9] = -Qdoti.wdot
        Kr_dot[4, 9:12] = Qdoti.vdot

        Kr_dot[5, 9:12] = 2 * Qdoti.wdot

        return Kr_dot

    def center_of_mass_position(self, Qi: SegmentNaturalCoordinates) -> np.ndarray:
        """
        This function returns the position of the center of mass of the segment in the global coordinate system.

        Returns
        -------
        np.ndarray
            Position of the center of mass of the segment in the global coordinate system [3x1]
        """
        return np.array(self.natural_center_of_mass.interpolate() @ Qi.to_array())

    def gravity_force(self) -> np.ndarray:
        """
        This function returns the gravity_force applied on the segment through gravity force.

        Returns
        -------
        np.ndarray
            Weight applied on the segment through gravity force [12 x 1]
        """

        return (self.natural_center_of_mass.interpolate().T * self.mass) @ np.array([0, 0, -9.81])

    def differential_algebraic_equation(
        self,
        Qi: Union[SegmentNaturalCoordinates, np.ndarray],
        Qdoti: Union[SegmentNaturalVelocities, np.ndarray],
        stabilization: dict = None,
    ) -> Tuple[SegmentNaturalAccelerations, np.ndarray]:
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
        if isinstance(Qi, SegmentNaturalVelocities):
            raise TypeError("Qi should be of type SegmentNaturalCoordinates")
        if isinstance(Qdoti, SegmentNaturalCoordinates):
            raise TypeError("Qdoti should be of type SegmentNaturalVelocities")

        # not able to verify if the types of Qi and Qdoti are np.ndarray
        if not isinstance(Qi, SegmentNaturalCoordinates):
            Qi = SegmentNaturalCoordinates(Qi)
        if not isinstance(Qdoti, SegmentNaturalVelocities):
            Qdoti = SegmentNaturalVelocities(Qdoti)

        Gi = self.mass_matrix
        Kr = self.rigid_body_constraint_jacobian(Qi)
        Krdot = self.rigid_body_constraint_jacobian_derivative(Qdoti)
        biais = -Krdot @ Qdoti.vector

        if stabilization is not None:
            biais -= stabilization["alpha"] * self.rigid_body_constraint(Qi) + stabilization[
                "beta"
            ] * self.rigid_body_constraint_derivative(Qi, Qdoti)

        A = zeros((18, 18))
        A[0:12, 0:12] = Gi
        A[12:18, 0:12] = Kr
        A[0:12, 12:18] = Kr.T
        A[12:, 12:18] = np.zeros((6, 6))

        B = np.concatenate([self.gravity_force(), biais], axis=0)

        # solve the linear system Ax = B with numpy
        x = np.linalg.solve(A, B)
        Qddoti = x[0:12]
        lambda_i = x[12:]
        return SegmentNaturalAccelerations(Qddoti), lambda_i

    def vector_from_name(self, vector_name: str) -> SegmentNaturalVector:
        return self._vectors.vector_from_name(vector_name)

    def add_natural_vector(self, vector: SegmentNaturalVector):
        """
        Add a new vector to the segment

        Parameters
        ----------
        vector
            The vector to add
        """
        if vector.parent_name is not None and vector.parent_name != self.name:
            raise ValueError(
                "The vector name should be the same as the 'key'. Alternatively, vector.name can be left undefined"
            )

        vector.parent_name = self.name
        self._vectors.add(vector)

    def add_natural_vector_from_segment_coordinates(
        self,
        name: str,
        direction: np.ndarray,
        normalize: bool = True,
        transformation_matrix_type: TransformationMatrixType = None,
    ):
        """
        Add a new marker to the segment

        Parameters
        ----------
        name: str
            The name of the vector
        direction: np.ndarray
            The location of the vector in the segment coordinate system
        normalize: bool
            True if the vector should be normalized, False otherwise
        transformation_matrix_type : TransformationMatrixType
            The type of the transformation matrix to compute, TransformationMatrixType.Buv by default
        """

        direction = direction / np.linalg.norm(direction) if normalize else direction
        direction = inv(self.compute_transformation_matrix(transformation_matrix_type)) @ direction

        natural_vector = SegmentNaturalVector(
            name=name,
            parent_name=self.name,
            direction=direction,
        )
        self._vectors.add(natural_vector)

    @property
    def nb_markers(self) -> int:
        return self._markers.nb_markers

    @property
    def nb_markers_technical(self) -> int:
        return self._markers.nb_markers_technical

    @property
    def marker_names(self) -> list[str]:
        return self._markers.marker_names

    @property
    def marker_names_technical(self) -> list[str]:
        return self._markers.marker_names_technical

    def marker_from_name(self, marker_name: str) -> NaturalMarker:
        return self._markers.marker_from_name(marker_name)

    def add_natural_marker(self, marker: NaturalMarker):
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
        self._markers.add(marker)

    def add_natural_marker_from_segment_coordinates(
        self,
        name: str,
        location: np.ndarray,
        is_distal_location: bool = False,
        is_technical: bool = True,
        is_anatomical: bool = False,
        transformation_matrix_type: TransformationMatrixType = None,
    ):
        """
        Add a new marker to the segment

        Parameters
        ----------
        name: str
            The name of the marker
        location: np.ndarray
            The location of the marker in the segment coordinate system
        is_distal_location: bool
            The location of the distal marker in the segment coordinate system
        is_technical: bool
            True if the marker is technical, False otherwise
        is_anatomical: bool
            True if the marker is anatomical, False otherwise
        transformation_matrix_type : TransformationMatrixType
            The type of the transformation matrix to compute, TransformationMatrixType.Buv by default
        """

        location = inv(self.compute_transformation_matrix(transformation_matrix_type)) @ location
        if is_distal_location:
            location += np.array([0, -1, 0])

        natural_marker = NaturalMarker(
            name=name,
            parent_name=self.name,
            position=location,
            is_technical=is_technical,
            is_anatomical=is_anatomical,
        )
        self._markers.add(natural_marker)

    def markers(self, Qi: SegmentNaturalCoordinates) -> np.ndarray:
        return self._markers.positions(Qi)

    def marker_constraints(
        self, marker_locations: np.ndarray, Qi: SegmentNaturalCoordinates, only_technical: bool = True
    ) -> np.ndarray:
        return self._markers.constraints(marker_locations, Qi, only_technical)

    def markers_jacobian(self, only_technical: bool = True) -> np.ndarray:
        return self._markers.jacobian(only_technical)

    def potential_energy(self, Qi: SegmentNaturalCoordinates) -> float:
        """
        This function returns the potential energy of the segment

        Parameters
        ----------
        Qi: SegmentNaturalCoordinates
            Natural coordinates of the segment

        Returns
        -------
        float
            Potential energy of the segment
        """
        return (self.mass * self.natural_center_of_mass.interpolate() @ Qi.vector)[2]

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
        return 0.5 * transpose(Qdoti.to_array()) @ self.mass_matrix @ Qdoti.to_array()

    def inverse_dynamics(
        self,
        Qi: SegmentNaturalCoordinates,
        Qddoti: SegmentNaturalAccelerations,
        subtree_intersegmental_generalized_forces: np.ndarray,
        segment_external_forces: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """

        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray]
            The generalized forces [3 x 1], torques [3 x 1], and lagrange multipliers [6 x 1]
        """

        proximal_interpolation_matrix = NaturalVector.proximal().interpolate()
        pseudo_interpolation_matrix = Qi.compute_pseudo_interpolation_matrix()
        rigid_body_constraints_jacobian = self.rigid_body_constraint_jacobian(Qi=Qi)

        # make a matrix out of it, todo: would be great to know if there is an analytical way to compute this matrix
        front_matrix = np.hstack(
            (
                proximal_interpolation_matrix.T,
                pseudo_interpolation_matrix.T,
                -rigid_body_constraints_jacobian.T,
            )
        )

        b = (
            (self.mass_matrix @ Qddoti)[:, np.newaxis]
            - self.gravity_force()[:, np.newaxis]
            - segment_external_forces
            - subtree_intersegmental_generalized_forces
        )

        # compute the generalized forces
        generalized_forces = np.linalg.inv(front_matrix) @ b

        return (generalized_forces[:3, 0], generalized_forces[3:6, 0], generalized_forces[6:, 0])
