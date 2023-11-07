from typing import Union, Tuple, Callable

import numpy as np
from numpy import cos, sin, eye, zeros, sum, dot, transpose
from numpy.linalg import inv

from ..model_creation.protocols import Data
from ..bionc_numpy.natural_coordinates import SegmentNaturalCoordinates
from ..bionc_numpy.natural_velocities import SegmentNaturalVelocities
from ..bionc_numpy.natural_accelerations import SegmentNaturalAccelerations
from ..bionc_numpy.homogenous_transform import HomogeneousTransform
from ..bionc_numpy.natural_marker import NaturalMarker, SegmentNaturalVector
from ..bionc_numpy.natural_vector import NaturalVector
from ..bionc_numpy.transformation_matrix import transformation_matrix
from ..utils.enums import TransformationMatrixType, transformation_matrix_str_to_enum

from ..protocols.natural_segment import AbstractNaturalSegment


class NaturalSegment(AbstractNaturalSegment):
    """
        Class used to define anatomical segment based on natural coordinate.

    Methods
    -------
    to_mx()
        This function returns the segment in MX format
    from_experimental_Q()
                This function sets the segment from the natural coordinates
        parameters_from_Q()
                It computes the parameters of the segment from SegmentNaturalCoordinates Q
    set_experimental_Q_function()
        This function sets the experimental Q function that evaluates Q from marker locations
    _Qi_from_markers()
        This function evaluates segment natural coordinates Q from markers locations.

    transformation_matrix()
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
        Add a new natual vector to the segment
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
        Computes inverse dynamics for one segment


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
    _markers : list
        list of markers in the segment
    _vector : list
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
        center_of_mass: np.ndarray = None,
        inertia: np.ndarray = None,
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
            center_of_mass=center_of_mass,
            inertia=inertia,
            index=index,
            is_ground=is_ground,
        )

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
            center_of_mass=self.center_of_mass,
            inertia=self.inertia,
        )
        for marker in self._markers:
            natural_segment.add_natural_marker(marker.to_mx())
        for vector in self._vectors:
            natural_segment.add_natural_vector(vector.to_mx())

        return natural_segment

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

    def transformation_matrix(self, matrix_type: str | TransformationMatrixType = None) -> np.ndarray:
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
            matrix_type = transformation_matrix_str_to_enum(matrix_type)

        if matrix_type is None:
            matrix_type = TransformationMatrixType.Buv  # NOTE: default value

        return transformation_matrix(
            matrix_type,
            length=self.length,
            alpha=self.alpha,
            beta=self.beta,
            gamma=self.gamma,
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
            rotation=self.transformation_matrix(transformation_matrix_type)
            @ np.concatenate((Q.u[:, np.newaxis], Q.v[:, np.newaxis], Q.w[:, np.newaxis]), axis=1),
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

        u = self.transformation_matrix @ T[0:3, 0]
        w = self.transformation_matrix @ T[0:3, 2]
        rp = self.transformation_matrix @ T[0:3, 4]
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

    def _pseudo_inertia_matrix(self) -> np.ndarray:
        """
        This function returns the pseudo-inertia matrix of the segment, denoted J_i.
        It transforms the inertia matrix of the segment in the segment coordinate system to the natural coordinate system.

        Returns
        -------
        np.ndarray
            Pseudo-inertia matrix of the segment in the natural coordinate system [3x3]
        """
        # todo: verify the formula
        middle_block = (
            self.inertia
            + self.mass * np.dot(self.center_of_mass.T, self.center_of_mass) * eye(3)
            - np.dot(self.center_of_mass.T, self.center_of_mass)
        )

        Binv = inv(self.transformation_matrix())
        Binv_transpose = np.transpose(Binv)

        return Binv @ (middle_block @ Binv_transpose)

    @property
    def pseudo_inertia_matrix(self) -> np.ndarray:
        """
        This function returns the pseudo-inertia matrix of the segment, denoted J_i.
        It transforms the inertia matrix of the segment in the segment coordinate system to the natural coordinate system.

        Returns
        -------
        np.ndarray
            Pseudo-inertia matrix of the segment in the natural coordinate system [3x3]
        """
        return self._pseudo_inertia_matrix

    def _natural_center_of_mass(self) -> NaturalVector:
        """
        This function computes the center of mass of the segment in the natural coordinate system.
        It transforms the center of mass of the segment in the segment coordinate system to the natural coordinate system.

        Returns
        -------
        np.ndarray
            Center of mass of the segment in the natural coordinate system [3x1]
        """
        return NaturalVector(inv(self.transformation_matrix()) @ self.center_of_mass)

    @property
    def natural_center_of_mass(self) -> NaturalVector:
        """
        This function returns the center of mass of the segment in the natural coordinate system.
        It transforms the center of mass of the segment in the segment coordinate system to the natural coordinate system.

        Returns
        -------
        np.ndarray
            Center of mass of the segment in the natural coordinate system [3x1]
        """
        return self._natural_center_of_mass

    def center_of_mass_position(self, Qi: SegmentNaturalCoordinates) -> np.ndarray:
        """
        This function returns the position of the center of mass of the segment in the global coordinate system.

        Returns
        -------
        np.ndarray
            Position of the center of mass of the segment in the global coordinate system [3x1]
        """
        return np.array(self.natural_center_of_mass.interpolate() @ Qi.to_array())

    def _update_mass_matrix(self) -> np.ndarray:
        """
        This function returns the generalized mass matrix of the segment, denoted G_i.

        Returns
        -------
        np.ndarray
            mass matrix of the segment [12 x 12]
        """

        Ji = self.pseudo_inertia_matrix
        n_ci = self.natural_center_of_mass

        Gi = zeros((12, 12))

        Gi[0:3, 0:3] = Ji[0, 0] * eye(3)
        Gi[0:3, 3:6] = (self.mass * n_ci[0] + Ji[0, 1]) * eye(3)
        Gi[0:3, 6:9] = -Ji[0, 1] * eye(3)
        Gi[0:3, 9:12] = Ji[0, 2] * eye(3)

        Gi[3:6, 3:6] = (self.mass + 2 * self.mass * n_ci[1] + Ji[1, 1]) * eye(3)
        Gi[3:6, 6:9] = -(self.mass * n_ci[1] + Ji[1, 1]) * eye(3)
        Gi[3:6, 9:12] = (self.mass * n_ci[2] + Ji[1, 2]) * eye(3)

        Gi[6:9, 6:9] = Ji[1, 1] * eye(3)
        Gi[6:9, 9:12] = -Ji[1, 2] * eye(3)

        Gi[9:12, 9:12] = Ji[2, 2] * eye(3)

        # symmetrize the matrix
        Gi[3:6, 0:3] = Gi[0:3, 3:6]
        Gi[6:9, 0:3] = Gi[0:3, 6:9]
        Gi[9:12, 0:3] = Gi[0:3, 9:12]

        Gi[6:9, 3:6] = Gi[3:6, 6:9]
        Gi[9:12, 3:6] = Gi[3:6, 9:12]

        Gi[9:12, 6:9] = Gi[6:9, 9:12]

        return Gi

    @property
    def mass_matrix(self) -> np.ndarray:
        """
        This function returns the generalized mass matrix of the segment, denoted G_i.

        Returns
        -------
        np.ndarray
            mass matrix of the segment [12 x 12]
        """

        return self._mass_matrix

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
        self._markers.append(marker)

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
                "The marker name should be the same as the 'key'. Alternatively, marker.name can be left undefined"
            )

        vector.parent_name = self.name
        self._vectors.append(vector)

    def add_natural_marker_from_segment_coordinates(
        self,
        name: str,
        location: np.ndarray,
        is_distal_location: bool = False,
        is_technical: bool = True,
        is_anatomical: bool = False,
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
        """

        location = inv(self.transformation_matrix()) @ location
        if is_distal_location:
            location += np.array([0, -1, 0])

        natural_marker = NaturalMarker(
            name=name,
            parent_name=self.name,
            position=location,
            is_technical=is_technical,
            is_anatomical=is_anatomical,
        )
        self.add_natural_marker(natural_marker)

    def add_natural_vector_from_segment_coordinates(
        self,
        name: str,
        direction: np.ndarray,
        normalize: bool = True,
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
        """

        direction = direction / np.linalg.norm(direction) if normalize else direction
        direction = inv(self.transformation_matrix()) @ direction

        natural_vector = SegmentNaturalVector(
            name=name,
            parent_name=self.name,
            direction=direction,
        )
        self.add_natural_vector(natural_vector)

    def markers(self, Qi: SegmentNaturalCoordinates) -> np.ndarray:
        """
        This function returns the position of the markers of the system as a function of the natural coordinates Q
        also referred as forward kinematics

        Parameters
        ----------
        Qi : SegmentNaturalCoordinates
            The natural coordinates of the segment [12 x n, 1]

        Returns
        -------
        np.ndarray
            The position of the markers [3, nbMarkers, nbFrames]
            in the global coordinate system/ inertial coordinate system
        """
        if not isinstance(Qi, SegmentNaturalCoordinates):
            Qi = SegmentNaturalCoordinates(Qi)

        markers = np.zeros((3, self.nb_markers, Qi.shape[1]))
        for i, marker in enumerate(self._markers):
            markers[:, i, :] = marker.position_in_global(Qi)

        return markers

    def marker_constraints(
        self, marker_locations: np.ndarray, Qi: SegmentNaturalCoordinates, only_technical: bool = True
    ) -> np.ndarray:
        """
        This function returns the marker constraints of the segment

        Parameters
        ----------
        marker_locations: np.ndarray
            Marker locations in the global/inertial coordinate system [3,N_markers]
        Qi: SegmentNaturalCoordinates
            Natural coordinates of the segment
        only_technical: bool
            If True, only the constraints of technical markers are returned, by default True

        Returns
        -------
        np.ndarray
            The defects of the marker constraints of the segment (3 x N_markers)
        """
        nb_markers = self.nb_markers_technical if only_technical else self.nb_markers
        markers = [m for m in self._markers if m.is_technical] if only_technical else self._markers

        if marker_locations.shape != (3, nb_markers):
            raise ValueError(f"marker_locations should be of shape (3, {nb_markers})")

        defects = np.zeros((3, nb_markers))

        for i, marker in enumerate(markers):
            defects[:, i] = marker.constraint(marker_location=marker_locations[:, i], Qi=Qi)

        return defects

    def markers_jacobian(self, only_technical: bool = True) -> np.ndarray:
        """
        This function returns the marker jacobian of the segment

        Parameters
        ----------
        only_technical: bool
            If True, only the constraints jacobian of technical markers are returned, by default True

        Returns
        -------
        np.ndarray
            The jacobian of the marker constraints of the segment [3, N_markers]
        """
        nb_markers = self.nb_markers_technical if only_technical else self.nb_markers
        markers = [m for m in self._markers if m.is_technical] if only_technical else self._markers
        return np.vstack([-marker.interpolation_matrix for marker in markers]) if nb_markers > 0 else np.array([])

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
