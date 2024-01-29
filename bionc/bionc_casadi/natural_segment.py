from typing import Union, Tuple

import numpy as np
from casadi import MX
from casadi import cos, transpose, vertcat, inv, dot, sum1, horzcat, solve

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
from .utils import to_numeric_MX
from ..protocols.natural_segment import AbstractNaturalSegment
from ..utils.enums import TransformationMatrixType


class NaturalSegment(AbstractNaturalSegment):
    """
        Class used to define anatomical segment based on natural coordinate.

    Methods
    -------
    compute_transformation_matrix()
        This function returns the transformation matrix, denoted Bi
    rigid_body_constraint()
        This function returns the rigid body constraints of the segment, denoted phi_r
    rigid_body_constraint_jacobian()
        This function returns the jacobian of rigid body constraints of the segment, denoted K_r

    add_natural_marker()
        This function adds a marker to the segment
    nb_markers
        This function returns the number of markers in the segment
    marker_constraints()
        This function returns the defects of the marker constraints of the segment, denoted Phi_m
    markers_jacobian()
        This function returns the jacobian of the marker constraints of the segment, denoted K_m

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
    _index : int
        index of the segment in the model
    _is_ground : bool
        is_ground to indicate if the segment is the ground segment
    _markers : NaturalSegmentMarkers
        markers of the segment
    _vectors : NaturalSegmentVectors
        vectors of the segment
    """

    def __init__(
            self,
            name: str = None,
            index: int = None,
            alpha: Union[MX, float, np.float64, np.ndarray] = np.pi / 2,
            beta: Union[MX, float, np.float64, np.ndarray] = np.pi / 2,
            gamma: Union[MX, float, np.float64, np.ndarray] = np.pi / 2,
            length: Union[MX, float, np.float64, np.ndarray] = None,
            mass: Union[MX, float, np.float64, np.ndarray] = None,
            natural_center_of_mass: Union[MX, np.ndarray] = None,
            natural_pseudo_inertia: Union[MX, np.ndarray] = None,
            inertial_transformation_matrix_type: TransformationMatrixType = None,
            is_ground: bool = False,
    ):
        if not isinstance(alpha, MX) and not isinstance(beta, MX) and not isinstance(gamma, MX):
            self._angle_sanity_check(alpha, beta, gamma)

        if natural_pseudo_inertia is not None:
            natural_pseudo_inertia = MX(natural_pseudo_inertia)
        if natural_center_of_mass is not None:
            natural_center_of_mass = NaturalVector(natural_center_of_mass)

        super().__init__(
            name=name,
            alpha=MX(alpha),
            beta=MX(beta),
            gamma=MX(gamma),
            length=MX(length),
            mass=mass,
            natural_center_of_mass=natural_center_of_mass,
            natural_pseudo_inertia=natural_pseudo_inertia,
            inertial_transformation_matrix_type=inertial_transformation_matrix_type,
            index=index,
            is_ground=is_ground,
        )

        self._markers = NaturalSegmentMarkers()
        self._vectors = NaturalSegmentVectors()

    def set_natural_inertial_parameters(
            self, mass: float, natural_center_of_mass: np.ndarray, natural_pseudo_inertia: np.ndarray
    ):
        self._mass = mass
        self._natural_center_of_mass = natural_center_of_mass
        self._natural_pseudo_inertia = natural_pseudo_inertia
        self._natural_inertial_parameters = NaturalInertialParameters(
            mass=mass,
            natural_center_of_mass=natural_center_of_mass,
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

    @classmethod
    def with_cartesian_inertial_parameters(
            cls,
            name: str = None,
            alpha: Union[MX, float, np.float64, np.ndarray] = np.pi / 2,
            beta: Union[MX, float, np.float64, np.ndarray] = np.pi / 2,
            gamma: Union[MX, float, np.float64, np.ndarray] = np.pi / 2,
            length: Union[MX, float, np.float64, np.ndarray] = None,
            mass: Union[MX, float, np.float64, np.ndarray] = None,
            center_of_mass: Union[MX, np.ndarray] = None,
            inertia: Union[MX, np.ndarray] = None,
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

        u, rp, rd, w = Q.to_components()

        length = np.linalg.norm(rp - rd)
        alpha = np.arccos(np.sum((rp - rd) * w, axis=0) / length)
        beta = np.arccos(np.sum(u * w, axis=0))
        gamma = np.arccos(np.sum(u * (rp - rd), axis=0) / length)

        return alpha, beta, gamma, length

    def compute_transformation_matrix(self, matrix_type: str | TransformationMatrixType = None) -> MX:
        """
        This function computes the transformation matrix, denoted Bi,
        from Natural Coordinate System to point to the orthogonal Segment Coordinate System.
        Example : if vector a expressed in (Pi, X, Y, Z), inv(B) * a is expressed in (Pi, ui, vi, wi)

        Returns
        -------
        MX
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
            self,
            Q: SegmentNaturalCoordinates,
            transformation_matrix_type: TransformationMatrixType | str = None,
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
            rotation=self.compute_transformation_matrix(transformation_matrix_type) @ horzcat(Q.u, Q.v, Q.w),
            translation=Q.rp,
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
        MX
            Location of the segment [3 x 1]
        """

        u = self.compute_transformation_matrix @ T[0:3, 0]
        w = self.compute_transformation_matrix @ T[0:3, 2]
        rp = self.compute_transformation_matrix @ T[0:3, 4]
        rd = (T @ MX([0, self.length, 0, 1]))[0:3]  # not sure of this line.

        return SegmentNaturalCoordinates((u, rp, rd, w))

    def rigid_body_constraint(self, Qi: Union[SegmentNaturalCoordinates, np.ndarray]) -> MX:
        """
        This function returns the rigid body constraints of the segment, denoted phi_r.

        Returns
        -------
        MX
            Rigid body constraints of the segment [6 x 1 x N_frame]
        """

        phir = MX.zeros(6)
        u, v, w = Qi.to_uvw()

        phir[0] = sum1(u ** 2) - 1
        phir[1] = dot(u, v) - self.length * cos(self.gamma)
        phir[2] = dot(u, w) - cos(self.beta)
        phir[3] = sum1(v ** 2) - self.length ** 2
        phir[4] = dot(v, w) - self.length * cos(self.alpha)
        phir[5] = sum1(w ** 2) - 1

        return phir

    @staticmethod
    def rigid_body_constraint_jacobian(Qi: SegmentNaturalCoordinates) -> MX:
        """
        This function returns the Jacobian matrix of the rigid body constraints denoted K_r

        Returns
        -------
        Kr : np.ndarray
            Jacobian matrix of the rigid body constraints denoted Kr [6 x 12 x N_frame]
        """
        # initialisation
        Kr = MX.zeros((6, 12))

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
    ) -> MX:
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
        MX
            Derivative of the rigid body constraints [6 x 1 x N_frame]
        """

        return self.rigid_body_constraint_jacobian(Qi) @ Qdoti

    @staticmethod
    def rigid_body_constraint_jacobian_derivative(Qdoti: SegmentNaturalVelocities) -> MX:
        """
        This function returns the derivative of the Jacobian matrix of the rigid body constraints denoted Kr_dot [6 x 12 x N_frame]

        Returns
        -------
        Kr_dot : np.ndarray
            derivative of the Jacobian matrix of the rigid body constraints denoted Kr_dot [6 x 12 ]
        """
        # initialisation
        Kr_dot = MX.zeros((6, 12))

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

    def center_of_mass_position(self, Qi: SegmentNaturalCoordinates) -> MX:
        """
        This function returns the position of the center of mass of the segment in the global coordinate system.

        Returns
        -------
        MX
            Position of the center of mass of the segment in the global coordinate system [3x1]
        """
        return self.natural_center_of_mass.interpolate() @ Qi

    def gravity_force(self) -> MX:
        """
        This function returns the gravity_force applied on the segment through gravity force.

        Returns
        -------
        MX
            Weight applied on the segment through gravity force [12 x 1]
        """

        return (self.natural_center_of_mass.interpolate().T * self.mass) @ MX([0, 0, -9.81])

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
        MX
            Differential algebraic equation of the segment [12 x 1]
        """

        Gi = self.mass_matrix
        Kr = self.rigid_body_constraint_jacobian(Qi)
        Krdot = self.rigid_body_constraint_jacobian_derivative(Qdoti)
        biais = Krdot @ Qdoti.vector

        if stabilization is not None:
            biais -= stabilization["alpha"] * self.rigid_body_constraint(Qi) + stabilization[
                "beta"
            ] * self.rigid_body_constraint_derivative(Qi, Qdoti)

        A = MX.zeros((18, 18))
        A[0:12, 0:12] = Gi
        A[12:18, 0:12] = Kr
        A[0:12, 12:18] = Kr.T
        A[12:, 12:18] = MX.zeros((6, 6))

        B = vertcat([self.gravity_force(), biais])

        # solve the linear system Ax = B with numpy
        raise NotImplementedError("This function is not implemented yet. You should invert the a matrix with casadi")
        # todo in casadi with symbolicqr 'solve(A, B, "symbolicqr")' ??
        x = np.linalg.solve(A, B)
        Qddoti = x[0:12]
        lambda_i = x[12:]
        return SegmentNaturalAccelerations(Qddoti), lambda_i

    @property
    def nb_vectors(self) -> int:
        return self._vectors.nb_vectors

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
                "The marker name should be the same as the 'key'. Alternatively, marker.name can be left undefined"
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
        transformation_matrix_type: TransformationMatrixType
            The type of the transformation matrix to compute, either "Buv" or TransformationMatrixType.Buv
        """

        direction = direction / np.linalg.norm(direction) if normalize else direction
        direction = to_numeric_MX(inv(self.compute_transformation_matrix(transformation_matrix_type))) @ direction

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
        transformation_matrix_type: TransformationMatrixType
            The type of the transformation matrix to compute, TransformationMatrixType.Buv by default
        """

        location = to_numeric_MX(inv(self.compute_transformation_matrix(transformation_matrix_type))) @ location
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

    def markers(self, Qi: SegmentNaturalCoordinates) -> MX:
        return self._markers.positions(Qi)

    def marker_constraints(
            self, marker_locations: np.ndarray, Qi: SegmentNaturalCoordinates, only_technical: bool = True
    ) -> MX:
        return self._markers.constraints(marker_locations, Qi, only_technical)

    def markers_jacobian(self, only_technical: bool = True) -> MX:
        return self._markers.jacobian(only_technical)

    def potential_energy(self, Qi: SegmentNaturalCoordinates) -> MX:
        """
        This function returns the potential energy of the segment

        Parameters
        ----------
        Qi: SegmentNaturalCoordinates
            Natural coordinates of the segment

        Returns
        -------
        MX
            Potential energy of the segment
        """
        return (self.mass * self.natural_center_of_mass.interpolate() @ Qi.vector)[2, 0]

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
        return 0.5 * transpose(Qdoti.to_array()) @ (self.mass_matrix @ Qdoti.to_array())

    def inverse_dynamics(
            self,
            Qi: SegmentNaturalCoordinates,
            Qddoti: SegmentNaturalAccelerations,
            subtree_intersegmental_generalized_forces: MX,
            segment_external_forces: MX,
    ) -> tuple[MX, MX, MX]:
        proximal_interpolation_matrix = NaturalVector.proximal().interpolate()
        pseudo_interpolation_matrix = Qi.compute_pseudo_interpolation_matrix()
        rigid_body_constraints_jacobian = self.rigid_body_constraint_jacobian(Qi=Qi)

        # make a matrix out of it, so that we can solve the system
        front_matrix = horzcat(
            proximal_interpolation_matrix.T, pseudo_interpolation_matrix.T, -rigid_body_constraints_jacobian.T
        )

        b = (
                (self.mass_matrix @ Qddoti)
                - self.gravity_force()
                - segment_external_forces
                - subtree_intersegmental_generalized_forces
        )
        # compute the generalized forces
        # x = A^-1 * b
        generalized_forces = solve(front_matrix, b, "symbolicqr")
        # generalized_forces = inv(front_matrix) @ b

        return generalized_forces[:3, 0], generalized_forces[3:6, 0], generalized_forces[6:, 0]
