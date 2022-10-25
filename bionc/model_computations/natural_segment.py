from copy import copy
from typing import Union, Tuple

import numpy as np
from numpy import cos, sin, matmul, eye, zeros, sum, ones
from numpy.linalg import inv

from ..utils.natural_coordinates import SegmentNaturalCoordinates
from ..utils.natural_velocities import SegmentNaturalVelocities
from ..utils.natural_accelerations import SegmentNaturalAccelerations
from ..utils.homogenous_transform import HomogeneousTransform
from ..model_computations.natural_axis import Axis
from ..model_computations.natural_marker import SegmentMarker, Marker


class NaturalSegment:
    """
        Class used to define anatomical segment based on natural coordinate.

    Methods
    -------
    transformation_matrix()
        This function returns the transformation matrix, denoted Bi
    rigid_body_constraint()
        This function returns the rigid body constraints of the segment, denoted phi_r
    rigid_body_constraint_jacobian()
        This function returns the jacobian of rigid body constraints of the segment, denoted K_r

    add_marker()
        This function adds a marker to the segment
    nb_markers()
        This function returns the number of markers in the segment
    marker_constraints()
        This function returns the defects of the marker constraints of the segment, denoted Phi_m
    marker_jacobian()
        This function returns the jacobian of the marker constraints of the segment, denoted K_m

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
    """

    def __init__(
        self,
        name: str = None,
        alpha: Union[float, np.float64] = np.pi / 2,
        beta: Union[float, np.float64] = np.pi / 2,
        gamma: Union[float, np.float64] = np.pi / 2,
        length: Union[float, np.float64] = None,
        mass: Union[float, np.float64] = None,
        center_of_mass: np.ndarray = None,
        inertia: np.ndarray = None,
    ):

        self._name = name

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
            self._center_of_mass_in_natural_coordinates_system = None
            self._interpolation_matrix_center_of_mass = None
        else:
            if center_of_mass.shape[0] != 3:
                raise ValueError("Center of mass must be 3x1")
            self._center_of_mass = center_of_mass
            self._center_of_mass_in_natural_coordinates_system = self._center_of_mass_in_natural_coordinates_system()
            self._interpolation_matrix_center_of_mass = self._interpolation_matrix_center_of_mass()

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

        if not isinstance(Q, SegmentNaturalCoordinates):
            Q = SegmentNaturalCoordinates(Q)

        u, rp, rd, w = Q.to_components()

        length = np.linalg.norm(rp - rd, axis=0)
        alpha = np.arccos(np.sum((rp - rd) * w, axis=0) / length)
        beta = np.arccos(np.sum(u * w, axis=0))
        gamma = np.arccos(np.sum(u * (rp - rd), axis=0) / length)

        return alpha, beta, gamma, length

    # def __str__(self):
    #     print("to do")

    @property
    def name(self):
        return self._name

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

    def _transformation_matrix(self) -> np.ndarray:
        """
        This function computes the transformation matrix, denoted Bi,
        from Natural Coordinate System to point to the orthogonal Segment Coordinate System.
        Example : if vector a expressed in (Pi, X, Y, Z), inv(B) * a is expressed in (Pi, ui, vi, wi)

        Returns
        -------
        np.ndarray
            Transformation matrix from natural coordinate to segment coordinate system [3x3]
        """
        return np.array(
            [
                [1, 0, 0],
                [self.length * cos(self.gamma), self.length * sin(self.gamma), 0],
                [
                    cos(self.beta),
                    (cos(self.alpha) - cos(self.beta) * cos(self.gamma)) / sin(self.beta),
                    np.sqrt(
                        1
                        - cos(self.beta) ** 2
                        - (cos(self.alpha) - cos(self.beta) * cos(self.gamma)) / sin(self.beta) ** 2
                    ),
                ],
            ]
        )

    @property
    def transformation_matrix(self) -> np.ndarray:
        """
        This function returns the transformation matrix, denoted Bi,
        from Natural Coordinate System to point to the orthogonal Segment Coordinate System.
        Example : if vector a expressed in (Pi, X, Y, Z), inv(B) * a is expressed in (Pi, ui, vi, wi)

        Returns
        -------
        np.ndarray
            Transformation matrix from natural coordinate to segment coordinate system [3x3]
        """
        return self._transformation_matrix

    def segment_coordinates_system(self, Q: SegmentNaturalCoordinates) -> HomogeneousTransform:
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
        if not isinstance(Q, SegmentNaturalCoordinates):
            Q = SegmentNaturalCoordinates(Q)

        return HomogeneousTransform.from_rt(
            rotation=self._transformation_matrix @ np.concatenate((Q.u, Q.v, Q.w), axis=1),
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
        if not isinstance(Qi, SegmentNaturalCoordinates):
            Qi = SegmentNaturalCoordinates(Qi)

        phir = zeros(6)
        # phir[0] = sum(Qi.u**2, 0) - 1
        # phir[1] = sum(Qi.u * (Qi.rp - Qi.rd), 0) - self.length * cos(self.gamma)
        # phir[2] = sum(Qi.u * Qi.w, 0) - cos(self.beta)
        # phir[3] = sum((Qi.rp - Qi.rd) ** 2, 0) - self.length**2
        # phir[4] = sum((Qi.rp - Qi.rd) * Qi.w, 0) - self.length * cos(self.alpha)
        # phir[5] = sum(Qi.w**2, 0) - 1

        phir[0] = sum(Qi.u**2, 0) - 1
        phir[1] = sum(Qi.u * Qi.v, 0) - self.length * cos(self.gamma)
        phir[2] = sum(Qi.u * Qi.w, 0) - cos(self.beta)
        phir[3] = sum(Qi.v**2, 0) - self.length**2
        phir[4] = sum(Qi.v * Qi.w, 0) - self.length * cos(self.alpha)
        phir[5] = sum(Qi.w**2, 0) - 1

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

    @staticmethod
    def rigid_body_constraint_jacobian_derivative(
        Qdoti: SegmentNaturalVelocities,
    ) -> np.ndarray:
        """
        This function returns the derivative of the Jacobian matrix of the rigid body constraints denoted Kr_dot [6 x 12 x N_frame]

        Returns
        -------
        Kr_dot : np.ndarray
            derivative of the Jacobian matrix of the rigid body constraints denoted Kr_dot [6 x 12 ]
        """
        if isinstance(Qdoti, SegmentNaturalCoordinates):
            raise TypeError("Qdoti should be a SegmentNaturalVelocities object")
            # not able to check if Qdoti is a SegmentNaturalVelocities if Qdoti is a np.ndarray
        if not isinstance(Qdoti, SegmentNaturalVelocities):
            Qdoti = SegmentNaturalVelocities(Qdoti)

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

        Binv = inv(self.transformation_matrix)
        Binv_transpose = np.transpose(Binv)

        return matmul(Binv, matmul(middle_block, Binv_transpose))

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

    def _center_of_mass_in_natural_coordinates_system(self) -> np.ndarray:
        """
        This function computes the center of mass of the segment in the natural coordinate system.
        It transforms the center of mass of the segment in the segment coordinate system to the natural coordinate system.

        Returns
        -------
        np.ndarray
            Center of mass of the segment in the natural coordinate system [3x1]
        """
        return matmul(inv(self.transformation_matrix), self.center_of_mass)

    @property
    def center_of_mass_in_natural_coordinates_system(self) -> np.ndarray:
        """
        This function returns the center of mass of the segment in the natural coordinate system.
        It transforms the center of mass of the segment in the segment coordinate system to the natural coordinate system.

        Returns
        -------
        np.ndarray
            Center of mass of the segment in the natural coordinate system [3x1]
        """
        return self._center_of_mass_in_natural_coordinates_system

    def _update_mass_matrix(self) -> np.ndarray:
        """
        This function returns the generalized mass matrix of the segment, denoted G_i.

        Returns
        -------
        np.ndarray
            mass matrix of the segment [12 x 12]
        """

        Ji = self.pseudo_inertia_matrix
        n_ci = self.center_of_mass_in_natural_coordinates_system

        Gi = zeros((12, 12))

        Gi[0:3, 0:3] = Ji[0, 0] * eye(3)
        Gi[0:3, 3:6] = (self.mass * n_ci[0] + Ji[0, 1]) * eye(3)
        Gi[0:3, 6:9] = -Ji[0, 1] * eye(3)
        Gi[0:3, 9:12] = -Ji[0, 2] * eye(3)
        Gi[3:6, 3:6] = (self.mass + 2 * self.mass * n_ci[1] + Ji[1, 1]) * eye(3)
        Gi[3:6, 6:9] = -(self.mass * n_ci[1] + Ji[1, 1]) * eye(3)
        Gi[3:6, 9:12] = (self.mass * n_ci[2] + Ji[1, 2]) * eye(3)
        Gi[6:9, 6:9] = Ji[1, 1] * eye(3)
        Gi[6:9, 9:12] = -Ji[1, 2] * eye(3)
        Gi[9:12, 9:12] = Ji[2, 2] * eye(3)

        # symmetrize the matrix
        Gi = np.tril(Gi) + np.tril(Gi, -1).T

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

    @staticmethod
    def interpolate(vector: np.ndarray) -> np.ndarray:
        """
        This function interpolates the vector to get the interpolation matrix, denoted Ni
        such as:
        Ni * Qi = location in the global frame

        Parameters
        ----------
        vector : np.ndarray
            Vector in the natural coordinate system to interpolate (Pi, ui, vi, wi)

        Returns
        -------
        interpolate_natural_vector: np.ndarray
            Interpolation  matrix [3 x 12], denoted Ni to get the location of the vector as linear combination of Q.
            vector in global frame = Ni * Qi
        """

        interpolation_matrix = np.zeros((3, 12))
        interpolation_matrix[0:3, 0:3] = vector[0] * eye(3)
        interpolation_matrix[0:3, 3:6] = (1 + vector[1]) * eye(3)
        interpolation_matrix[0:3, 6:9] = -vector[1] * eye(3)
        interpolation_matrix[0:3, 9:12] = vector[2] * eye(3)

        return interpolation_matrix

    def _interpolation_matrix_center_of_mass(self) -> np.ndarray:
        """
        This function returns the interpolation matrix for the center of mass of the segment, denoted N_i^Ci.
        It allows to apply the gravity force at the center of mass of the segment.

        Returns
        -------
        np.ndarray
            Interpolation matrix for the center of mass of the segment in the natural coordinate system [12 x 3]
        """
        n_ci = self.center_of_mass_in_natural_coordinates_system
        return self.interpolate(n_ci)

    @property
    def interpolation_matrix_center_of_mass(self) -> np.ndarray:
        """
        This function returns the interpolation matrix for the center of mass of the segment, denoted N_i^Ci.
        It allows to apply the gravity force at the center of mass of the segment.

        Returns
        -------
        np.ndarray
            Interpolation matrix for the center of mass of the segment in the natural coordinate system [12 x 3]
        """
        return self._interpolation_matrix_center_of_mass

    def weight(self) -> np.ndarray:
        """
        This function returns the weight applied on the segment through gravity force.

        Returns
        -------
        np.ndarray
            Weight applied on the segment through gravity force [12 x 1]
        """

        return np.matmul(
            self.interpolation_matrix_center_of_mass.T * self.mass,
            np.array([0, 0, -9.81]),
        )

    def differential_algebraic_equation(
        self,
        Qi: Union[SegmentNaturalCoordinates, np.ndarray],
        Qdoti: Union[SegmentNaturalVelocities, np.ndarray],
    ) -> Tuple[SegmentNaturalAccelerations, np.ndarray]:
        """
        This function returns the differential algebraic equation of the segment

        Parameters
        ----------
        Qi: SegmentNaturalCoordinates
            Natural coordinates of the segment
        Qdoti: SegmentNaturalCoordinates
            Derivative of the natural coordinates of the segment

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
        biais = np.matmul(Krdot, Qdoti.vector)

        A = zeros((18, 18))
        A[0:12, 0:12] = Gi
        A[12:18, 0:12] = Kr
        A[0:12, 12:18] = Kr.T
        A[12:, 12:18] = np.zeros((6, 6))

        B = np.concatenate([self.weight(), biais], axis=0)

        # solve the linear system Ax = B with numpy
        x = np.linalg.solve(A, B)
        Qddoti = x[0:12]
        lambda_i = x[12:]
        return SegmentNaturalAccelerations(Qddoti), lambda_i

    def add_marker(self, marker: SegmentMarker):
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

    def nb_markers(self) -> int:
        """
        Returns the number of markers of the natural segment

        Returns
        -------
        int
            Number of markers of the segment
        """
        return len(self._markers)

    def marker_constraints(self, marker_locations: np.ndarray, Qi: SegmentNaturalCoordinates) -> np.ndarray:
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
        if marker_locations.shape != (3, self.nb_markers()):
            raise ValueError(f"marker_locations should be of shape (3, {self.nb_markers()})")

        defects = np.zeros((3, self.nb_markers()))

        for i, marker in enumerate(self._markers):
            defects[:, i] = marker.constraint(marker_location=marker_locations[:, i], Qi=Qi)

        return defects

    def marker_jacobian(self):
        """
        This function returns the marker jacobian of the segment

        Returns
        -------
        np.ndarray
            The jacobian of the marker constraints of the segment (3 x N_markers)
        """
        return np.vstack([-marker.interpolation_matrix for marker in self._markers])
