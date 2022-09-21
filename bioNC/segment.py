from abc import ABC, abstractmethod
from typing import Union

import numpy as np
from numpy import cos, sin, matmul, eye, zeros, sum, ones
from numpy.linalg import inv

# from .HomogeneousMatrix import HomogeneousMatrix
from .utils.natural_coordinates import SegmentNaturalCoordinates


# from .utils.inertia_matrix import dumas

# # TODO : Add the external forces that are exerted on the solide ==> Prb should be given in global frame
# # if not during a MBO there is a risk that the position of the forces might be changed
#
# m, rCs, Is, Js_temp = dumas(weight, np.mean(self.length), sexe, segment_name_inertia)


class NaturalSegment:
    class Generic(ABC):
        """
        Abstract class for a rigid body segment in natural coordinates system

        Methods
        -------

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
                Transformation matrix from natural coordinate to segment coordinate system
            """

        @abstractmethod
        def rigidBodyConstraint(self, Qi: SegmentNaturalCoordinates) -> np.ndarray:
            """
            This function returns the rigid body constraints of the segment, denoted phi_r.

            Returns
            -------
            np.ndarray
                Rigid body constraints of the segment
            """

        @staticmethod
        @abstractmethod
        def rigidBodyConstraintJacobian(Qi: SegmentNaturalCoordinates) -> np.ndarray:
            """
            This function returns the Jacobian matrix of the rigid body constraints denoted K_r

            Returns
            -------
            Kr : np.ndarray
                Jacobian matrix of the rigid body constraints denoted Kr
            """

        @staticmethod
        @abstractmethod
        def rigidBodyConstraintJacobianDerivative(Qdoti: SegmentNaturalCoordinates) -> np.ndarray:
            """
            This function returns the derivative of the Jacobian matrix of the rigid body constraints denoted Kr_dot [6 x 12 x N_frame]

            Returns
            -------
            Kr_dot : np.ndarray
                derivative of the Jacobian matrix of the rigid body constraints denoted Kr_dot
            """

        @staticmethod
        @abstractmethod
        def pseudo_inertia_matrix(self) -> np.ndarray:
            """
            This function returns the pseudo-inertia matrix of the segment, denoted J_i.
            It transforms the inertia matrix of the segment in the segment coordinate system to the natural coordinate system.

            Returns
            -------
            np.ndarray
                Pseudo-inertia matrix of the segment in the natural coordinate system
            """

        @staticmethod
        @abstractmethod
        def center_of_mass_in_natural_coordinates_system(self) -> np.ndarray:
            """
            This function returns the center of mass of the segment in the natural coordinate system.
            It transforms the center of mass of the segment in the segment coordinate system to the natural coordinate system.

            Returns
            -------
            np.ndarray
                Center of mass of the segment in the natural coordinate system [3x1]
            """

        @staticmethod
        @abstractmethod
        def generalized_mass_matrix(self):
            """
            This function returns the generalized mass matrix of the segment, denoted G_i.

            Returns
            -------
            np.ndarray
                generalized mass matrix of the segment
            """

        @staticmethod
        @abstractmethod
        def interpolation_matrix_center_of_mass(self):
            """
            This function returns the interpolation matrix for the center of mass of the segment, denoted N_i^Ci.
            It allows to apply the gravity force at the center of mass of the segment.

            Returns
            -------
            np.ndarray
                Interpolation matrix for the center of mass of the segment in the natural coordinate system
            """

        @staticmethod
        @abstractmethod
        def weight(self):
            """
            This function returns the weight applied on the segment through gravity force.

            Returns
            -------
            np.ndarray
                Weight applied on the segment through gravity force
            """

        @abstractmethod
        def differential_algebraic_equation(self, Qi: SegmentNaturalCoordinates, Qdoti: SegmentNaturalCoordinates):
            """
            This function returns the differential algebraic equation of the segment

            Returns
            -------
            np.ndarray
                Differential algebraic equation of the segment
            """

    class ThreeD(Generic):
        """
            Class used to define anatomical segment based on natural coordinate.

        Methods
        -------
        transformation_matrix()
            This function returns the transformation matrix, denoted Bi
        rigidBodyConstraint()
            This function returns the rigid body constraints of the segment, denoted phi_r
        rigidBodyConstraintJacobian()
            This function returns the jacobian of rigid body constraints of the segment, denoted K_r

        Attributes
        ----------
        segment_name : str
            name of the segment
        length : float
            length of the segment
        alpha : float
            angle between u and w
        beta : float
            angle between w and (rp-rd)
        gamma : float
            angle between (rp-rd) and u
        mass : float
            mass of the segment in Segment Coordinate System
        center_of_mass : np.ndarray
            center of mass of the segment in Segment Coordinate System
        inertia: np.ndarray
            inertia matrix of the segment in Segment Coordinate System
        """

        def __init__(
            self,
            segment_name: str,
            alpha: float,
            beta: float,
            gamma: float,
            length: float,
            mass: float,
            center_of_mass: np.ndarray,
            inertia: np.ndarray,
        ):

            self.segment_name = segment_name

            self.length = length
            self.alpha = alpha
            self.beta = beta
            self.gamma = gamma
            self._transformation_matrix = self._transformation_matrix()

            self.mass = mass
            if center_of_mass.shape[0] != 3:
                raise ValueError("Center of mass must be 3x1")
            self.center_of_mass = center_of_mass
            self._center_of_mass_in_natural_coordinates_system = self._center_of_mass_in_natural_coordinates_system()
            self._interpolation_matrix_center_of_mass = self._interpolation_matrix_center_of_mass()

            if inertia.shape != (3, 3):
                raise ValueError("Inertia matrix must be 3x3")
            self.inertia = inertia
            self._pseudo_inertia_matrix = self._pseudo_inertia_matrix()
            self._generalized_mass_matrix = self._generalized_mass_matrix()

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
            return self._transformation_matrix

        def rigidBodyConstraint(self, Qi: Union[SegmentNaturalCoordinates, np.ndarray]) -> np.ndarray:
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
            phir[0] = sum(Qi.u**2, 0) - 1
            phir[1] = sum(Qi.u * (Qi.rp - Qi.rd), 0) - self.length * cos(self.gamma)
            phir[2] = sum(Qi.u * Qi.w, 0) - cos(self.beta)
            phir[3] = sum((Qi.rp - Qi.rd) ** 2, 0) - self.length**2
            phir[4] = sum((Qi.rp - Qi.rd) * Qi.w, 0) - self.length * cos(self.alpha)
            phir[5] = sum(Qi.w**2, 0) - 1

            return phir

        @staticmethod
        def rigidBodyConstraintJacobian(Qi: SegmentNaturalCoordinates) -> np.ndarray:
            """
            This function returns the Jacobian matrix of the rigid body constraints denoted K_r

            Returns
            -------
            Kr : np.ndarray
                Jacobian matrix of the rigid body constraints denoted Kr [6 x 12 x N_frame]
            """
            # initialisation
            Kr = zeros((6, 12))

            Kr[0, 0:3] = 2 * Qi.u

            Kr[1, 0:3] = Qi.rp - Qi.rd
            Kr[1, 3:6] = Qi.u
            Kr[1, 6:9] = -Qi.u

            Kr[2, 0:3] = Qi.w
            Kr[2, 9:12] = Qi.u

            Kr[3, 3:6] = 2 * (Qi.rp - Qi.rd)
            Kr[3, 6:9] = -2 * (Qi.rp - Qi.rd)

            Kr[4, 3:6] = Qi.w
            Kr[4, 6:9] = -Qi.w
            Kr[4, 9:12] = Qi.rp - Qi.rd

            Kr[5, 9:12] = 2 * Qi.w

            return Kr

        @staticmethod
        def rigidBodyConstraintJacobianDerivative(Qdoti: SegmentNaturalCoordinates) -> np.ndarray:
            """
            This function returns the derivative of the Jacobian matrix of the rigid body constraints denoted Kr_dot [6 x 12 x N_frame]

            Returns
            -------
            Kr_dot : np.ndarray
                derivative of the Jacobian matrix of the rigid body constraints denoted Kr_dot [6 x 12 ]
            """
            # initialisation
            Kr_dot = zeros((6, 12))

            Kr_dot[0, 0:3] = 2 * Qdoti.u
            Kr_dot[1, 0:3] = Qdoti.rp - Qdoti.rd
            Kr_dot[1, 3:6] = Qdoti.u
            Kr_dot[1, 6:9] = -Qdoti.u
            Kr_dot[2, 0:3] = Qdoti.w
            Kr_dot[2, 9:12] = Qdoti.u

            Kr_dot[3, 3:6] = 2 * (Qdoti.rp - Qdoti.rd)
            Kr_dot[3, 6:9] = -2 * (Qdoti.rp - Qdoti.rd)
            Kr_dot[4, 3:6] = Qdoti.w
            Kr_dot[4, 6:9] = -Qdoti.w
            Kr_dot[4, 9:12] = Qdoti.rp - Qdoti.rd
            Kr_dot[5, 9:12] = 2 * Qdoti.w

            return Kr_dot

        def _pseudo_inertia_matrix(self):
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
        def pseudo_inertia_matrix(self):
            """
            This function returns the pseudo-inertia matrix of the segment, denoted J_i.
            It transforms the inertia matrix of the segment in the segment coordinate system to the natural coordinate system.

            Returns
            -------
            np.ndarray
                Pseudo-inertia matrix of the segment in the natural coordinate system [3x3]
            """
            return self._pseudo_inertia_matrix

        def _center_of_mass_in_natural_coordinates_system(self):
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
        def center_of_mass_in_natural_coordinates_system(self):
            """
            This function returns the center of mass of the segment in the natural coordinate system.
            It transforms the center of mass of the segment in the segment coordinate system to the natural coordinate system.

            Returns
            -------
            np.ndarray
                Center of mass of the segment in the natural coordinate system [3x1]
            """
            return self._center_of_mass_in_natural_coordinates_system

        def _generalized_mass_matrix(self):
            """
            This function returns the generalized mass matrix of the segment, denoted G_i.

            Returns
            -------
            np.ndarray
                generalized mass matrix of the segment [12 x 12]
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
        def generalized_mass_matrix(self):
            """
            This function returns the generalized mass matrix of the segment, denoted G_i.

            Returns
            -------
            np.ndarray
                generalized mass matrix of the segment [12 x 12]
            """

            return self._generalized_mass_matrix

        def _interpolation_matrix_center_of_mass(self):
            """
            This function returns the interpolation matrix for the center of mass of the segment, denoted N_i^Ci.
            It allows to apply the gravity force at the center of mass of the segment.

            Returns
            -------
            np.ndarray
                Interpolation matrix for the center of mass of the segment in the natural coordinate system [12 x 3]
            """
            n_ci = self.center_of_mass_in_natural_coordinates_system

            interpolation_matrix = np.zeros((12, 3))
            interpolation_matrix[0:3, 0:3] = n_ci[0] * eye(3)
            interpolation_matrix[3:6, 0:3] = (1 + n_ci[1]) * eye(3)
            interpolation_matrix[6:9, 0:3] = -n_ci[1] * eye(3)
            interpolation_matrix[9:12, 0:3] = n_ci[2] * eye(3)

            return interpolation_matrix

        @property
        def interpolation_matrix_center_of_mass(self):
            """
            This function returns the interpolation matrix for the center of mass of the segment, denoted N_i^Ci.
            It allows to apply the gravity force at the center of mass of the segment.

            Returns
            -------
            np.ndarray
                Interpolation matrix for the center of mass of the segment in the natural coordinate system [12 x 3]
            """
            return self._interpolation_matrix_center_of_mass

        def weight(self):
            """
            This function returns the weight applied on the segment through gravity force.

            Returns
            -------
            np.ndarray
                Weight applied on the segment through gravity force [12 x 1]
            """

            return np.matmul(self.interpolation_matrix_center_of_mass * self.mass, np.array([0, 0, -9.81]))

        def differential_algebraic_equation(
            self, Qi: Union[SegmentNaturalCoordinates, np.array], Qdoti: Union[SegmentNaturalCoordinates, np.array]
        ):
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

            if not isinstance(Qi, SegmentNaturalCoordinates):
                Qi = SegmentNaturalCoordinates(Qi)
            if not isinstance(Qdoti, SegmentNaturalCoordinates):
                Qdoti = SegmentNaturalCoordinates(Qdoti)

            Gi = self.generalized_mass_matrix
            Kr = self.rigidBodyConstraintJacobian(Qi)
            Kr_transpose = np.transpose(Kr)
            Krdot = self.rigidBodyConstraintJacobianDerivative(Qdoti)
            biais = np.matmul(Krdot, Qdoti.vector)

            A = zeros((18, 18))
            A[0:12, 0:12] = Gi
            A[12:, 0:12] = Kr
            A[0:12, 12:] = Kr_transpose
            A[12:, 12:] = np.zeros((6, 6))

            B = np.concatenate([self.weight(), biais], axis=0)

            # solve the linear system Ax = B with numpy
            x = np.linalg.solve(A, B)
            Qddoti = x[0:12]
            lambda_i = x[12:]
            return Qddoti, lambda_i

        def location_from_homogenous_transform(self, T: np.ndarray):
            """
            This function returns the location of the segment in natural coordinate from its homogenous transform

            Parameters
            ----------
            T: np.ndarray
                Homogenous transform of the segment Ti which transforms from the local frame (Oi, Xi, Yi, Zi)
                to the global frame (Xi, Yi, Zi)

            Returns
            -------
            np.ndarray
                Location of the segment [3 x 1]
            """

            u = self.transformation_matrix * T[0:3, 0]
            # v = self.transformation_matrix() * T[0:3, 1]
            w = self.transformation_matrix * T[0:3, 2]
            rp = self.transformation_matrix * T[0:3, 4]
            rd = np.matmul(T, np.array([0, self.length, 0, 1]))[0:3]

            return SegmentNaturalCoordinates((u, rp, rd, w))

        # def generalizedInertiaMatrix(self):
        #
        #     return Gi

        # def weightGeneralizedForces(self, Qi: SegmentGeneralizedCoordinates) -> np.ndarray:
        #     return P

        # def inverse_dynamics(self, Q, Qdot, Qddot) -> np.ndarray:
        #     return ID(Q, Qdot, Qddot)

        # def ode(self) -> np.ndarray:
        #     return Qddot_i, lambda_i

    class TwoD(Generic):
        """
            Class used to define anatomical segment based on natural coordinate.

        Methods
        -------
        transformation_matrix()
            This function returns the transformation matrix, denoted Bi
        rigidBodyConstraint()
            This function returns the rigid body constraints of the segment, denoted phi_r
        rigidBodyConstraintJacobian()
            This function returns the jacobian of rigid body constraints of the segment, denoted K_r

        Attributes
        ----------
        segment_name : str
            name of the segment
        length : float
            length of the segment
        gamma : float
            angle between (rp-rd) and u
        mass : float
            mass of the segment in Segment Coordinate System
        center_of_mass : np.ndarray
            center of mass of the segment in Segment Coordinate System
        inertia: np.ndarray
            inertia matrix of the segment in Segment Coordinate System
        """

        def __init__(
            self,
            segment_name: str,
            gamma: float,
            length: float,
            mass: float,
            center_of_mass: np.ndarray,
            inertia: np.ndarray,
        ):

            self.segment_name = segment_name

            self.length = length
            self.gamma = gamma
            #
            if inertia.shape[0] != 1:
                raise ValueError("Inertia matrix must be 1x1")
            self.inertia = inertia
            #
            self.mass = mass
            #
            if center_of_mass.shape[0] != 2:
                raise ValueError("Center of mass must be 3x1")
            self.center_of_mass = center_of_mass

        def transformation_matrix(self):
            """
            This function returns the transformation matrix, denoted Bi,
            from Natural Coordinate System to point to the orthogonal Segment Coordinate System.
            Example : if vector a expressed in (Pi, X, Y), inv(B) * a is expressed in (Pi, ui, vi)

            Returns
            -------
            np.ndarray
                Transformation matrix from natural coordinate to segment coordinate system [2 x 2]
            """
            return np.array(
                [
                    [1, 0],
                    [self.length * cos(self.gamma), self.length * sin(self.gamma)],
                ]
            )

        def rigidBodyConstraint(self, Qi: SegmentNaturalCoordinates) -> np.ndarray:
            """
            This function returns the rigid body constraints of the segment, denoted phi_r.

            Returns
            -------
            np.ndarray
                Rigid body constraints of the segment [3 x 1 ]
            """
            phir = zeros((6, 1, Qi.shape[1]))
            phir[0, :, :] = sum(Qi.u**2, 0) - ones((Qi.u.shape[1]))
            phir[1, :, :] = sum(Qi.u * (Qi.rp - Qi.rd), 0) - self.length * cos(self.gamma)
            # phir[2, :, :] = sum(Qi.u * Qi.w, 0) - cos(self.beta)
            phir[3, :, :] = sum((Qi.rp - Qi.rd) ** 2, 0) - self.length**2
            # phir[4, :, :] = sum((Qi.rp - Qi.rd) * Qi.w, 0) - self.length * cos(self.alpha)
            # phir[5, :, :] = sum(Qi.w ** 2, 0) - ones(Qi.u.shape[1])

            return phir

        @staticmethod
        def rigidBodyConstraintJacobian(Qi: SegmentNaturalCoordinates) -> np.ndarray:
            """
            This function returns the Jacobian matrix of the rigid body constraints denoted K_r

            Returns
            -------
            Kr : np.ndarray
                Jacobian matrix of the rigid body constraints denoted Kr [3 x 6 ]
            """
            # todo: make it 2d
            # initialisation
            Kr = zeros((6, 12, Qi.shape[1]))

            Kr[0, 0:2, :] = 2 * Qi.u

            Kr[1, 0:2, :] = Qi.rp - Qi.rd
            Kr[1, 3:6, :] = Qi.u
            Kr[1, 6:9, :] = -Qi.u

            Kr[2, 0:3, :] = Qi.w
            Kr[2, 9:12, :] = Qi.u

            Kr[3, 3:6, :] = 2 * (Qi.rp - Qi.rd)
            Kr[3, 6:9, :] = -2 * (Qi.rp - Qi.rd)

            Kr[4, 3:6, :] = Qi.w
            Kr[4, 6:9, :] = -Qi.w
            Kr[4, 9:12, :] = Qi.rp - Qi.rd

            Kr[5, 9:12, :] = 2 * Qi.w

            return Kr

    #
    #     @staticmethod
    #     def rigidBodyConstraintJacobianDerivative(Qdoti: SegmentNaturalCoordinates) -> np.ndarray:
    #         """
    #         This function returns the derivative of the Jacobian matrix of the rigid body constraints denoted Kr_dot [6 x 12 ]
    #
    #         Returns
    #         -------
    #         Kr_dot : np.ndarray
    #             derivative of the Jacobian matrix of the rigid body constraints denoted Kr_dot [6 x 12 ]
    #         """
    #         # initialisation
    #         Kr_dot = zeros((6, 12, Qdoti.shape[1]))
    #
    #         Kr_dot[0, 0:3, :] = 2 * Qdoti.u
    #         Kr_dot[1, 0:3, :] = Qdoti.rp - Qdoti.rd
    #         Kr_dot[1, 3:6, :] = Qdoti.u
    #         Kr_dot[1, 6:9, :] = -Qdoti.u
    #         Kr_dot[2, 0:3, :] = Qdoti.w
    #         Kr_dot[2, 9:12, :] = Qdoti.u
    #
    #         Kr_dot[3, 3:6, :] = 2 * (Qdoti.rp - Qdoti.rd)
    #         Kr_dot[3, 6:9, :] = -2 * (Qdoti.rp - Qdoti.rd)
    #         Kr_dot[4, 3:6, :] = Qdoti.w
    #         Kr_dot[4, 6:9, :] = -Qdoti.w
    #         Kr_dot[4, 9:12, :] = Qdoti.rp - Qdoti.rd
    #         Kr_dot[5, 9:12, :] = 2 * Qdoti.w
    #
    #         return Kr_dot
    #
    #     def pseudo_inertia_matrix(self):
    #         """
    #         This function returns the pseudo-inertia matrix of the segment, denoted J_i.
    #         It transforms the inertia matrix of the segment in the segment coordinate system to the natural coordinate system.
    #
    #         Returns
    #         -------
    #         np.ndarray
    #             Pseudo-inertia matrix of the segment in the natural coordinate system [3x3]
    #         """
    #         # todo: verify the formula
    #         middle_block = (
    #                 self.inertia
    #                 + self.mass * np.dot(self.center_of_mass.T, self.center_of_mass) * eye(3)
    #                 - np.dot(self.center_of_mass.T, self.center_of_mass)
    #         )
    #
    #         Binv = inv(self.transformation_matrix())
    #         Binv_transpose = np.transpose(Binv)
    #
    #         return matmul(Binv, matmul(middle_block, Binv_transpose))
    #
    #     def center_of_mass_in_natural_coordinates_system(self):
    #         """
    #         This function returns the center of mass of the segment in the natural coordinate system.
    #         It transforms the center of mass of the segment in the segment coordinate system to the natural coordinate system.
    #
    #         Returns
    #         -------
    #         np.ndarray
    #             Center of mass of the segment in the natural coordinate system [3x1]
    #         """
    #         return matmul(inv(self.transformation_matrix()), self.center_of_mass)
    #
    #     def generalized_mass_matrix(self):
    #         """
    #         This function returns the generalized mass matrix of the segment, denoted G_i.
    #
    #         Returns
    #         -------
    #         np.ndarray
    #             generalized mass matrix of the segment [12 x 12]
    #         """
    #
    #         Ji = self.pseudo_inertia_matrix()
    #         n_ci = self.center_of_mass_in_natural_coordinates_system()
    #
    #         Gi = zeros((12, 12))
    #
    #         Gi[0:3, 0:3] = Ji[0, 0] * eye(3)
    #         Gi[0:3, 3:6] = (self.mass * n_ci[0] + Ji[0, 1]) * eye(3)
    #         Gi[0:3, 6:9] = -Ji[0, 1] * eye(3)
    #         Gi[0:3, 9:12] = -Ji[0, 2] * eye(3)
    #         Gi[3:6, 3:6] = (self.mass + 2 * self.mass * n_ci[1] + Ji[1, 1]) * eye(3)
    #         Gi[3:6, 6:9] = -(self.mass * n_ci[1] + Ji[1, 1]) * eye(3)
    #         Gi[3:6, 9:12] = (self.mass * n_ci[2] + Ji[1, 2]) * eye(3)
    #         Gi[6:9, 6:9] = Ji[1, 1] * eye(3)
    #         Gi[6:9, 9:12] = -Ji[1, 2] * eye(3)
    #         Gi[9:12, 9:12] = Ji[2, 2] * eye(3)
    #
    #         # symmetrize the matrix
    #         Gi = np.tril(Gi) + np.tril(Gi, -1).T
    #
    #         return Gi
    #
    #     def interpolation_matrix_center_of_mass(self):
    #         """
    #         This function returns the interpolation matrix for the center of mass of the segment, denoted N_i^Ci.
    #         It allows to apply the gravity force at the center of mass of the segment.
    #
    #         Returns
    #         -------
    #         np.ndarray
    #             Interpolation matrix for the center of mass of the segment in the natural coordinate system [12 x 3]
    #         """
    #         n_ci = self.center_of_mass_in_natural_coordinates_system()
    #
    #         interpolation_matrix = np.zeros((12, 3))
    #         interpolation_matrix[0:3, 0:3] = n_ci[0] * eye(3)
    #         interpolation_matrix[3:6, 0:3] = (1 + n_ci[1]) * eye(3)
    #         interpolation_matrix[6:9, 0:3] = -n_ci[1] * eye(3)
    #         interpolation_matrix[9:12, 0:3] = n_ci[2] * eye(3)
    #
    #         return interpolation_matrix
    #
    #     def weight(self):
    #         """
    #         This function returns the weight applied on the segment through gravity force.
    #
    #         Returns
    #         -------
    #         np.ndarray
    #             Weight applied on the segment through gravity force [12 x 1]
    #         """
    #
    #         return self.interpolation_matrix_center_of_mass() * self.mass * np.array([0, 0, -9.81])
    #
    #     def differential_algebraic_equation(self, Qi: SegmentNaturalCoordinates, Qdoti: SegmentNaturalCoordinates):
    #         """
    #         This function returns the differential algebraic equation of the segment
    #
    #         Returns
    #         -------
    #         np.ndarray
    #             Differential algebraic equation of the segment [12 x 1]
    #         """
    #         Gi = self.generalized_mass_matrix()
    #         Kr = self.rigidBodyConstraintJacobian(Qi)
    #         Kr_transpose = np.transpose(Kr)
    #         Krdot = self.rigidBodyConstraintJacobianDerivative(Qdoti)
    #         biais = Krdot * Qdoti
    #
    #         A = np.array([[Gi, Kr_transpose],
    #                       Kr, np.zeros((6, 6))])
    #
    #         B = np.concatenate([self.weight(), biais], axis=1)
    #
    #         # solve the linear system Ax = B with numpy
    #         x = np.linalg.solve(A, B)
    #         Qddoti = x[0:6, 0]
    #         lambda_i = x[6:12, 0]
    #         return Qddoti, lambda_i
