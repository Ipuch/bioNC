from typing import Union

import numpy as np
from casadi import MX
from casadi import transpose, dot, inv

from numpy.linalg import inv

from ..bionc_casadi.natural_vector import NaturalVector
from .utils import to_numeric_MX, to_numeric


class NaturalInertialParameters:
    """
    This class represents the inertial parameters of a segment in the natural coordinate system.

    Methods
    -------
    from_cartesian_inertial_parameters
        Creates a NaturalInertialParameters object from cartesian inertial parameters.
    mass
        This function returns the mass of the segment in the natural coordinate system.
    natural_center_of_mass
        This function returns the center of mass of the segment in the natural coordinate system.
    natural_pseudo_inertia
        This function returns the pseudo-inertia matrix of the segment, denoted J_i.
    _update_mass_matrix
        This function returns the generalized mass matrix of the segment, denoted G_i (private).
    mass_matrix
        This function returns the generalized mass matrix of the segment, denoted G_i.
    compute_cartesian_inertia_from_pseudo
        Computes the cartesian inertia matrix from pseudo-inertia matrix considering a transformation matrix.
    center_of_mass
        Computes the center of mass of the segment in the specified segment coordinate system.
    inertia
        Computes the inertia of the segment in the specified segment coordinate system.
    compute_natural_center_of_mass
        Computes the center of mass of the segment in the natural coordinate system from cartesian center of mass.
    compute_pseudo_inertia_matrix
        Computes the pseudo-inertia matrix of the segment in the natural coordinate system from cartesian inertia.
    to_mx
        This function returns the NaturalInertialParameters in MX format.

    Attributes
    ----------
    _mass
        The mass of the segment in kg
    _natural_center_of_mass
        The center of mass of the segment in the natural coordinate system [3x1]
    _natural_pseudo_inertia
        The pseudo-inertia matrix of the segment in the natural coordinate system [3x3]
    _mass_matrix
        The generalized mass matrix of the segment [12x12]
    _initial_transformation_matrix
        The transformation matrix from the natural coordinate system to the segment coordinate system [3x3],
        that has been initially used to compute the natural inertial parameters.
    """

    def __init__(
        self,
        mass: Union[np.ndarray, float, np.float64] = None,
        natural_center_of_mass: Union[np.ndarray, MX] = None,
        natural_pseudo_inertia: Union[np.ndarray, MX] = None,
        initial_transformation_matrix: [np.ndarray, MX] = None,
    ):
        if mass is None:
            raise ValueError("mass must be provided")
        if natural_center_of_mass is None:
            raise ValueError("natural_center_of_mass must be provided")
        if natural_pseudo_inertia is None:
            raise ValueError("natural_pseudo_inertia must be provided")

        if isinstance(mass, np.ndarray):
            mass = mass.item()

        self._mass = MX(mass)

        if natural_center_of_mass.shape[0] != 3:
            raise ValueError("Center of mass must be 3x1")

        self._natural_center_of_mass = MX(natural_center_of_mass)

        if natural_pseudo_inertia.shape != (3, 3):
            raise ValueError("Pseudo inertia matrix must be 3x3")

        self._natural_pseudo_inertia = MX(natural_pseudo_inertia)
        self._mass_matrix = self._update_mass_matrix()

        if initial_transformation_matrix is not None:
            if initial_transformation_matrix.shape != (3, 3):
                raise ValueError("Transformation matrix must be 3x3")

            self._initial_transformation_matrix = MX(initial_transformation_matrix)
        else:
            self._initial_transformation_matrix = None

    @property
    def mass(self) -> float:
        """
        This function returns the mass of the segment in the natural coordinate system.

        Returns
        -------
        float
            Mass of the segment in the natural coordinate system
        """
        return self._mass

    @property
    def natural_center_of_mass(self) -> MX:
        """
        This function returns the center of mass of the segment in the natural coordinate system.

        Returns
        -------
        np.ndarray
            Center of mass of the segment in the natural coordinate system [3x1]
        """
        return self._natural_center_of_mass

    @property
    def natural_pseudo_inertia(self) -> MX:
        """
        This function returns the pseudo-inertia matrix of the segment, denoted J_i.
        It transforms the inertia matrix of the segment in the segment coordinate system to the natural coordinate system.

        Returns
        -------
        np.ndarray
            Pseudo-inertia matrix of the segment in the natural coordinate system [3x3]
        """
        return self._natural_pseudo_inertia

    def _update_mass_matrix(self) -> MX:
        """
        This function returns the generalized mass matrix of the segment, denoted G_i.

        Returns
        -------
        MX
            mass matrix of the segment [12 x 12]
        """

        Ji = self.natural_pseudo_inertia
        n_ci = self.natural_center_of_mass

        Gi = MX.zeros((12, 12))

        Gi[0:3, 0:3] = Ji[0, 0] * MX.eye(3)
        Gi[0:3, 3:6] = (self.mass * n_ci[0] + Ji[0, 1]) * MX.eye(3)
        Gi[0:3, 6:9] = -Ji[0, 1] * MX.eye(3)
        Gi[0:3, 9:12] = Ji[0, 2] * MX.eye(3)

        Gi[3:6, 3:6] = (self.mass + 2 * self.mass * n_ci[1] + Ji[1, 1]) * MX.eye(3)
        Gi[3:6, 6:9] = -(self.mass * n_ci[1] + Ji[1, 1]) * MX.eye(3)
        Gi[3:6, 9:12] = (self.mass * n_ci[2] + Ji[1, 2]) * MX.eye(3)

        Gi[6:9, 6:9] = Ji[1, 1] * MX.eye(3)
        Gi[6:9, 9:12] = -Ji[1, 2] * MX.eye(3)

        Gi[9:12, 9:12] = Ji[2, 2] * MX.eye(3)

        # symmetrize the matrix without the diagonal blocks
        Gi[3:6, 0:3] = Gi[0:3, 3:6]
        Gi[6:9, 0:3] = Gi[0:3, 6:9]
        Gi[9:12, 0:3] = Gi[0:3, 9:12]

        Gi[6:9, 3:6] = Gi[3:6, 6:9]
        Gi[9:12, 3:6] = Gi[3:6, 9:12]

        Gi[9:12, 6:9] = Gi[6:9, 9:12]

        return Gi

    @property
    def mass_matrix(self) -> MX:
        """
        This function returns the generalized mass matrix of the segment, denoted G_i.

        Returns
        -------
        np.ndarray
            mass matrix of the segment [12 x 12]
        """

        return self._mass_matrix

    @staticmethod
    def compute_cartesian_inertia_from_pseudo(
        mass: float,
        cartesian_center_of_mass: MX,
        pseudo_inertia: MX,
        transformation_mat: MX,
    ) -> MX:
        """
        Computes the cartesian inertia matrix from pseudo-inertia matrix.

        Parameters
        ----------
        mass : float
            Mass of the segment in Segment Coordinate System
        cartesian_center_of_mass : MX
            Center of mass of the segment in Segment Coordinate System
        pseudo_inertia : MX
            Pseudo-inertia matrix of the segment
        transformation_mat : MX
            Transformation matrix from natural coordinate to segment coordinate system [3x3]

        Returns
        -------
        np.ndarray
            Pseudo-inertia matrix of the segment in the natural coordinate system [3x3]
        """
        B = transformation_mat
        middle_block = B @ (pseudo_inertia @ transpose(B))
        inertia = (
            middle_block
            - mass * transpose(cartesian_center_of_mass) @ cartesian_center_of_mass * MX.eye(3)
            + transpose(cartesian_center_of_mass) @ cartesian_center_of_mass
        )
        return inertia

    def center_of_mass(self, transformation_matrix: MX = None) -> MX:
        """
        Computes the center of mass of the segment in the specified segment coordinate system.

        Parameters
        ----------
        transformation_matrix : np.ndarray
            Transformation matrix from natural coordinate to segment coordinate system [3x3]

        Returns
        -------
        np.ndarray
            The location of the center of mass of the segment in the segment coordinate system [3x1]

        """
        if transformation_matrix is None:
            transformation_matrix = self._initial_transformation_matrix
        if transformation_matrix is None:
            raise ValueError("compute_transformation_matrix must be provided")

        return transformation_matrix @ self.natural_center_of_mass

    def inertia(self, transformation_matrix: MX = None) -> MX:
        """

        Parameters
        ----------
        transformation_matrix : np.ndarray
            Transformation matrix from natural coordinate to segment coordinate system [3x3]
        """
        if transformation_matrix is None:
            transformation_matrix = self._initial_transformation_matrix
        if transformation_matrix is None:
            raise ValueError("transformation_matrix must be provided")

        return self.compute_cartesian_inertia_from_pseudo(
            mass=self.mass,
            cartesian_center_of_mass=self.center_of_mass(transformation_matrix),
            pseudo_inertia=self.natural_pseudo_inertia,
            transformation_mat=transformation_matrix,
        )

    @classmethod
    def from_cartesian_inertial_parameters(
        cls,
        mass: Union[np.ndarray, float, np.float64] = None,
        center_of_mass: np.ndarray = None,
        inertia_matrix: np.ndarray = None,
        inertial_transformation_matrix: np.ndarray = None,
    ):
        if inertia_matrix.shape != (3, 3):
            raise ValueError("Inertia matrix must be 3x3")

        natural_center_of_mass = cls.compute_natural_center_of_mass(center_of_mass, inertial_transformation_matrix)
        pseudo_inertia_matrix = cls.compute_pseudo_inertia_matrix(
            mass=mass,
            cartesian_center_of_mass=center_of_mass,
            cartesian_inertia=inertia_matrix,
            transformation_mat=inertial_transformation_matrix,
        )

        return cls(
            mass=mass,
            natural_center_of_mass=natural_center_of_mass,
            natural_pseudo_inertia=pseudo_inertia_matrix,
            initial_transformation_matrix=inertial_transformation_matrix,
        )

    @staticmethod
    def compute_natural_center_of_mass(center_of_mass: np.ndarray, transformation_matrix: np.ndarray) -> NaturalVector:
        """
        This function computes the center of mass of the segment in the natural coordinate system.
        It transforms the center of mass of the segment in the segment coordinate system to the natural coordinate system.

        Parameters
        ----------
        center_of_mass : np.ndarray
            Center of mass of the segment in the segment coordinate system [3x1]
        transformation_matrix : np.ndarray
            Transformation matrix from natural coordinate to segment coordinate system [3x3]
        """
        # todo: write analytical inverses of transformation matrix
        return NaturalVector(inv(to_numeric(transformation_matrix)) @ center_of_mass)

    @staticmethod
    def compute_pseudo_inertia_matrix(
        mass: float,
        cartesian_center_of_mass: np.ndarray,
        cartesian_inertia: np.ndarray,
        transformation_mat: np.ndarray,
    ):
        """
        This function returns the pseudo-inertia matrix of the segment, denoted J_i.
        It transforms the inertia matrix of the segment in the segment coordinate system to the natural coordinate system.

        Parameters
        ----------
        mass : float
            mass of the segment in Segment Coordinate System
        cartesian_center_of_mass : np.ndarray
            center of mass of the segment in Segment Coordinate System
        cartesian_inertia : np.ndarray
            inertia matrix of the segment in Segment Coordinate System
        transformation_mat : np.ndarray
            Transformation matrix from natural coordinate to segment coordinate system [3x3]

        References
        ----------
            Dumas, R., Chèze, L., 2007 3D inverse dynamics in non-orthonormal segment coordinate system in section 2.2.2

        """
        center_of_mass = cartesian_center_of_mass
        inertia = cartesian_inertia

        middle_block = (
            inertia + mass * dot(center_of_mass, center_of_mass) * MX.eye(3) - dot(center_of_mass, center_of_mass)
        )

        Binv = inv(to_numeric(transformation_mat))
        Binv_transpose = transpose(Binv)

        return Binv @ middle_block @ Binv_transpose
