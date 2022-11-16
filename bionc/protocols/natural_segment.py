# import ABC
from abc import ABC, abstractmethod
from typing import Tuple, Union
import numpy as np


class AbstractNaturalSegment(ABC):
    """
        Class used to define anatomical segment based on natural coordinate.

    Methods
    -------
    set_name
        This method is used to set the name of the segment.
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


    add_marker()
        This function adds a marker to the segment
    nb_markers()
        This function returns the number of markers in the segment
    marker_constraints()
        This function returns the defects of the marker constraints of the segment, denoted Phi_m
    marker_jacobian()
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
    def name(self):
        """
        This function returns the name of the segment

        Returns
        -------
        str
            Name of the segment
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
    def _center_of_mass_in_natural_coordinates_system(self):
        """
        This function computes the center of mass of the segment in the natural coordinate system.
        It transforms the center of mass of the segment in the segment coordinate system to the natural coordinate system.

        Returns
        -------
        np.ndarray
            Center of mass of the segment in the natural coordinate system [3x1]
        """

    @abstractmethod
    def center_of_mass_in_natural_coordinates_system(self):
        """
        This function returns the center of mass of the segment in the natural coordinate system.
        It transforms the center of mass of the segment in the segment coordinate system to the natural coordinate system.

        Returns
        -------
        np.ndarray
            Center of mass of the segment in the natural coordinate system [3x1]
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
    def _interpolation_matrix_center_of_mass(self):
        """
        This function returns the interpolation matrix for the center of mass of the segment, denoted N_i^Ci.
        It allows to apply the gravity force at the center of mass of the segment.

        Returns
        -------
        np.ndarray
            Interpolation matrix for the center of mass of the segment in the natural coordinate system [12 x 3]
        """

    @abstractmethod
    def interpolation_matrix_center_of_mass(self):
        """
        This function returns the interpolation matrix for the center of mass of the segment, denoted N_i^Ci.
        It allows to apply the gravity force at the center of mass of the segment.

        Returns
        -------
        np.ndarray
            Interpolation matrix for the center of mass of the segment in the natural coordinate system [12 x 3]
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

    @abstractmethod
    def add_marker(self, marker):
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
    def marker_jacobian(self):
        """
        This function returns the marker jacobian of the segment

        Returns
        -------
        np.ndarray
            The jacobian of the marker constraints of the segment (3 x N_markers)
        """
