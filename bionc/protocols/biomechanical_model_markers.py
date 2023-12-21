import numpy as np
from abc import ABC, abstractmethod
from casadi import MX

from .biomechanical_model_segments import GenericBiomechanicalModelSegments
from .natural_coordinates import NaturalCoordinates


class GenericBiomechanicalModelMarkers(ABC):
    """
    This is an abstract base class that provides the basic structure and methods for all markers of a biomechanical models.
    The implemented methods are not specific to numpy or casadi.

    Attributes
    ----------
    segments : dict
        A dictionary containing the segments of the model. The keys are the names of the segments and the values are the corresponding segment objects.

    Methods
    -------
    nb_markers(self) -> int
        Returns the number of markers in the model.
    nb_markers_technical(self) -> int
        Returns the number of technical markers in the model.
    marker_names(self) -> list[str]
        Returns the names of the markers in the model.
    marker_names_technical(self) -> list[str]
        Returns the names of the technical markers in the model.
    marker_technical_index
        This function returns the index of the marker with the given name
    markers(self, Q: NaturalCoordinates)
        Returns the position of the markers of the system as a function of the natural coordinates Q.
    constraints(self, markers: np.ndarray | MX, Q: NaturalCoordinates)
        Returns the marker constraints of all segments.
    constraints_jacobian(self)
        Returns the Jacobian matrix the markers constraints.
    center_of_mass_position(self, Q: NaturalCoordinates)
        Returns the position of the center of mass of each segment as a function of the natural coordinates Q.
    Q_from_markers(self, markers: np.ndarray) -> NaturalCoordinates
        Returns the natural coordinates of the system as a function of the markers positions.
    """

    def __init__(
        self,
        segments: GenericBiomechanicalModelSegments = None,
    ):
        self.segments = segments

    @property
    def nb_markers(self) -> int:
        """
        This function returns the number of markers in the model
        """
        nb_markers = 0
        for key in self.segments.segments_no_ground:
            nb_markers += self.segments[key].nb_markers
        return nb_markers

    @property
    def nb_markers_technical(self) -> int:
        """
        This function returns the number of technical markers in the model
        """
        nb_markers = 0
        for key in self.segments.segments_no_ground:
            nb_markers += self.segments[key].nb_markers_technical
        return nb_markers

    @property
    def names(self) -> list[str]:
        """
        This function returns the names of the markers in the model
        """
        marker_names = []
        for key in self.segments.segments_no_ground:
            marker_names += self.segments[key].marker_names
        return marker_names

    @property
    def names_technical(self) -> list[str]:
        """
        This function returns the names of the technical markers in the model
        """
        marker_names = []
        for key in self.segments.segments_no_ground:
            marker_names += self.segments[key].marker_names_technical
        return marker_names

    def marker_technical_index(self, name: str) -> int:
        """
        This function returns the index of the marker with the given name

        Parameters
        ----------
        name : str
            The name of the marker

        Returns
        -------
        int
            The index of the marker with the given name
        """
        return self.names_technical.index(name)

    @abstractmethod
    def markers(self, Q: NaturalCoordinates):
        """
        This function returns the position of the markers of the system as a function of the natural coordinates Q
        also referred as forward kinematics

        Parameters
        ----------
        Q : NaturalCoordinates
            The natural coordinates of the segment [12 x n, 1]

        Returns
        -------
            The position of the markers [3, nbMarkers, nbFrames]
            in the global coordinate system/ inertial coordinate system
        """

    @abstractmethod
    def constraints(self, markers: np.ndarray | MX, Q: NaturalCoordinates, only_technical: bool):
        """
        This function returns the marker constraints of all segments, denoted Phi_r
        as a function of the natural coordinates Q.

        markers : np.ndarray | MX
            The markers positions [3, nb_markers]
        Q : NaturalCoordinates
            The natural coordinates of the segment [12 * nb_segments, 1]
        only_technical : bool
            If True, only the technical markers are considered

        Returns
        -------
            Defects of the marker constraints [nb_markers x 3, 1]
        """

    @abstractmethod
    def constraints_jacobian(self, only_technical: bool):
        """
        This function returns the Jacobian matrix the markers constraints, denoted k_m.

        only_technical : bool
            If True, only technical markers are considered, by default True,
            because we only want to use technical markers for inverse kinematics, this choice can be revised.

        Returns
        -------
            Jacobian of the constraints of the marker [nb_markers x 3, nb_Q]
        """

    @abstractmethod
    def center_of_mass_position(self, Q: NaturalCoordinates):
        """
        This function returns the position of the center of mass of each segment as a function of the natural coordinates Q

        Parameters
        ----------
        Q : NaturalCoordinates
            The natural coordinates of the segment [12 x n, 1]

        Returns
        -------
            The position of the center of mass [3, nbSegments]
            in the global coordinate system/ inertial coordinate system
        """

    @abstractmethod
    def Q_from_markers(self, markers: np.ndarray) -> NaturalCoordinates:
        """
        This function returns the natural coordinates of the system as a function of the markers positions
        also referred as inverse kinematics
        but the constraints are not enforced,
        this can be used as an initial guess for proper inverse kinematics.

        Parameters
        ----------
        markers : np.ndarray
            The markers positions [3, nb_markers]

        Returns
        -------
        NaturalCoordinates
            The natural coordinates of the segment [12 x n, 1]

        See Also
        --------
        ..bionc_numpy.inverse_kinematics
        """
