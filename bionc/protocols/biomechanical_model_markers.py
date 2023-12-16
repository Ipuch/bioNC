import numpy as np
from abc import ABC, abstractmethod
from casadi import MX
from typing import Any

from bionc.protocols.natural_coordinates import NaturalCoordinates


class GenericBiomechanicalModelMarkers(ABC):
    """


    Attributes
    ----------
    ???

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
    markers(self, Q: NaturalCoordinates)
        Returns the position of the markers of the system as a function of the natural coordinates Q.
    markers_constraints(self, markers: np.ndarray | MX, Q: NaturalCoordinates)
        Returns the marker constraints of all segments.
    markers_constraints_jacobian(self)
        Returns the Jacobian matrix the markers constraints.
    marker_technical_index
        This function returns the index of the marker with the given name
    """

    def __init__(
        self,
        segments: dict[str:Any, ...] = None,
    ):
        from .natural_segment import AbstractNaturalSegment  # Imported here to prevent from circular imports
        from .joint import JointBase  # Imported here to prevent from circular imports

        self.segments: dict[str:AbstractNaturalSegment, ...] = {} if segments is None else segments
        self.joints: dict[str:JointBase, ...] = {} if joints is None else joints
        # From Pythom 3.7 the insertion order in a dict is preserved. This is important because when writing a new
        # the order of the segment matters
        self._mass_matrix = self._update_mass_matrix()

    @property
    def nb_markers(self) -> int:
        """
        This function returns the number of markers in the model
        """
        nb_markers = 0
        for key in self.segments_no_ground:
            nb_markers += self.segments[key].nb_markers
        return nb_markers

    @property
    def nb_markers_technical(self) -> int:
        """
        This function returns the number of technical markers in the model
        """
        nb_markers = 0
        for key in self.segments_no_ground:
            nb_markers += self.segments[key].nb_markers_technical
        return nb_markers

    @property
    def marker_names(self) -> list[str]:
        """
        This function returns the names of the markers in the model
        """
        marker_names = []
        for key in self.segments_no_ground:
            marker_names += self.segments[key].marker_names
        return marker_names

    @property
    def marker_names_technical(self) -> list[str]:
        """
        This function returns the names of the technical markers in the model
        """
        marker_names = []
        for key in self.segments_no_ground:
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
        return self.marker_names_technical.index(name)

    @abstractmethod
    def markers_constraints(self, markers: np.ndarray | MX, Q: NaturalCoordinates):
        """
        This function returns the marker constraints of all segments, denoted Phi_r
        as a function of the natural coordinates Q.

        markers : np.ndarray | MX
            The markers positions [3,nb_markers]

        Q : NaturalCoordinates
            The natural coordinates of the segment [12 * nb_segments, 1]

        Returns
        -------
            Rigid body constraints of the segment [nb_markers x 3, 1]
        """

    @abstractmethod
    def markers_constraints_jacobian(self):
        """
        This function returns the Jacobian matrix the markers constraints, denoted k_m.

        Returns
        -------
            Joint constraints of the marker [nb_markers x 3, nb_Q]
        """
