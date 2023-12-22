import numpy as np
from abc import ABC, abstractmethod

from ..natural_coordinates import SegmentNaturalCoordinates
from ..natural_markers import AbstractNaturalMarker


class AbstractNaturalSegmentMarkers(ABC):
    """
    Abstract class used to define anatomical segment based on natural markers.

    This class provides an interface for adding markers to a segment, retrieving the number of markers,
    retrieving the names of the markers, and performing operations related to the markers' positions,
    constraints, and jacobian.

    Methods
    -------
    add()
        This function adds a marker to the segment
    add_from_segment_coordinates()
        Add a new marker to the segment
    nb_markers
        This function returns the number of markers of the segment
    nb_technical_markers
        This function returns the number of technical markers of the segment
    marker_names
        This function returns the names of the markers of the segment
    marker_names_technical
        This function returns the names of the technical markers of the segment
    marker_from_name()
        This function returns the marker with the given name
    markers()
        This function returns the position of the markers of the system as a function of the natural coordinates Q also referred as forward kinematics
    constraints()
        This function returns the defects of the marker constraints of the segment, denoted Phi_m
    jacobian()
        This function returns the jacobian of the marker constraints of the segment, denoted K_m

    Attributes
    ----------
    _markers : list[NaturalMarker]
        name of the segment
    """

    def __init__(self):
        self._markers = []

    def add(self, marker: AbstractNaturalMarker):
        """
        Add a new marker to the segment

        Parameters
        ----------
        marker
            The marker to add
        """
        self._markers.append(marker)

    @property
    def nb_markers(self) -> int:
        """
        Get the number of markers in the segment.

        Returns
        -------
        int
            The number of markers in the segment.
        """
        return len(self._markers)

    @property
    def nb_markers_technical(self) -> int:
        """
        Get the number of technical markers in the segment.

        Returns
        -------
        int
            The number of technical markers in the segment.
        """
        return len(self.marker_names_technical)

    @property
    def marker_names(self) -> list[str]:
        """
        Get the names of the markers in the segment.

        Returns
        -------
        list[str]
            The names of the markers in the segment.
        """
        return [marker.name for marker in self._markers]

    @property
    def marker_names_technical(self) -> list[str]:
        """
        Get the names of the technical markers in the segment.

        Returns
        -------
        list[str]
            The names of the technical markers in the segment.
        """
        return [marker.name for marker in self._markers if marker.is_technical]

    @abstractmethod
    def positions(self, Qi: SegmentNaturalCoordinates):
        """
        This function returns the position of the markers of the system as a function of the natural coordinates Q
        also referred as forward kinematics

        Parameters
        ----------
        Qi : SegmentNaturalCoordinates
            The natural coordinates of the segment [12 x n, 1]

        Returns
        -------
            The position of the markers [3, nbMarkers, nbFrames]
            in the global coordinate system/ inertial coordinate system
        """

    @abstractmethod
    def constraints(self, marker_locations, Qi: SegmentNaturalCoordinates, only_technical: bool):
        """
        This function returns the marker constraints of the segment

        Parameters
        ----------
        marker_locations: MX or np.ndarray
            Marker locations in the global/inertial coordinate system (3 x N_markers)
        Qi: SegmentNaturalCoordinates
            Natural coordinates of the segment
        only_technical: bool
            If True, only the constraints of technical markers are returned, by default True
        """

    @abstractmethod
    def jacobian(self, only_technical: bool):
        """
        This function returns the marker jacobian of the segment
        """

    def marker_from_name(self, marker_name: str) -> AbstractNaturalMarker:
        """
        This function returns the marker with the given name

        Parameters
        ----------
        marker_name: str
            Name of the marker

        Returns
        -------
        AbstractNaturalMarker
            The marker with the given name.

        Raises
        ------
        ValueError
            If no marker with the given name is found.
        """
        for marker in self._markers:
            if marker.name == marker_name:
                return marker

        raise ValueError(f"No marker with name {marker_name} was found")

    def __iter__(self):
        """
        Get an iterator for the markers in the segment.

        Returns
        -------
        iterator
            An iterator for the markers in the segment.
        """
        return iter(self._markers)
