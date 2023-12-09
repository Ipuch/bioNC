from casadi import MX
from casadi import vertcat

from .natural_coordinates import SegmentNaturalCoordinates
from .natural_marker import NaturalMarker
from ..protocols.natural_segment_markers import AbstractNaturalSegmentMarkers


class NaturalSegmentMarkers(AbstractNaturalSegmentMarkers):
    """
    Class used to define markers of a segment based on natural coordinate.

    Attributes
    ----------
    _markers : list[NaturalMarker]
        name of the segment
    """

    def __init__(self):
        super(NaturalSegmentMarkers, self).__init__()

    def positions(self, Qi: SegmentNaturalCoordinates) -> MX:
        """
        This function returns the position of the markers of the system as a function of the natural coordinates Q
        also referred as forward kinematics

        Parameters
        ----------
        Qi : SegmentNaturalCoordinates
            The natural coordinates of the segment [12 x n, 1]

        Returns
        -------
        MX
            The position of the markers [3, nbMarkers, nbFrames]
            in the global coordinate system/ inertial coordinate system
        """
        if not isinstance(Qi, SegmentNaturalCoordinates):
            Qi = SegmentNaturalCoordinates(Qi)

        markers = MX.zeros((3, self.nb_markers))
        for i, marker in enumerate(self._markers):
            markers[:, i] = marker.position_in_global(Qi)

        return markers

    def constraints(self, marker_locations: MX, Qi: SegmentNaturalCoordinates, only_technical: bool = True) -> MX:
        """
        This function returns the marker constraints of the segment

        Parameters
        ----------
        marker_locations: MX
            Marker locations in the global/inertial coordinate system [3,N_markers]
        Qi: SegmentNaturalCoordinates
            Natural coordinates of the segment
        only_technical: bool
            If True, only the constraints of technical markers are returned, by default True

        Returns
        -------
        MX
            The defects of the marker constraints of the segment (3 x N_markers)
        """
        nb_markers = self.nb_markers_technical if only_technical else self.nb_markers
        markers = [m for m in self._markers if m.is_technical] if only_technical else self._markers

        if marker_locations.shape != (3, nb_markers):
            raise ValueError(f"marker_locations should be of shape (3, {nb_markers})")

        defects = MX.zeros((3, nb_markers))

        for i, marker in enumerate(markers):
            defects[:, i] = marker.constraint(marker_location=marker_locations[:, i], Qi=Qi)

        return defects

    def jacobian(self, only_technical: bool = True) -> MX:
        """
        This function returns the marker jacobian of the segment

        Parameters
        ----------
        only_technical: bool
            If True, only the constraints jacobian of technical markers are returned, by default True

        Returns
        -------
        MX
            The jacobian of the marker constraints of the segment [3, N_markers]
        """
        nb_markers = self.nb_markers_technical if only_technical else self.nb_markers
        markers = [m for m in self._markers if m.is_technical] if only_technical else self._markers
        return vertcat(*[-marker.interpolation_matrix for marker in markers]) if nb_markers > 0 else MX.zeros((0, 12))
