import numpy as np

from .biomechanical_model_segments import BiomechanicalModelSegments
from .natural_coordinates import NaturalCoordinates
from ..protocols.biomechanical_model_markers import GenericBiomechanicalModelMarkers


class BiomechanicalModelMarkers(GenericBiomechanicalModelMarkers):
    """
    This is an abstract base class that provides the basic structure and methods for all markers of a biomechanical model.
    The implemented methods are specific to numpy.

    Methods
    -------
    markers(self, Q: NaturalCoordinates)
        Returns the position of the markers of the system as a function of the natural coordinates Q.
    markers_constraints(self, markers: np.ndarray | MX, Q: NaturalCoordinates)
        Returns the marker constraints of all segments.
    markers_constraints_jacobian(self)
        Returns the Jacobian matrix the markers constraints.
    Q_from_markers(self, markers: np.ndarray)
        Returns the natural coordinates of the system as a function of the markers positions.
    """

    def __init__(
        self,
        segments: BiomechanicalModelSegments = None,
    ):
        segments = BiomechanicalModelSegments() if segments is None else segments
        super().__init__(segments=segments)

    def markers(self, Q: NaturalCoordinates) -> np.ndarray:
        """
        This function returns the position of the markers of the system as a function of the natural coordinates Q
        also referred as forward kinematics

        Parameters
        ----------
        Q : NaturalCoordinates
            The natural coordinates of the segment [12 x n, 1]

        Returns
        -------
        np.ndarray
            The position of the markers [3, nbMarkers, nbFrames]
            in the global coordinate system/ inertial coordinate system
        """
        markers = np.zeros((3, self.nb_markers, Q.shape[1]))

        for segment in self.segments.segments_no_ground.values():
            idx = self.indexes(segment.index, only_technical=False)
            markers[:, idx] = segment.markers(Q.vector(segment.index))

        return markers

    def center_of_mass_position(self, Q: NaturalCoordinates) -> np.ndarray:
        """
        This function returns the position of the center of mass of each segment as a function of the natural coordinates Q

        Parameters
        ----------
        Q : NaturalCoordinates
            The natural coordinates of the segment [12 x n, 1]

        Returns
        -------
        np.ndarray
            The position of the center of mass [3, nbSegments]
            in the global coordinate system/ inertial coordinate system
        """
        com = np.zeros((3, self.segments.nb_segments, Q.shape[1]))
        for i, segment in enumerate(self.segments.segments_no_ground.values()):
            position = segment.center_of_mass_position(Q.vector(i))
            com[:, i, :] = position if len(position.shape) == 2 else position[:, np.newaxis]

        return com

    def constraints(self, markers: np.ndarray, Q: NaturalCoordinates, only_technical: bool = True) -> np.ndarray:
        """
        This function returns the marker constraints of all segments, denoted Phi_r
        as a function of the natural coordinates Q.

        markers : np.ndarray
            The markers positions [3,nb_markers]
        Q : NaturalCoordinates
            The natural coordinates of the segment [12 x n, 1]
        only_technical : bool
            If True, only technical markers are considered, by default True,
            because we only want to use technical markers for inverse kinematics, this choice can be revised.

        Returns
        -------
        np.ndarray
            Defects of the marker constraints [nb_markers x 3, 1]
        """
        nb_markers = self.nb_markers_technical if only_technical else self.nb_markers
        if markers.shape[1] != nb_markers:
            raise ValueError(
                f"markers should have {nb_markers} columns. "
                f"And should include the following markers: "
                f"{self.names_technical if only_technical else self.names}"
            )

        phi_m = np.zeros((nb_markers * 3))

        for i_segment, segment in enumerate(self.segments.segments_no_ground.values()):
            nb_segment_markers = segment.nb_markers_technical if only_technical else segment.nb_markers
            if nb_segment_markers == 0:
                continue
            constraint_idx = self.direction_index(i_segment, only_technical)
            marker_idx = self.indexes(i_segment, only_technical)

            markers_temp = markers[:, marker_idx]
            phi_m[constraint_idx] = segment.marker_constraints(markers_temp, Q.vector(i_segment)).flatten("F")

        return phi_m

    def constraints_jacobian(self, only_technical: bool = True) -> np.ndarray:
        """
        This function returns the Jacobian matrix the markers constraints, denoted K_m.

        Parameters
        ----------
        only_technical : bool
            If True, only technical markers are considered, by default True,
            because we only want to use technical markers for inverse kinematics, this choice can be revised.

        Returns
        -------
        np.ndarray
            Jacobian of the constraints of the marker [nb_markers x 3, nb_Q]
        """
        nb_markers = self.nb_markers_technical if only_technical else self.nb_markers

        km = np.zeros((3 * nb_markers, 12 * self.segments.nb_segments))
        for i_segment, segment in enumerate(self.segments.segments_no_ground.values()):
            nb_segment_markers = segment.nb_markers_technical if only_technical else segment.nb_markers
            if nb_segment_markers == 0:
                continue

            constraint_idx = self.direction_index(i_segment, only_technical)
            segment_idx = segment.coordinates_slice

            km[constraint_idx, segment_idx] = segment.markers_jacobian()

        return km

    def Q_from_markers(self, markers: np.ndarray) -> NaturalCoordinates:
        """
        This function returns the natural coordinates of the system as a function of the markers positions
        also referred as inverse kinematics
        but the constraints are not enforced, this can be used as an initial guess for proper inverse kinematics.

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
        if markers.shape[1] != self.nb_markers_technical:
            raise ValueError(
                f"markers should have {self.nb_markers_technical} columns, "
                f"and should include the following markers: {self.names_technical}"
            )

        # convert markers to Data
        from ..model_creation.generic_data import GenericData

        marker_data = GenericData(markers, self.names_technical)

        Q = []

        for segment in self.segments.segments_no_ground.values():
            Q.append(segment._Qi_from_markers(marker_data, self))

        return NaturalCoordinates.from_qi(tuple(Q))
