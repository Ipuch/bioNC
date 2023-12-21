import numpy as np
from casadi import MX

from .biomechanical_model_segments import BiomechanicalModelSegments
from .natural_coordinates import NaturalCoordinates
from ..protocols.biomechanical_model_markers import GenericBiomechanicalModelMarkers


class BiomechanicalModelMarkers(GenericBiomechanicalModelMarkers):
    """


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

    def markers(self, Q: NaturalCoordinates) -> MX:
        """
        This function returns the position of the markers of the system as a function of the natural coordinates Q
        also referred as forward kinematics

        Parameters
        ----------
        Q : NaturalCoordinates
            The natural coordinates of the segment [12 x n, 1]

        Returns
        -------
        MX
            The position of the markers [3, nbMarkers, nbFrames]
            in the global coordinate system/ inertial coordinate system
        """
        markers = MX.zeros((3, self.nb_markers))
        nb_markers = 0
        for segment in self.segments.segments_no_ground.values():
            idx = slice(nb_markers, nb_markers + segment.nb_markers)
            markers[:, idx] = segment.markers(Q.vector(segment.index))
            nb_markers += segment.nb_markers

        return markers

    def center_of_mass_position(self, Q: NaturalCoordinates) -> MX:
        """
        This function returns the position of the center of mass of each segment as a function of the natural coordinates Q

        Parameters
        ----------
        Q : NaturalCoordinates
            The natural coordinates of the segment [12 x n, 1]

        Returns
        -------
        MX
            The position of the center of mass [3, nbSegments]
            in the global coordinate system/ inertial coordinate system
        """
        com = MX.zeros((3, self.segments.nb_segments))
        for i, segment in enumerate(self.segments.segments_no_ground.values()):
            position = segment.center_of_mass_position(Q.vector(i))
            com[:, i] = position

        return com

    def constraints(self, markers: np.ndarray | MX, Q: NaturalCoordinates, only_technical: bool = True) -> MX:
        """
        This function returns the marker constraints of all segments, denoted Phi_r
        as a function of the natural coordinates Q.

        markers : np.ndarray | MX
           The markers positions [3,nb_markers]
        Q : NaturalCoordinates
           The natural coordinates of the segment [12 x n, 1]
        only_technical : bool
           If True, only technical markers are considered, by default True,
           because we only want to use technical markers for inverse kinematics, this choice can be revised.

        Returns
        -------
        MX
           Defects of the marker constraints [nb_markers x 3, 1]
        """
        if not isinstance(markers, MX):
            markers = MX(markers)

        nb_markers = self.nb_markers_technical if only_technical else self.nb_markers
        if markers.shape[1] != nb_markers:
            raise ValueError(
                f"markers should have {nb_markers} columns. "
                f"And should include the following markers: "
                f"{self.names_technical if only_technical else self.names}"
            )

        phi_m = MX.zeros((nb_markers * 3, 1))
        marker_count = 0

        for i_segment, segment in enumerate(self.segments.segments_no_ground.values()):
            nb_segment_markers = segment.nb_markers_technical if only_technical else self.segments[name].nb_markers
            if nb_segment_markers == 0:
                continue
            constraint_idx = slice(marker_count * 3, (marker_count + nb_segment_markers) * 3)
            marker_idx = slice(marker_count, marker_count + nb_segment_markers)

            markers_temp = markers[:, marker_idx]
            phi_m[constraint_idx] = segment.marker_constraints(
                markers_temp, Q.vector(i_segment), only_technical=only_technical
            )[
                :
            ]  # [:] to flatten the array

            marker_count += nb_segment_markers

        return phi_m

    def constraints_jacobian(self, only_technical: bool = True) -> MX:
        """
        This function returns the Jacobian matrix the markers constraints, denoted K_m.

        Parameters
        ----------
        only_technical : bool
            If True, only technical markers are considered, by default True,
            because we only want to use technical markers for inverse kinematics, this choice can be revised.

        Returns
        -------
        MX
            Jacobian of the constraints of the marker [nb_markers x 3, nb_Q]
        """
        nb_markers = self.nb_markers_technical if only_technical else self.nb_markers

        km = MX.zeros((3 * nb_markers, 12 * self.segments.nb_segments))
        marker_count = 0
        for i_segment, segment in enumerate(self.segments.segments_no_ground.values()):
            nb_segment_markers = segment.nb_markers_technical if only_technical else segment.nb_markers
            if nb_segment_markers == 0:
                continue
            constraint_idx = slice(marker_count * 3, (marker_count + nb_segment_markers) * 3)
            segment_idx = slice(12 * i_segment, 12 * (i_segment + 1))
            km[constraint_idx, segment_idx] = segment.markers_jacobian()
            marker_count += nb_segment_markers

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
        raise NotImplementedError("This function is not implemented yet with casadi.")
        # todo: we should replace lambda function with casadi function,
        #  and use casadi type DM instead of the current use of np.ndarray
        #  so it would be possible to use this function with MX type.
