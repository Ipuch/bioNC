from enum import Enum

import numpy as np

from bionc.bionc_numpy import (
    BiomechanicalModel,
    NaturalSegment,
    SegmentNaturalCoordinates,
)


class InverseKinematicsMethods(Enum):
    """
    This class lists the available inverse kinematics methods
    """

    NEWTON_RAPHSON = "newton_raphson"
    OTHER_METHOD = "other_method"


class InverseKinematics:
    """
    This class solve the inverse kinematics problem based on inverse kinematics methods

    Parameters
    ----------
    biomechanical_model
        The biomechanical model
    markers
        The markers [3, n_markers, n_frames]
    """

    def __init__(
        self,
        biomechanical_model: BiomechanicalModel,
        markers: np.ndarray,
    ):
        self.biomechanical_model = biomechanical_model
        self.markers = markers

        # self.Q_init = np.zeros(self.biomechanical_model.nb_Q, self.markers.shape[2])
        self.Q_init = self.biomechanical_model.Q_from_xp(markers)

    def solve(self, method: InverseKinematicsMethods = InverseKinematicsMethods.NEWTON_RAPHSON):
        """
        This function solves the inverse kinematics problem

        Parameters
        ----------
        method
            The method to use to solve the inverse kinematics problem
        """

        # initialize matrices
