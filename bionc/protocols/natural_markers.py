from abc import ABC, abstractmethod
from typing import Callable

import numpy as np

from .biomechanical_model import GenericBiomechanicalModel
from ..model_creation.protocols import Data
from .natural_coordinates import SegmentNaturalCoordinates


class AbstractNaturalMarker(ABC):
    """
    Class used to create a segment markers for the natural segments

    Methods
    -------
    from_data()
        Creates a segment marker from the data
    constraint()
        Computes the constraint for the marker given the segment natural coordinates and experimental marker location

    """

    @abstractmethod
    def from_data(
        cls,
        data: Data,
        name: str,
        function: Callable,
        parent_name: str,
        kinematic_chain: GenericBiomechanicalModel,
        Q_xp: SegmentNaturalCoordinates = None,
        is_technical: bool = True,
        is_anatomical: bool = False,
    ):
        """
        This is a constructor for the MarkerReal class. It evaluates the function that defines the marker to get an
        actual position

        Parameters
        ----------
        data
            The data to pick the data from
        name
            The name of the new marker
        function
            The function (f(m) -> np.ndarray, where m is a dict of markers (XYZ1 x time)) that defines the marker
        parent_name
            The name of the parent the marker is attached to
        kinematic_chain
            The model as it is constructed at that particular time. It is useful if some values must be obtained from
            previously computed values
        Q_xp: SegmentNaturalCoordinates
            The segment natural coordinates identified from data
        is_technical
            If the marker should be flagged as a technical marker
        is_anatomical
            If the marker should be flagged as an anatomical marker
        """

    def constraint(self, marker_location: np.ndarray, Qi: SegmentNaturalCoordinates) -> np.ndarray:
        """
        This function computes the constraint for the marker

        Parameters
        ----------
        marker_location: np.ndarray
            The location of the marker in the global/inertial coordinate system
        Qi
            The segment natural coordinates

        Returns
        -------
        The constraint for the marker
        """


class AbstractSegmentNaturalVector(ABC):
    """
    Class used to create a segment vector for the natural segments

    """

    def position_in_global(self, Qi: SegmentNaturalCoordinates):
        """
        This function computes the position of the vector in the global coordinate system

        Parameters
        ----------
        Qi : SegmentNaturalCoordinates
            The segment natural coordinates

        Returns
        -------
        The position of the vector in the global coordinate system
        """
