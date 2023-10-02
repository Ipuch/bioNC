from typing import Callable, Union

import numpy as np

from bionc.bionc_numpy.biomechanical_model import BiomechanicalModel
from ..bionc_numpy.natural_marker import NaturalMarker, Marker

# from .biomechanical_model_template import BiomechanicalModelTemplate
from .protocols import Data
from ..bionc_numpy.natural_segment import NaturalSegment

# from ..utils.natural_coordinates import SegmentNaturalCoordinates
from ..protocols.natural_coordinates import SegmentNaturalCoordinates


class MarkerTemplate:
    def __init__(
        self,
        name: str = None,
        function: Callable | str = None,
        parent_name: str = None,
        is_technical: bool = True,
        is_anatomical: bool = False,
        marker_type: str = "NaturalMarker",
    ):
        """
        This is a pre-constructor for the NaturalMarker class. It allows to create a generic model by marker names

        Parameters
        ----------
        name
            The name of the new marker
        function
            The function (f(m) -> np.ndarray, where m is a dict of markers) that defines the marker with.
            If a str is provided, the position of the corresponding marker is used
        parent_name
            The name of the parent the marker is attached to
        is_technical
            If the marker should be flagged as a technical marker
        is_anatomical
            If the marker should be flagged as an anatomical marker
        """
        self.name = name
        function = function if function is not None else self.name
        self.function = (lambda m, bio: m[function]) if isinstance(function, str) else function
        self.parent_name = parent_name
        self.is_technical = is_technical
        self.is_anatomical = is_anatomical
        self.marker_type = marker_type

    def to_marker(
        self, data: Data, kinematic_chain: BiomechanicalModel, natural_segment: NaturalSegment = None
    ) -> Union[NaturalMarker, Marker]:
        return Marker.from_data(
            data=data,
            name=self.name,
            function=self.function,
            kinematic_chain=kinematic_chain,
            is_technical=self.is_technical,
            is_anatomical=self.is_anatomical,
        )

    def to_natural_marker(
        self, data: Data, kinematic_chain: BiomechanicalModel, Q_xp: SegmentNaturalCoordinates = None
    ) -> Union[NaturalMarker, Marker]:
        return NaturalMarker.from_data(
            data,
            self.name,
            self.function,
            self.parent_name,
            kinematic_chain,
            Q_xp=Q_xp,
            is_technical=self.is_technical,
            is_anatomical=self.is_anatomical,
        )

    @staticmethod
    def normal_to(m, bio, mk1: np.ndarray | str, mk2: np.ndarray | str, mk3: np.ndarray | str) -> np.ndarray:
        """
        Compute the normal vector to a plane defined by three points.

        Parameters
        ----------
        m: dict[str, float]
            Dictionnaries containing the location of markers in global frames
        bio: BiomechanicalModel
            The model as it is constructed at that particular time. It is useful if some values must be obtained from previously computed values
        mk1: np.ndarray | str
            First point of the plane
        mk2: np.ndarray | str
            Second point of the plane
        mk3: np.ndarray | str
            Third point of the plane

        Returns
        -------
        normalized_plan_vector : np.ndarray
            Normalized vector normal to the plane
        """
        if isinstance(mk1, str):
            mk1 = m[mk1]
        if isinstance(mk2, str):
            mk2 = m[mk2]
        if isinstance(mk3, str):
            mk3 = m[mk3]

        normalized_plan_vector = np.ones((4, mk1.shape[1]))
        for i, (mk1i, mk2i, mk3i) in enumerate(zip(mk1.T, mk2.T, mk3.T)):
            v1 = mk2i[:3] - mk1i[:3]
            v2 = mk3i[:3] - mk1i[:3]
            normalized_plan_vector[:3, i] = np.cross(v1, v2) / np.linalg.norm(np.cross(v1, v2))

        return normalized_plan_vector

    @staticmethod
    def middle_of(m, bio, m1: str, m2: str):
        """
        Compute the position of the middle of two points.

        Parameters
        ----------
        m: dict[str, float]
            Dictionnaries containing the location of markers in global frames
        bio: BiomechanicalModel
            The model as it is constructed at that particular time. It is useful if some values must be obtained from previously computed values
        m1: np.ndarray | str
            First point
        m2: np.ndarray | str
            Second point

        Returns
        -------
        middle_point : np.ndarray
            middle point between m1 and m2
        """
        return (m[m1] + m[m2]) / 2
