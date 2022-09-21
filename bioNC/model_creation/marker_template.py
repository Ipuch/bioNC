from typing import Callable

from ..model_computations.biomechanical_model import BiomechanicalModel
from ..model_computations.marker import Marker
from .protocols import Data
from ..model_computations.segment_coordinate_system import NaturalSegment


class MarkerTemplate:
    def __init__(
        self,
        name: str = None,
        function: Callable | str = None,
        parent_name: str = None,
        is_technical: bool = True,
        is_anatomical: bool = False,
    ):
        """
        This is a pre-constructor for the Marker class. It allows to create a generic model by marker names

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

    def to_marker(
        self, data: Data, kinematic_chain: BiomechanicalModel, parent_scs: NaturalSegment = None
    ) -> Marker:
        return Marker.from_data(
            data,
            self.name,
            self.function,
            self.parent_name,
            kinematic_chain,
            parent_scs,
            is_technical=self.is_technical,
            is_anatomical=self.is_anatomical,
        )
