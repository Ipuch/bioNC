import numpy as np
from typing import Callable

from ..biomechanical_model import BiomechanicalModel
from ..natural_marker import Marker
from ...model_creation.protocols import Data


class Axis:
    def __init__(self, start: Marker, end: Marker):
        """
        Parameters
        ----------
        start:
            The initial NaturalMarker
        end:
            The final NaturalMarker
        """
        self.start_point = start
        self.end_point = end

    def axis(self) -> np.ndarray:
        """
        Returns the axis vector
        """
        start = self.start_point.position
        end = self.end_point.position
        return (end - start) / np.linalg.norm(end[:3] - start[:3], axis=0)

    @classmethod
    def from_data(
        cls,
        data: Data,
        function: Callable,
        kinematic_chain: BiomechanicalModel,
    ) -> "Axis":
        """
        Compute the axis from actual data

        Parameters
        ----------
        data: Data
            The actual data
        function: Callable
            The function (f(m) -> np.ndarray, where m is a dict of markers) that defines the axis
        kinematic_chain: BiomechanicalModel
            The model as it is constructed at that particular time. It is useful if some values must be obtained from
            previously computed values
        """
        position: np.ndarray = function(data.values, kinematic_chain)
        position_zero = np.zeros(position.shape)

        return cls(Marker(name=None, position=position_zero), Marker(name=None, position=position))
