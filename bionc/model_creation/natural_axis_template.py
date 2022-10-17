from typing import Callable

from ..model_computations.natural_axis import Axis
from ..model_computations.biomechanical_model import BiomechanicalModel
from .marker_template import MarkerTemplate
from .protocols import Data
from ..model_computations.natural_segment import NaturalSegment


class AxisTemplate:
    def __init__(self, start: Callable | str, end: Callable | str):
        """
        Parameters
        ----------
        start
            The function (f(m) -> np.ndarray, where m is a dict of markers) that defines the starting point of the axis.
            If a str is provided, the position of the corresponding marker is used
        end
            The function (f(m) -> np.ndarray, where m is a dict of markers) that defines the end point of the axis.
            If a str is provided, the position of the corresponding marker is used
        """
        self.start = MarkerTemplate(function=start, marker_type="Marker")
        self.end = MarkerTemplate(function=end, marker_type="Marker")

    def to_axis(self, data: Data, kinematic_chain: BiomechanicalModel, parent_scs: NaturalSegment = None) -> Axis:
        """
        Compute the axis from actual data
        Parameters
        ----------
        data
            The actual data
        kinematic_chain
            The model as it is constructed at that particular time. It is useful if some values must be obtained from
            previously computed values
        parent_scs
            The transformation from global to local
        """

        start = self.start.to_marker(data, kinematic_chain, parent_scs)
        end = self.end.to_marker(data, kinematic_chain, parent_scs)
        return Axis(start, end)
