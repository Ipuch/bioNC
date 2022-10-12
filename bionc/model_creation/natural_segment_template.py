from typing import Callable

from ..model_computations.natural_axis import Axis
from .natural_axis_template import AxisTemplate
from ..model_computations.biomechanical_model import BiomechanicalModel
from .marker_template import MarkerTemplate
from .protocols import Data
from ..model_computations.natural_segment import NaturalSegment


class NaturalSegmentTemplate:
    def __init__(
        self,
        u_axis: AxisTemplate,
        proximal_point: Callable | str,
        distal_point: Callable | str,
        w_axis: AxisTemplate,
    ):
        """
        Set the SegmentCoordinateSystemReal matrix of the segment. To compute the third axis, a first cross product of
        the first_axis with the second_axis is performed. All the axes are then normalized. Then, either the first or
        second axis (depending on [axis_to_keep]) is recomputed with a cross product to get an
        orthonormal system of axes. The system is finally moved to the origin

        Parameters
        ----------
        u_axis
            The first axis of the segment, denoted u
        proximal_point
            The function to compute the origin of the segment, proximal location of the segment, denoted rp
        distal_point
            The function to compute the distal end of the segment, distal location of the segment, denoted rd
        w_axis
            The third axis of the segment, denoted w

        """

        self.u_axis = u_axis
        self.proximal_point = MarkerTemplate(function=proximal_point, marker_type="Marker")
        self.distal_point = MarkerTemplate(function=distal_point, marker_type="Marker")
        self.w_axis = w_axis

    def update(self, data: Data, kinematic_chain: BiomechanicalModel) -> NaturalSegment:
        """
        Collapse the generic SegmentCoordinateSystem to an actual SegmentCoordinateSystemReal with value
        based on the model and the data

        Parameters
        ----------
        data
            The actual data
        kinematic_chain
            The model as it is constructed at that particular time. It is useful if some values must be obtained from
            previously computed values
        parent_scs
            The SegmentCoordinateSystemReal of the parent to compute the local transformation
        Returns
        -------
        The collapsed SegmentCoordinateSystemReal
        """

        return NaturalSegment.from_markers(
            u_axis=self.u_axis.to_axis(data, kinematic_chain),
            proximal_point=self.proximal_point.to_marker(data, kinematic_chain),
            distal_point=self.distal_point.to_marker(data, kinematic_chain),
            w_axis=self.w_axis.to_axis(data, kinematic_chain),
        )
