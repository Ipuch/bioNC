from typing import Callable

from ..model_computations.natural_axis import NaturalAxis
from .natural_axis_template import NaturalAxisTemplate
from ..model_computations.biomechanical_model import BiomechanicalModel
from .marker_template import MarkerTemplate
from .protocols import Data
from ..model_computations.segment_coordinate_system import NaturalSegmentCoordinateSystem


class NaturalSegmentCoordinateSystemTemplate:
    def __init__(
        self,
        u_axis: NaturalAxisTemplate,
        rp: Callable | str,
        rd: Callable | str,
        w_axis: NaturalAxisTemplate,
    ):
        """
        Set the SegmentCoordinateSystemReal matrix of the segment. To compute the third axis, a first cross product of
        the first_axis with the second_axis is performed. All the axes are then normalized. Then, either the first or
        second axis (depending on [axis_to_keep]) is recomputed with a cross product to get an
        orthonormal system of axes. The system is finally moved to the origin

        Parameters
        ----------
        u_axis
            The first axis of the segment
        rp
            The function to compute the origin of the segment, proximal location of the segment
        rd
            The function to compute the distal end of the segment, distal location of the segment
        w_axis
            The third axis of the segment

        """

        self.rp = MarkerTemplate(function=rp)
        self.rd = MarkerTemplate(function=rd)
        self.u_axis = u_axis
        self.w_axis = w_axis

    def to_sncs(
        self, data: Data, kinematic_chain: BiomechanicalModel) -> NaturalSegmentCoordinateSystem:
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

        return NaturalSegmentCoordinateSystem.from_markers(
            self.origin.to_marker(data, kinematic_chain),
            self.first_axis.to_axis(data, kinematic_chain),
            self.second_axis.to_axis(data, kinematic_chain),
            self.axis_to_keep,
        )
