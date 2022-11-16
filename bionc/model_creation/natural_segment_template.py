from typing import Callable

from .natural_axis_template import AxisTemplate
from bionc.bionc_numpy.biomechanical_model import BiomechanicalModel

# from ..utils.natural_coordinates import SegmentNaturalCoordinates
from ..protocols.natural_coordinates import SegmentNaturalCoordinates
from .marker_template import MarkerTemplate
from .protocols import Data
from ..bionc_numpy.natural_segment import NaturalSegment


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

    def experimental_Q(self, data: Data, kinematic_chain: BiomechanicalModel) -> SegmentNaturalCoordinates:
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

        Returns
        -------
        SegmentNaturalCoordinates
        The Segment Natural Coordinates Q (12 x n_frames)
        """
        from ..bionc_numpy import SegmentNaturalCoordinates

        self.Q = SegmentNaturalCoordinates.from_components(
            u=self.u_axis.to_axis(data, kinematic_chain).axis()[:3, :],
            rp=self.proximal_point.to_marker(data, kinematic_chain).position[:3, :],
            rd=self.distal_point.to_marker(data, kinematic_chain).position[:3, :],
            w=self.w_axis.to_axis(data, kinematic_chain).axis()[:3, :],
        )
        return self.Q

    def update(self) -> NaturalSegment:
        """
        Collapse the generic SegmentCoordinateSystem to an actual SegmentCoordinateSystemReal with value
        based on the model and the data

        Parameters
        ----------
        Q: SegmentNaturalCoordinates
            The experimental data of Q coordinates

        Returns
        -------
        NaturalSegment
        The collapsed SegmentCoordinateSystemReal
        """

        return NaturalSegment.from_experimental_Q(self.Q)
