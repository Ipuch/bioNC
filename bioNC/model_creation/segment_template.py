from .inertia_parameters_template import InertiaParametersTemplate
from .marker_template import MarkerTemplate
from .segment_coordinate_system_template import NaturalSegmentCoordinateSystemTemplate


class SegmentTemplate:
    def __init__(
        self,
        name: str = None,
        segment_coordinate_system: NaturalSegmentCoordinateSystemTemplate = None,
        inertia_parameters: InertiaParametersTemplate = None,
    ):
        """
        Create a new generic segment.

        Parameters
        ----------
        name
            The name of the segment
        segment_coordinate_system
            The natural segment coordinate system
        inertia_parameters
            The inertia parameters of the segment
        """

        self.name = name
        self.markers = []
        self.segment_coordinate_system = segment_coordinate_system
        self.inertia_parameters = inertia_parameters

    def add_marker(self, marker: MarkerTemplate):
        """
        Add a new marker to the segment

        Parameters
        ----------
        marker
            The marker to add
        """
        if marker.parent_name is not None and marker.parent_name != self.name:
            raise ValueError(
                "The marker name should be the same as the 'key'. Alternatively, marker.name can be left undefined"
            )

        marker.parent_name = self.name
        self.markers.append(marker)
