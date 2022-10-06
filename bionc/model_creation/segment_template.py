from .inertia_parameters_template import InertiaParametersTemplate
from .marker_template import MarkerTemplate
from .natural_segment_template import NaturalSegmentTemplate


class SegmentTemplate:
    def __init__(
        self,
        name: str = None,
        natural_segment: NaturalSegmentTemplate = None,
        inertia_parameters: InertiaParametersTemplate = None,
    ):
        """
        Create a new generic segment.

        Parameters
        ----------
        name
            The name of the segment
        natural_segment
            The natural segment coordinate system
        inertia_parameters
            The inertia parameters of the segment
        """

        self.name = name
        self.markers = []
        self.natural_segment = natural_segment
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
