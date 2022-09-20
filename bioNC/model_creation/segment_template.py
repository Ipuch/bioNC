from .inertia_parameters import InertiaParametersTemplate
from .marker import MarkerTemplate
from .mesh import Mesh
from .segment_coordinate_system import SegmentCoordinateSystem


class SegmentTemplate:
    def __init__(
        self,
        name: str = None,
        parent_name: str = "",
        translations: str = "",
        rotations: str = "",
        segment_coordinate_system: SegmentCoordinateSystem = None,
        inertia_parameters: InertiaParametersTemplate = None,
        mesh: Mesh = None,
    ):
        """
        Create a new generic segment.

        Parameters
        ----------
        name
            The name of the segment
        parent_name
            The name of the segment the current segment is attached to
        translations
            The sequence of translation
        rotations
            The sequence of rotation
        inertia_parameters
            The inertia parameters of the segment
        mesh
            The mesh points of the segment
        """

        self.name = name
        self.parent_name = parent_name
        self.translations = translations
        self.rotations = rotations
        self.markers = []
        self.segment_coordinate_system = segment_coordinate_system
        self.inertia_parameters = inertia_parameters
        self.mesh = mesh

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
