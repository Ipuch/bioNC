# The actual model to inherit from
from .biomechanical_model import BiomechanicalModel

# Some classes to define the BiomechanicalModel
from .natural_axis import Axis
from .natural_marker import SegmentMarker, Marker
from .segment import Segment
from .natural_segment import NaturalSegment
from .inertia_parameters import InertiaParameters
