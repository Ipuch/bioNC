# The actual model to inherit from
from .biomechanical_model import BiomechanicalModel

# Some classes to define the BiomechanicalModel
from .natural_axis import NaturalAxis
from .marker import Marker
from .segment import Segment
from .segment_coordinate_system import NaturalSegmentCoordinateSystem
from .inertia_parameters import InertiaParameters
