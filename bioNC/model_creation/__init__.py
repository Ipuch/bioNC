# The actual model to inherit from
from .biomechanical_model_template import BiomechanicalModelTemplate

# Some classes to define the BiomechanicalModel
from .natural_axis_template import NaturalAxisTemplate
from .marker_template import MarkerTemplate
from .protocols import Data, GenericDynamicModel
from .segment_template import SegmentTemplate
from .segment_coordinate_system_template import NaturalSegmentCoordinateSystemTemplate
from .inertia_parameters_template import InertiaParametersTemplate

# The accepted data formating
from .c3d_data import C3dData
