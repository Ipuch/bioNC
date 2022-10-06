# The actual model to inherit from
from .biomechanical_model_template import BiomechanicalModelTemplate

# Some classes to define the BiomechanicalModel
from .natural_axis_template import AxisTemplate
from .marker_template import MarkerTemplate
from .protocols import Data, GenericDynamicModel
from .segment_template import SegmentTemplate
from .natural_segment_template import NaturalSegmentTemplate
from .inertia_parameters_template import InertiaParametersTemplate

# The accepted data formating
from .c3d_data import C3dData
