# The actual model to inherit from
from .biomechanical_model import BiomechanicalModelTemplate

# Some classes to define the BiomechanicalModel
from .axis import AxisTemplate
from .marker import MarkerTemplate
from .mesh import Mesh
from .protocols import Data, GenericDynamicModel
from .segment import Segment
from .segment_coordinate_system import SegmentCoordinateSystem
from .inertia_parameters import InertiaParametersTemplate

# Add also the "Real" version of classes to create models from values
from .biomechanical_model_real import BiomechanicalModelReal
from .axis_real import NaturalAxis
from .marker_real import MarkerReal
from .mesh_real import MeshReal
from .segment_real import Segment
from .segment_coordinate_system_real import SegmentCoordinateSystemReal
from .inertia_parameters_real import InertiaParametersReal

# The accepted data formating
from .c3d_data import C3dData
