from .natural_coordinates import SegmentNaturalCoordinates, NaturalCoordinates
from .natural_velocities import SegmentNaturalVelocities, NaturalVelocities
from .natural_accelerations import SegmentNaturalAccelerations, NaturalAccelerations
from .homogenous_transform import HomogeneousTransform
from .natural_segment import NaturalSegment

# The actual model to inherit from
from .biomechanical_model import BiomechanicalModel

# Some classes to define the BiomechanicalModel
from .natural_axis import Axis
from .natural_marker import SegmentMarker, Marker
from .natural_segment import NaturalSegment
from .inertia_parameters import InertiaParameters
from .interpolation_matrix import interpolate_natural_vector, to_natural_vector
from .joint import Joint
