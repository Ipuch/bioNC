# from bionc.bionc_numpy.interpolation_matrix import interpolate_natural_vector, to_natural_vector
from .utils.vnop_array import vnop_array

from .model_creation import (
    AxisTemplate,
    MarkerTemplate,
    SegmentTemplate,
    NaturalSegmentTemplate,
    InertiaParametersTemplate,
    BiomechanicalModelTemplate,
    C3dData,
    Data,
    GenericDynamicModel,
)
from .bionc_numpy import (
    Axis,
    NaturalMarker,
    Marker,
    NaturalSegment,
    InertiaParameters,
    BiomechanicalModel,
    JointType,
)

from .protocols import natural_coordinates
from bionc import bionc_casadi
from bionc import bionc_numpy

from .protocols.natural_coordinates import SegmentNaturalCoordinates, NaturalCoordinates
from .protocols.natural_velocities import SegmentNaturalVelocities, NaturalVelocities
from .protocols.natural_accelerations import SegmentNaturalAccelerations, NaturalAccelerations
from .protocols.homogenous_transform import HomogeneousTransform

from casadi.casadi import MX as MX_type
from numpy import ndarray

# global variable to store the type of the math interface
casadi_type = MX_type
numpy_type = ndarray
