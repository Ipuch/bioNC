from .utils.interpolation_matrix import interpolate_natural_vector, to_natural_vector
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
from .model_computations import (
    Axis,
    SegmentMarker,
    Marker,
    Segment,
    NaturalSegment,
    InertiaParameters,
    BiomechanicalModel,
)

from .math_interface.math_interface import (
    zeros,
    eye,
    array,
    symmetrize_upp,
    vertcat,
    horzcat,
)

from .math_interface.protocols import natural_coordinates
from .math_interface import using_casadi as bionc_casadi
from .math_interface import using_numpy as bionc_numpy

from bionc.math_interface.protocols.natural_coordinates import SegmentNaturalCoordinates, NaturalCoordinates
from bionc.math_interface.protocols.natural_velocities import SegmentNaturalVelocities, NaturalVelocities
from bionc.math_interface.protocols.natural_accelerations import SegmentNaturalAccelerations, NaturalAccelerations
from bionc.math_interface.protocols.homogenous_transform import HomogeneousTransform

from casadi.casadi import MX as MX_type
from numpy import ndarray

# global variable to store the type of the math interface
casadi_type = MX_type
numpy_type = ndarray
