# from .utils.natural_coordinates import (
#     # SegmentNaturalCoordinates,
#     # NaturalCoordinates,
# )
from .utils.natural_velocities import SegmentNaturalVelocities, NaturalVelocities
from .utils.natural_accelerations import SegmentNaturalAccelerations, NaturalAccelerations
from .utils.homogenous_transform import HomogeneousTransform
from .utils.interpolation_matrix import interpolate_natural_vector, to_natural_vector
from .utils.natural_coordinates import vnop_array

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


from .math_interface import internal
from .math_interface import using_casadi as bionc_casadi
from .math_interface import using_numpy as bionc_numpy

# I don't know if it's useful to import the following yet
from .math_interface.internal import SegmentNaturalCoordinates, NaturalCoordinates

from casadi.casadi import MX as MX_type
from numpy import ndarray

# global variable to store the type of the math interface
casadi_type = MX_type
numpy_type = ndarray
