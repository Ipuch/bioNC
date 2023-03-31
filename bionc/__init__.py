from .model_creation import (
    AxisTemplate,
    AxisFunctionTemplate,
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
    ExternalForceList, ExternalForce,
)

from .protocols import natural_coordinates
from bionc import bionc_casadi
from bionc import bionc_numpy

from .bionc_numpy.natural_coordinates import SegmentNaturalCoordinates, NaturalCoordinates
from .bionc_numpy.natural_velocities import SegmentNaturalVelocities, NaturalVelocities
from .bionc_numpy.natural_accelerations import SegmentNaturalAccelerations, NaturalAccelerations
from .bionc_numpy.homogenous_transform import HomogeneousTransform

from .utils.enums import NaturalAxis, CartesianAxis

from casadi.casadi import MX as MX_type
from numpy import ndarray

# global variable to store the type of the math interface
casadi_type = MX_type
numpy_type = ndarray

from .vizualization import Viz
from .bionc_numpy import InverseKinematics
