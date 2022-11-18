from enum import Enum
from .joint import Joint


class JointType(Enum):
    """
    This class represents the different types of joints
    """

    WELD = "not implemented yet"
    REVOLUTE = Joint.Hinge
    PRISMATIC = "not implemented yet"
    CARDAN = Joint.Universal
    SPHERICAL = Joint.Spherical
    PLANAR = "planar"
