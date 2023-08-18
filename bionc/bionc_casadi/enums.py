from enum import Enum
from .joint import Joint, GroundJoint


class JointType(Enum):
    """
    This class represents the different types of joints
    """

    # WELD = "not implemented yet"
    GROUND_FREE = GroundJoint.Free
    GROUND_WELD = GroundJoint.Weld
    GROUND_REVOLUTE = GroundJoint.Hinge
    GROUND_SPHERICAL = GroundJoint.Spherical
    CONSTANT_LENGTH = Joint.ConstantLength
    REVOLUTE = Joint.Hinge
    # PRISMATIC = "not implemented yet"
    UNIVERSAL = Joint.Universal
    SPHERICAL = Joint.Spherical
    SPHERE_ON_PLANE = Joint.SphereOnPlane

    # PLANAR = "planar"
