from enum import Enum


class NaturalAxis(Enum):
    U = "U"
    V = "V"
    W = "W"


class CartesianAxis(Enum):
    X = "X"
    Y = "Y"
    Z = "Z"


class EulerSequence(Enum):
    XYX = "xyx"
    XZX = "xzx"
    XYZ = "xyz"
    XZY = "xzy"
    YXY = "yxy"
    YZX = "yzx"
    YXZ = "yxz"
    YZY = "yzy"
    ZXZ = "zxz"
    ZXY = "zxy"
    ZYZ = "zyz"


class ISBEulerSequence:
    """returns the corresponding enums
    e.g. GLENO_HUMERAL returns EulerSequence.XYX
    """

    def __init__(self):
        pass

    @property
    def GLENO_HUMERAL(self) -> EulerSequence:
        return EulerSequence.YXY

    @property
    def SCAPULO_THORACIC(self) -> EulerSequence:
        return EulerSequence.YXZ

    @property
    def ACROMIO_CLAVICULAR(self) -> EulerSequence:
        return EulerSequence.YXZ

    @property
    def STERNO_CLAVICULAR(self) -> EulerSequence:
        return EulerSequence.YXZ

    @property
    def THORACO_HUMERAL(self) -> EulerSequence:
        return EulerSequence.YXY

    # todo: add the rest of the joints
