from enum import Enum


class NaturalAxis(Enum):
    """Natural axis"""

    U = "U"  # eventually corresponds to X
    V = "V"  # eventually corresponds to Y
    W = "W"  # eventually corresponds to Z


class CartesianAxis(Enum):
    X = "X"
    Y = "Y"
    Z = "Z"


class EulerSequence(Enum):
    """Euler intrinsic sequence of rotations"""

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


class TransformationMatrixType(Enum):
    """Transformation matrix denoted B according to Raphael Dumas' work"""

    Buv = "Buv"  # implemented
    Bvu = "Bvu"  # implemented
    Bwu = "Bwu"  # implemented
    Buw = "Buw"  # not implemented
    Bvw = "Bvw"  # not implemented
    Bwv = "Bwv"  # not implemented


def transformation_matrix_str_to_enum(matrix_type: str) -> TransformationMatrixType:

    if matrix_type == "Buv":
        return TransformationMatrixType.Buv
    elif matrix_type == "Bvu":
        return TransformationMatrixType.Bvu
    elif matrix_type == "Bwu":
        return TransformationMatrixType.Bwu
    elif matrix_type == "Buw":
        return TransformationMatrixType.Buw
    elif matrix_type == "Bvw":
        return TransformationMatrixType.Bvw
    elif matrix_type == "Bwv":
        return TransformationMatrixType.Bwv
    else:
        raise ValueError(f"Unknown matrix type: {matrix_type}")
