from .enums import NaturalAxis, TransformationMatrixType


class TransformationMatrixUtil:
    """
    A utility class to get the corresponding TransformationMatrixType from a plane and an axis to keep
    to build the orthogonal segment coordinate system.

    It requires a plane (tuple of NaturalAxis) and an axis to keep (NaturalAxis).
    The two axes of the plane are used to perform a cross product to get the third axis.
    The kept axis is equivalent in the orthogonal segment coordinate system and the natural coordinate system.

    For example, if the plane is (NaturalAxis.U, NaturalAxis.V) and the axis to keep is NaturalAxis.U,
    the corresponding TransformationMatrixType is Buv.

    Methods
    -------
    to_enum()
        Get the corresponding TransformationMatrixType from the plane and the axis to keep.

    """

    def __init__(self, plane: tuple[NaturalAxis, NaturalAxis], axis_to_keep: NaturalAxis):
        check_plane(plane)
        check_axis_to_keep(axis_to_keep)

        self.plane = plane
        self.axis_to_keep = axis_to_keep
        """
        Set the plane and the axis to keep.
        
        Parameters
        ----------
        plane : tuple[NaturalAxis, NaturalAxis]
            The plane to use to build the orthogonal segment coordinate system.
        axis_to_keep : NaturalAxis
            The axis to keep in the orthogonal segment coordinate system.
        """

    def to_enum(self) -> TransformationMatrixType:
        transformation_matrix_map = {
            (NaturalAxis.U, NaturalAxis.V, NaturalAxis.U): TransformationMatrixType.Buv,
            (NaturalAxis.U, NaturalAxis.V, NaturalAxis.V): TransformationMatrixType.Bvu,
            (NaturalAxis.W, NaturalAxis.U, NaturalAxis.U): TransformationMatrixType.Buw,
            (NaturalAxis.W, NaturalAxis.U, NaturalAxis.W): TransformationMatrixType.Bwu,
            (NaturalAxis.V, NaturalAxis.W, NaturalAxis.V): TransformationMatrixType.Bvw,
            (NaturalAxis.V, NaturalAxis.W, NaturalAxis.W): TransformationMatrixType.Bwv,
        }

        key = (self.plane[0], self.plane[1], self.axis_to_keep)
        return transformation_matrix_map.get(key)


def check_length(plane: tuple[NaturalAxis, NaturalAxis]):
    if len(plane) != 2:
        raise ValueError(f"Plane must be a tuple[NaturalAxis] of length 2, got {len(plane)}")


def check_type(plane: tuple[NaturalAxis, NaturalAxis]):
    if not all(isinstance(axis, NaturalAxis) for axis in plane):
        raise ValueError(f"Plane must be a tuple of NaturalAxis, got {plane}")


def check_different_axis(plane: tuple[NaturalAxis, NaturalAxis]):
    if plane[0] == plane[1]:
        raise ValueError(f"Plane must be a tuple of different axis, got {plane}")


def check_indirect_frame(plane: tuple[NaturalAxis, NaturalAxis]):
    if (
        (plane[0] == NaturalAxis.V and plane[1] == NaturalAxis.U)
        or (plane[0] == NaturalAxis.U and plane[1] == NaturalAxis.W)
        or (plane[0] == NaturalAxis.W and plane[1] == NaturalAxis.V)
    ):
        raise ValueError(f"Invert Axis in plane, because it would lead to an indirect frame, got {plane}")


def check_plane(plane: tuple[NaturalAxis, NaturalAxis]):
    """Check if the plane is valid"""
    check_length(plane)
    check_type(plane)
    check_different_axis(plane)
    check_indirect_frame(plane)


def check_axis_to_keep(axis_to_keep: NaturalAxis):
    """Check if the axis to keep is valid"""
    if not isinstance(axis_to_keep, NaturalAxis):
        raise ValueError(f"Axis to keep must be of type NaturalAxis, got {axis_to_keep}")
