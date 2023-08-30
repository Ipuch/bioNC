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
        if NaturalAxis.U in self.plane and NaturalAxis.V in self.plane:
            if self.axis_to_keep == NaturalAxis.U:
                return TransformationMatrixType.Buv
            elif self.axis_to_keep == NaturalAxis.V:
                return TransformationMatrixType.Bvu

        elif NaturalAxis.U in self.plane and NaturalAxis.W in self.plane:
            if self.axis_to_keep == NaturalAxis.U:
                return TransformationMatrixType.Buw
            elif self.axis_to_keep == NaturalAxis.W:
                return TransformationMatrixType.Bwu

        elif NaturalAxis.V in self.plane and NaturalAxis.W in self.plane:
            if self.axis_to_keep == NaturalAxis.V:
                return TransformationMatrixType.Bvw
            elif self.axis_to_keep == NaturalAxis.W:
                return TransformationMatrixType.Bwv


def check_plane(plane: tuple[NaturalAxis, NaturalAxis]):
    """Check if the plane is valid"""
    if len(plane) != 2:
        raise ValueError(f"Plane must be a tuple of length 2, got {len(plane)}")
    if not all(isinstance(axis, NaturalAxis) for axis in plane):
        raise ValueError(f"Plane must be a tuple of NaturalAxis, got {plane}")
    if plane[0] == plane[1]:
        raise ValueError(f"Plane must be a tuple of different axis, got {plane}")
    if (
        (plane[0] == NaturalAxis.V and plane[1] == NaturalAxis.U)
        or (plane[0] == NaturalAxis.U and plane[1] == NaturalAxis.W)
        or (plane[0] == NaturalAxis.W and plane[1] == NaturalAxis.V)
    ):
        raise ValueError(f"Invert Axis in plane, because it would lead to an indirect frame, got {plane}")


def check_axis_to_keep(axis_to_keep: NaturalAxis):
    """Check if the axis to keep is valid"""
    if not isinstance(axis_to_keep, NaturalAxis):
        raise ValueError(f"Axis to keep must be of type NaturalAxis, got {axis_to_keep}")
