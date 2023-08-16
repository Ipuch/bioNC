from ..utils.enums import NaturalAxis


class TransformationMatrix:
    def __init__(self, matrix):
        self.matrix = matrix

    @classmethod
    def from_plane_and_axis_to_keep(cls, plane: tuple[NaturalAxis, NaturalAxis], axis_to_keep: NaturalAxis):
        """
        Create a transformation matrix from a plane and an axis to keep

        Parameters
        ----------
        plane: tuple[NaturalAxis, NaturalAxis]
            The plane to define the cross product of the orthogonal axis
        axis_to_keep:
            The axis to keep in the plane

        Returns
        -------

        """

        check_plane(plane)


def check_plane(plane: tuple[NaturalAxis, NaturalAxis]):
    """ Check if the plane is valid """
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


# write unit tests for check_plane
def test_check_plane():

    plane = (NaturalAxis.U, NaturalAxis.U)
    with pytest.raises(ValueError, match=f"Plane must be a tuple of different axis, got {plane}"):
        check_plane(plane)
    plane = (NaturalAxis.U)
    with pytest.raises(ValueError, match=f"Plane must be a tuple of length 2, got {len(plane)}"):
        check_plane(plane)
    plane = (NaturalAxis.U, 1)
    with pytest.raises(ValueError, match=f"Plane must be a tuple of NaturalAxis, got {plane}"):
        check_plane(plane)
    plane = (NaturalAxis.V, NaturalAxis.U)
    with pytest.raises(ValueError, match=f"Invert Axis in plane, because it would lead to an indirect frame, got {plane}"):
        check_plane(plane)
    plane = (NaturalAxis.W, NaturalAxis.U)
    with pytest.raises(ValueError, match=f"Invert Axis in plane, because it would lead to an indirect frame, got {plane}"):
        check_plane(plane)
    plane = (NaturalAxis.V, NaturalAxis.W)
    with pytest.raises(ValueError, match=f"Invert Axis in plane, because it would lead to an indirect frame, got {plane}"):
        check_plane(plane)


