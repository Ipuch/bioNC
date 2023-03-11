from typing import Callable

import numpy as np
from casadi import MX, vertcat, horzcat

from .biomechanical_model import BiomechanicalModel
from ..model_creation.protocols import Data

from ..protocols.natural_coordinates import SegmentNaturalCoordinates
from ..protocols.natural_markers import AbstractNaturalMarker, AbstractSegmentNaturalVector
from .natural_vector import NaturalVector

# todo: need a list of markers MarkerList


class NaturalMarker(AbstractNaturalMarker):
    """
    Class used to create a segment markers for the natural segments

    Methods
    -------
    from_data()
        Creates a segment marker from the data
    constraint()
        Computes the constraint for the marker given the segment natural coordinates and experimental marker location

    Attributes
    ----------
    name: str
        The name of the marker
    parent_name: str
        The name of the parent segment on which the marker is attached
    position: np.ndarray
        The 3d position of the marker in the non orthogonal segment coordinate system
    is_technical: bool
        If the marker should be flagged as a technical marker
    is_anatomical: bool
        If the marker should be flagged as an anatomical landmark
    """

    def __init__(
        self,
        name: str,
        parent_name: str,
        position: tuple[int | float, int | float, int | float] | np.ndarray | MX | NaturalVector = None,
        is_technical: bool = True,
        is_anatomical: bool = False,
    ):
        """
        Parameters
        ----------
        name
            The name of the new marker
        parent_name
            The name of the parent the marker is attached to
        position
            The 3d position of the marker in the non orthogonal segment coordinate system
        is_technical
            If the marker should be flagged as a technical marker
        is_anatomical
            If the marker should be flagged as an anatomical marker
        """
        self.name = name
        self.parent_name = parent_name

        if not isinstance(position, NaturalVector):
            position = NaturalVector(position)

        self.position = position
        self.interpolation_matrix = position.interpolate()

        self.is_technical = is_technical
        self.is_anatomical = is_anatomical

        self.is_technical = is_technical
        self.is_anatomical = is_anatomical

    @classmethod
    def from_data(
        cls,
        data: Data,
        name: str,
        function: Callable,
        parent_name: str,
        kinematic_chain: BiomechanicalModel,
        Q_xp: SegmentNaturalCoordinates = None,
        is_technical: bool = True,
        is_anatomical: bool = False,
    ):
        """
        This is a constructor for the MarkerReal class. It evaluates the function that defines the marker to get an
        actual position

        Parameters
        ----------
        data
            The data to pick the data from
        name
            The name of the new marker
        function
            The function (f(m) -> np.ndarray, where m is a dict of markers (XYZ1 x time)) that defines the marker
        parent_name
            The name of the parent the marker is attached to
        kinematic_chain
            The model as it is constructed at that particular time. It is useful if some values must be obtained from
            previously computed values
        Q_xp: SegmentNaturalCoordinates
            The segment natural coordinates identified from data
        is_technical
            If the marker should be flagged as a technical marker
        is_anatomical
            If the marker should be flagged as an anatomical marker
        """

        # Get the position of the markers and do some sanity checks
        p: np.ndarray = function(data.values, kinematic_chain)
        if not isinstance(p, np.ndarray):
            raise RuntimeError(f"The function {function} must return a np.ndarray of dimension 4xT (XYZ1 x time)")
        if len(p.shape) == 1:
            p = p[:, np.newaxis]

        if len(p.shape) != 2 or p.shape[0] != 4:
            raise RuntimeError(f"The function {function} must return a np.ndarray of dimension 4xT (XYZ1 x time)")

        natural_positions = Q_xp.to_natural_vector(vector=p[:3, :])
        # mean
        natural_position = NaturalVector(natural_positions.mean(axis=1))

        if np.isnan(natural_position).all():
            raise RuntimeError(f"All the values for {function} returned nan which is not permitted")
        return cls(
            name,
            parent_name,
            position=MX(natural_position),
            is_technical=is_technical,
            is_anatomical=is_anatomical,
        )

    def constraint(self, marker_location: np.ndarray, Qi: SegmentNaturalCoordinates) -> MX:
        """
        This function computes the constraint for the marker

        Parameters
        ----------
        marker_location: np.ndarray
            The location of the marker in the global/inertial coordinate system
        Qi
            The segment natural coordinates

        Returns
        -------
        The constraint for the marker
        """
        if marker_location.shape[0] != 3:
            raise ValueError("The marker location must be a 3d vector")
        if marker_location.shape.__len__() > 1:
            if marker_location.shape[1] != 1:
                raise ValueError("The marker location must be a 3d vector with only one column")

        return marker_location - self.interpolation_matrix @ Qi.vector

    def position_in_global(self, Qi: SegmentNaturalCoordinates) -> MX:
        """
        This function computes the position of the marker in the global coordinate system

        Parameters
        ----------
        Qi : SegmentNaturalCoordinates
            The segment natural coordinates

        Returns
        -------
        The position of the marker in the global coordinate system
        """
        return self.interpolation_matrix @ Qi

    def __str__(self):
        # Define the print function, so it automatically formats things in the file properly
        out_string = f"marker {self.name}\n"
        out_string += f"\tparent {self.parent_name}\n"

        p = np.array(self.position)
        p = p if len(p.shape) == 1 else np.nanmean(p, axis=1)
        p = p if len(p.shape) == 1 else np.nanmean(p, axis=0)
        out_string += f"\tposition {p[0]:0.4f} {p[1]:0.4f} {p[2]:0.4f}\n"
        out_string += f"\ttechnical {1 if self.is_technical else 0}\n"
        out_string += f"\tanatomical {1 if self.is_anatomical else 0}\n"
        out_string += "endmarker\n"
        return out_string

    def __add__(self, other: np.ndarray | tuple):
        if isinstance(other, tuple):
            other = np.array(other)

        if isinstance(other, np.ndarray):
            return NaturalMarker(name=self.name, parent_name=self.parent_name, position=self.position + other)
        elif isinstance(other, NaturalMarker):
            return NaturalMarker(name=self.name, parent_name=self.parent_name, position=self.position + other.position)
        else:
            raise NotImplementedError(f"The addition for {type(other)} is not implemented")

    def __sub__(self, other):
        if isinstance(other, tuple):
            other = np.array(other)

        if isinstance(other, np.ndarray):
            return NaturalMarker(name=self.name, parent_name=self.parent_name, position=self.position - other)
        elif isinstance(other, NaturalMarker):
            return NaturalMarker(name=self.name, parent_name=self.parent_name, position=self.position - other.position)
        else:
            raise NotImplementedError(f"The subtraction for {type(other)} is not implemented")


class Marker:
    def __init__(
        self,
        name: str,
        position: tuple[int | float, int | float, int | float] | np.ndarray | MX = None,
        is_technical: bool = True,
        is_anatomical: bool = False,
    ):
        """
        Parameters
        ----------
        name
            The name of the new marker
        position
            The 3d position of the marker in the orhtogonal coordinate system (XYZ1 x time)) that defines the marker
        is_technical
            If the marker should be flagged as a technical marker
        is_anatomical
            If the marker should be flagged as an anatomical marker
        """
        self.name = name

        if position is None:
            raise ValueError("A position must be provided")

        if position.shape[0] != 4:
            if position.shape[0] == 3:  # If the position is a 3d vector, add a 1 at the bottom
                position = vertcat((position, MX.ones((1, position.shape[1]))))
            else:
                raise ValueError("The position must be (XYZ x time) or (XYZ1 x time)")

        self.position = position
        self.is_technical = is_technical
        self.is_anatomical = is_anatomical

    @classmethod
    def from_data(
        cls,
        data: Data,
        name: str,
        function: Callable,
        kinematic_chain: BiomechanicalModel,
        is_technical: bool = True,
        is_anatomical: bool = False,
    ):
        """
        This is a constructor for the MarkerReal class. It evaluates the function that defines the marker to get an
        actual position

        Parameters
        ----------
        data
            The data to pick the data from
        name
            The name of the new marker
        function
            The function (f(m) -> np.ndarray, where m is a dict of markers (XYZ1 x time)) that defines the marker
        kinematic_chain
            The model as it is constructed at that particular time. It is useful if some values must be obtained from
            previously computed values
        is_technical
            If the marker should be flagged as a technical marker
        is_anatomical
            If the marker should be flagged as an anatomical marker
        """

        # Get the position of the markers and do some sanity checks
        position: np.ndarray = function(data.values, kinematic_chain)
        if not isinstance(position, np.ndarray):
            raise RuntimeError(f"The function {function} must return a np.ndarray of dimension 4xT (XYZ1 x time)")
        if len(position.shape) == 1:
            position = position[:, np.newaxis]

        if len(position.shape) != 2 or position.shape[0] != 4:
            raise RuntimeError(f"The function {function} must return a np.ndarray of dimension 4xT (XYZ1 x time)")

        position[3, :] = 1  # Do not trust user and make sure the last value is a perfect one

        if np.isnan(position).all():
            raise RuntimeError(f"All the values for {function} returned nan which is not permitted")

        return cls(
            name,
            MX(position),
            is_technical=is_technical,
            is_anatomical=is_anatomical,
        )

    def __str__(self):
        # Define the print function, so it automatically formats things in the file properly
        out_string = f"marker {self.name}\n"

        p = np.array(self.position)
        p = p if len(p.shape) == 1 else np.nanmean(p, axis=1)
        p = p if len(p.shape) == 1 else np.nanmean(p, axis=0)
        out_string += f"\tposition {p[0]:0.4f} {p[1]:0.4f} {p[2]:0.4f}\n"
        out_string += f"\ttechnical {1 if self.is_technical else 0}\n"
        out_string += f"\tanatomical {1 if self.is_anatomical else 0}\n"
        out_string += "endmarker\n"
        return out_string

    def __add__(self, other: np.ndarray | tuple):
        if isinstance(other, tuple):
            other = np.array(other)

        if isinstance(other, np.ndarray):
            return Marker(name=self.name, position=self.position + other)
        elif isinstance(other, Marker):
            return Marker(name=self.name, position=self.position + other.position)
        else:
            raise NotImplementedError(f"The addition for {type(other)} is not implemented")

    def __sub__(self, other):
        if isinstance(other, tuple):
            other = np.array(other)

        if isinstance(other, np.ndarray):
            return Marker(name=self.name, position=self.position - other)
        elif isinstance(other, Marker):
            return Marker(name=self.name, position=self.position - other.position)
        else:
            raise NotImplementedError(f"The subtraction for {type(other)} is not implemented")


class SegmentNaturalVector(AbstractSegmentNaturalVector):
    """
    Class used to create a segment vector for the natural segments

    Methods
    -------


    Attributes
    ----------
    name: str
        The name of the marker
    parent_name: str
        The name of the parent segment on which the marker is attached
    position: np.ndarray
        The 3d position of the vector in the non orthogonal segment coordinate system
    interpolation_matrix: np.ndarray
        The interpolation matrix that is used to interpolate the vector in the global coordinate system
    """

    def __init__(
        self,
        name: str,
        parent_name: str,
        direction: tuple[int | float, int | float, int | float] | np.ndarray | NaturalVector = None,
    ):
        """
        Parameters
        ----------
        name
            The name of the new marker
        parent_name
            The name of the parent the marker is attached to
        direction
            The 3d position of the vector in the non orthogonal segment coordinate system
        """
        self.name = name
        self.parent_name = parent_name

        if not isinstance(direction, NaturalVector):
            direction = NaturalVector(direction)

        self.position = direction
        self.interpolation_matrix = direction.interpolate().rot

    def position_in_global(self, Qi: SegmentNaturalCoordinates) -> MX:
        """
        This function computes the position of the vector in the global coordinate system

        Parameters
        ----------
        Qi : SegmentNaturalCoordinates
            The segment natural coordinates

        Returns
        -------
        The position of the vector in the global coordinate system
        """
        return self.interpolation_matrix @ Qi
