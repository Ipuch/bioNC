from typing import Callable

import numpy as np

from .biomechanical_model import BiomechanicalModel
from ..model_creation.protocols import Data
from ..utils.interpolation_matrix import interpolate_natural_vector


# todo: need a list of markers MarkerList


class NaturalMarker:
    def __init__(
        self,
        name: str,
        parent_name: str,
        position: tuple[int | float, int | float, int | float] | np.ndarray = None,
        interpolation_matrix: np.ndarray = None,
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
        interpolation_matrix
            The interpolation matrix to use for the marker
        is_technical
            If the marker should be flagged as a technical marker
        is_anatomical
            If the marker should be flagged as an anatomical marker
        """
        self.name = name
        self.parent_name = parent_name

        if position is None and interpolation_matrix is None:
            raise ValueError("Either a position or an interpolation matrix must be provided")
        elif position is not None and interpolation_matrix is not None:
            self.position = position if isinstance(position, np.ndarray) else np.array(position)
            self.interpolation_matrix = interpolate_natural_vector(self.position)
        elif position is None and interpolation_matrix is not None:
            self.interpolation_matrix = interpolation_matrix
            # todo compute the position from the interpolation matrix
        else:
            raise ValueError("position and interpolation matrix cannot both be provided")

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
        natural_segment: "NaturalSegment" = None,
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
        natural_segment
            The natural segment the marker is attached to
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

        p[3, :] = 1  # Do not trust user and make sure the last value is a perfect one
        raise NotImplementedError("This is not implemented yet, todo")
        # projected_p = (natural_segment.transpose if natural_segment is not None else np.identity(4)) @ p
        # todo: add a marker in a natural segment with interpolation matrix etc...
        # this has to be done with the developpement of natural segments
        # natural_segment.to_non_orthogonal_ccordinate_system(p)
        # last question: does it has to be the mean position in the local coordinate system ?
        if np.isnan(projected_p).all():
            raise RuntimeError(f"All the values for {function} returned nan which is not permitted")
        return cls(
            name,
            parent_name,
            projected_p,
            is_technical=is_technical,
            is_anatomical=is_anatomical,
        )

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
