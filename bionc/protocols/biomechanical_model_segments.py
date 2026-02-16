import numpy as np
from abc import ABC, abstractmethod
from casadi import MX
from typing import Any

from bionc.protocols.natural_coordinates import NaturalCoordinates
from bionc.protocols.natural_velocities import NaturalVelocities


class GenericBiomechanicalModelSegments(ABC):
    """
    This is an abstract base class that provides the basic structure and methods for all biomechanical models.
    It contains the segments and the joints of the model. The implemented methods are not specific to numpy or casadi.

    Attributes
    ----------
    segments : dict
        A dictionary containing the segments of the model. The keys are the names of the segments and the values are the corresponding segment objects.

    Methods
    -------
    __getitem__(self, name: str)
        Returns the segment with the given name.
    __setitem__(self, name: str, segment: Any)
        Adds a segment to the model.
    keys(self)
        Returns the keys of the segments dictionary.
    values(self)
        Returns the values of the segments dictionary.
    items(self)
        Returns the items of the segments dictionary.
    has_ground_segment(self) -> bool
        Returns true if the model has a ground segment.
    get_ground_segment(self)
        Returns the ground segment of the model.
    segments_no_ground(self)
        Returns the dictionary of all the segments except the ground segment.
    children(self, segment: str | int) -> list[int]
        Returns the children of the given segment.
    parents(self, segment: str | int) -> list[int]
        Returns the parents of the given segment.
    nb_segments(self) -> int
        Returns the number of segments in the model.
    segment_names(self) -> list[str]
        Returns the names of the segments in the model.
    nb_rigid_body_constraints(self) -> int
        Returns the number of rigid body constraints in the model.
    nb_Q(self) -> int
        Returns the number of generalized coordinates in the model.
    nb_Qdot(self) -> int
        Returns the number of generalized velocities in the model.
    nb_Qddot(self) -> int
        Returns the number of generalized accelerations in the model.
    segment_from_index(self, index: int)
        Returns the segment with the given index.
    rigid_body_constraints(self, Q: NaturalCoordinates)
        Returns the rigid body constraints of all segments.
    rigid_body_constraints_derivative(self, Q: NaturalCoordinates, Qdot: NaturalCoordinates)
        Returns the derivative of the rigid body constraints.
    rigid_body_constraints_jacobian(self, Q: NaturalCoordinates)
        Returns the rigid body constraints of all segments.
    kinetic_energy(self, Qdot: NaturalVelocities) -> Union[np.ndarray, MX]
        Returns the kinetic energy of the system.
    potential_energy(self, Q: NaturalCoordinates) -> Union[np.ndarray, MX]
        Returns the potential energy of the system.
    lagrangian(self, Q: NaturalCoordinates, Qdot: NaturalVelocities)
        Returns the lagrangian of the system.
    energy(self, Q: NaturalCoordinates, Qdot: NaturalVelocities)
        Returns the total energy of the model.
    """

    def __init__(
        self,
        segments: dict[str:Any, ...] = None,
    ):
        from .natural_segment import AbstractNaturalSegment  # Imported here to prevent from circular imports

        self.segments: dict[str:AbstractNaturalSegment, ...] = {} if segments is None else segments
        # From Pythom 3.7 the insertion order in a dict is preserved. This is important because when writing a new
        # the order of the segment matters

    def __getitem__(self, name: str):
        """
        This function returns the segment with the given name

        Parameters
        ----------
        name : str
            Name of the segment

        Returns
        -------
        NaturalSegment
            The segment with the given name
        """
        return self.get_ground_segment() if name == "ground" else self.segments[name]

    def get_ground_segment(self):
        """
        This function returns the ground segment of the model

        Returns
        -------
        NaturalSegment
            The ground segment of the model
        """
        ground_segment = next((segment for segment in self.segments.values() if segment.is_ground), None)
        if ground_segment is None:
            raise ValueError("No ground segment found")
        return ground_segment

    def __setitem__(self, name: str, segment: Any):
        """
        This function adds a segment to the model

        Parameters
        ----------
        name : str
            Name of the segment
        segment : NaturalSegment
            The segment to add
        """
        if segment.name == name:  # Make sure the name of the segment fits the internal one
            segment.set_index(len(self.segments))
            self.segments[name] = segment
        else:
            raise ValueError("The name of the segment does not match the name of the segment")

    def keys(self):
        return self.segments.keys()

    def values(self):
        return self.segments.values()

    def items(self):
        return self.segments.items()

    def __len__(self):
        return len(self.segments)

    @property
    def has_ground_segment(self) -> bool:
        """
        This function returns true if the model has a ground segment

        Returns
        -------
        bool
            True if the model has a ground segment
        """
        return any(segment._is_ground for segment in self.segments.values())

    @property
    def segments_no_ground(self):
        """
        This function returns the dictionary of all the segments except the ground segment

        Returns
        -------
        dict[str: NaturalSegment, ...]
            The dictionary of all the segments except the ground segment
        """
        return {name: segment for name, segment in self.segments.items() if not segment._is_ground}

    @property
    def nb_segments(self) -> int:
        """
        This function returns the number of segments in the model
        """
        return len(self.segments) - 1 if self.has_ground_segment else len(self.segments)

    @property
    def segment_names(self) -> list[str]:
        """
        This function returns the names of the segments in the model
        """
        return list(self.segments.keys())

    @property
    def nb_rigid_body_constraints(self) -> int:
        """
        This function returns the number of rigid body constraints in the model
        """
        return 6 * self.nb_segments

    @property
    def nb_Q(self) -> int:
        """
        This function returns the number of generalized coordinates in the model
        """
        return 12 * self.nb_segments

    @property
    def nb_Qdot(self) -> int:
        """
        This function returns the number of generalized velocities in the model
        """
        return 12 * self.nb_segments

    @property
    def nb_Qddot(self) -> int:
        """
        This function returns the number of generalized accelerations in the model
        """
        return 12 * self.nb_segments

    def segment_from_index(self, index: int):
        """
        This function returns the segment with the given index

        Parameters
        ----------
        index : int
            The index of the segment

        Returns
        -------
        Segment
            The segment with the given index
        """
        for segment in self.segments.values():
            if segment.index == index:
                return segment
        raise ValueError(f"The segment index does not exist, the model has only {self.nb_segments} segments")

    @property
    def normalized_coordinates(self) -> tuple[tuple[int, ...]]:
        """Returns the indices of the normalized coordinates for all ui, and wi."""
        idx = []
        for i in range(self.nb_segments):
            # create list from i* 12 to (i+1) * 12
            segment_idx = [i for i in range(i * 12, (i + 1) * 12)]
            idx.append(segment_idx[0:3])
            idx.append(segment_idx[9:12])

        return idx

    @abstractmethod
    def rigid_body_constraints(self, Q: NaturalCoordinates):
        """
        This function returns the rigid body constraints of all segments, denoted Phi_r
        as a function of the natural coordinates Q.

        Parameters
        ----------
        Q : NaturalCoordinates
            The natural coordinates of the segment [12 * nb_segments, 1]

        Returns
        -------
        np.ndarray | MX
            Rigid body constraints of the segment [6 * nb_segments, 1]
        """

    @abstractmethod
    def rigid_body_constraints_derivative(self, Q: NaturalCoordinates, Qdot: NaturalCoordinates):
        """
        This function returns the derivative of the rigid body constraints denoted Phi_r_dot

        Parameters
        ----------
        Q : NaturalCoordinates
            The natural coordinates of the model
        Qdot : NaturalVelocities
            The natural velocities of the model

        Returns
        -------
            Derivative of the rigid body constraints
        """

    @abstractmethod
    def rigid_body_constraints_jacobian(self, Q: NaturalCoordinates):
        """
        This function returns the rigid body constraints of all segments, denoted K_r
        as a function of the natural coordinates Q.

        Parameters
        ----------
        Q : NaturalCoordinates
            The natural coordinates of the segment [12 * nb_segments, 1]

        Returns
        -------
        np.ndarray | MX
            Rigid body constraints of the segment [6 * nb_segments, nbQ]
        """


