from abc import ABC, abstractmethod

from .biomechanical_model_segments import GenericBiomechanicalModelSegments
from .natural_coordinates import NaturalCoordinates


class GenericBiomechanicalModelMuscles(ABC):
    """
    Abstract base class that holds the muscles of a biomechanical model and
    exposes muscle-related queries (lengths, moment arms). Implemented methods
    are not specific to numpy or casadi.

    Attributes
    ----------
    segments : GenericBiomechanicalModelSegments
        The segments of the model. Required to resolve segment indices when
        muscles attach to specific segments.
    muscles : dict
        Ordered dictionary of muscles keyed by muscle name.
    """

    def __init__(
        self,
        segments: GenericBiomechanicalModelSegments = None,
        muscles: dict = None,
    ):
        self.segments = segments
        self.muscles = {} if muscles is None else muscles

    def add_muscle(self, muscle) -> None:
        """Add a muscle to the model. The muscle name must be unique."""
        if muscle.name in self.muscles:
            raise ValueError(f"A muscle named {muscle.name!r} already exists in the model.")
        self.muscles[muscle.name] = muscle

    @property
    def nb_muscles(self) -> int:
        return len(self.muscles)

    @property
    def names(self) -> list[str]:
        return list(self.muscles.keys())

    @abstractmethod
    def lengths(self, Q: NaturalCoordinates, model):
        """Return the length of every muscles, stacked along axis 0."""

    @abstractmethod
    def moment_arms(self, Q: NaturalCoordinates, model):
        """Return ``- d L / d Q`` stacked as (nb_muscles, 12 * nb_segments)."""
