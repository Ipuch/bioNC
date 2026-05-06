from abc import ABC, abstractmethod

from .biomechanical_model import GenericBiomechanicalModel
from .natural_coordinates import NaturalCoordinates


class AbstractMuscle(ABC):
    """
    A straight-line muscle going through an ordered list of via points.

    A via point is anchored either to a segment (described by a NaturalVector
    expressing its position in segment-local non-orthogonal coordinates) or to
    the GROUND (a fixed 3D position in the global frame).

    The muscle length is the sum of the Euclidean distances between consecutive
    via points expressed in the global frame. The moment arm is exposed as
    ``-d length / d Q`` (a vector of size 12 * nb_segments), which the user can
    project onto joint axes if needed.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the muscle."""

    @property
    @abstractmethod
    def via_points(self) -> list:
        """Ordered list of via points (origin first, insertion last)."""

    @abstractmethod
    def via_points_in_global(self, Q: NaturalCoordinates, model: GenericBiomechanicalModel):
        """Return the global position of every via point as a (3, n) array."""

    @abstractmethod
    def length(self, Q: NaturalCoordinates, model: GenericBiomechanicalModel):
        """Return the muscle-tendon length for the given natural coordinates."""

    @abstractmethod
    def moment_arm(self, Q: NaturalCoordinates, model: GenericBiomechanicalModel):
        """
        Return the moment arm with respect to the natural coordinates Q,
        i.e. ``- d length / d Q`` (shape: 12 * nb_segments,).
        """
