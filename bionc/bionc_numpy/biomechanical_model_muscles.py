import numpy as np

from .biomechanical_model_segments import BiomechanicalModelSegments
from .natural_coordinates import NaturalCoordinates
from ..protocols.biomechanical_model_muscles import GenericBiomechanicalModelMuscles


class BiomechanicalModelMuscles(GenericBiomechanicalModelMuscles):
    """Numpy container holding the muscles of a biomechanical model."""

    def __init__(
        self,
        segments: BiomechanicalModelSegments = None,
        muscles: dict = None,
    ):
        segments = BiomechanicalModelSegments() if segments is None else segments
        super().__init__(segments=segments, muscles=muscles)

    def lengths(self, Q: NaturalCoordinates, model) -> np.ndarray:
        if not self.muscles:
            return np.zeros(0)
        return np.array([m.length(Q, model) for m in self.muscles.values()])

    def moment_arms(self, Q: NaturalCoordinates, model) -> np.ndarray:
        if not self.muscles:
            return np.zeros((0, 12 * model.nb_segments))
        return np.vstack([m.moment_arm(Q, model) for m in self.muscles.values()])
