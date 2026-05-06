from casadi import MX, vertcat

from .biomechanical_model_segments import BiomechanicalModelSegments
from .natural_coordinates import NaturalCoordinates
from ..protocols.biomechanical_model_muscles import GenericBiomechanicalModelMuscles


class BiomechanicalModelMuscles(GenericBiomechanicalModelMuscles):
    """Casadi container holding the muscles of a biomechanical model."""

    def __init__(
        self,
        segments: BiomechanicalModelSegments = None,
        muscles: dict = None,
    ):
        segments = BiomechanicalModelSegments() if segments is None else segments
        super().__init__(segments=segments, muscles=muscles)

    def lengths(self, Q: NaturalCoordinates, model) -> MX:
        if not self.muscles:
            return MX.zeros(0, 1)
        return vertcat(*[m.length(Q, model) for m in self.muscles.values()])

    def moment_arms(self, Q: NaturalCoordinates, model) -> MX:
        if not self.muscles:
            return MX.zeros(0, 12 * model.nb_segments)
        return vertcat(*[m.moment_arm(Q, model).T for m in self.muscles.values()])
