from typing import Union

import numpy as np
from casadi import MX, sqrt, jacobian, vertcat, horzcat, sum1

from ..protocols.muscle import AbstractMuscle
from .natural_coordinates import NaturalCoordinates
from .natural_vector import NaturalVector

GROUND_NAMES = ("GROUND", "ground", "Ground")


class MuscleViaPoint:
    def __init__(
        self,
        name: str,
        parent_name: str,
        position: Union[tuple, np.ndarray, MX, NaturalVector],
    ):
        self.name = name
        self.parent_name = parent_name
        self.is_ground = parent_name in GROUND_NAMES

        if self.is_ground:
            pos = np.asarray(position, dtype=float).reshape(3)
            self.position = pos
            self.interpolation_matrix = None
        else:
            if not isinstance(position, NaturalVector):
                position = NaturalVector(position)
            self.position = position
            self.interpolation_matrix = position.interpolate()

    def position_in_global(self, Q: NaturalCoordinates, model) -> MX:
        if self.is_ground:
            return MX(self.position.reshape(3, 1))
        seg_idx = model.segments[self.parent_name].index
        Qi = Q.vector(seg_idx)
        return self.interpolation_matrix @ Qi


class Muscle(AbstractMuscle):
    def __init__(self, name: str, via_points: list[MuscleViaPoint]):
        if len(via_points) < 2:
            raise ValueError("A muscle requires at least two via points (origin and insertion).")
        self._name = name
        self._via_points = list(via_points)

    @property
    def name(self) -> str:
        return self._name

    @property
    def via_points(self) -> list[MuscleViaPoint]:
        return self._via_points

    def via_points_in_global(self, Q: NaturalCoordinates, model) -> MX:
        cols = [self._via_points[k].position_in_global(Q, model) for k in range(len(self._via_points))]
        return horzcat(*cols)

    def length(self, Q: NaturalCoordinates, model) -> MX:
        points = self.via_points_in_global(Q, model)
        total = MX(0)
        for s in range(points.shape[1] - 1):
            d = points[:, s + 1] - points[:, s]
            total = total + sqrt(sum1(d * d))
        return total

    def moment_arm(self, Q: NaturalCoordinates, model) -> MX:
        """Returns ``- d length / d Q`` as an MX of shape (12 * nb_segments, 1)."""
        ell = self.length(Q, model)
        Q_vec = Q if isinstance(Q, MX) else MX(Q)
        return -jacobian(ell, Q_vec).T
