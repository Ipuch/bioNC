from typing import Union

import numpy as np

from ..protocols.muscle import AbstractMuscle
from .natural_coordinates import NaturalCoordinates
from .natural_vector import NaturalVector

GROUND_NAMES = ("GROUND", "ground", "Ground")


class MuscleViaPoint:
    """
    A muscle via point.

    Parameters
    ----------
    name
        Name of the via point.
    parent_name
        Name of the segment to which the via point is attached. Use "GROUND" for
        a fixed point in the global frame.
    position
        - If ``parent_name`` is a segment: 3D position in segment local
          (non-orthogonal) coordinates, as a tuple/np.ndarray/NaturalVector.
        - If ``parent_name`` is GROUND: 3D position in the global frame.
    """

    def __init__(
        self,
        name: str,
        parent_name: str,
        position: Union[tuple, np.ndarray, NaturalVector],
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

    def position_in_global(self, Q: NaturalCoordinates, model) -> np.ndarray:
        if self.is_ground:
            return self.position.reshape(3)
        seg_idx = model.segments[self.parent_name].index
        Qi = Q.vector(seg_idx)
        return np.asarray(self.interpolation_matrix @ Qi).reshape(3)

    def to_mx(self):
        from ..bionc_casadi.muscle import MuscleViaPoint as MuscleViaPointMX

        return MuscleViaPointMX(
            name=self.name,
            parent_name=self.parent_name,
            position=self.position if self.is_ground else self.position,
        )


class Muscle(AbstractMuscle):
    """
    A straight-line muscle defined as an ordered list of via points.
    """

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

    def via_points_in_global(self, Q: NaturalCoordinates, model) -> np.ndarray:
        points = np.zeros((3, len(self._via_points)))
        for k, vp in enumerate(self._via_points):
            points[:, k] = vp.position_in_global(Q, model)
        return points

    def length(self, Q: NaturalCoordinates, model) -> float:
        points = self.via_points_in_global(Q, model)
        diffs = points[:, 1:] - points[:, :-1]
        return float(np.sum(np.linalg.norm(diffs, axis=0)))

    def moment_arm(self, Q: NaturalCoordinates, model) -> np.ndarray:
        """
        Returns ``- d length / d Q`` of size (12 * nb_segments,).

        For a straight-line muscle, the contribution of segment i to ``d length / d Q_i``
        is the sum over via points k attached to segment i of ``N_k.T @ u_k``, where
        ``u_k`` is the (signed) sum of unit vectors of the segments of the muscle
        path adjacent to via point k:

            u_k =  (p_k - p_{k-1}) / |p_k - p_{k-1}|   (incoming, if any)
                 - (p_{k+1} - p_k) / |p_{k+1} - p_k|   (outgoing, if any)
        """
        nb_seg = model.nb_segments
        nb_Q = 12 * nb_seg
        dl_dQ = np.zeros(nb_Q)

        points = self.via_points_in_global(Q, model)
        n = points.shape[1]

        # Unit vectors along each muscle segment (between consecutive via points)
        unit = np.zeros((3, n - 1))
        for s in range(n - 1):
            d = points[:, s + 1] - points[:, s]
            norm = np.linalg.norm(d)
            if norm < 1e-12:
                unit[:, s] = 0.0
            else:
                unit[:, s] = d / norm

        for k, vp in enumerate(self._via_points):
            if vp.is_ground:
                continue
            u_k = np.zeros(3)
            if k > 0:
                u_k += unit[:, k - 1]
            if k < n - 1:
                u_k -= unit[:, k]
            seg_idx = model.segments[vp.parent_name].index
            N = vp.interpolation_matrix
            contrib = np.asarray(N).T @ u_k
            dl_dQ[seg_idx * 12 : (seg_idx + 1) * 12] += contrib.reshape(12)

        return -dl_dQ

    def __str__(self) -> str:
        out = f"muscle {self._name}\n"
        for vp in self._via_points:
            p = np.asarray(vp.position).reshape(-1)
            out += f"\tviapoint {vp.name} parent {vp.parent_name} position {p[0]:.4f} {p[1]:.4f} {p[2]:.4f}\n"
        out += "endmuscle\n"
        return out

    def to_mx(self):
        from ..bionc_casadi.muscle import Muscle as MuscleMX

        return MuscleMX(name=self._name, via_points=[vp.to_mx() for vp in self._via_points])
