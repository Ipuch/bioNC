import numpy as np

from pyorerun.abstract.markers import rgb255_to_hex_rgba
from pyorerun.xp_components.force_vector import Vector

from ..protocols.biomechanical_model import GenericBiomechanicalModel
from .animations import NaturalVectorColors


class _ColoredVector(Vector):
    """Vector with a per-instance color (the base ``Vector`` hardcodes one)."""

    def __init__(self, name, num, vector_origin, vector_endpoint, color_rgb255):
        magnitude = vector_endpoint - vector_origin
        super().__init__(name=name, num=num, vector_origins=vector_origin, vector_magnitudes=magnitude)
        self._color = rgb255_to_hex_rgba(np.asarray(color_rgb255))

    def to_component(self, frame):
        import rerun as rr

        return rr.Arrows3D(
            origins=self.vector_origins[:, frame],
            vectors=self.vector_magnitude[:, frame],
            colors=self._color,
        )

    def to_chunk(self, **kwargs):
        import rerun as rr

        return {
            self.name: [
                *rr.Arrows3D.columns(
                    origins=self.vector_origins.T.tolist(),
                    vectors=self.vector_magnitude.T.tolist(),
                    colors=[self._color for _ in range(self.nb_frames)],
                )
            ]
        }


def add_natural_vectors(
    phase_rerun,
    model: GenericBiomechanicalModel,
    Q: np.ndarray,
    scale_u: float = 0.2,
    scale_v: float = 1.0,
    scale_w: float = 0.2,
) -> None:
    """
    Add the u, v, w natural-coordinate vectors of every segment to a pyorerun phase.

    Vectors are anchored at the proximal point ``rp`` and colored according to
    ``NaturalVectorColors`` (red / green / blue for u / v / w).

    Parameters
    ----------
    phase_rerun
        A ``pyorerun.PhaseRerun`` instance already configured for the trajectory.
    model
        The biomechanical model providing segment names.
    Q
        Natural coordinates trajectory of shape ``(nb_Q, nb_frames)``.
    scale_u, scale_v, scale_w
        Multiplicative scale factors applied to each natural vector for display.
        ``v`` is the segment vector ``rd - rp`` and is unscaled by default
        (``scale_v=1.0``) so it spans the full segment.
    """
    Q = np.asarray(Q, dtype=float)
    if Q.ndim != 2 or Q.shape[0] != 12 * model.nb_segments:
        raise ValueError(
            f"Q must have shape (12*nb_segments, nb_frames); got {Q.shape} for nb_segments={model.nb_segments}."
        )

    for i, seg_name in enumerate(model.segment_names):
        u = Q[12 * i + 0 : 12 * i + 3, :]
        rp = Q[12 * i + 3 : 12 * i + 6, :]
        rd = Q[12 * i + 6 : 12 * i + 9, :]
        w = Q[12 * i + 9 : 12 * i + 12, :]

        for axis_name, endpoint, color in (
            (f"{seg_name}_u", rp + scale_u * u, NaturalVectorColors.U.value),
            (f"{seg_name}_v", rp + scale_v * (rd - rp), NaturalVectorColors.V.value),
            (f"{seg_name}_w", rp + scale_w * w, NaturalVectorColors.W.value),
        ):
            phase_rerun.xp_data.add_data(
                _ColoredVector(
                    name=f"{phase_rerun.name}/{axis_name}",
                    num=i,
                    vector_origin=rp,
                    vector_endpoint=endpoint,
                    color_rgb255=color,
                )
            )
