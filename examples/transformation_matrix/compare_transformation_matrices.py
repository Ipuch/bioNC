"""
Side-by-side, animated comparison of every available ``TransformationMatrixType``.

Each implemented type gets its own copy of the same non-orthogonal natural
frame (u, v, w drawn in ``NaturalVectorColors``), arranged in a row along the
global x axis. The orthogonal segment frame derived from that type is drawn
at the same anchor with **always-RGB** axes (x = red, y = green, z = blue),
so the user can read each tile independently and see how the orthogonal frame
re-aligns when the natural angles (alpha, beta, gamma) wobble in time via a
cosine sweep.
"""

import numpy as np

from bionc import TransformationMatrixType
from bionc.bionc_numpy.transformation_matrix import (
    TRANSFORMATION_MAP,
    compute_transformation_matrix,
)
from bionc.vizualization.animations import NaturalVectorColors
from bionc.vizualization.pyorerun_natural_vectors import _ColoredVector


RGB_RED = (255, 0, 0)
RGB_GREEN = (0, 255, 0)
RGB_BLUE = (0, 0, 255)


def _implemented_types() -> list[TransformationMatrixType]:
    """Return the TransformationMatrixType values whose B(length, alpha, beta, gamma) is implemented."""
    available = []
    for t in TransformationMatrixType:
        try:
            TRANSFORMATION_MAP[t](1.0, np.pi / 2, np.pi / 2, np.pi / 2)
        except NotImplementedError:
            continue
        available.append(t)
    return available


def _angle_sweep(nb_frames: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Cosine-driven (alpha, beta, gamma) trajectory; returns radians + the time axis."""
    t = np.linspace(0.0, 1.0, nb_frames)
    cos = np.cos(2.0 * np.pi * t)
    alpha = np.deg2rad(70.0 + 10.0 * cos)
    beta = np.deg2rad(80.0 + 10.0 * np.cos(2.0 * np.pi * t + np.pi / 3))
    gamma = np.deg2rad(60.0 + 10.0 * np.cos(2.0 * np.pi * t + 2.0 * np.pi / 3))
    return t, alpha, beta, gamma


def _natural_and_orthogonal_axes(
    types: list[TransformationMatrixType],
    alpha: np.ndarray,
    beta: np.ndarray,
    gamma: np.ndarray,
) -> tuple[np.ndarray, dict[TransformationMatrixType, np.ndarray]]:
    """
    Return:
      natural -- (3, 3, nb_frames) with columns [u, v_basis, w] from B_Buv
      ortho   -- { type: (3, 3, nb_frames) rotation matrix expressing the type's
                  orthogonal axes (e_x, e_y, e_z as columns) in our chosen global,
                  which coincides with the Buv segment-Cartesian frame. }
    """
    nb_frames = alpha.size
    natural = np.zeros((3, 3, nb_frames))
    ortho = {t: np.zeros((3, 3, nb_frames)) for t in types}

    for k in range(nb_frames):
        B_ref = compute_transformation_matrix(TransformationMatrixType.Buv, 1.0, alpha[k], beta[k], gamma[k])
        natural[:, :, k] = B_ref
        for t in types:
            B_t = compute_transformation_matrix(t, 1.0, alpha[k], beta[k], gamma[k])
            # Express the type's orthogonal axes in our reference (Buv) global frame.
            ortho[t][:, :, k] = B_ref @ np.linalg.inv(B_t)
    return natural, ortho


def _add_arrow(prr, name: str, num: int, origin: np.ndarray, endpoint: np.ndarray, color_rgb255: tuple) -> None:
    prr.xp_data.add_data(
        _ColoredVector(
            name=f"{prr.name}/{name}",
            num=num,
            vector_origin=origin,
            vector_endpoint=endpoint,
            color_rgb255=color_rgb255,
        )
    )


def main(nb_frames: int = 90, tile_spacing: float = 2.5, ortho_scale: float = 0.6):
    types = _implemented_types()
    t, alpha, beta, gamma = _angle_sweep(nb_frames)
    natural, ortho = _natural_and_orthogonal_axes(types, alpha, beta, gamma)

    nat_u = natural[:, 0, :]
    nat_v = natural[:, 1, :]
    nat_w = natural[:, 2, :]

    from pyorerun import PhaseRerun

    prr = PhaseRerun(t_span=t)

    for i, type_enum in enumerate(types):
        rp = np.tile(np.array([[i * tile_spacing], [0.0], [0.0]]), (1, nb_frames))

        _add_arrow(prr, f"{type_enum.name}/u", i, rp, rp + nat_u, NaturalVectorColors.U.value)
        _add_arrow(prr, f"{type_enum.name}/v", i, rp, rp + nat_v, NaturalVectorColors.V.value)
        _add_arrow(prr, f"{type_enum.name}/w", i, rp, rp + nat_w, NaturalVectorColors.W.value)

        rot = ortho[type_enum]
        _add_arrow(prr, f"{type_enum.name}/x", i, rp, rp + ortho_scale * rot[:, 0, :], RGB_RED)
        _add_arrow(prr, f"{type_enum.name}/y", i, rp, rp + ortho_scale * rot[:, 1, :], RGB_GREEN)
        _add_arrow(prr, f"{type_enum.name}/z", i, rp, rp + ortho_scale * rot[:, 2, :], RGB_BLUE)

    print("Tiles laid out along x; each tile shares the same natural u/v/w but shows")
    print("a different orthogonal frame derived from its TransformationMatrixType.")
    print("Natural axes use NaturalVectorColors; orthogonal axes are always RGB.")
    print("Available types (left to right):", ", ".join(t.name for t in types))
    skipped = [t.name for t in TransformationMatrixType if t not in types]
    if skipped:
        print("Skipped (not implemented):", ", ".join(skipped))
    prr.rerun()


if __name__ == "__main__":
    main()
