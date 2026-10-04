"""
Figures of the README section on the transformation matrix B, written to docs/:

- b_matrix_gram_schmidt.gif: how B_uv builds the orthonormal segment coordinate system (SCS) on the natural frame
  (Gram-Schmidt: X = u, Y in the plane (u, v), Z = X x Y), that the columns of B are u, v, w read in (X, Y, Z), and how
  a point p given in (X, Y, Z) is reached by n1 u + n2 v + n3 w, with n = inv(B) p;
- b_matrix_types.gif: the SCS built by each implemented B on the same natural frame, while its angles change;
- b_matrix_frames.png: the global frame, the SCS and the natural frame of a segment, and a point given in the SCS
  reached from rp by n1 u + n2 v + n3 w.

Natural vectors use the colors of bionc.vizualization (u magenta, v lime, w cyan), orthonormal axes are always RGB
(dashed). Run: python readme_figures.py (needs matplotlib and pillow)
"""

from pathlib import Path

import numpy as np

from bionc import TransformationMatrixType
from bionc.bionc_numpy.transformation_matrix import TRANSFORMATION_MAP, compute_transformation_matrix
from bionc.vizualization.animations import NaturalVectorColors

DOCS = Path(__file__).parent.parent.parent / "docs"
NATURAL_COLORS = {name: tuple(c / 255 for c in NaturalVectorColors[name.upper()].value) for name in ("u", "v", "w")}
RGB = ("#ff3b3b", "#2ee65a", "#3b8bff")
BACKGROUND = "#0d1117"  # GitHub dark
TEXT, SUBTEXT = "white", "#b0b8c4"
ALPHA, BETA, GAMMA = np.radians(70), np.radians(80), np.radians(60)


def _implemented_types() -> list[TransformationMatrixType]:
    types = []
    for matrix_type in TransformationMatrixType:
        try:
            TRANSFORMATION_MAP[matrix_type](1.0, np.pi / 2, np.pi / 2, np.pi / 2)
        except NotImplementedError:
            continue
        types.append(matrix_type)
    return types


def _scene(ax, center=(0, 0, 0), half_size: float = 0.8, elevation: float = 20, azimuth: float = -60, zoom=1.6):
    ax.set_facecolor(BACKGROUND)
    ax.set_axis_off()
    for set_lim, c in zip((ax.set_xlim, ax.set_ylim, ax.set_zlim), center):
        set_lim(c - half_size, c + half_size)
    ax.set_box_aspect((1, 1, 1), zoom=zoom)
    ax.view_init(elev=elevation, azim=azimuth)


def _arrow(ax, origin, vector, color, width=3.0, label=None, label_offset=1.12, dashed=False):
    vector = np.asarray(vector, dtype=float)
    if np.linalg.norm(vector) < 1e-9:
        return
    ax.quiver(
        *origin,
        *vector,
        color=color,
        linewidth=width,
        arrow_length_ratio=0.1,
        linestyle="--" if dashed else "-",
    )
    if label:
        position = np.asarray(origin) + label_offset * vector
        ax.text(*position, label, color=color, fontsize=13, fontweight="bold", ha="center", va="center")


def _orthonormal_axes(ax, origin, rotation, length, labels="XYZ"):
    for axis, color, label in zip(np.asarray(rotation).T, RGB, labels):
        _arrow(ax, origin, length * axis, color, width=1.5, label=label, dashed=True)


def _natural_frame(ax, origin, uvw, scale=1.0, labels=True):
    for name, vector in zip("uvw", np.asarray(uvw).T):
        _arrow(ax, origin, scale * vector, NATURAL_COLORS[name], width=3.5, label=name if labels else None)


def _title(fig, text: str, sub: str = "", top: float = 0.95):
    fig.text(0.5, top, text, color=TEXT, fontsize=14, ha="center", va="top", fontweight="bold")
    if sub:
        fig.text(0.5, top - 0.06, sub, color=SUBTEXT, fontsize=10.5, ha="center", va="top")


def _ease(x: float) -> float:
    """0 -> 1 smoothly"""
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


def _dotted(ax, points, color, alpha=1.0):
    ax.plot(*np.asarray(points).T, ":", color=color, linewidth=1.2, alpha=alpha)


def gram_schmidt_gif(file: Path, nb_frames: int = 200, fps: int = 20):
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    B = compute_transformation_matrix(TransformationMatrixType.Buv, 1.0, ALPHA, BETA, GAMMA)
    u, v, w = B.T
    natural = np.array([0.35, 0.55, 0.5])  # chosen positive for a readable path
    point = B @ natural  # the point in (X, Y, Z), so that n = inv(B) p
    route = np.array([np.zeros(3), u * natural[0], B[:, :2] @ natural[:2], point])
    stages = [
        ("The natural frame (u, v, w)", "measured on the segment, not orthogonal"),
        ("X = u", "B_uv keeps u exactly"),
        ("Y: in the plane (u, v), orthogonal to X", "Gram-Schmidt: v minus its projection on X"),
        ("Z = X x Y", "(X, Y, Z) is the orthonormal segment coordinate system"),
        ("B: u, v, w read in (X, Y, Z)", "its columns. B is triangular, it carries L, alpha, beta, gamma"),
        ("A point p given in (X, Y, Z)", "is reached by n1 u + n2 v + n3 w, with n = inv(B) p"),
    ]
    bounds = np.linspace(0, 1, len(stages) + 1)

    fig = plt.figure(figsize=(6.4, 6.4), facecolor=BACKGROUND)
    ax = fig.add_axes([0, 0.06, 1, 0.8], projection="3d")

    def draw(frame):
        ax.cla()
        fig.texts.clear()
        s = min(frame / (nb_frames - 20), 0.999)  # the last image is held
        _scene(ax, center=(0.35, 0.35, 0.45), half_size=0.75, azimuth=-62 + 35 * s, zoom=1.15)
        stage = int(np.searchsorted(bounds, s, side="right") - 1)
        progress = [_ease((s - bounds[i]) / (bounds[i + 1] - bounds[i]) * 1.6) for i in range(len(stages))]
        _title(fig, *stages[stage])

        grow = progress[0]
        for name, vector in zip("uvw", (u, v, w)):
            _arrow(ax, np.zeros(3), grow * vector, NATURAL_COLORS[name], width=3.5, label=name if grow > 0.9 else None)

        for index, (axis, label) in enumerate(zip(np.eye(3), "XYZ")):
            if stage >= index + 1 and progress[index + 1] > 0:
                length = 1.25 * progress[index + 1]
                _arrow(ax, np.zeros(3), length * axis, RGB[index], width=1.6, dashed=True, label=label)
        if stage == 2:
            plane = np.array([[-0.2, -0.2, 0], [1.0, -0.2, 0], [1.0, 1.0, 0], [-0.2, 1.0, 0]])
            ax.add_collection3d(Poly3DCollection([plane], color="white", alpha=0.08))
            _dotted(ax, [[v[0], 0, 0], v], "white", alpha=progress[2])

        if stage == 4:
            for column, name in zip(B.T, "uvw"):
                _dotted(ax, [[column[0], 0, 0], [column[0], column[1], 0], column], NATURAL_COLORS[name])
            lines = "\n".join("[ " + "  ".join(f"{B[i, j]:+.2f}" for j in range(3)) + " ]" for i in range(3))
            fig.text(0.05, 0.05, "B =\n" + lines, color=TEXT, fontsize=11, family="monospace", va="bottom")
            fig.text(0.05, 0.02, "      u      v      w", color=SUBTEXT, fontsize=11, family="monospace", va="bottom")

        if stage == 5:
            a = progress[5]
            ax.scatter(*point, color=TEXT, s=80, depthshade=False)
            ax.text(*(point + np.array([0.0, 0.0, 0.08])), "p", color=TEXT, fontsize=14, fontweight="bold")
            _dotted(ax, [[point[0], 0, 0], [point[0], point[1], 0], point], SUBTEXT)
            for leg, name in enumerate("uvw"):
                leg_progress = min(max(3 * a - leg, 0.0), 1.0)
                if leg_progress > 0:
                    end = route[leg] + leg_progress * (route[leg + 1] - route[leg])
                    ax.plot(*np.array([route[leg], end]).T, color=NATURAL_COLORS[name], linewidth=3)
            fig.text(
                0.5,
                0.03,
                f"n = inv(B) p = ({natural[0]:.2f}, {natural[1]:.2f}, {natural[2]:.2f})",
                color=TEXT,
                fontsize=11,
                ha="center",
                family="monospace",
            )
        return []

    animation = FuncAnimation(fig, draw, frames=nb_frames, interval=1000 / fps)
    animation.save(file, writer=PillowWriter(fps=fps), dpi=72, savefig_kwargs=dict(facecolor=BACKGROUND))
    plt.close(fig)


def types_gif(file: Path, nb_frames: int = 90, fps: int = 15):
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    types = _implemented_types()
    descriptions = {
        TransformationMatrixType.Buv: "X = u,  v in (X, Y)",
        TransformationMatrixType.Bvu: "Y = v,  u in (X, Y)",
        TransformationMatrixType.Bwu: "Z = w,  u in (Z, X)",
        TransformationMatrixType.Buw: "X = u,  w in (X, Z)",
    }
    t = np.linspace(0, 1, nb_frames, endpoint=False)
    alpha = np.radians(70 + 12 * np.cos(2 * np.pi * t))
    beta = np.radians(80 + 12 * np.cos(2 * np.pi * t + np.pi / 3))
    gamma = np.radians(60 + 12 * np.cos(2 * np.pi * t + 2 * np.pi / 3))

    fig = plt.figure(figsize=(3.2 * len(types), 4.2), facecolor=BACKGROUND)
    width = 1 / len(types)
    axes = [fig.add_axes([i * width, 0.13, width, 0.62], projection="3d") for i in range(len(types))]

    def draw(frame):
        fig.texts.clear()
        _title(
            fig,
            "One natural frame (u, v, w), one orthonormal frame per B",
            f"alpha {np.degrees(alpha[frame]):.0f} deg    beta {np.degrees(beta[frame]):.0f} deg    "
            f"gamma {np.degrees(gamma[frame]):.0f} deg",
            top=0.96,
        )
        reference = compute_transformation_matrix(
            TransformationMatrixType.Buv, 1.0, alpha[frame], beta[frame], gamma[frame]
        )
        for i, (ax, matrix_type) in enumerate(zip(axes, types)):
            ax.cla()
            _scene(ax, center=(0.3, 0.3, 0.5), half_size=0.8, zoom=1.1)
            _natural_frame(ax, np.zeros(3), reference)
            # [u v w] = R B, so the axes of this SCS are the columns of R = [u v w] inv(B)
            B = compute_transformation_matrix(matrix_type, 1.0, alpha[frame], beta[frame], gamma[frame])
            _orthonormal_axes(ax, np.zeros(3), reference @ np.linalg.inv(B), length=1.15)
            center = (i + 0.5) * width
            fig.text(center, 0.08, matrix_type.name, color=TEXT, fontsize=12, fontweight="bold", ha="center")
            fig.text(center, 0.02, descriptions.get(matrix_type, ""), color=SUBTEXT, fontsize=10, ha="center")
        return []

    animation = FuncAnimation(fig, draw, frames=nb_frames, interval=1000 / fps)
    animation.save(file, writer=PillowWriter(fps=fps), dpi=72, savefig_kwargs=dict(facecolor=BACKGROUND))
    plt.close(fig)


def frames_png(file: Path):
    import matplotlib.pyplot as plt

    length = 0.45
    B = compute_transformation_matrix(
        TransformationMatrixType.Buv, length, np.radians(80), np.radians(97), np.radians(75)
    )
    axis = np.ones(3) / np.sqrt(3)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    pose = np.eye(3) + np.sin(np.radians(35)) * K + (1 - np.cos(np.radians(35))) * K @ K
    rp = np.array([0.45, 0.35, 0.25])
    uvw = pose @ B  # u, v, w in the global frame
    natural = np.array([0.25, 0.3, 0.2])  # chosen positive for a readable path
    point = B @ natural  # the point in the SCS, so that n = inv(B) p
    target = rp + uvw @ natural

    fig = plt.figure(figsize=(7.5, 6.8), facecolor=BACKGROUND)
    ax = fig.add_axes([0, 0.12, 1, 0.72], projection="3d")
    _scene(ax, center=(0.4, 0.3, 0.3), half_size=0.4, elevation=18, azimuth=-58, zoom=1.25)

    for vector, label in zip(np.eye(3) * 0.15, "xyz"):
        _arrow(ax, np.zeros(3), vector, "#8b949e", width=1.2, label=label)
    ax.text(0, 0, -0.05, "global", color="#8b949e", fontsize=10, ha="center")
    _orthonormal_axes(ax, rp, pose, length=0.3)
    _natural_frame(ax, rp, uvw / np.linalg.norm(uvw, axis=0) * 0.22)
    ax.text(*(rp + np.array([-0.03, 0, -0.04])), "rp", color=TEXT, fontsize=11, ha="right")

    # the coordinates of p in (X, Y, Z), and its path along u, then v, then w
    corners = [rp, rp + pose[:, 0] * point[0], rp + pose[:, :2] @ point[:2], target]
    _dotted(ax, corners, SUBTEXT)
    route = np.array([rp, rp + uvw[:, 0] * natural[0], rp + uvw[:, :2] @ natural[:2], target])
    for (start, end), name in zip(zip(route[:-1], route[1:]), "uvw"):
        ax.plot(*np.array([start, end]).T, color=NATURAL_COLORS[name], linewidth=2.2)
    ax.scatter(*target, color=TEXT, s=90, depthshade=False)
    ax.text(*(target + np.array([0.02, 0, 0.03])), "p", color=TEXT, fontsize=14, fontweight="bold")

    _title(fig, "Three frames, one point", "p is given in the segment coordinate system (X, Y, Z), dotted")
    for line, y in (
        ("natural -> SCS:   p = B n,  n = inv(B) p", 0.085),
        ("natural -> global:   r = rp + n1 u + n2 v + n3 w  (solid)", 0.05),
        ("SCS -> global:   r = rp + R p,  R = [u v w] inv(B)", 0.015),
    ):
        fig.text(0.5, y, line, color=SUBTEXT, fontsize=10.5, ha="center", family="monospace")
    fig.savefig(file, dpi=110, facecolor=BACKGROUND)
    plt.close(fig)


def main(folder: Path = DOCS):
    folder = Path(folder)
    folder.mkdir(exist_ok=True)
    gram_schmidt_gif(folder / "b_matrix_gram_schmidt.gif")
    types_gif(folder / "b_matrix_types.gif")
    frames_png(folder / "b_matrix_frames.png")
    print("written to", folder)


if __name__ == "__main__":
    main()
