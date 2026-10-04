"""
From segment coordinates to natural coordinates: what the transformation matrix B does.

A segment in natural coordinates is described by Q = (u, rp, rd, w). The three vectors u, v = rp - rd and w form the
natural frame: they are not orthogonal in general (the angles alpha between v and w, beta between u and w, gamma
between u and v are measured on the subject), and v has the length L of the segment.

Anatomical data are given in an orthonormal segment coordinate system (SCS) instead: a centre of mass, a marker or a
muscle via point p = (X, Y, Z) relative to rp. B links the two frames. Its columns are u, v and w expressed in the SCS
(Dumas & Cheze 2007), e.g. for Buv:

    B = [[1, L cos(gamma), cos(beta)],
         [0, L sin(gamma), ...      ],
         [0, 0,            ...      ]]

so that

    p = B n,   i.e.   n = inv(B) p          (natural coordinates of an SCS point)
    position in the global frame = rp + n1 u + n2 v + n3 w = rp + [u v w] n
    SCS rotation in the global frame: R = [u v w] inv(B)

For orthonormal matrices, the inverse is the transpose. B is not orthonormal, so inv(B) is NOT B.T. Using B.T in
place of B silently misplaces every point given in the SCS. And when the segment is orthogonal (alpha = beta = gamma
= 90 deg), B = diag(1, L, 1) is symmetric: B.T == B, so the mistake is invisible.

This example draws, for a non-orthogonal segment:
1. the SCS, the natural frame, and a point p rebuilt from its natural coordinates with inv(B) (exact) and with
   inv(B.T) (the transposed mistake);
2. the error of the transposed mistake as the segment departs from orthogonality.
"""

import numpy as np

from bionc import TransformationMatrixType
from bionc.bionc_numpy import NaturalSegment, SegmentNaturalCoordinates
from bionc.bionc_numpy.transformation_matrix import compute_transformation_matrix


def natural_frame(
    alpha: float, beta: float, gamma: float, length: float
) -> tuple[np.ndarray, SegmentNaturalCoordinates]:
    """
    B, and the natural coordinates of a segment placed so that its SCS is the global frame: rp = 0, and u, v, w are
    the columns of B
    """
    B = compute_transformation_matrix(TransformationMatrixType.Buv, length, alpha, beta, gamma)
    u, v, w = B[:, 0], B[:, 1], B[:, 2]
    return B, SegmentNaturalCoordinates.from_components(u=u, rp=np.zeros(3), rd=-v, w=w)


def to_global(Q: SegmentNaturalCoordinates, natural: np.ndarray) -> np.ndarray:
    """rp + n1 u + n2 v + n3 w"""
    return np.asarray(Q.rp).reshape(3) + Q.to_uvw_matrix() @ natural


def transposed_mistake_error(alpha: float, beta: float, gamma: float, length: float, point: np.ndarray) -> float:
    """Distance between the point p and its reconstruction with inv(B.T) instead of inv(B)"""
    B, Q = natural_frame(alpha, beta, gamma, length)
    return float(np.linalg.norm(to_global(Q, np.linalg.inv(B.T) @ point) - point))


def main(show: bool = True) -> dict:
    alpha, beta, gamma, length = np.radians(80.2), np.radians(97.4), np.radians(74.5), 0.45
    point = np.array([0.02, -0.2, 0.01])  # e.g. a centre of mass given in the SCS [m]

    B, Q = natural_frame(alpha, beta, gamma, length)
    natural = np.linalg.inv(B) @ point
    natural_transposed = np.linalg.inv(B.T) @ point
    rebuilt, rebuilt_transposed = to_global(Q, natural), to_global(Q, natural_transposed)

    # bioNC does the same: the natural centre of mass of a segment defined with cartesian inertial parameters
    segment = NaturalSegment.with_cartesian_inertial_parameters(
        name="segment",
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        length=length,
        mass=1.0,
        center_of_mass=point,
        inertia=np.eye(3) * 0.01,
    )
    bionc_center_of_mass = to_global(Q, np.array(segment.natural_center_of_mass).squeeze())

    results = dict(
        natural=natural,
        error=float(np.linalg.norm(rebuilt - point)),
        error_transposed=float(np.linalg.norm(rebuilt_transposed - point)),
        error_bionc=float(np.linalg.norm(bionc_center_of_mass - point)),
    )
    print("B =\n", B.round(4))
    print("natural coordinates n = inv(B) p:", natural.round(4))
    print(f"rp + [u v w] n is back on p, error {results['error']:.1e} m")
    print(f"with inv(B.T) instead of inv(B), error {results['error_transposed'] * 1000:.1f} mm")
    print(f"bioNC natural centre of mass, error {results['error_bionc']:.1e} m")

    # the error of the transposed mistake as the segment departs from orthogonality
    deviations = np.linspace(0, 30, 61)
    errors = np.array(
        [
            transposed_mistake_error(np.radians(90 - d), np.radians(90 + d / 2), np.radians(90 - d), length, point)
            for d in deviations
        ]
    )
    results.update(deviations=deviations, errors=errors)

    if show:
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(12, 5.5))
        ax = fig.add_subplot(1, 2, 1, projection="3d")
        origin = np.zeros(3)
        for axis, color, label in zip(np.eye(3) * 0.15, ("r", "g", "b"), ("X", "Y", "Z")):
            ax.quiver(*origin, *axis, color=color, linewidth=1, linestyle="dotted")
            ax.text(*(axis * 1.15), label, color=color)
        for vector, color, label in zip(Q.to_uvw_matrix().T, ("tab:red", "tab:green", "tab:blue"), ("u", "v", "w")):
            vector = vector * (0.15 / np.linalg.norm(vector))
            ax.quiver(*origin, *vector, color=color, linewidth=2.5)
            ax.text(*(vector * 1.2), label, color=color, fontweight="bold")
        # path rp -> n1 u -> + n2 v -> + n3 w that leads to the point
        uvw = Q.to_uvw_matrix()
        path = np.array([origin, uvw[:, 0] * natural[0], uvw[:, :2] @ natural[:2], uvw @ natural])
        ax.plot(*path.T, "k--", linewidth=1, label="n1 u + n2 v + n3 w")
        ax.scatter(*point, color="k", s=60, label="p given in the SCS")
        ax.scatter(*rebuilt, color="tab:green", s=20, marker="x", label="rebuilt with inv(B)")
        ax.scatter(*rebuilt_transposed, color="tab:red", s=60, marker="x", label="rebuilt with inv(B.T)")
        ax.set_title("Segment coordinates (dotted, orthonormal)\nvs natural frame u, v, w (bold)")
        ax.legend(fontsize=8, loc="upper left")
        ax.set_xlim(-0.2, 0.2)
        ax.set_ylim(-0.25, 0.15)
        ax.set_zlim(-0.2, 0.2)
        ax.set_box_aspect((1, 1, 1))

        ax_error = fig.add_subplot(1, 2, 2)
        ax_error.plot(deviations, errors * 1000)
        ax_error.set_xlabel("departure from orthogonality d [deg]\n(alpha = 90 - d, beta = 90 + d/2, gamma = 90 - d)")
        ax_error.set_ylabel("misplacement of p with inv(B.T) [mm]")
        ax_error.set_title("Invisible on orthogonal segments (d = 0):\nB is then symmetric, B.T == B")
        ax_error.grid(True)
        fig.tight_layout()
        plt.show()

    return results


if __name__ == "__main__":
    main()
