"""
From one frame to another: segment coordinates, natural coordinates and the global frame.

A segment in natural coordinates is described by Q = (u, rp, rd, w). The three vectors u, v = rp - rd and w form the
natural frame: they are not orthogonal in general (alpha is the angle between v and w, beta between u and w, gamma
between u and v), and v has the length L of the segment.

Anatomical data are given in an orthonormal segment coordinate system (SCS) instead: a centre of mass, a marker or a
muscle via point p = (X, Y, Z), relative to rp. The transformation matrix B links the two. Its columns are u, v and w
expressed in the SCS (Dumas & Cheze 2007). Three frames, three relations:

    natural -> SCS        p = B n                       n = inv(B) p
    natural -> global     r = rp + [u v w] n
    SCS     -> global     r = rp + R p                  R = [u v w] inv(B)

For an orthonormal matrix the inverse is the transpose, but B is not orthonormal: inv(B) is NOT B.T. Using B.T in place
of B silently misplaces every point given in the SCS, and the mistake is invisible on orthogonal segments
(alpha = beta = gamma = 90 deg), where B = diag(1, L, 1) is symmetric. bioNC carried it until issue #178.

The example is written twice, side by side:
- Part 1, plain numpy: B is built by hand from (alpha, beta, gamma, L) and every frame change is a line of linear algebra;
- Part 2, bioNC: the same frame changes with the bioNC API, checked against part 1.
Then two figures: the frames and a point rebuilt with inv(B) and with inv(B.T), and the error of the transposed mistake
as the segment departs from orthogonality.
"""

import numpy as np

from bionc.bionc_numpy import NaturalSegment, SegmentNaturalCoordinates

# the segment: non-orthogonal angles and length
ALPHA, BETA, GAMMA, LENGTH = np.radians(80.2), np.radians(97.4), np.radians(74.5), 0.45
# its pose in the global frame: the SCS is rotated by R0 and its origin rp is at RP
RP = np.array([0.1, 0.2, 0.3])
# a point given in the SCS, e.g. a centre of mass [m]
POINT = np.array([0.02, -0.2, 0.01])


def rotation_about(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rodrigues' formula"""
    axis = np.asarray(axis, dtype=float) / np.linalg.norm(axis)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K


R0 = rotation_about([1, 1, 1], np.radians(40))


# ----------------------------------------------------------------------------------------------------------------------
# Part 1: plain numpy
# ----------------------------------------------------------------------------------------------------------------------
def transformation_matrix_by_hand(alpha: float, beta: float, gamma: float, length: float) -> np.ndarray:
    """
    B whose columns are u, v, w expressed in the SCS (Buv: X along u, Y in the plane (u, v)):
    u = (1, 0, 0), v = L (cos(gamma), sin(gamma), 0), and w the unit vector with u.w = cos(beta), v.w = L cos(alpha)
    """
    u = np.array([1.0, 0, 0])
    v = length * np.array([np.cos(gamma), np.sin(gamma), 0])
    w_x = np.cos(beta)
    w_y = (np.cos(alpha) - np.cos(beta) * np.cos(gamma)) / np.sin(gamma)
    w = np.array([w_x, w_y, np.sqrt(1 - w_x**2 - w_y**2)])
    return np.column_stack((u, v, w))


def natural_coordinates_of_the_pose(B: np.ndarray, rotation: np.ndarray, rp: np.ndarray) -> np.ndarray:
    """Q = (u, rp, rd, w) in the global frame: the columns of B rotated by the pose of the SCS, rd = rp - v"""
    u, v, w = rotation @ B[:, 0], rotation @ B[:, 1], rotation @ B[:, 2]
    return np.concatenate((u, rp, rp - v, w))


def frame_changes_with_numpy(point: np.ndarray) -> dict:
    B = transformation_matrix_by_hand(ALPHA, BETA, GAMMA, LENGTH)
    Q = natural_coordinates_of_the_pose(B, R0, RP)
    u, rp, rd, w = Q[0:3], Q[3:6], Q[6:9], Q[9:12]
    uvw = np.column_stack((u, rp - rd, w))

    natural = np.linalg.solve(B, point)  # SCS -> natural: n = inv(B) p
    in_global_from_natural = rp + uvw @ natural  # natural -> global
    rotation = uvw @ np.linalg.inv(B)  # SCS rotation: R = [u v w] inv(B)
    in_global_from_scs = rp + rotation @ point  # SCS -> global
    natural_from_global = np.linalg.solve(uvw, in_global_from_scs - rp)  # global -> natural
    natural_transposed = np.linalg.solve(B.T, point)  # the mistake: inv(B.T) instead of inv(B)

    return dict(
        B=B,
        Q=Q,
        natural=natural,
        rotation=rotation,
        in_global=in_global_from_natural,
        in_global_from_scs=in_global_from_scs,
        natural_from_global=natural_from_global,
        in_global_transposed=rp + uvw @ natural_transposed,
    )


# ----------------------------------------------------------------------------------------------------------------------
# Part 2: the same with bioNC
# ----------------------------------------------------------------------------------------------------------------------
def frame_changes_with_bionc(point: np.ndarray, Q: np.ndarray) -> dict:
    segment = NaturalSegment.with_cartesian_inertial_parameters(
        name="segment",
        alpha=ALPHA,
        beta=BETA,
        gamma=GAMMA,
        length=LENGTH,
        mass=1.0,
        center_of_mass=point,  # given in the SCS, converted to natural coordinates with inv(B)
        inertia=np.eye(3) * 0.01,
    )
    segment.add_natural_marker_from_segment_coordinates(name="marker", location=point)
    Qi = SegmentNaturalCoordinates.from_components(u=Q[0:3], rp=Q[3:6], rd=Q[6:9], w=Q[9:12])

    B = segment.compute_transformation_matrix()  # B, columns u, v, w in the SCS
    natural = segment.compute_transformation_matrix_inverse() @ point  # SCS -> natural
    scs = segment.segment_coordinates_system(Qi)  # SCS -> global, homogeneous transform (R, rp)
    marker_in_global = np.asarray(segment.markers(Qi)).reshape(3)  # natural -> global, forward kinematics
    center_of_mass_in_global = np.asarray(segment.natural_center_of_mass.interpolate().to_array() @ Q).reshape(3)
    natural_from_global = np.asarray(Qi.to_natural_vector(marker_in_global)).reshape(3)  # global -> natural

    return dict(
        B=B,
        natural=natural,
        natural_marker=np.asarray(segment.marker_from_name("marker").position).reshape(3),
        natural_center_of_mass=np.asarray(segment.natural_center_of_mass).reshape(3),
        rotation=scs.rot,
        origin=np.asarray(scs.translation).reshape(3),
        in_global=marker_in_global,
        center_of_mass_in_global=center_of_mass_in_global,
        natural_from_global=natural_from_global,
    )


def transposed_mistake_error(alpha: float, beta: float, gamma: float, length: float, point: np.ndarray) -> float:
    """Distance between the point p and its reconstruction with inv(B.T) instead of inv(B), in the SCS"""
    B = transformation_matrix_by_hand(alpha, beta, gamma, length)
    return float(np.linalg.norm(B @ np.linalg.solve(B.T, point) - point))


def main(show: bool = True) -> dict:
    with_numpy = frame_changes_with_numpy(POINT)
    with_bionc = frame_changes_with_bionc(POINT, with_numpy["Q"])

    comparisons = [
        ("B (columns u, v, w in the SCS)", with_numpy["B"], with_bionc["B"]),
        ("SCS -> natural, n = inv(B) p", with_numpy["natural"], with_bionc["natural"]),
        ("  natural marker position", with_numpy["natural"], with_bionc["natural_marker"]),
        ("  natural centre of mass", with_numpy["natural"], with_bionc["natural_center_of_mass"]),
        ("SCS rotation, R = [u v w] inv(B)", with_numpy["rotation"], with_bionc["rotation"]),
        ("  equals the pose R0", R0, with_bionc["rotation"]),
        ("SCS origin, rp", RP, with_bionc["origin"]),
        ("natural -> global, rp + [u v w] n", with_numpy["in_global"], with_bionc["in_global"]),
        ("  centre of mass in global", with_numpy["in_global"], with_bionc["center_of_mass_in_global"]),
        ("SCS -> global, rp + R p", with_numpy["in_global_from_scs"], with_bionc["in_global"]),
        ("global -> natural", with_numpy["natural_from_global"], with_bionc["natural_from_global"]),
    ]
    print(f"{'frame change':40s} | max |numpy - bionc|")
    for label, a, b in comparisons:
        print(f"{label:40s} | {np.abs(np.asarray(a) - np.asarray(b)).max():.1e}")

    transposed_error = float(np.linalg.norm(with_numpy["in_global_transposed"] - with_numpy["in_global"]))
    print(f"\nwith inv(B.T) instead of inv(B), the point lands {transposed_error * 1000:.1f} mm away")

    # the error of the transposed mistake as the segment departs from orthogonality
    deviations = np.linspace(0, 30, 61)
    errors = np.array(
        [
            transposed_mistake_error(np.radians(90 - d), np.radians(90 + d / 2), np.radians(90 - d), LENGTH, POINT)
            for d in deviations
        ]
    )

    results = dict(
        natural=with_numpy["natural"],
        max_numpy_bionc_difference=max(float(np.abs(np.asarray(a) - np.asarray(b)).max()) for _, a, b in comparisons),
        error=float(np.linalg.norm(with_numpy["in_global_from_scs"] - with_numpy["in_global"])),
        error_bionc=float(np.linalg.norm(with_bionc["center_of_mass_in_global"] - with_numpy["in_global_from_scs"])),
        error_transposed=transposed_error,
        deviations=deviations,
        errors=errors,
    )

    if show:
        plot(with_numpy, deviations, errors)

    return results


def plot(with_numpy: dict, deviations: np.ndarray, errors: np.ndarray):
    import matplotlib.pyplot as plt

    Q, natural = with_numpy["Q"], with_numpy["natural"]
    u, rp, rd, w = Q[0:3], Q[3:6], Q[6:9], Q[9:12]
    uvw = np.column_stack((u, rp - rd, w))

    fig = plt.figure(figsize=(12, 5.5))
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    for axis, color, label in zip(np.eye(3) * 0.1, ("r", "g", "b"), ("x", "y", "z")):
        ax.quiver(0, 0, 0, *axis, color=color, linewidth=1)
        ax.text(*(axis * 1.2), label, color=color)
    for axis, color, label in zip(with_numpy["rotation"].T * 0.15, ("r", "g", "b"), ("X", "Y", "Z")):
        ax.quiver(*rp, *axis, color=color, linewidth=1, linestyle="dotted")
        ax.text(*(rp + axis * 1.15), label, color=color)
    for vector, color, label in zip(uvw.T, ("tab:red", "tab:green", "tab:blue"), ("u", "v", "w")):
        vector = vector * (0.15 / np.linalg.norm(vector))
        ax.quiver(*rp, *vector, color=color, linewidth=2.5)
        ax.text(*(rp + vector * 1.25), label, color=color, fontweight="bold")
    # path rp -> + n1 u -> + n2 v -> + n3 w that leads to the point
    path = np.array([rp, rp + uvw[:, 0] * natural[0], rp + uvw[:, :2] @ natural[:2], rp + uvw @ natural])
    ax.plot(*path.T, "k--", linewidth=1, label="rp + n1 u + n2 v + n3 w")
    ax.scatter(*with_numpy["in_global_from_scs"], color="k", s=60, label="p given in the SCS: rp + R p")
    ax.scatter(*with_numpy["in_global"], color="tab:green", s=20, marker="x", label="rebuilt with inv(B)")
    ax.scatter(*with_numpy["in_global_transposed"], color="tab:red", s=60, marker="x", label="rebuilt with inv(B.T)")
    ax.set_title("Global frame (x, y, z), SCS (X, Y, Z, dotted)\nand natural frame u, v, w (bold)")
    ax.legend(fontsize=8, loc="upper left")
    for set_lim, c in zip((ax.set_xlim, ax.set_ylim, ax.set_zlim), rp):
        set_lim(c - 0.3, c + 0.3)
    ax.set_box_aspect((1, 1, 1))

    ax_error = fig.add_subplot(1, 2, 2)
    ax_error.plot(deviations, errors * 1000)
    ax_error.set_xlabel("departure from orthogonality d [deg]\n(alpha = 90 - d, beta = 90 + d/2, gamma = 90 - d)")
    ax_error.set_ylabel("misplacement of p with inv(B.T) [mm]")
    ax_error.set_title("Invisible on orthogonal segments (d = 0):\nB is then symmetric, B.T == B")
    ax_error.grid(True)
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
