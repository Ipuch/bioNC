"""
Distal femur view of a knee parallel mechanism, in rerun.

Everything is expressed in the femur segment coordinate system, so that the femur stands still and the tibia moves
on it. For each joint of the model between the femur and the tibia:
- SphereOnPlane: the condyle sphere (center, radius), a slice of the tibial plane around its contact point,
  and the point of the sphere closest to the plane (where they touch when the constraint holds);
- ConstantLength: the ligament as a segment between its femoral and tibial insertions, with its length.
"""

import numpy as np
import rerun as rr

from bionc.bionc_numpy import BiomechanicalModel, NaturalCoordinates
from bionc.bionc_numpy.joints import Joint

SPHERE_COLOR = [200, 200, 230, 90]
PLANE_COLOR = [230, 160, 60, 140]
CONTACT_COLOR = [220, 40, 40]
LIGAMENT_COLORS = [[40, 120, 220], [40, 180, 90], [160, 60, 200], [220, 120, 40], [90, 90, 90]]


def _in_plane_basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Two unit vectors orthogonal to the normal and to each other"""
    helper = np.array([1.0, 0, 0]) if abs(normal[0]) < 0.9 else np.array([0, 1.0, 0])
    e1 = np.cross(normal, helper)
    e1 /= np.linalg.norm(e1)
    return e1, np.cross(normal, e1)


def knee_geometry(model: BiomechanicalModel, Q: NaturalCoordinates, femur: str) -> dict:
    """
    Positions of the mechanism, expressed in the femur segment coordinate system.

    Returns
    -------
    dict with keys "spheres" and "ligaments", lists of dicts (one per joint)
    """
    femur_segment = model.segments[femur]
    femur_frame = femur_segment.segment_coordinates_system(Q.vector(femur_segment.index))
    rotation, origin = femur_frame.rot, femur_frame.translation.reshape(3)

    def to_femur(point_in_global: np.ndarray) -> np.ndarray:
        return rotation.T @ (np.asarray(point_in_global).reshape(3) - origin)

    spheres, ligaments = [], []
    for name, joint in model.joints.items():
        if joint.parent is None or joint.parent.name != femur:
            continue
        Q_parent, Q_child = Q.vector(joint.parent.index), Q.vector(joint.child.index)

        if isinstance(joint, Joint.SphereOnPlane):
            center = to_femur(joint.sphere_center.position_in_global(Q_parent))
            plane_point = to_femur(joint.plane_point.position_in_global(Q_child))
            normal = rotation.T @ np.asarray(joint.plane_normal.position_in_global(Q_child)).reshape(3)
            normal /= np.linalg.norm(normal)
            spheres.append(
                dict(
                    name=name,
                    center=center,
                    radius=joint.sphere_radius,
                    plane_point=plane_point,
                    normal=normal,
                    # point of the sphere closest to the plane, on the plane when the constraint holds
                    contact=center - joint.sphere_radius * normal,
                    # signed distance sphere surface - plane, i.e. the joint constraint
                    gap=float((center - plane_point) @ normal - joint.sphere_radius),
                )
            )
        elif isinstance(joint, Joint.ConstantLength):
            femoral = to_femur(joint.parent_point.position_in_global(Q_parent))
            tibial = to_femur(joint.child_point.position_in_global(Q_child))
            ligaments.append(
                dict(name=name, femoral=femoral, tibial=tibial, length=float(np.linalg.norm(tibial - femoral)))
            )

    return dict(spheres=spheres, ligaments=ligaments)


def log_distal_femur_view(
    model: BiomechanicalModel,
    all_Q: np.ndarray,
    time_steps: np.ndarray,
    femur: str = "THIGH",
    plane_half_size: float = 0.02,
    application_id: str = "bionc_knee_distal_femur",
    save_path: str = None,
):
    """
    Log the distal femur view in rerun, one frame per time step.

    Parameters
    ----------
    model: BiomechanicalModel
        The knee model, with SphereOnPlane and ConstantLength joints between the femur and the tibia
    all_Q: np.ndarray
        Natural coordinates over time [12 * nb_segments x nb_frames]
    time_steps: np.ndarray
        Time of each frame [s]
    femur: str
        Name of the femur segment, the view is expressed in its segment coordinate system
    plane_half_size: float
        Half size of the square slice drawn for each tibial plane [m]
    application_id: str
        The rerun application id
    save_path: str
        If given, the recording is saved to this .rrd file instead of spawning the viewer
    """
    rr.init(application_id)
    if save_path is None:
        rr.spawn()
    else:
        rr.save(save_path)

    rr.log(
        "femur/axes", rr.Arrows3D(vectors=np.eye(3) * 0.03, colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]]), static=True
    )

    geometry = knee_geometry(model, NaturalCoordinates(all_Q[:, 0]), femur)
    # the condyle spheres and the femoral insertions are fixed in the femur frame
    for sphere in geometry["spheres"]:
        rr.log(
            f"femur/condyles/{sphere['name']}",
            rr.Ellipsoids3D(
                centers=[sphere["center"]],
                half_sizes=[[sphere["radius"]] * 3],
                colors=[SPHERE_COLOR],
                fill_mode="solid",
                labels=[sphere["name"]],
            ),
            static=True,
        )

    for i, t in enumerate(time_steps):
        rr.set_time("time", duration=float(t))
        geometry = knee_geometry(model, NaturalCoordinates(all_Q[:, i]), femur)

        for sphere in geometry["spheres"]:
            e1, e2 = _in_plane_basis(sphere["normal"])
            corners = [
                sphere["plane_point"] + plane_half_size * (a * e1 + b * e2)
                for a, b in ((-1, -1), (1, -1), (1, 1), (-1, 1))
            ]
            rr.log(
                f"tibia/planes/{sphere['name']}",
                rr.Mesh3D(
                    vertex_positions=corners,
                    triangle_indices=[[0, 1, 2], [0, 2, 3]],
                    vertex_colors=[PLANE_COLOR] * 4,
                ),
            )
            rr.log(
                f"tibia/contacts/{sphere['name']}",
                rr.Points3D(
                    [sphere["contact"]],
                    radii=0.002,
                    colors=[CONTACT_COLOR],
                    labels=[f"{sphere['name']} gap {sphere['gap'] * 1000:.3f} mm"],
                ),
            )
            rr.log(f"gap_mm/{sphere['name']}", rr.Scalars(sphere["gap"] * 1000))

        for j, ligament in enumerate(geometry["ligaments"]):
            color = LIGAMENT_COLORS[j % len(LIGAMENT_COLORS)]
            rr.log(
                f"ligaments/{ligament['name']}",
                rr.LineStrips3D(
                    [[ligament["femoral"], ligament["tibial"]]],
                    radii=0.0015,
                    colors=[color],
                    labels=[f"{ligament['name']} {ligament['length'] * 1000:.1f} mm"],
                ),
            )
            rr.log(
                f"ligaments/{ligament['name']}/insertions",
                rr.Points3D([ligament["femoral"], ligament["tibial"]], radii=0.0025, colors=[color, color]),
            )
            rr.log(f"ligament_length_mm/{ligament['name']}", rr.Scalars(ligament["length"] * 1000))
