import os
from pathlib import Path

import numpy as np

from bionc import (
    AxisTemplate,
    BiomechanicalModelTemplate,
    MarkerTemplate,
    SegmentTemplate,
    NaturalSegmentTemplate,
    C3dData,
    BiomechanicalModel,
    JointType,
)
import ezc3d

# from .de_leva import DeLevaTable todo: add this to the example
#
# This examples shows how to
#     1. Create a model from scratch using a template with marker names (model_creation_from_data)


def harrington2007(RASIS: np.ndarray, LASIS: np.ndarray, RPSIS: np.ndarray, LPSIS: np.ndarray) -> tuple:
    """
    This function computes the hip joint center from the RASIS, LASIS, RPSIS and LPSIS markers
    RASIS: RASIS marker

    Parameters
    ----------
    RASIS: np.ndarray
        RASIS marker location in meters
    LASIS: np.ndarray
        LASIS marker location in meters
    RPSIS: np.ndarray
        RPSIS marker location in meters
    LPSIS: np.ndarray
        LPSIS marker location in meters

    Returns
    -------
    tuple(np.ndarray, np.ndarray)
        The right and left hip joint center in global coordinates system in meters
    """
    # convert inputs in millimeters
    rasis = RASIS[:3, :] * 1000
    lasis = LASIS[:3, :] * 1000
    rpsis = RPSIS[:3, :] * 1000
    lpsis = LPSIS[:3, :] * 1000

    # Right-handed Pelvis reference system definition
    Sacrum = (rpsis + lpsis) / 2
    # Global Pelvis center position
    OP = (rasis + lasis) / 2

    rhjc_global = np.zeros((4, rasis.shape[1]))
    lhjc_global = np.zeros((4, rasis.shape[1]))

    for i in range(rasis.shape[1]):
        provv = (rasis[:3, i] - Sacrum[:3, i]) / np.linalg.norm(rasis[:3, i] - Sacrum[:3, i])
        ib = (rasis[:3, i] - lasis[:3, i]) / np.linalg.norm(rasis[:3, i] - lasis[:3, i])

        kb = np.cross(ib, provv) / np.linalg.norm(np.cross(ib, provv))
        jb = np.cross(kb, ib) / np.linalg.norm(np.cross(kb, ib))

        OB = OP[:3, i]
        # Rotation + translation in homogenous matrix
        Pelvis = np.array(
            [[ib[0], jb[0], kb[0], OB[0]], [ib[1], jb[1], kb[1], OB[1]], [ib[2], jb[2], kb[2], OB[2]], [0, 0, 0, 1]]
        )

        # Transformation from global to pelvis reference system
        OPB = np.linalg.inv(Pelvis) @ np.hstack((OB, 1))

        PW = np.linalg.norm(rasis[:3, i] - lasis[:3, i])  # PW: width of pelvis (distance among ASIS)
        PD = np.linalg.norm(
            Sacrum[:3, i] - OP[:3, i]
        )  # PD: pelvis depth = distance between mid points joining PSIS and ASIS

        # Harrington formula in mm
        diff_ap = -0.24 * PD - 9.9
        diff_v = -0.3 * PW - 10.9
        diff_ml = 0.33 * PW + 7.3

        # vector that must be subtract to OP to obtain hjc in pelvis CS
        vett_diff_pelvis_sx = np.array([-diff_ml, diff_ap, diff_v, 1])
        vett_diff_pelvis_dx = np.array([diff_ml, diff_ap, diff_v, 1])

        # hjc in pelvis CS (4x4)
        rhjc_pelvis = OPB[:3] + vett_diff_pelvis_dx[:3]
        lhjc_pelvis = OPB[:3] + vett_diff_pelvis_sx[:3]

        # transformation from pelvis to global CS
        rhjc_global[:3, i] = Pelvis[:3, :3] @ rhjc_pelvis + OB
        lhjc_global[:3, i] = Pelvis[:3, :3] @ lhjc_pelvis + OB

    rhjc_global[:3, :] /= 1000
    lhjc_global[:3, :] /= 1000
    rhjc_global[-1, :] = 1
    lhjc_global[-1, :] = 1

    return rhjc_global, lhjc_global


def model_creation_from_measured_data(c3d_filename: str = "statref.c3d") -> BiomechanicalModel:
    """
    Create a model from a data file and we build the biomechanical model as a template using the marker names
    """

    # Fill the kinematic chain model
    model = BiomechanicalModelTemplate()
    # de_leva = DeLevaTable(total_mass=100, sex="female")

    right_hip_joint = lambda m, bio: harrington2007(m["RFWT"], m["LFWT"], m["RBWT"], m["LBWT"])[0]
    left_hip_joint = lambda m, bio: harrington2007(m["RFWT"], m["LFWT"], m["RBWT"], m["LBWT"])[1]
    right_knee_joint = lambda m, bio: MarkerTemplate.middle_of(m, bio, "RKNI", "RKNE")
    right_ankle_joint = lambda m, bio: MarkerTemplate.middle_of(m, bio, "RANE", "RANI")
    # m['R_HJC'] = right_hip_joint
    # m['L_HJC'] = left_hip_joint

    model["PELVIS"] = SegmentTemplate(
        natural_segment=NaturalSegmentTemplate(
            u_axis=AxisTemplate(
                # from the middle of posterior illiac spine to the middle of anterior illiac spine
                start=lambda m, bio: MarkerTemplate.middle_of(m, bio, "RBWT", "LBWT"),
                end=lambda m, bio: MarkerTemplate.middle_of(m, bio, "RFWT", "LFWT"),
            ),
            # middle of the right and left posterior superior iliac spine
            # or sacroiliac joint
            proximal_point=lambda m, bio: MarkerTemplate.middle_of(m, bio, "RBWT", "LBWT"),
            # Hip joint center projected in the sagittal plane of the pelvis ==> NO
            # middle of the right and left hip joint center
            # todo: how to compute the sagittal plane of the pelvis?
            # We do not need it
            distal_point=lambda m, bio: right_hip_joint(m, bio),
            # normal to the sagittal plane of the pelvis
            # todo
            w_axis=AxisTemplate(start=left_hip_joint, end=right_hip_joint),
        )
    )

    model["THIGH"] = SegmentTemplate(
        natural_segment=NaturalSegmentTemplate(
            u_axis=AxisTemplate(
                start=right_hip_joint,
                # u_axis is defined from the normal of the plane formed by the hip center, the medial epicondyle and the
                # lateral epicondyle
                end=lambda m, bio: MarkerTemplate.normal_to(m, bio, right_hip_joint(m, bio), "RKNE", "RKNI"),
            ),
            proximal_point=right_hip_joint,
            # the knee joint computed from the medial femoral epicondyle and the lateral femoral epicondyle
            distal_point=lambda m, bio: MarkerTemplate.middle_of(m, bio, "RKNE", "RKNI"),
            w_axis=AxisTemplate(start="RKNI", end="RKNE"),
        )
    )

    model["THIGH"].add_marker(MarkerTemplate(name="HIP_CENTER", function=right_hip_joint, parent_name="THIGH"))
    model["THIGH"].add_marker(MarkerTemplate("RKNI", parent_name="THIGH"))
    model["THIGH"].add_marker(MarkerTemplate("RKNE", parent_name="THIGH"))
    model["THIGH"].add_marker(MarkerTemplate("KNEE_JOINT", function=right_knee_joint, parent_name="THIGH"))

    model["SHANK"] = SegmentTemplate(
        natural_segment=NaturalSegmentTemplate(
            u_axis=AxisTemplate(
                start=right_knee_joint,
                # u_axis is defined from the normal of the plane formed by the hip center, the medial epicondyle and the
                # lateral epicondyle
                end=lambda m, bio: MarkerTemplate.normal_to(m, bio, right_knee_joint(m, bio), "RANE", "RANI"),
            ),
            proximal_point=right_knee_joint,
            # the knee joint computed from the medial femoral epicondyle and the lateral femoral epicondyle
            distal_point=lambda m, bio: MarkerTemplate.middle_of(m, bio, "RANE", "RANI"),
            w_axis=AxisTemplate(start="RANE", end="RANI"),
        )
    )
    model["SHANK"].add_marker(MarkerTemplate("KNEE_JOINT", right_knee_joint, parent_name="SHANK"))
    model["SHANK"].add_marker(MarkerTemplate("RANE", parent_name="SHANK"))
    model["SHANK"].add_marker(MarkerTemplate("RANI", parent_name="SHANK"))
    model["SHANK"].add_marker(MarkerTemplate("ANKLE_JOINT", function=right_ankle_joint, parent_name="SHANK"))

    model["FOOT"] = SegmentTemplate(
        natural_segment=NaturalSegmentTemplate(
            u_axis=AxisTemplate(
                start=right_ankle_joint,
                # u_axis is defined from calcaneous (CAL) to the middle of M1 and M5
                end=lambda m, bio: (m["RHEE"] - (m["RTARI"] + m["RTAR"]) / 2)
                / np.linalg.norm(m["RHEE"] - (m["RTARI"] + m["RTAR"]) / 2),
            ),
            proximal_point=right_ankle_joint,
            #  middle of M1 and M5
            distal_point=lambda m, bio: MarkerTemplate.middle_of(m, bio, "RTARI", "RTAR"),
            w_axis=AxisTemplate(start="RTARI", end="RTAR"),
        )
    )

    model["FOOT"].add_marker(MarkerTemplate("RHEE", parent_name="FOOT"))
    model["FOOT"].add_marker(MarkerTemplate("RTARI", parent_name="FOOT"))
    model["FOOT"].add_marker(MarkerTemplate("RTAR", parent_name="FOOT"))
    model["FOOT"].add_marker(MarkerTemplate("ANKLE_JOINT", function=right_ankle_joint, parent_name="FOOT"))

    model.add_joint(
        name="hip",
        joint_type=JointType.SPHERICAL,
        parent="PELVIS",
        child="THIGH",
    )

    model.add_joint(
        name="knee",
        joint_type=JointType.SPHERICAL,
        parent="THIGH",
        child="SHANK",
    )

    model.add_joint(
        name="ankle",
        joint_type=JointType.SPHERICAL,
        parent="SHANK",
        child="FOOT",
    )

    c3d_data = C3dData(f"{c3d_filename}")

    # Put the model together, print it and print it to a bioMod file
    natural_model = model.update(c3d_data)

    return natural_model


def generate_c3d_file():

    # Load an empty c3d structure
    c3d = ezc3d.c3d()

    marker_tuple = ("RFWT", "LFWT", "RBWT", "LBWT", "RKNE", "RKNI", "RANE", "RANI", "RHEE", "RTARI", "RTAR")

    # Fill it with random data
    c3d["parameters"]["POINT"]["RATE"]["value"] = [100]
    c3d["parameters"]["POINT"]["LABELS"]["value"] = marker_tuple

    c3d["data"]["points"] = np.ones((4, len(marker_tuple), 2))
    c3d["data"]["points"][:3, 0, :] = np.array(
        [[0.18416385, 0.19876392], [1.33434277, 1.338479], [0.91699817, 0.91824384]]
    )
    c3d["data"]["points"][:3, 1, :] = np.array(
        [[0.18485233, 0.1985842], [1.1036825, 1.10781494], [0.91453168, 0.91681091]],
    )
    c3d["data"]["points"][:3, 2, :] = np.array(
        [[0.38178949, 0.39600946], [1.28057019, 1.2837561], [0.9454278, 0.94480548]],
    )
    c3d["data"]["points"][:3, 3, :] = np.array(
        [[0.38251419, 0.39637326], [1.18559143, 1.18885852], [0.94143542, 0.94134717]]
    )
    c3d["data"]["points"][:3, 4, :] = np.array(
        [[0.28976505, 0.29850735], [1.40114758, 1.40165784], [0.47594894, 0.47537778]],
    )
    c3d["data"]["points"][:3, 5, :] = np.array(
        [[0.28883856, 0.29766995], [1.29643408, 1.29758105], [0.43036957, 0.43072437]],
    )
    c3d["data"]["points"][:3, 6, :] = np.array(
        [[0.35603992, 0.3566329], [1.44538721, 1.44557263], [0.10275449, 0.10202186]],
    )
    c3d["data"]["points"][:3, 7, :] = np.array(
        [[0.32703876, 0.32869626], [1.3525918, 1.35138013], [0.1117594, 0.11084975]]
    )
    c3d["data"]["points"][:3, 8, :] = np.array(
        [[0.41810855, 0.41600098], [1.3925741, 1.39322546], [0.07911389, 0.07784219]],
    )
    c3d["data"]["points"][:3, 9, :] = np.array(
        [[0.22581064, 0.22588875], [1.36999072, 1.37007214], [0.03077233, 0.03103891]],
    )
    c3d["data"]["points"][:3, 10, :] = np.array(
        [[0.23365552, 0.23372018], [1.49159607, 1.49141943], [0.03238689, 0.03257346]],
    )

    # Write the c3d file
    filename = f"{Path(__file__).parent.resolve()}/statref.c3d"
    c3d.write(filename)

    return filename


def main():
    # create a c3d file with data
    filename = generate_c3d_file()
    # Create the model from a c3d file and markers as template
    model = model_creation_from_measured_data(filename)
    # remove the c3d file
    os.remove(filename)

    model.save("models/lower_limb.nc")


if __name__ == "__main__":
    main()
