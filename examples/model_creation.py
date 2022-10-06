import os

import numpy as np
import biorbd
from bionc import (
    AxisTemplate,
    BiomechanicalModelTemplate,
    C3dData,
    MarkerTemplate,
    SegmentTemplate,
    NaturalSegmentTemplate,
)
import ezc3d

# from .de_leva import DeLevaTable todo: add this to the example
#
# This examples shows how to
#     1. Create a model from scratch using a template with marker names (model_creation_from_data)

#
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
    RASIS *= 1000
    LASIS *= 1000
    RPSIS *= 1000
    LPSIS *= 1000

    # Right-handed Pelvis reference system definition
    Sacrum = (RPSIS + LPSIS) / 2
    # Global Pelvis center position
    OP = (RASIS + LASIS) / 2

    provv = (RASIS - Sacrum) / np.linalg.norm(RASIS - Sacrum)
    ib = (RASIS - LASIS) / np.linalg.norm(RASIS - LASIS)

    kb = np.cross(ib, provv) / np.linalg.norm(np.cross(ib, provv))
    jb = np.cross(kb, ib) / np.linalg.norm(np.cross(kb, ib))

    OB = OP
    # Rotation + translation in homogenous matrix
    Pelvis = np.array(
        [[ib[0], jb[0], kb[0], OB[0]], [ib[1], jb[1], kb[1], OB[1]], [ib[2], jb[2], kb[2], OB[2]], [0, 0, 0, 1]]
    )

    # Transformation from global to pelvis reference system
    OPB = np.linalg.inv(Pelvis) @ np.array([OB, 1])

    PW = np.linalg.norm(RASIS - LASIS)  # PW: width of pelvis (distance among ASIS)
    PD = np.linalg.norm(Sacrum - OP)  # PD: pelvis depth = distance between mid points joining PSIS and ASIS

    # Harrington formula
    diff_ap = -0.24 * PD - 9.9
    diff_v = -0.3 * PW - 10.9
    diff_ml = 0.33 * PW + 7.3

    # vector that must be subtract to OP to obtain hjc in pelvis CS
    vett_diff_pelvis_sx = np.array([-diff_ml, diff_ap, diff_v, 1])
    vett_diff_pelvis_dx = np.array([diff_ml, diff_ap, diff_v, 1])

    # hjc in pelvis CS (4x4)
    rhjc_pelvis = OPB + vett_diff_pelvis_dx
    lhjc_pelvis = OPB + vett_diff_pelvis_sx

    # transformation from pelvis to global CS
    rhjc_global = Pelvis[:3, :3] @ rhjc_pelvis + OB
    lhjc_global = Pelvis[:3, :3] @ lhjc_pelvis + OB

    return rhjc_global / 1000, lhjc_global / 1000


def model_creation_from_measured_data():
    """
    Create a model from a data file and we build the biomechanical model as a template using the marker names
    """

    # Fill the kinematic chain model
    model = BiomechanicalModelTemplate()
    # de_leva = DeLevaTable(total_mass=100, sex="female")

    model["PELVIS"] = SegmentTemplate(
        natural_segment=NaturalSegmentTemplate(
            u_axis=AxisTemplate(
                # from the middle of posterior illiac spine to the middle of anterior illiac spine
                start=lambda m, bio: MarkerTemplate.middle_of(m, bio, "RPSIS", "LPSIS"),
                end=lambda m, bio: MarkerTemplate.middle_of(m, bio, "RASIS", "LASIS"),
            ),
            proximal_point=lambda m, bio: MarkerTemplate.middle_of(m, bio, "RPSIS", "LPSIS"),
            # Hip joint center projected in the sagittal plane of the pelvis
            # todo: how to compute the sagittal plane of the pelvis?
            distal_point=lambda m, bio: MarkerTemplate.middle_of(m, bio, "LFE", "MFE"),
            # normal to the sagittal plane of the pelvis
            # todo
            w_axis=AxisTemplate(start="MFE", end="LFE"),
        )
    )

    right_hip_joint = lambda m, bio: harrington2007(m["RASIS"], m["LASIS"], m["RPSIS"], m["LPSIS"])[0]

    model["THIGH"] = SegmentTemplate(
        natural_segment=NaturalSegmentTemplate(
            u_axis=AxisTemplate(
                start=right_hip_joint,
                # u_axis is defined from the normal of the plane formed by the hip center, the medial epicondyle and the
                # lateral epicondyle
                end=lambda m, bio: MarkerTemplate.normal_to(m, bio, "HIP_CENTER", "LFE", "MFE"),
            ),
            proximal_point="HIP_CENTER",
            # the knee joint computed from the medial femoral epicondyle and the lateral femoral epicondyle
            distal_point=lambda m, bio: MarkerTemplate.middle_of(m, bio, "LFE", "MFE"),
            w_axis=AxisTemplate(start="MFE", end="LFE"),
        )
    )

    model["THIGH"].add_marker(MarkerTemplate("HIP_CENTER", function=right_hip_joint, parent_name="THIGH"))
    model["THIGH"].add_marker(MarkerTemplate("MFE", parent_name="THIGH"))
    model["THIGH"].add_marker(MarkerTemplate("LFE", parent_name="THIGH"))
    model["THIGH"].add_marker(
        MarkerTemplate(
            "KNEE_JOINT", function=lambda m, bio: MarkerTemplate.middle_of(m, bio, "MFE", "LFE"), parent_name="THIGH"
        )
    )

    model["SHANK"] = SegmentTemplate(
        natural_segment=NaturalSegmentTemplate(
            u_axis=AxisTemplate(
                start="KNEE_CENTER",
                # u_axis is defined from the normal of the plane formed by the hip center, the medial epicondyle and the
                # lateral epicondyle
                end=lambda m, bio: MarkerTemplate.normal_to(m, bio, "KNEE_CENTER", "LM", "MM"),
            ),
            proximal_point="KNEE_CENTER",
            # the knee joint computed from the medial femoral epicondyle and the lateral femoral epicondyle
            distal_point=lambda m, bio: MarkerTemplate.middle_of(m, bio, "LM", "MM"),
            w_axis=AxisTemplate(start="LM", end="MM"),
        )
    )
    model["SHANK"].add_marker(
        MarkerTemplate(
            "KNEE_JOINT", function=lambda m, bio: MarkerTemplate.middle_of(m, bio, "MFE", "LFE"), parent_name="SHANK"
        )
    )
    model["SHANK"].add_marker(MarkerTemplate("LM", parent_name="SHANK"))
    model["SHANK"].add_marker(MarkerTemplate("MM", parent_name="SHANK"))
    model["SHANK"].add_marker(
        MarkerTemplate(
            "ANKLE_JOINT", function=lambda m, bio: MarkerTemplate.middle_of(m, bio, "LM", "MM"), parent_name="SHANK"
        )
    )

    model["FOOT"] = SegmentTemplate(
        natural_segment=NaturalSegmentTemplate(
            u_axis=AxisTemplate(
                start="ANKLE_JOINT",
                # u_axis is defined from calcaneous (CAL) to the middle of M1 and M5
                end=lambda m, bio: (m["CAL"] - (m["M1"] + m["M5"]) / 2)
                / np.linalg.norm(m["CAL"] - (m["M1"] + m["M5"]) / 2),
            ),
            proximal_point="ANKLE_JOINT",
            #  middle of M1 and M5
            distal_point=lambda m, bio: MarkerTemplate.middle_of(m, bio, "M1", "M5"),
            w_axis=AxisTemplate(start="M1", end="M5"),
        )
    )

    model["FOOT"].add_marker(MarkerTemplate("CAL", parent_name="FOOT"))
    model["FOOT"].add_marker(MarkerTemplate("M1", parent_name="FOOT"))
    model["FOOT"].add_marker(MarkerTemplate("M5", parent_name="FOOT"))
    model["FOOT"].add_marker(
        MarkerTemplate(
            "ANKLE_JOINT", function=lambda m, bio: MarkerTemplate.middle_of(m, bio, "LM", "MM"), parent_name="FOOT"
        )
    )

    # Put the model together, print it and print it to a bioMod file
    natural_model = model.update(C3dData("my_file.c3d"))

    return natural_model


def main():
    # Create the model from a data file and markers as template
    model_creation_from_measured_data()


if __name__ == "__main__":
    main()
