import os
from pathlib import Path

import ezc3d
import numpy as np

from pyomeca import Markers

"""
Test the inverse kinematics
"""

from bionc import (
    Axis,
    AxisTemplate,
    AxisFunctionTemplate,
    BiomechanicalModelTemplate,
    MarkerTemplate,
    Marker,
    SegmentTemplate,
    NaturalSegmentTemplate,
    C3dData,
    BiomechanicalModel,
    JointType,
    EulerSequence,
    TransformationMatrixUtil,
    TransformationMatrixType,
    NaturalAxis,
)


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
    # no need as my inputs are already in mm
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
    mid_hjc_global = np.zeros((4, rasis.shape[1]))

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
        mid_hjc_global[:3, i] = (rhjc_global[:3, i] + lhjc_global[:3, i]) / 2

    rhjc_global[:3, :] /= 1000
    lhjc_global[:3, :] /= 1000
    mid_hjc_global[:3, :] /= 1000
    rhjc_global[-1, :] = 1
    lhjc_global[-1, :] = 1
    mid_hjc_global[-1, :] = 1

    return rhjc_global, lhjc_global, mid_hjc_global


def model_creation_markers(c3d_filename: str) -> BiomechanicalModel:
    model = BiomechanicalModelTemplate()

    right_hip_joint = lambda m, bio: harrington2007(m["RIAS"], m["LIAS"], m["RIPS"], m["LIPS"])[0]
    left_hip_joint = lambda m, bio: harrington2007(m["RIAS"], m["LIAS"], m["RIPS"], m["LIPS"])[1]
    mid_hip_joint = lambda m, bio: harrington2007(m["RIAS"], m["LIAS"], m["RIPS"], m["LIPS"])[2]
    right_knee_joint = lambda m, bio: MarkerTemplate.middle_of(m, bio, "RFME", "RFLE")
    right_ankle_joint = lambda m, bio: MarkerTemplate.middle_of(m, bio, "RFAL", "RTAM")
    left_knee_joint = lambda m, bio: MarkerTemplate.middle_of(m, bio, "LFME", "LFLE")
    left_ankle_joint = lambda m, bio: MarkerTemplate.middle_of(m, bio, "LFAL", "LTAM")

    model["PELVIS"] = SegmentTemplate(
        natural_segment=NaturalSegmentTemplate(
            u_axis=AxisTemplate(
                start=lambda m, bio: MarkerTemplate.middle_of(m, bio, "RIPS", "LIPS"),
                end=lambda m, bio: MarkerTemplate.middle_of(m, bio, "RIAS", "LIAS"),
            ),
            proximal_point=lambda m, bio: MarkerTemplate.middle_of(m, bio, "RIPS", "LIPS"),
            distal_point=lambda m, bio: mid_hip_joint(m, bio),
            w_axis=AxisTemplate(start=left_hip_joint, end=right_hip_joint),
        )
    )

    model["PELVIS"].add_marker(MarkerTemplate(name="RIAS", parent_name="PELVIS", is_technical=True))
    model["PELVIS"].add_marker(MarkerTemplate(name="LIAS", parent_name="PELVIS", is_technical=True))
    model["PELVIS"].add_marker(MarkerTemplate(name="RIPS", parent_name="PELVIS", is_technical=True))
    model["PELVIS"].add_marker(MarkerTemplate(name="LIPS", parent_name="PELVIS", is_technical=True))
    model["PELVIS"].add_marker(
        MarkerTemplate(
            name="RIGHT_HIP_JOINT",
            function=right_hip_joint,
            parent_name="PELVIS",
            is_technical=False,
            is_anatomical=True,
        )
    )
    model["PELVIS"].add_marker(
        MarkerTemplate(
            name="LEFT_HIP_JOINT", function=left_hip_joint, parent_name="PELVIS", is_technical=False, is_anatomical=True
        )
    )

    model["RTHIGH"] = SegmentTemplate(
        natural_segment=NaturalSegmentTemplate(
            u_axis=AxisFunctionTemplate(
                function=lambda m, bio: MarkerTemplate.normal_to(m, bio, right_hip_joint(m, bio), "RFLE", "RFME")
            ),
            proximal_point=right_hip_joint,
            distal_point=lambda m, bio: MarkerTemplate.middle_of(m, bio, "RFLE", "RFME"),
            w_axis=AxisTemplate(start="RFME", end="RFLE"),
        )
    )

    model["RTHIGH"].add_marker(
        MarkerTemplate(
            name="RIGHT_HIP_CENTER",
            function=right_hip_joint,
            parent_name="RTHIGH",
            is_technical=False,
            is_anatomical=True,
        )
    )
    model["RTHIGH"].add_marker(MarkerTemplate("RFLE", parent_name="RTHIGH", is_technical=True))
    model["RTHIGH"].add_marker(MarkerTemplate("RFME", parent_name="RTHIGH", is_technical=True))
    model["RTHIGH"].add_marker(
        MarkerTemplate(
            "RIGHT_KNEE_JOINT", function=right_knee_joint, parent_name="RTHIGH", is_technical=False, is_anatomical=True
        )
    )

    model["LTHIGH"] = SegmentTemplate(
        natural_segment=NaturalSegmentTemplate(
            u_axis=AxisFunctionTemplate(
                function=lambda m, bio: MarkerTemplate.normal_to(m, bio, left_hip_joint(m, bio), "LFME", "LFLE")
            ),
            proximal_point=left_hip_joint,
            distal_point=lambda m, bio: MarkerTemplate.middle_of(m, bio, "LFLE", "LFME"),
            w_axis=AxisTemplate(start="LFLE", end="LFME"),
        )
    )

    model["LTHIGH"].add_marker(
        MarkerTemplate(
            name="LEFT_HIP_CENTER",
            function=left_hip_joint,
            parent_name="LTHIGH",
            is_technical=False,
            is_anatomical=True,
        )
    )
    model["LTHIGH"].add_marker(MarkerTemplate("LFLE", parent_name="LTHIGH", is_technical=True))
    model["LTHIGH"].add_marker(MarkerTemplate("LFME", parent_name="LTHIGH", is_technical=True))
    model["LTHIGH"].add_marker(
        MarkerTemplate(
            "LEFT_KNEE_JOINT", function=left_knee_joint, parent_name="LTHIGH", is_technical=False, is_anatomical=True
        )
    )

    model["RSHANK"] = SegmentTemplate(
        natural_segment=NaturalSegmentTemplate(
            u_axis=AxisFunctionTemplate(
                function=lambda m, bio: MarkerTemplate.normal_to(m, bio, right_knee_joint(m, bio), "RFAL", "RTAM")
            ),
            proximal_point=right_knee_joint,
            distal_point=lambda m, bio: MarkerTemplate.middle_of(m, bio, "RFAL", "RTAM"),
            w_axis=AxisTemplate(start="RTAM", end="RFAL"),
        )
    )
    model["RSHANK"].add_marker(
        MarkerTemplate(
            "RIGHT_KNEE_JOINT", right_knee_joint, parent_name="RSHANK", is_technical=False, is_anatomical=True
        )
    )
    model["RSHANK"].add_marker(MarkerTemplate("RFAL", parent_name="RSHANK", is_technical=True))
    model["RSHANK"].add_marker(MarkerTemplate("RTAM", parent_name="RSHANK", is_technical=True))
    model["RSHANK"].add_marker(
        MarkerTemplate(
            "RIGHT_ANKLE_JOINT",
            function=right_ankle_joint,
            parent_name="RSHANK",
            is_technical=False,
            is_anatomical=True,
        )
    )

    model["LSHANK"] = SegmentTemplate(
        natural_segment=NaturalSegmentTemplate(
            u_axis=AxisFunctionTemplate(
                function=lambda m, bio: MarkerTemplate.normal_to(m, bio, left_knee_joint(m, bio), "LTAM", "LFAL")
            ),
            proximal_point=left_knee_joint,
            # the knee joint computed from the medial femoral epicondyle and the lateral femoral epicondyle
            distal_point=lambda m, bio: MarkerTemplate.middle_of(m, bio, "LFAL", "LTAM"),
            w_axis=AxisTemplate(start="LFAL", end="LTAM"),
        )
    )
    model["LSHANK"].add_marker(
        MarkerTemplate("LEFT_KNEE_JOINT", left_knee_joint, parent_name="LSHANK", is_technical=False, is_anatomical=True)
    )
    model["LSHANK"].add_marker(MarkerTemplate("LFAL", parent_name="LSHANK", is_technical=True))
    model["LSHANK"].add_marker(MarkerTemplate("LTAM", parent_name="LSHANK", is_technical=True))
    model["LSHANK"].add_marker(
        MarkerTemplate(
            "LEFT_ANKLE_JOINT", function=left_ankle_joint, parent_name="LSHANK", is_technical=False, is_anatomical=True
        )
    )

    model["RFOOT"] = SegmentTemplate(
        natural_segment=NaturalSegmentTemplate(
            # u_axis is defined from calcaneous (CAL) to the middle of M1 and M5
            u_axis=AxisTemplate(
                start="RFCC",
                end=lambda m, bio: MarkerTemplate.middle_of(m, bio, "RFM1", "RFM5"),
            ),
            proximal_point=right_ankle_joint,
            #  middle of M1 and M5
            distal_point=lambda m, bio: MarkerTemplate.middle_of(m, bio, "RFM1", "RFM5"),
            w_axis=AxisTemplate(start="RFM1", end="RFM5"),
        )
    )

    model["RFOOT"].add_marker(MarkerTemplate("RFCC", parent_name="RFOOT", is_technical=True))
    model["RFOOT"].add_marker(MarkerTemplate("RFM1", parent_name="RFOOT", is_technical=True))
    model["RFOOT"].add_marker(MarkerTemplate("RFM5", parent_name="RFOOT", is_technical=True))
    model["RFOOT"].add_marker(
        MarkerTemplate(
            "RIGHT_ANKLE_JOINT", function=right_ankle_joint, parent_name="RFOOT", is_technical=False, is_anatomical=True
        )
    )

    model["LFOOT"] = SegmentTemplate(
        natural_segment=NaturalSegmentTemplate(
            # u_axis is defined from calcaneous (CAL) to the middle of M1 and M5
            u_axis=AxisTemplate(
                start="LFCC",
                end=lambda m, bio: MarkerTemplate.middle_of(m, bio, "LFM1", "LFM5"),
            ),
            proximal_point=left_ankle_joint,
            #  middle of M1 and M5
            distal_point=lambda m, bio: MarkerTemplate.middle_of(m, bio, "LFM1", "LFM5"),
            w_axis=AxisTemplate(start="LFM5", end="LFM1"),
        )
    )
    #
    model["LFOOT"].add_marker(MarkerTemplate("LFCC", parent_name="LFOOT", is_technical=True))
    model["LFOOT"].add_marker(MarkerTemplate("LFM1", parent_name="LFOOT", is_technical=True))
    model["LFOOT"].add_marker(MarkerTemplate("LFM5", parent_name="LFOOT", is_technical=True))
    model["LFOOT"].add_marker(
        MarkerTemplate(
            "LEFT_ANKLE_JOINT", function=left_ankle_joint, parent_name="LFOOT", is_technical=False, is_anatomical=True
        )
    )

    model.add_joint(
        name="right_hip",
        joint_type=JointType.SPHERICAL,
        parent="PELVIS",
        child="RTHIGH",
        parent_point="RIGHT_HIP_JOINT",
        child_point="RIGHT_HIP_CENTER",
        projection_basis=EulerSequence.ZXY,
        parent_basis=TransformationMatrixType.Bwu,
        child_basis=TransformationMatrixType.Buv,
    )

    model.add_joint(
        name="left_hip",
        joint_type=JointType.SPHERICAL,
        parent="PELVIS",
        child="LTHIGH",
        parent_point="LEFT_HIP_JOINT",
        child_point="LEFT_HIP_CENTER",
        projection_basis=EulerSequence.ZXY,
        parent_basis=TransformationMatrixType.Bwu,
        child_basis=TransformationMatrixType.Buv,
    )

    model.add_joint(
        name="right_knee",
        joint_type=JointType.REVOLUTE,
        parent="RTHIGH",
        child="RSHANK",
        parent_axis=[NaturalAxis.W, NaturalAxis.W],
        child_axis=[NaturalAxis.V, NaturalAxis.U],
        theta=[np.pi / 2, np.pi / 2],
        projection_basis=EulerSequence.ZXY,
        parent_basis=TransformationMatrixType.Bwu,
        child_basis=TransformationMatrixType.Buv,
    )

    model.add_joint(
        name="left_knee",
        joint_type=JointType.REVOLUTE,
        parent="LTHIGH",
        child="LSHANK",
        parent_axis=[NaturalAxis.W, NaturalAxis.W],
        child_axis=[NaturalAxis.V, NaturalAxis.U],
        theta=[np.pi / 2, np.pi / 2],
        projection_basis=EulerSequence.ZXY,
        parent_basis=TransformationMatrixType.Bwu,
        child_basis=TransformationMatrixType.Buv,
    )

    model.add_joint(
        name="right_ankle",
        joint_type=JointType.REVOLUTE,
        parent="RSHANK",
        child="RFOOT",
        parent_axis=[NaturalAxis.W, NaturalAxis.W],
        child_axis=[NaturalAxis.V, NaturalAxis.U],
        theta=[np.pi / 2, np.pi / 2],
        projection_basis=EulerSequence.ZXY,
        parent_basis=TransformationMatrixType.Bwu,
        child_basis=TransformationMatrixType.Buw,
    )

    model.add_joint(
        name="left_ankle",
        joint_type=JointType.REVOLUTE,
        parent="LSHANK",
        child="LFOOT",
        parent_axis=[NaturalAxis.W, NaturalAxis.W],
        child_axis=[NaturalAxis.V, NaturalAxis.U],
        theta=[np.pi / 2, np.pi / 2],
        projection_basis=EulerSequence.ZXY,
        parent_basis=TransformationMatrixType.Bwu,
        child_basis=TransformationMatrixType.Buw,
    )

    c3d_data = C3dData(f"{c3d_filename}")

    # Put the model together, print it and print it to a bioMod file
    natural_model = model.update(c3d_data)

    return natural_model


def generate_c3d_file():
    """
    This function generates a c3d file with full body open pose markerset
    This is made to not  overload the repository with a c3d file

    Returns
    -------
    c3d: ezc3d.c3d
        The c3d file
    """
    # Load an empty c3d structure
    c3d = ezc3d.c3d()

    marker_tuple = (
        "RIAS",
        "LIAS",
        "RIPS",
        "LIPS",
        "RFLE",
        "RFME",
        "LFLE",
        "LFME",
        "RFAL",
        "RTAM",
        "LFAL",
        "LTAM",
        "RFCC",
        "RFM1",
        "RFM5",
        "LFCC",
        "LFM1",
        "LFM5",
    )

    # Fill it with random data
    c3d["parameters"]["POINT"]["RATE"]["value"] = [300]
    c3d["parameters"]["POINT"]["LABELS"]["value"] = marker_tuple
    c3d["parameters"]["POINT"]["UNITS"]["value"] = ["m"]

    c3d["data"]["points"] = np.ones((4, len(marker_tuple), 2))
    c3d["data"]["points"][:3, :, :] = np.array(
        [
            [
                [-0.99512893, -0.99511677],
                [-1.02508819, -1.0245254],
                [-1.1552875, -1.15531087],
                [-1.16930199, -1.169083],
                [-1.07241082, -1.0696162],
                [-1.08410621, -1.08112752],
                [-0.94997555, -0.95119309],
                [-0.96873051, -0.96978855],
                [-1.35851967, -1.35799468],
                [-1.3114084, -1.3112303],
                [-0.97949064, -0.98210907],
                [-0.92955995, -0.93213236],
                [-1.39764822, -1.39714336],
                [-1.23721457, -1.23893678],
                [-1.28200543, -1.28266609],
                [-1.01061773, -1.01343119],
                [-0.8215977, -0.82410777],
                [-0.86654609, -0.86893791],
            ],
            [
                [-0.08428027, -0.08365151],
                [0.16915414, 0.16987239],
                [-0.04029829, -0.03917728],
                [0.0741705, 0.07526213],
                [-0.10409034, -0.10354505],
                [0.00659003, 0.00701484],
                [0.1593492, 0.15984261],
                [0.05167168, 0.05210308],
                [-0.13896838, -0.13885656],
                [-0.07625225, -0.07610498],
                [0.16105932, 0.16058722],
                [0.09675215, 0.0964836],
                [-0.09523564, -0.09531972],
                [-0.09987623, -0.09949642],
                [-0.16837367, -0.167971],
                [0.13030158, 0.12955573],
                [0.11789952, 0.11763644],
                [0.18628153, 0.18631244],
            ],
            [
                [0.92126387, 0.92181492],
                [0.95468545, 0.95495188],
                [0.98916602, 0.98961705],
                [1.00039744, 1.00072575],
                [0.46156183, 0.46162155],
                [0.43770415, 0.43769968],
                [0.48856014, 0.48850757],
                [0.47824231, 0.47793096],
                [0.12470137, 0.12716728],
                [0.10742468, 0.10935797],
                [0.05826186, 0.05896187],
                [0.07246689, 0.07210413],
                [0.12971023, 0.13277037],
                [0.03050142, 0.03068059],
                [0.03568999, 0.03723054],
                [0.0399006, 0.04101478],
                [0.03398312, 0.03370571],
                [0.02054886, 0.02053879],
            ],
        ]
    )

    # Write the c3d file
    filename = f"{Path(__file__).parent.resolve()}/statref_markers.c3d"
    c3d.write(filename)

    return filename


def main():
    # create a c3d file with data
    filename = generate_c3d_file()
    # Create the model from a c3d file and markers as template
    model = model_creation_markers(filename)

    # load experimental markers
    markers_xp = Markers.from_c3d(filename).to_numpy()

    # remove the c3d file
    os.remove(filename)

    # dump the model in a pickle format
    model.save("../models/lower_limb.nc")


if __name__ == "__main__":
    main()
