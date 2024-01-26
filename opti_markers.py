import os
from pathlib import Path

import ezc3d
import numpy as np
from pyomeca import Markers
from bionc import InverseKinematics, Viz
from bionc.bionc_numpy.enums import InitialGuessModeType

from bionc.utils.export_c3d_from_bionc_model import (
    add_natural_coordinate_to_c3d,
    add_technical_markers_to_c3d,
    get_points_ezc3d,
)


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
    NaturalCoordinates,
    NaturalSegment,
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

def model_creation(c3d_filename: str) -> BiomechanicalModel:
    model = BiomechanicalModelTemplate()

    right_hip_joint = lambda m, bio: harrington2007(m["RIAS"], m["LIAS"], m["RIPS"], m["LIPS"])[0]
    left_hip_joint = lambda m, bio: harrington2007(m["RIAS"], m["LIAS"], m["RIPS"], m["LIPS"])[1]
    mid_hip_joint = lambda m, bio: harrington2007(m["RIAS"], m["LIAS"], m["RIPS"], m["LIPS"])[2]
    right_knee_joint = lambda m, bio: MarkerTemplate.middle_of(m, bio, "RFME", "RFLE")
    right_ankle_joint = lambda m, bio: MarkerTemplate.middle_of(m, bio, "RFAL", "RTAM")
    left_knee_joint = lambda m, bio: MarkerTemplate.middle_of(m, bio, "LFME", "LFLE")
    left_ankle_joint = lambda m, bio: MarkerTemplate.middle_of(m, bio, "LFAL", "LTAM")

    
    
    # vpelvis = lambda m, bio: Axis(
    #     start=Marker(name="midHip", position=mid_hip_joint(m, bio)), end=Marker(name="TV12", position=m["TV12"])
    # ).axis()


    # modèle basé sur la table de Raphaël et ce qu'on ferait classiquement avec des marqueurs
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

    # modèle basé sur le markerless
    # midSAT = lambda m, bio: MarkerTemplate.middle_of(m, bio, "RSAT", "LSAT") # milieu des acromions
    # vpelvis = lambda m, bio: Axis(
    #     start=Marker(name="midHip", position=mid_hip_joint(m, bio)),  end=Marker(name="midSAT", position=midSAT(m,bio))
    # ).axis()
    # wpelvis = lambda m, bio: Axis(
    #     start=Marker(name="leftHip", position=left_hip_joint(m, bio)), end=Marker(name="rightHip", position=right_hip_joint(m,bio))
    # ).axis()
    # model["PELVIS"] = SegmentTemplate(
    #     natural_segment=NaturalSegmentTemplate(
    #         u_axis=AxisFunctionTemplate(
    #             function=lambda m, bio: AxisTemplate.normalized_cross_product(m, bio, vpelvis(m, bio), wpelvis(m, bio))
    #         ),
    #         proximal_point=lambda m, bio: MarkerTemplate.middle_of(m, bio, "RIPS", "LIPS"),
    #         distal_point=lambda m, bio: mid_hip_joint(m, bio),

    #         w_axis=AxisTemplate(start=left_hip_joint, end=right_hip_joint),
    #     )
    # )


    model["PELVIS"].add_marker(MarkerTemplate(name="RSAT", parent_name="PELVIS", is_technical=True))
    model["PELVIS"].add_marker(MarkerTemplate(name="LSAT", parent_name="PELVIS", is_technical=True))
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
            # the knee joint computed from the medial femoral epicondyle and the lateral femoral epicondyle
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
            # the knee joint computed from the medial femoral epicondyle and the lateral femoral epicondyle
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
            # the knee joint computed from the medial femoral epicondyle and the lateral femoral epicondyle
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
        child_point = "RIGHT_HIP_CENTER",
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
        child_point = "LEFT_HIP_CENTER",
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

def generate_c3d_angles(angles, name_file_export):
    c3d = ezc3d.c3d()
    angles_name = [
        "pelvis",
        # "torso",
        "r_hip",
        "l_hip",
        "r_knee",
        "l_knee",
        "r_ankle",
        "l_ankle",
        # "r_shoulder",
        # "l_shoulder",
        # "r_elbow",
        # "l_elbow",
    ]
    size = angles.shape[2]
    nb_joints = angles.shape[1]

    anglesc3d = np.ones([4, len(angles_name), size])
    for i in range(3):
        for j in range(len(angles_name)):
            anglesc3d[i, j, :] = angles[i, j, :]

    c3d["data"]["points"] = anglesc3d
    c3d["parameters"]["POINT"]["RATE"]["value"] = [60]
    c3d["parameters"]["POINT"]["LABELS"]["value"] = angles_name
    c3d.write(name_file_export)

def main():
    num_mouv = 6
    mouv = ['bend', 'cmjs', 'lufe', 'luyo', 'stai', 'stsk', 'walk']
    filename =  "D:/Users/chaumeil/these/DATA_ESB/"+ mouv[num_mouv]+ "/subject05_"+mouv[num_mouv]+"_m.c3d"
    c3d_data = ezc3d.c3d(filename)
    nb_frames = c3d_data["data"]["points"].shape[2]
    
    model = model_creation(filename)
    markers_xp = Markers.from_c3d(filename, usecols=model.marker_names_technical).to_numpy()
    Q_initialize = model.Q_from_markers(markers_xp[:, :, :])

    ik_solver = InverseKinematics(
        model,
        experimental_markers=markers_xp[0:3, :, :],
        solve_frame_per_frame=True,
        active_direct_frame_constraints=True,
    )
    Q_sol = ik_solver.solve(Q_init=Q_initialize, initial_guess_mode=InitialGuessModeType.USER_PROVIDED_FIRST_FRAME_ONLY, method="ipopt")
    angles = np.zeros((3, 7, nb_frames))
    for i in range(nb_frames):
        angles[:, :, i] = (
            model.natural_coordinates_to_joint_angles(NaturalCoordinates(Q_sol[:, i])) * 180 / (np.pi)
        )

    generate_c3d_angles(angles, "D:/Users/chaumeil/these/DATA_ESB/"+ mouv[num_mouv]+ "/angles_markers.c3d")

    acq = ezc3d.c3d(filename)
    add_natural_coordinate_to_c3d(acq, model, Q_sol)
    acq.write("D:/Users/chaumeil/these/DATA_ESB/"+ mouv[num_mouv]+ "/Q_markers.c3d")



if __name__ == "__main__":
    main()