import os
from pathlib import Path

import ezc3d
import numpy as np
from pyomeca import Markers
from bionc import InverseKinematics, Viz
from bionc.bionc_numpy.enums import InitialGuessModeType
# from bionc.bionc_numpy.enums import InitialGuessModeType

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

# from bionc import *


def model_creation(c3d_filename: str, is_static: bool, is_init: bool) -> BiomechanicalModel:
    model = BiomechanicalModelTemplate()

    ######### Right and Left Foot #########

    model["RFOOT"] = SegmentTemplate(
        natural_segment=NaturalSegmentTemplate(
            u_axis=AxisTemplate(start="right_heel", end="midRToes"),
            proximal_point=lambda m, bio: m["right_ankle"],
            distal_point=lambda m, bio: m["midRToes"],
            w_axis=AxisFunctionTemplate(
                function=lambda m, bio: MarkerTemplate.normal_to(m, bio, "right_ankle", "midRToes", "right_knee")
            ),
        )
    )
    model["RFOOT"].add_marker(
        MarkerTemplate(name="right_ankle", parent_name="RFOOT", is_technical=True, is_anatomical=False)
    )
    model["RFOOT"].add_marker(
        MarkerTemplate(name="right_heel", parent_name="RFOOT", is_technical=True, is_anatomical=False)
    )

    model["RFOOT"].add_marker(
        MarkerTemplate(name="midRToes", parent_name="RFOOT", is_technical=True, is_anatomical=False)
    )

    model["LFOOT"] = SegmentTemplate(
        natural_segment=NaturalSegmentTemplate(
            u_axis=AxisTemplate(start="left_heel", end="midLToes"),
            proximal_point=lambda m, bio: m["left_ankle"],
            distal_point=lambda m, bio: m["midLToes"],
            w_axis=AxisFunctionTemplate(
                function=lambda m, bio: MarkerTemplate.normal_to(m, bio, "left_ankle", "midLToes", "left_knee")
            ),
        )
    )
    model["LFOOT"].add_marker(
        MarkerTemplate(name="left_ankle", parent_name="LFOOT", is_technical=True, is_anatomical=False)
    )
    model["LFOOT"].add_marker(
        MarkerTemplate(name="left_heel", parent_name="LFOOT", is_technical=True, is_anatomical=False)
    )
    model["LFOOT"].add_marker(
        MarkerTemplate(name="midLToes", parent_name="LFOOT", is_technical=True, is_anatomical=False)
    )

    ######## Right and Left Shank #########

    wrshank = lambda m, bio: MarkerTemplate.normal_to(m, bio, "right_ankle", "midRToes", "right_knee")
    vrshank = lambda m, bio: Axis(
        start=Marker(name="right_ankle", position=m["right_ankle"]),
        end=Marker(name="right_knee", position=m["right_knee"]),
    ).axis()

    model["RSHANK"] = SegmentTemplate(
        natural_segment=NaturalSegmentTemplate(
            u_axis=AxisFunctionTemplate(
                function=lambda m, bio: AxisTemplate.normalized_cross_product(m, bio, vrshank(m, bio), wrshank(m, bio))
            ),
            proximal_point=lambda m, bio: m["right_knee"],
            distal_point=lambda m, bio: m["right_ankle"],
            w_axis=AxisFunctionTemplate(function=wrshank),
        )
    )
    model["RSHANK"].add_marker(
        MarkerTemplate(name="right_knee", parent_name="RSHANK", is_technical=True, is_anatomical=False)
    )
    model["RSHANK"].add_marker(
        MarkerTemplate(name="right_ankle", parent_name="RSHANK", is_technical=True, is_anatomical=False)
    )

    # model["RSHANK"].beta = 85 * np.pi / 180

    vlshank = lambda m, bio: Axis(
        start=Marker(name="left_ankle", position=m["left_ankle"]), end=Marker(name="left_knee", position=m["left_knee"])
    ).axis()

    wlshank = lambda m, bio: MarkerTemplate.normal_to(m, bio, "left_ankle", "midLToes", "left_knee")

    model["LSHANK"] = SegmentTemplate(
        natural_segment=NaturalSegmentTemplate(
            u_axis=AxisFunctionTemplate(
                function=lambda m, bio: AxisTemplate.normalized_cross_product(m, bio, vlshank(m, bio), wlshank(m, bio))
            ),
            proximal_point=lambda m, bio: m["left_knee"],
            distal_point=lambda m, bio: m["left_ankle"],
            w_axis=AxisFunctionTemplate(function=wlshank),
        )
    )
    model["LSHANK"].add_marker(
        MarkerTemplate(name="left_knee", parent_name="LSHANK", is_technical=True, is_anatomical=False)
    )
    model["LSHANK"].add_marker(
        MarkerTemplate(name="left_ankle", parent_name="LSHANK", is_technical=True, is_anatomical=False)
    )
    # model["LSHANK"].beta = 85 * np.pi / 180

    ############## Right and Left Thigh ##############

    vrthigh = lambda m, bio: Axis(
        start=Marker(name="right_knee", position=m["right_knee"]), end=Marker(name="right_hip", position=m["right_hip"])
    ).axis()
    wrthigh = lambda m, bio: MarkerTemplate.normal_to(m, bio, "right_knee", "right_hip", "right_ankle")

    if is_init == True:
        model["RTHIGH"] = SegmentTemplate(
            natural_segment=NaturalSegmentTemplate(
                u_axis=AxisFunctionTemplate(
                    function=lambda m, bio: AxisTemplate.normalized_cross_product(
                        m, bio, vrthigh(m, bio), wrshank(m, bio)
                    )
                ),
                proximal_point=lambda m, bio: m["right_hip"],
                distal_point=lambda m, bio: m["right_knee"],
                w_axis=AxisFunctionTemplate(function=wrshank),
            )
        )
    else:
        model["RTHIGH"] = SegmentTemplate(
            natural_segment=NaturalSegmentTemplate(
                u_axis=AxisFunctionTemplate(
                    function=lambda m, bio: AxisTemplate.normalized_cross_product(
                        m, bio, vrthigh(m, bio), wrthigh(m, bio)
                    )
                ),
                proximal_point=lambda m, bio: m["right_hip"],
                distal_point=lambda m, bio: m["right_knee"],
                w_axis=AxisFunctionTemplate(function=wrthigh),
            )
        )

    model["RTHIGH"].add_marker(
        MarkerTemplate(name="right_knee", parent_name="RTHIGH", is_technical=True, is_anatomical=False)
    )
    model["RTHIGH"].add_marker(
        MarkerTemplate(name="right_hip", parent_name="RTHIGH", is_technical=True, is_anatomical=False)
    )

    # model["RTHIGH"].alpha = 95*np.pi/180
    # model["RTHIGH"].beta = 95 * np.pi / 180
    # model["RTHIGH"].gamma = 95*np.pi/180

    vlthigh = lambda m, bio: Axis(
        start=Marker(name="left_knee", position=m["left_knee"]), end=Marker(name="left_hip", position=m["left_hip"])
    ).axis()
    wlthigh = lambda m, bio: MarkerTemplate.normal_to(m, bio, "left_knee", "left_hip", "left_ankle")

    if is_init == True:
        model["LTHIGH"] = SegmentTemplate(
            natural_segment=NaturalSegmentTemplate(
                u_axis=AxisFunctionTemplate(
                    function=lambda m, bio: AxisTemplate.normalized_cross_product(
                        m, bio, vlthigh(m, bio), wlshank(m, bio)
                    )
                ),
                proximal_point=lambda m, bio: m["left_hip"],
                distal_point=lambda m, bio: m["left_knee"],
                w_axis=AxisFunctionTemplate(function=wlshank),
            )
        )
    else:
        model["LTHIGH"] = SegmentTemplate(
            natural_segment=NaturalSegmentTemplate(
                u_axis=AxisFunctionTemplate(
                    function=lambda m, bio: AxisTemplate.normalized_cross_product(
                        m, bio, vlthigh(m, bio), wlthigh(m, bio)
                    )
                ),
                proximal_point=lambda m, bio: m["left_hip"],
                distal_point=lambda m, bio: m["left_knee"],
                w_axis=AxisFunctionTemplate(function=wlthigh),
            )
        )

    model["LTHIGH"].add_marker(
        MarkerTemplate(name="left_knee", parent_name="LTHIGH", is_technical=True, is_anatomical=False)
    )
    model["LTHIGH"].add_marker(
        MarkerTemplate(name="left_hip", parent_name="LTHIGH", is_technical=True, is_anatomical=False)
    )

    # model["LTHIGH"].alpha = 100*np.pi/180
    # model["LTHIGH"].beta = 95 * np.pi / 180
    # model["LTHIGH"].gamma = 100*np.pi/180

    ################## Pelvis ###########################
    midHip = lambda m, bio: MarkerTemplate.middle_of(m, bio, "right_hip", "left_hip")
    vpelvis = lambda m, bio: Axis(
        start=Marker(name="midHip", position=midHip(m, bio)), end=Marker(name="chest", position=m["chest"])
    ).axis()
    wpelvis = lambda m, bio: Axis(
        start=Marker(name="left_hip", position=m["left_hip"]), end=Marker(name="right_hip", position=m["right_hip"])
    ).axis()

    model["PELVIS"] = SegmentTemplate(
        natural_segment=NaturalSegmentTemplate(
            u_axis=AxisFunctionTemplate(
                function=lambda m, bio: AxisTemplate.normalized_cross_product(m, bio, vpelvis(m, bio), wpelvis(m, bio))
            ),
            proximal_point=lambda m, bio: m["chest"], # point médian entre les deux hanches et les deux épaules
            distal_point=lambda m, bio: MarkerTemplate.middle_of(m, bio, "right_hip", "left_hip"),
            w_axis=AxisTemplate(start="left_hip", end="right_hip"),
        )
    )
    model["PELVIS"].add_marker(
        MarkerTemplate(name="right_hip", parent_name="PELVIS", is_technical=True, is_anatomical=False)
    )
    model["PELVIS"].add_marker(
        MarkerTemplate(name="left_hip", parent_name="PELVIS", is_technical=True, is_anatomical=False)
    )
    model["PELVIS"].add_marker(
        MarkerTemplate(name="chest", parent_name="PELVIS", is_technical=True, is_anatomical=False)
    )

    # ################ TORSO #####################

    vtorso = lambda m, bio: Axis(
        start=Marker(name="chest", position=m["chest"]), end=Marker(name="neck", position=m["neck"])
    ).axis()
    wtorso = lambda m, bio: Axis(
        start=Marker(name="left_shoulder", position=m["left_shoulder"]),
        end=Marker(name="right_shoulder", position=m["right_shoulder"]),
    ).axis()

    model["TORSO"] = SegmentTemplate(
        natural_segment=NaturalSegmentTemplate(
            u_axis=AxisFunctionTemplate(
                function=lambda m, bio: AxisTemplate.normalized_cross_product(m, bio, vtorso(m, bio), wtorso(m, bio))
            ),
            proximal_point=lambda m, bio: m["neck"],
            distal_point=lambda m, bio: m["chest"],
            w_axis=AxisTemplate(start="left_shoulder", end="right_shoulder"),
        )
    )

    model["TORSO"].add_marker(
        MarkerTemplate(name="right_shoulder", parent_name="TORSO", is_technical=True, is_anatomical=False)
    )
    model["TORSO"].add_marker(
        MarkerTemplate(name="left_shoulder", parent_name="TORSO", is_technical=True, is_anatomical=False)
    )
    model["TORSO"].add_marker(MarkerTemplate(name="chest", parent_name="TORSO", is_technical=True))
    model["TORSO"].add_marker(MarkerTemplate(name="neck", parent_name="TORSO", is_technical=True))

    # model["TORSO"].beta = 85*np.pi/180

    # # ################ Head ##################

    # vhead = lambda m, bio: Axis(start=Marker(name='neck',position=m["neck"]), end=Marker(name='top_head',position=m["top_head"])).axis()
    # whead = lambda m, bio: Axis(start=Marker(name='left_ear',position=m["left_ear"]), end=Marker(name='right_ear',position=m["right_ear"])).axis()

    # model["HEAD"] = SegmentTemplate(
    #     natural_segment= NaturalSegmentTemplate(
    #         u_axis = AxisFunctionTemplate(
    #             function = lambda m, bio: AxisTemplate.normalized_cross_product(m, bio, vhead(m, bio), whead(m, bio))
    #         ),
    #         proximal_point= lambda m, bio: m["top_head"],
    #         distal_point= lambda m, bio: m["neck"],
    #         w_axis=AxisTemplate(start = "left_ear", end="right_ear"),
    #     )
    # )

    # model["HEAD"].add_marker(MarkerTemplate(name="right_ear", parent_name="HEAD", is_technical=True, is_anatomical=False))
    # model["HEAD"].add_marker(MarkerTemplate(name="left_ear", parent_name="HEAD", is_technical=True, is_anatomical=False))
    # model["HEAD"].add_marker(MarkerTemplate(name="right_eye", parent_name="HEAD", is_technical=True, is_anatomical=False))
    # model["HEAD"].add_marker(MarkerTemplate(name="left_eye", parent_name="HEAD", is_technical=True, is_anatomical=False))
    # model["HEAD"].add_marker(MarkerTemplate(name="top_head", parent_name="HEAD", is_technical=True, is_anatomical=False))
    # model["HEAD"].add_marker(MarkerTemplate(name="neck", parent_name="HEAD", is_technical=True))

    # # ################## Right and Left Upper arm and Forearm ######################

    vruarm = lambda m, bio: Axis(
        start=Marker(name="right_elbow", position=m["right_elbow"]),
        end=Marker(name="right_shoulder", position=m["right_shoulder"]),
    ).axis()
    vrforearm = lambda m, bio: Axis(
        start=Marker(name="right_wrist", position=m["right_wrist"]),
        end=Marker(name="right_elbow", position=m["right_elbow"]),
    ).axis()

    wruarm = lambda m, bio: AxisTemplate.normalized_cross_product(m, bio, vruarm(m, bio), vrforearm(m, bio))
    wrforearm = lambda m, bio: AxisTemplate.normalized_cross_product(m, bio, vruarm(m, bio), vrforearm(m, bio))

    uruarm = lambda m, bio: AxisTemplate.normalized_cross_product(m, bio, vruarm(m, bio), wtorso(m, bio))

    if is_init == True:
        model["RUARM"] = SegmentTemplate(
            natural_segment=NaturalSegmentTemplate(
                u_axis=AxisFunctionTemplate(
                    function=lambda m, bio: AxisTemplate.normalized_cross_product(
                        m, bio, vruarm(m, bio), wtorso(m, bio)
                    )
                ),
                proximal_point=lambda m, bio: m["right_shoulder"],
                distal_point=lambda m, bio: m["right_elbow"],
                w_axis=AxisFunctionTemplate(
                    function=lambda m, bio: AxisTemplate.normalized_cross_product(
                        m, bio, uruarm(m, bio), vruarm(m, bio)
                    )
                ),
            )
        )
    else:
        model["RUARM"] = SegmentTemplate(  # version 1 du modèle
            natural_segment=NaturalSegmentTemplate(
                u_axis=AxisFunctionTemplate(
                    function=lambda m, bio: AxisTemplate.normalized_cross_product(
                        m, bio, vruarm(m, bio), wruarm(m, bio)
                    )
                ),
                proximal_point=lambda m, bio: m["right_shoulder"],
                distal_point=lambda m, bio: m["right_elbow"],
                w_axis=AxisFunctionTemplate(
                    function=lambda m, bio: AxisTemplate.normalized_cross_product(
                        m, bio, vruarm(m, bio), vrforearm(m, bio)
                    )
                ),
            )
        )

    model["RUARM"].add_marker(
        MarkerTemplate(name="right_shoulder", parent_name="RUARM", is_technical=True, is_anatomical=False)
    )
    model["RUARM"].add_marker(
        MarkerTemplate(name="right_elbow", parent_name="RUARM", is_technical=True, is_anatomical=False)
    )

    # model["RUARM"].beta = 95*np.pi/180

    if is_init == True:
        model["RFOREARM"] = SegmentTemplate(
            natural_segment=NaturalSegmentTemplate(
                u_axis=AxisFunctionTemplate(
                    function=lambda m, bio: AxisTemplate.normalized_cross_product(
                        m, bio, vrforearm(m, bio), wrforearm(m, bio)
                    )
                ),
                proximal_point=lambda m, bio: m["right_elbow"],
                distal_point=lambda m, bio: m["right_wrist"],
                w_axis=AxisFunctionTemplate(
                    function=lambda m, bio: AxisTemplate.normalized_cross_product(
                        m, bio, uruarm(m, bio), vruarm(m, bio)
                    )
                ),
            )
        )
    else:
        model["RFOREARM"] = SegmentTemplate(
            natural_segment=NaturalSegmentTemplate(
                u_axis=AxisFunctionTemplate(
                    function=lambda m, bio: AxisTemplate.normalized_cross_product(
                        m, bio, vrforearm(m, bio), wrforearm(m, bio)
                    )
                ),
                proximal_point=lambda m, bio: m["right_elbow"],
                distal_point=lambda m, bio: m["right_wrist"],
                w_axis=AxisFunctionTemplate(
                    function=lambda m, bio: AxisTemplate.normalized_cross_product(
                        m, bio, vruarm(m, bio), vrforearm(m, bio)
                    )
                ),
            )
        )

    model["RFOREARM"].add_marker(
        MarkerTemplate(name="right_elbow", parent_name="RFOREARM", is_technical=True, is_anatomical=False)
    )
    model["RFOREARM"].add_marker(
        MarkerTemplate(name="right_wrist", parent_name="RFOREARM", is_technical=True, is_anatomical=False)
    )

    vluarm = lambda m, bio: Axis(
        start=Marker(name="left_elbow", position=m["left_elbow"]),
        end=Marker(name="left_shoulder", position=m["left_shoulder"]),
    ).axis()
    vlforearm = lambda m, bio: Axis(
        start=Marker(name="left_wrist", position=m["left_wrist"]),
        end=Marker(name="left_elbow", position=m["left_elbow"]),
    ).axis()

    wluarm = lambda m, bio: AxisTemplate.normalized_cross_product(m, bio, vluarm(m, bio), vlforearm(m, bio))
    wlforearm = lambda m, bio: AxisTemplate.normalized_cross_product(m, bio, vluarm(m, bio), vlforearm(m, bio))

    uluarm = lambda m, bio: AxisTemplate.normalized_cross_product(m, bio, vluarm(m, bio), wtorso(m, bio))

    if is_init == True:
        model["LUARM"] = SegmentTemplate(
            natural_segment=NaturalSegmentTemplate(
                u_axis=AxisFunctionTemplate(
                    function=lambda m, bio: AxisTemplate.normalized_cross_product(
                        m, bio, vluarm(m, bio), wtorso(m, bio)
                    )
                ),
                proximal_point=lambda m, bio: m["left_shoulder"],
                distal_point=lambda m, bio: m["left_elbow"],
                w_axis=AxisFunctionTemplate(
                    function=lambda m, bio: AxisTemplate.normalized_cross_product(
                        m, bio, uluarm(m, bio), vluarm(m, bio)
                    )
                ),
            )
        )
    else:
        model["LUARM"] = SegmentTemplate(
            natural_segment=NaturalSegmentTemplate(
                u_axis=AxisFunctionTemplate(
                    function=lambda m, bio: AxisTemplate.normalized_cross_product(
                        m, bio, vluarm(m, bio), wluarm(m, bio)
                    )
                ),
                proximal_point=lambda m, bio: m["left_shoulder"],
                distal_point=lambda m, bio: m["left_elbow"],
                w_axis=AxisFunctionTemplate(
                    function=lambda m, bio: AxisTemplate.normalized_cross_product(
                        m, bio, vluarm(m, bio), vlforearm(m, bio)
                    )
                ),
            )
        )

    model["LUARM"].add_marker(
        MarkerTemplate(name="left_shoulder", parent_name="LUARM", is_technical=True, is_anatomical=False)
    )
    model["LUARM"].add_marker(
        MarkerTemplate(name="left_elbow", parent_name="LUARM", is_technical=True, is_anatomical=False)
    )

    # model["LUARM"].beta = 95*np.pi/180

    if is_init == True:
        model["LFOREARM"] = SegmentTemplate(
            natural_segment=NaturalSegmentTemplate(
                u_axis=AxisFunctionTemplate(
                    function=lambda m, bio: AxisTemplate.normalized_cross_product(
                        m, bio, vlforearm(m, bio), wlforearm(m, bio)
                    )
                ),
                proximal_point=lambda m, bio: m["left_elbow"],
                distal_point=lambda m, bio: m["left_wrist"],
                w_axis=AxisFunctionTemplate(
                    function=lambda m, bio: AxisTemplate.normalized_cross_product(
                        m, bio, uluarm(m, bio), vluarm(m, bio)
                    )
                ),
            )
        )
    else:
        model["LFOREARM"] = SegmentTemplate(
            natural_segment=NaturalSegmentTemplate(
                u_axis=AxisFunctionTemplate(
                    function=lambda m, bio: AxisTemplate.normalized_cross_product(
                        m, bio, vlforearm(m, bio), wlforearm(m, bio)
                    )
                ),
                proximal_point=lambda m, bio: m["left_elbow"],
                distal_point=lambda m, bio: m["left_wrist"],
                w_axis=AxisFunctionTemplate(
                    function=lambda m, bio: AxisTemplate.normalized_cross_product(
                        m, bio, vluarm(m, bio), vlforearm(m, bio)
                    )
                ),
            )
        )

    model["LFOREARM"].add_marker(
        MarkerTemplate(name="left_elbow", parent_name="LFOREARM", is_technical=True, is_anatomical=False)
    )
    model["LFOREARM"].add_marker(
        MarkerTemplate(name="left_wrist", parent_name="LFOREARM", is_technical=True, is_anatomical=False)
    )

    # ################### Model joints #####################

    model.add_joint(
        name="rhip",
        joint_type=JointType.SPHERICAL,
        parent="PELVIS",
        child="RTHIGH",
        parent_point="right_hip",
        child_point="right_hip",
        projection_basis=EulerSequence.ZXY,
        parent_basis=TransformationMatrixType.Bwu,
        child_basis=TransformationMatrixType.Buv,
    )

    model.add_joint(
        name="lhip",
        joint_type=JointType.SPHERICAL,
        parent="PELVIS",
        child="LTHIGH",
        parent_point="left_hip",
        child_point="left_hip",
        projection_basis=EulerSequence.ZXY,
        parent_basis=TransformationMatrixType.Bwu,
        child_basis=TransformationMatrixType.Buv,
    )

    model.add_joint(
        name="rknee",
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
        name="lknee",
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
        name="rankle",
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
        name="lankle",
        joint_type=JointType.REVOLUTE,
        parent="LSHANK",
        child="LFOOT",
        parent_axis=[NaturalAxis.W, NaturalAxis.W],
        child_axis=[NaturalAxis.V, NaturalAxis.U],
        theta=[np.pi / 2, np.pi / 2],
        projection_basis=EulerSequence.ZXY,
        parent_basis=TransformationMatrixUtil(
            plane=(NaturalAxis.W, NaturalAxis.U),
            axis_to_keep=NaturalAxis.W,
        ).to_enum(),
        child_basis=TransformationMatrixType.Buw,
    )

    model.add_joint(
        name="rshoulder",
        joint_type=JointType.SPHERICAL,
        parent="TORSO",
        child="RUARM",
        parent_point="right_shoulder",
        child_point="right_shoulder",
        projection_basis=EulerSequence.ZXY,
        parent_basis=TransformationMatrixType.Bwu,
        child_basis=TransformationMatrixType.Buv,
    )

    model.add_joint(
        name="lshoulder",
        joint_type=JointType.SPHERICAL,
        parent="TORSO",
        child="LUARM",
        parent_point="left_shoulder",
        child_point="left_shoulder",
        projection_basis=EulerSequence.ZXY,
        parent_basis=TransformationMatrixType.Bwu,
        child_basis=TransformationMatrixType.Buv,
    )

    # model.add_joint(
    #     name="neck",
    #     joint_type=JointType.SPHERICAL,
    #     parent="TORSO",
    #     child="HEAD",
    #     parent_point="neck",
    #     child_point="neck",
    #     projection_basis=EulerSequence.ZXY,
    #     parent_basis=TransformationMatrixType.Bwu,
    #     child_basis=TransformationMatrixType.Bwu,
    # )

    model.add_joint(
        name="relbow",
        joint_type=JointType.REVOLUTE,
        parent="RUARM",
        child="RFOREARM",
        parent_axis=[NaturalAxis.W, NaturalAxis.W],
        child_axis=[NaturalAxis.V, NaturalAxis.U],
        theta=[np.pi / 2, np.pi / 2],
        projection_basis=EulerSequence.ZXY,
        parent_basis=TransformationMatrixType.Bwu,
        child_basis=TransformationMatrixType.Buv,
    )

    model.add_joint(
        name="lelbow",
        joint_type=JointType.REVOLUTE,
        parent="LUARM",
        child="LFOREARM",
        parent_axis=[NaturalAxis.W, NaturalAxis.W],
        child_axis=[NaturalAxis.V, NaturalAxis.U],
        theta=[np.pi / 2, np.pi / 2],
        projection_basis=EulerSequence.ZXY,
        parent_basis=TransformationMatrixType.Bwu,
        child_basis=TransformationMatrixType.Buv,
    )

    c3d_data = C3dData(f"{c3d_filename}")

    # Put the model together, print it and print it to a bioMod file
    natural_model = model.update(c3d_data)

    return natural_model


allCameras = [26578, 26579, 26580, 26581, 26582, 26583, 26584, 26585, 26586, 26587]
ind_cameras = [8, 9]


def generate_c3d_angles(angles, name_file_export):
    c3d = ezc3d.c3d()

    size = angles.shape[2]
    nb_joints = angles.shape[1]

    anglesc3d = np.ones([4, 12, size])
    for i in range(3):
        for j in range(12):
            anglesc3d[i, j, :] = angles[i, j, :]

    angles_name = [
        "pelvis",
        "torso",
        "r_hip",
        "l_hip",
        "r_knee",
        "l_knee",
        "r_ankle",
        "l_ankle",
        "r_shoulder",
        "l_shoulder",
        "r_elbow",
        "l_elbow",
    ]

    c3d["data"]["points"] = anglesc3d
    c3d["parameters"]["POINT"]["RATE"]["value"] = [60]
    c3d["parameters"]["POINT"]["LABELS"]["value"] = angles_name
    c3d.write(name_file_export)


def main():
    num_mouv = 6
    mouv = ['bend', 'cmjs', 'lufe', 'luyo', 'stai', 'stsk', 'walk']
    # frame_max_for_all_mouvs = [1108,441,730,997,1688,855,792]
    # frame_max = frame_max_for_all_mouvs[num_mouv]

    # create a c3d file with data
    filename_static = "D:/Users/chaumeil/these/openpose/sujet5/wDLT_results_static.c3d"
    filename_dynamic = "D:/Users/chaumeil/these/DATA_ESB/"+ mouv[num_mouv]+ "/triang_2Cams_gap_filled.c3d"
    model_optim = model_creation(filename_static, False, False)
    model_init = model_creation(filename_static, False, True)

    marker_xp_optim = Markers.from_c3d(filename_dynamic, usecols=model_optim.marker_names_technical).to_numpy()
    marker_xp_initialize = Markers.from_c3d(filename_dynamic, usecols=model_init.marker_names_technical).to_numpy()

    hmp_c3d_path = "D:/Users/chaumeil/these/DATA_ESB/"+ mouv[num_mouv]+ "/heatmaps_parameters.c3d"
    c3d_data = ezc3d.c3d(hmp_c3d_path)

    nb_frames = c3d_data["data"]["points"].shape[2]  # to be modified if needed
    # nb_frames = 670
    # compute the natural coordinates
    Q_initialize = model_init.Q_from_markers(marker_xp_initialize[:, :, 0:nb_frames])

    ik_solver_base = InverseKinematics(
        model_optim,
        experimental_markers=marker_xp_optim[0:3, :, 0:nb_frames],
        solve_frame_per_frame=True,
        active_direct_frame_constraints=True,
    )

    Qbase = ik_solver_base.solve(Q_init=Q_initialize, initial_guess_mode=InitialGuessModeType.USER_PROVIDED_FIRST_FRAME_ONLY, method="ipopt")

    # modifier les joints en rajoutant projection basis, parent basis et child basis
    angles = np.zeros((3, 12, nb_frames))
    for i in range(nb_frames):
        angles[:, :, i] = (
            model_optim.natural_coordinates_to_joint_angles(NaturalCoordinates(Qbase[:, i])) * 180 / (np.pi)
        )

    generate_c3d_angles(angles, "D:/Users/chaumeil/these/DATA_ESB/"+ mouv[num_mouv]+ "/angles_triang.c3d")

    acq = ezc3d.c3d(filename_dynamic)
    add_natural_coordinate_to_c3d(acq, model_optim, Qbase)
    acq.write("D:/Users/chaumeil/these/DATA_ESB/"+ mouv[num_mouv]+ "/Q_triang.c3d")
    # da = 1
    from bionc import Viz

    # display the experimental markers in red and the model markers in green
    # almost superimposed because the model is well defined on the experimental data

    # à décommenter
    # bionc_viz = Viz(model_optim, show_center_of_mass=False)
    # bionc_viz.animate(Qbase, markers_xp=marker_xp_initialize[:, :, 0:nb_frames])

    da = ik_solver_base.sol()
    do = 1


if __name__ == "__main__":
    