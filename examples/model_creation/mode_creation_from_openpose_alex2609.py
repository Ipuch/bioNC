import os
from pathlib import Path

import ezc3d
import numpy as np
from pyomeca import Markers
from bionc import InverseKinematics, Viz
import time


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
    NaturalCoordinates
)


# rajouter le booléen ici comme ce que voulait faire Alex
def model_creation_optim(
    c3d_filename: str 
) -> BiomechanicalModel:
    model = BiomechanicalModelTemplate()

    ######### Right and Left Foot #########

    midRToes = lambda m, bio: MarkerTemplate.middle_of(m, bio, "right_Btoe", "right_Stoe")
    midLToes = lambda m, bio: MarkerTemplate.middle_of(m, bio, "left_Btoe", "left_Stoe")

    model["RFOOT"] = SegmentTemplate(
        natural_segment=NaturalSegmentTemplate(
            u_axis=AxisTemplate(start="right_heel", end=midRToes),  
            proximal_point=lambda m, bio: m["right_ankle"],  
            distal_point=lambda m, bio: MarkerTemplate.middle_of(m, bio, "right_Btoe", "right_Stoe"),
            w_axis=AxisFunctionTemplate(
                function=lambda m, bio: MarkerTemplate.normal_to(m, bio, "right_ankle", midRToes(m, bio), "right_knee")
            ),
        )
    )
    model["RFOOT"].add_marker(
        MarkerTemplate(name="right_ankle", parent_name="RFOOT", is_technical=True, is_anatomical=False)
    )
    model["RFOOT"].add_marker(
        MarkerTemplate(name="right_heel", parent_name="RFOOT", is_technical=True, is_anatomical=False)
    )
    # model["RFOOT"].add_marker(MarkerTemplate(name = "right_Btoe", parent_name="RFOOT", is_technical=True, is_anatomical=False))
    # model["RFOOT"].add_marker(MarkerTemplate(name = "right_Stoe", parent_name="RFOOT", is_technical=True, is_anatomical=False))

    model["LFOOT"] = SegmentTemplate(
        natural_segment=NaturalSegmentTemplate(
            u_axis=AxisTemplate(start="left_heel", end=midLToes),  # ici midRToes ne dépend pas de m, bio ==> problème?
            proximal_point=lambda m, bio: m["left_ankle"],
            distal_point=lambda m, bio: MarkerTemplate.middle_of(m, bio, "left_Btoe", "left_Stoe"),
            w_axis=AxisFunctionTemplate(
                function=lambda m, bio: MarkerTemplate.normal_to(m, bio, "left_ankle", midLToes(m, bio), "left_knee")
            ),
        )
    )
    model["LFOOT"].add_marker(
        MarkerTemplate(name="left_ankle", parent_name="LFOOT", is_technical=True, is_anatomical=False)
    )
    model["LFOOT"].add_marker(
        MarkerTemplate(name="left_heel", parent_name="LFOOT", is_technical=True, is_anatomical=False)
    )
    # model["LFOOT"].add_marker(MarkerTemplate(name = "left_Btoe", parent_name="LFOOT", is_technical=True, is_anatomical=False))
    # model["LFOOT"].add_marker(MarkerTemplate(name = "left_Stoe", parent_name="LFOOT", is_technical=True, is_anatomical=False))

    ######## Right and Left Shank #########

    vrshank = lambda m, bio: AxisTemplate.from_start_to_end(
        m, bio, start="right_ankle", end="right_knee"
    )  
    wrshank = lambda m, bio: MarkerTemplate.normal_to(m, bio, "right_ankle", midRToes(m, bio), "right_knee")

    model["RSHANK"] = SegmentTemplate(
        natural_segment=NaturalSegmentTemplate(
            u_axis=AxisFunctionTemplate(
                function=lambda m, bio: AxisTemplate.normal_to_vectors(m, bio, vrshank(m, bio), wrshank(m, bio))
            ),
            proximal_point=lambda m, bio: m["right_knee"],
            distal_point=lambda m, bio: m["right_ankle"],
            w_axis=AxisFunctionTemplate(
                function=lambda m, bio: MarkerTemplate.normal_to(m, bio, "right_ankle", midRToes(m, bio), "right_knee")
            ),
        )
    )
    model["RSHANK"].add_marker(
        MarkerTemplate(name="right_knee", parent_name="RSHANK", is_technical=True, is_anatomical=False)
    )
    model["RSHANK"].add_marker(
        MarkerTemplate(name="right_ankle", parent_name="RSHANK", is_technical=True, is_anatomical=False)
    )

    vlshank = lambda m, bio: AxisTemplate.from_start_to_end(
        m, bio, start="left_ankle", end="left_knee"
    )  
    wlshank = lambda m, bio: MarkerTemplate.normal_to(m, bio, "left_ankle", midLToes(m, bio), "left_knee")

    model["LSHANK"] = SegmentTemplate(
        natural_segment=NaturalSegmentTemplate(
            u_axis=AxisFunctionTemplate(
                function=lambda m, bio: AxisTemplate.normal_to_vectors(m, bio, vlshank(m, bio), wlshank(m, bio))
            ),
            proximal_point=lambda m, bio: m["left_knee"],
            distal_point=lambda m, bio: m["left_ankle"],
            w_axis=AxisFunctionTemplate(
                function=lambda m, bio: MarkerTemplate.normal_to(m, bio, "left_ankle", midLToes(m, bio), "left_knee")
            ),
        )
    )
    model["LSHANK"].add_marker(
        MarkerTemplate(name="left_knee", parent_name="LSHANK", is_technical=True, is_anatomical=False)
    )
    model["LSHANK"].add_marker(
        MarkerTemplate(name="left_ankle", parent_name="LSHANK", is_technical=True, is_anatomical=False)
    )

    ############## Right and Left Thigh ##############

    vrthigh = lambda m, bio: AxisTemplate.from_start_to_end(m, bio, start="right_knee", end="right_hip")
    wrthigh = lambda m, bio: MarkerTemplate.normal_to(m, bio, "right_knee", "right_hip", "right_ankle")

    model["RTHIGH"] = SegmentTemplate(
        natural_segment=NaturalSegmentTemplate(
            u_axis=AxisFunctionTemplate(
                function=lambda m, bio: AxisTemplate.normal_to_vectors(m, bio, vrthigh(m, bio), wrthigh(m, bio))
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

    vlthigh = lambda m, bio: AxisTemplate.from_start_to_end(m, bio, start="left_knee", end="left_hip")
    wlthigh = lambda m, bio: MarkerTemplate.normal_to(m, bio, "left_knee", "left_hip","left_ankle")

    model["LTHIGH"] = SegmentTemplate(
        natural_segment=NaturalSegmentTemplate(
            u_axis=AxisFunctionTemplate(
                function=lambda m, bio: AxisTemplate.normal_to_vectors(m, bio, vlthigh(m, bio), wlthigh(m, bio))
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

    ################## Pelvis ###########################
    midHip = lambda m, bio: MarkerTemplate.middle_of(m, bio, "right_hip", "left_hip")
    vpelvis = lambda m, bio: AxisTemplate.from_start_to_end(m, bio, start=midHip(m, bio), end="chest")

    wpelvis = lambda m, bio: AxisTemplate.from_start_to_end(m, bio, start="left_hip", end="right_hip")

    model["PELVIS"] = SegmentTemplate(
        natural_segment=NaturalSegmentTemplate(
            u_axis=AxisFunctionTemplate(
                function=lambda m, bio: AxisTemplate.normal_to_vectors(m, bio, vpelvis(m, bio), wpelvis(m, bio))
            ),
            proximal_point=lambda m, bio: m["chest"],
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


    vtorso = lambda m, bio: AxisTemplate.from_start_to_end(m, bio, start = "chest", end = "neck")
    wtorso = lambda m, bio: AxisTemplate.from_start_to_end(m, bio, start = "left_shoulder", end = "right_shoulder")
    
    model["TORSO"] = SegmentTemplate(
        natural_segment = NaturalSegmentTemplate(
            u_axis = AxisFunctionTemplate(
                function = lambda m, bio: AxisTemplate.normal_to_vectors(m, bio, vtorso(m, bio), wtorso(m, bio))
            ),
            proximal_point= lambda m, bio : m["neck"],
            distal_point= lambda m, bio : m["chest"],
            w_axis = AxisTemplate(start = "left_shoulder", end="right_shoulder"),
        )
    )
    
    model["TORSO"].add_marker(MarkerTemplate(name="right_shoulder", parent_name="TORSO", is_technical=True, is_anatomical=False))
    model["TORSO"].add_marker(MarkerTemplate(name="left_shoulder", parent_name="TORSO", is_technical=True, is_anatomical=False))
    model["TORSO"].add_marker(MarkerTemplate(name="chest", parent_name="TORSO", is_technical=True))
    model["TORSO"].add_marker(MarkerTemplate(name="neck", parent_name="TORSO", is_technical=True))
    
    ################ Head ##################
    
    vhead = lambda m, bio: AxisTemplate.from_start_to_end(m, bio, start = "neck", end = "top_head")
    whead = lambda m, bio: AxisTemplate.from_start_to_end(m, bio, start = "left_ear", end="right_ear")
    model["HEAD"] = SegmentTemplate(
        natural_segment= NaturalSegmentTemplate(
            u_axis = AxisFunctionTemplate(
                function = lambda m, bio: AxisTemplate.normal_to_vectors(m, bio, vhead(m, bio), whead(m, bio))
            ),
            proximal_point= lambda m, bio: m["top_head"],
            distal_point= lambda m, bio: m["neck"],
            w_axis=AxisTemplate(start = "left_ear", end="right_ear"),
        )
    )
    
    model["HEAD"].add_marker(MarkerTemplate(name="right_ear", parent_name="HEAD", is_technical=True, is_anatomical=False))
    model["HEAD"].add_marker(MarkerTemplate(name="left_ear", parent_name="HEAD", is_technical=True, is_anatomical=False))
    model["HEAD"].add_marker(MarkerTemplate(name="right_eye", parent_name="HEAD", is_technical=True, is_anatomical=False))
    model["HEAD"].add_marker(MarkerTemplate(name="left_eye", parent_name="HEAD", is_technical=True, is_anatomical=False))
    model["HEAD"].add_marker(MarkerTemplate(name="top_head", parent_name="HEAD", is_technical=True, is_anatomical=False))
    model["HEAD"].add_marker(MarkerTemplate(name="neck", parent_name="HEAD", is_technical=True))
    
    
    ################## Right and Left Upper arm and Forearm ######################
    
    vruarm = lambda m, bio: AxisTemplate.from_start_to_end(m, bio,start = "right_elbow", end = "right_shoulder")
    vrforearm = lambda m, bio: AxisTemplate.from_start_to_end(m, bio,start = "right_wrist", end = "right_elbow")
    wruarm = lambda m, bio: AxisTemplate.normal_to_vectors(m, bio, vruarm(m, bio), vrforearm(m, bio))
    wrforearm = lambda m, bio: AxisTemplate.normal_to_vectors(m, bio, vruarm(m, bio), vrforearm(m, bio))
    
    model["RUARM"] = SegmentTemplate( # version 1 du modèle
        natural_segment=NaturalSegmentTemplate(
            u_axis = AxisFunctionTemplate(
                function = lambda m, bio: AxisTemplate.normal_to_vectors(m, bio, vruarm(m, bio), wruarm(m, bio))
            ), 
            proximal_point= lambda m, bio: m["right_shoulder"],
            distal_point = lambda m, bio: m["right_elbow"],
            w_axis = AxisFunctionTemplate(
                function = lambda m, bio: AxisTemplate.normal_to_vectors(m, bio, vruarm(m, bio), vrforearm(m, bio))
                ),
        )
    )
    
    model["RUARM"].add_marker(MarkerTemplate(name="right_shoulder", parent_name="RUARM", is_technical=True, is_anatomical=False))
    model["RUARM"].add_marker(MarkerTemplate(name="right_elbow", parent_name="RUARM", is_technical=True, is_anatomical=False))
    

    model["RFOREARM"] = SegmentTemplate(
        natural_segment=NaturalSegmentTemplate(
            u_axis =AxisFunctionTemplate(
                function = lambda m, bio : AxisTemplate.normal_to_vectors(m, bio, vrforearm(m, bio), wrforearm(m, bio))
            ),
            proximal_point= lambda m, bio: m["right_elbow"],
            distal_point= lambda m, bio: m["right_wrist"],
            w_axis=AxisFunctionTemplate(
                function = lambda m, bio: AxisTemplate.normal_to_vectors(m, bio, vruarm(m, bio), vrforearm(m, bio))
                ),
        )
    )
    
    model["RFOREARM"].add_marker(MarkerTemplate(name="right_elbow", parent_name="RFOREARM", is_technical=True, is_anatomical=False))
    model["RFOREARM"].add_marker(MarkerTemplate(name="right_wrist", parent_name="RFOREARM", is_technical=True, is_anatomical=False))
    
    
    vluarm = lambda m, bio: AxisTemplate.from_start_to_end(m, bio,start = "left_elbow", end = "left_shoulder")
    vlforearm = lambda m, bio: AxisTemplate.from_start_to_end(m, bio,start = "left_wrist", end = "left_elbow")
    wluarm = lambda m, bio: AxisTemplate.normal_to_vectors(m, bio, vluarm(m, bio), vlforearm(m, bio))
    wlforearm = lambda m, bio: AxisTemplate.normal_to_vectors(m, bio, vluarm(m, bio), vlforearm(m, bio))
    
    model["LUARM"] = SegmentTemplate(
        natural_segment=NaturalSegmentTemplate(
            u_axis = AxisFunctionTemplate(
                function = lambda m, bio: AxisTemplate.normal_to_vectors(m, bio, vluarm(m, bio), wluarm(m, bio))
            ), 
            proximal_point= lambda m, bio: m["left_shoulder"],
            distal_point = lambda m, bio: m["left_elbow"],
            w_axis = AxisFunctionTemplate(
                function = lambda m, bio: AxisTemplate.normal_to_vectors(m, bio, vluarm(m, bio), vlforearm(m, bio))
                ),
        )
    )
    
    model["LUARM"].add_marker(MarkerTemplate(name="left_shoulder", parent_name="LUARM", is_technical=True, is_anatomical=False))
    model["LUARM"].add_marker(MarkerTemplate(name="left_elbow", parent_name="LUARM", is_technical=True, is_anatomical=False))
    
    model["LFOREARM"] = SegmentTemplate(
        natural_segment=NaturalSegmentTemplate(
            u_axis =AxisFunctionTemplate(
                function = lambda m, bio : AxisTemplate.normal_to_vectors(m, bio, vlforearm(m, bio), wlforearm(m, bio))
            ),
            proximal_point= lambda m, bio: m["left_elbow"],
            distal_point= lambda m, bio: m["left_wrist"],
            w_axis=AxisFunctionTemplate(
                function = lambda m, bio: AxisTemplate.normal_to_vectors(m, bio, vluarm(m, bio), vlforearm(m, bio))
                ),
        )
    )
    
    model["LFOREARM"].add_marker(MarkerTemplate(name="left_elbow", parent_name="LFOREARM", is_technical=True, is_anatomical=False))
    model["LFOREARM"].add_marker(MarkerTemplate(name="left_wrist", parent_name="LFOREARM", is_technical=True, is_anatomical=False))
    
    

    ################### Model joints #####################

    model.add_joint(
        name="rhip",
        joint_type=JointType.SPHERICAL,
        parent="PELVIS",
        child="RTHIGH",
        parent_point="right_hip",
        child_point="right_hip",
    )

    model.add_joint(
        name="lhip",
        joint_type=JointType.SPHERICAL,
        parent="PELVIS",
        child="LTHIGH",
        parent_point="left_hip",
        child_point="left_hip",
    )

    model.add_joint(
        name="rknee",
        joint_type=JointType.REVOLUTE,
        parent="RTHIGH",
        child="RSHANK",
        parent_axis=[NaturalAxis.W, NaturalAxis.W],
        child_axis=[NaturalAxis.V, NaturalAxis.U],
        theta=[np.pi / 2, np.pi / 2],
    )

    model.add_joint(
        name="lknee",
        joint_type=JointType.REVOLUTE,
        parent="LTHIGH",
        child="LSHANK",
        parent_axis=[NaturalAxis.W, NaturalAxis.W],
        child_axis=[NaturalAxis.V, NaturalAxis.U],
        theta=[np.pi / 2, np.pi / 2],
    )

    model.add_joint(
        name="rankle",
        joint_type=JointType.REVOLUTE,
        parent="RSHANK",
        child="RFOOT",
        parent_axis=[NaturalAxis.W, NaturalAxis.W],
        child_axis=[NaturalAxis.V, NaturalAxis.U],
        theta=[np.pi / 2, np.pi / 2],
    )

    model.add_joint(
        name="lankle",
        joint_type=JointType.REVOLUTE,
        parent="LSHANK",
        child="LFOOT",
        parent_axis=[NaturalAxis.W, NaturalAxis.W],
        child_axis=[NaturalAxis.V, NaturalAxis.U],
        theta=[np.pi / 2, np.pi / 2],
    )

    model.add_joint(
        name="rshoulder",
        joint_type=JointType.SPHERICAL,
        parent="TORSO",
        child="RUARM",
        parent_point="right_shoulder",
        child_point="right_shoulder",
    )

    model.add_joint(
        name="lshoulder",
        joint_type=JointType.SPHERICAL,
        parent="TORSO",
        child="LUARM",
        parent_point="left_shoulder",
        child_point="left_shoulder",
    )


    model.add_joint(
        name="neck",
        joint_type=JointType.SPHERICAL,
        parent="TORSO",
        child="HEAD",
        parent_point="neck",
        child_point="neck",
    )

    model.add_joint(
        name="relbow",
        joint_type=JointType.REVOLUTE,
        parent="RUARM",
        child="RFOREARM",
        parent_axis=[NaturalAxis.W, NaturalAxis.W],
        child_axis=[NaturalAxis.V, NaturalAxis.U],
        theta=[np.pi / 2, np.pi / 2],
    )

    model.add_joint(
        name="lelbow",
        joint_type=JointType.REVOLUTE,
        parent="LUARM",
        child="LFOREARM",
        parent_axis=[NaturalAxis.W, NaturalAxis.W],
        child_axis=[NaturalAxis.V, NaturalAxis.U],
        theta=[np.pi / 2, np.pi / 2],
    )

    c3d_data = C3dData(f"{c3d_filename}")

    # Put the model together, print it and print it to a bioMod file
    natural_model = model.update(c3d_data)

    return natural_model


def model_creation_static(
    c3d_filename: str 
) -> BiomechanicalModel:
    model = BiomechanicalModelTemplate()

    ######### Right and Left Foot #########
    midRToes = lambda m, bio: MarkerTemplate.middle_of(m, bio, "right_Btoe", "right_Stoe")
    midLToes = lambda m, bio: MarkerTemplate.middle_of(m, bio, "left_Btoe", "left_Stoe")

    model["RFOOT"] = SegmentTemplate(
        natural_segment=NaturalSegmentTemplate(
            u_axis=AxisTemplate(start="right_heel", end=midRToes),  # ici midRToes ne dépend pas de m, bio ==> problème?
            # proximal_point =  lambda m, bio: ["right_ankle"] ,
            proximal_point=lambda m, bio: m["right_ankle"],  # ==> quel est le bon formalisme?
            distal_point=lambda m, bio: MarkerTemplate.middle_of(m, bio, "right_Btoe", "right_Stoe"),
            w_axis=AxisFunctionTemplate(
                function=lambda m, bio: MarkerTemplate.normal_to(m, bio, "right_ankle", midRToes(m, bio), "right_knee")
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
        MarkerTemplate(name="right_Btoe", parent_name="RFOOT", is_technical=True, is_anatomical=False)
    )
    model["RFOOT"].add_marker(
        MarkerTemplate(name="right_Stoe", parent_name="RFOOT", is_technical=True, is_anatomical=False)
    )

    model["LFOOT"] = SegmentTemplate(
        natural_segment=NaturalSegmentTemplate(
            u_axis=AxisTemplate(start="left_heel", end=midLToes),  # ici midRToes ne dépend pas de m, bio ==> problème?
            proximal_point=lambda m, bio: m["left_ankle"],
            distal_point=lambda m, bio: MarkerTemplate.middle_of(m, bio, "left_Btoe", "left_Stoe"),
            w_axis=AxisFunctionTemplate(
                function=lambda m, bio: MarkerTemplate.normal_to(m, bio, "left_ankle", midLToes(m, bio), "left_knee")
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
        MarkerTemplate(name="left_Btoe", parent_name="LFOOT", is_technical=True, is_anatomical=False)
    )
    model["LFOOT"].add_marker(
        MarkerTemplate(name="left_Stoe", parent_name="LFOOT", is_technical=True, is_anatomical=False)
    )

    ######## Right and Left Shank #########

    vrshank = lambda m, bio: AxisTemplate.from_start_to_end(
        m, bio, start="right_ankle", end="right_knee"
    )  # vérifier que v va bien du distal au proximal
    wrshank = lambda m, bio: MarkerTemplate.normal_to(m, bio, "right_ankle", midRToes(m, bio), "right_knee")

    model["RSHANK"] = SegmentTemplate(
        natural_segment=NaturalSegmentTemplate(
            u_axis=AxisFunctionTemplate(
                function=lambda m, bio: AxisTemplate.normal_to_vectors(m, bio, vrshank(m, bio), wrshank(m, bio))
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

    #######

    vlshank = lambda m, bio: AxisTemplate.from_start_to_end(
        m, bio, start="left_ankle", end="left_knee"
    )  # vérifier que v va bien du distal au proximal
    wlshank = lambda m, bio: MarkerTemplate.normal_to(m, bio, "left_ankle", midLToes(m, bio), "left_knee")

    model["LSHANK"] = SegmentTemplate(
        natural_segment=NaturalSegmentTemplate(
            u_axis=AxisFunctionTemplate(
                function=lambda m, bio: AxisTemplate.normal_to_vectors(m, bio, vlshank(m, bio), wlshank(m, bio))
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

    ############## Right and Left Thigh ##############

    vrthigh = lambda m, bio: AxisTemplate.from_start_to_end(m, bio, start="right_knee", end="right_hip")
    wrthigh = lambda m, bio: MarkerTemplate.normal_to(m, bio, "right_knee", "right_hip", "right_ankle")

    model["RTHIGH"] = SegmentTemplate(
        natural_segment=NaturalSegmentTemplate(
            u_axis=AxisFunctionTemplate(
                function=lambda m, bio: AxisTemplate.normal_to_vectors(m, bio, vrthigh(m, bio), wrthigh(m, bio))
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

    vlthigh = lambda m, bio: AxisTemplate.from_start_to_end(m, bio, start="left_knee", end="left_hip")
    wlthigh = lambda m, bio: MarkerTemplate.normal_to(m, bio, "left_knee", "left_hip", "left_ankle")

    model["LTHIGH"] = SegmentTemplate(
        natural_segment=NaturalSegmentTemplate(
            u_axis=AxisFunctionTemplate(
                function=lambda m, bio: AxisTemplate.normal_to_vectors(m, bio, vlthigh(m, bio), wlthigh(m, bio))
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

    ################## Pelvis ###########################
    midHip = lambda m, bio: MarkerTemplate.middle_of(m, bio, "right_hip", "left_hip")
    vpelvis = lambda m, bio: AxisTemplate.from_start_to_end(m, bio, start=midHip(m, bio), end="chest")

    wpelvis = lambda m, bio: AxisTemplate.from_start_to_end(m, bio, start="left_hip", end="right_hip")

    model["PELVIS"] = SegmentTemplate(
        natural_segment=NaturalSegmentTemplate(
            u_axis=AxisFunctionTemplate(
                function=lambda m, bio: AxisTemplate.normal_to_vectors(m, bio, vpelvis(m, bio), wpelvis(m, bio))
            ),
            proximal_point=lambda m, bio: m["chest"],
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


    vtorso = lambda m, bio: AxisTemplate.from_start_to_end(m, bio, start = "chest", end = "neck")
    wtorso = lambda m, bio: AxisTemplate.from_start_to_end(m, bio, start = "left_shoulder", end = "right_shoulder")
    
    model["TORSO"] = SegmentTemplate(
        natural_segment = NaturalSegmentTemplate(
            u_axis = AxisFunctionTemplate(
                function = lambda m, bio: AxisTemplate.normal_to_vectors(m, bio, vtorso(m, bio), wtorso(m, bio))
            ),
            proximal_point= lambda m, bio : m["neck"],
            distal_point= lambda m, bio : m["chest"],
            w_axis = AxisTemplate(start = "left_shoulder", end="right_shoulder"),
        )
    )
    
    model["TORSO"].add_marker(MarkerTemplate(name="right_shoulder", parent_name="TORSO", is_technical=True, is_anatomical=False))
    model["TORSO"].add_marker(MarkerTemplate(name="left_shoulder", parent_name="TORSO", is_technical=True, is_anatomical=False))
    model["TORSO"].add_marker(MarkerTemplate(name="chest", parent_name="TORSO", is_technical=True))
    model["TORSO"].add_marker(MarkerTemplate(name="neck", parent_name="TORSO", is_technical=True))
    
    ################ Head ##################
    
    vhead = lambda m, bio: AxisTemplate.from_start_to_end(m, bio, start = "neck", end = "top_head")
    whead = lambda m, bio: AxisTemplate.from_start_to_end(m, bio, start = "left_ear", end="right_ear")
    model["HEAD"] = SegmentTemplate(
        natural_segment= NaturalSegmentTemplate(
            u_axis = AxisFunctionTemplate(
                function = lambda m, bio: AxisTemplate.normal_to_vectors(m, bio, vhead(m, bio), whead(m, bio))
            ),
            proximal_point= lambda m, bio: m["top_head"],
            distal_point= lambda m, bio: m["neck"],
            w_axis=AxisTemplate(start = "left_ear", end="right_ear"),
        )
    )
    
    model["HEAD"].add_marker(MarkerTemplate(name="right_ear", parent_name="HEAD", is_technical=True, is_anatomical=False))
    model["HEAD"].add_marker(MarkerTemplate(name="left_ear", parent_name="HEAD", is_technical=True, is_anatomical=False))
    model["HEAD"].add_marker(MarkerTemplate(name="right_eye", parent_name="HEAD", is_technical=True, is_anatomical=False))
    model["HEAD"].add_marker(MarkerTemplate(name="left_eye", parent_name="HEAD", is_technical=True, is_anatomical=False))
    model["HEAD"].add_marker(MarkerTemplate(name="top_head", parent_name="HEAD", is_technical=True, is_anatomical=False))
    model["HEAD"].add_marker(MarkerTemplate(name="neck", parent_name="HEAD", is_technical=True))
    
    
    ################## Right and Left Upper arm and Forearm ######################
    
    vruarm = lambda m, bio: AxisTemplate.from_start_to_end(m, bio,start = "right_elbow", end = "right_shoulder")
    vrforearm = lambda m, bio: AxisTemplate.from_start_to_end(m, bio,start = "right_wrist", end = "right_elbow")
    wruarm = lambda m, bio: AxisTemplate.normal_to_vectors(m, bio, vruarm(m, bio), vrforearm(m, bio))
    wrforearm = lambda m, bio: AxisTemplate.normal_to_vectors(m, bio, vruarm(m, bio), vrforearm(m, bio))
    
    model["RUARM"] = SegmentTemplate( # version 1 du modèle
        natural_segment=NaturalSegmentTemplate(
            u_axis = AxisFunctionTemplate(
                function = lambda m, bio: AxisTemplate.normal_to_vectors(m, bio, vruarm(m, bio), wruarm(m, bio))
            ), 
            proximal_point= lambda m, bio: m["right_shoulder"],
            distal_point = lambda m, bio: m["right_elbow"],
            w_axis = AxisFunctionTemplate(
                function = lambda m, bio: AxisTemplate.normal_to_vectors(m, bio, vruarm(m, bio), vrforearm(m, bio))
                ),
        )
    )
    
    model["RUARM"].add_marker(MarkerTemplate(name="right_shoulder", parent_name="RUARM", is_technical=True, is_anatomical=False))
    model["RUARM"].add_marker(MarkerTemplate(name="right_elbow", parent_name="RUARM", is_technical=True, is_anatomical=False))
    

    model["RFOREARM"] = SegmentTemplate(
        natural_segment=NaturalSegmentTemplate(
            u_axis =AxisFunctionTemplate(
                function = lambda m, bio : AxisTemplate.normal_to_vectors(m, bio, vrforearm(m, bio), wrforearm(m, bio))
            ),
            proximal_point= lambda m, bio: m["right_elbow"],
            distal_point= lambda m, bio: m["right_wrist"],
            w_axis=AxisFunctionTemplate(
                function = lambda m, bio: AxisTemplate.normal_to_vectors(m, bio, vruarm(m, bio), vrforearm(m, bio))
                ),
        )
    )
    
    model["RFOREARM"].add_marker(MarkerTemplate(name="right_elbow", parent_name="RFOREARM", is_technical=True, is_anatomical=False))
    model["RFOREARM"].add_marker(MarkerTemplate(name="right_wrist", parent_name="RFOREARM", is_technical=True, is_anatomical=False))
    
    
    vluarm = lambda m, bio: AxisTemplate.from_start_to_end(m, bio,start = "left_elbow", end = "left_shoulder")
    vlforearm = lambda m, bio: AxisTemplate.from_start_to_end(m, bio,start = "left_wrist", end = "left_elbow")
    wluarm = lambda m, bio: AxisTemplate.normal_to_vectors(m, bio, vluarm(m, bio), vlforearm(m, bio))
    wlforearm = lambda m, bio: AxisTemplate.normal_to_vectors(m, bio, vluarm(m, bio), vlforearm(m, bio))
    
    model["LUARM"] = SegmentTemplate(
        natural_segment=NaturalSegmentTemplate(
            u_axis = AxisFunctionTemplate(
                function = lambda m, bio: AxisTemplate.normal_to_vectors(m, bio, vluarm(m, bio), wluarm(m, bio))
            ), 
            proximal_point= lambda m, bio: m["left_shoulder"],
            distal_point = lambda m, bio: m["left_elbow"],
            w_axis = AxisFunctionTemplate(
                function = lambda m, bio: AxisTemplate.normal_to_vectors(m, bio, vluarm(m, bio), vlforearm(m, bio))
                ),
        )
    )
    
    model["LUARM"].add_marker(MarkerTemplate(name="left_shoulder", parent_name="LUARM", is_technical=True, is_anatomical=False))
    model["LUARM"].add_marker(MarkerTemplate(name="left_elbow", parent_name="LUARM", is_technical=True, is_anatomical=False))
    
    model["LFOREARM"] = SegmentTemplate(
        natural_segment=NaturalSegmentTemplate(
            u_axis =AxisFunctionTemplate(
                function = lambda m, bio : AxisTemplate.normal_to_vectors(m, bio, vlforearm(m, bio), wlforearm(m, bio))
            ),
            proximal_point= lambda m, bio: m["left_elbow"],
            distal_point= lambda m, bio: m["left_wrist"],
            w_axis=AxisFunctionTemplate(
                function = lambda m, bio: AxisTemplate.normal_to_vectors(m, bio, vluarm(m, bio), vlforearm(m, bio))
                ),
        )
    )
    
    model["LFOREARM"].add_marker(MarkerTemplate(name="left_elbow", parent_name="LFOREARM", is_technical=True, is_anatomical=False))
    model["LFOREARM"].add_marker(MarkerTemplate(name="left_wrist", parent_name="LFOREARM", is_technical=True, is_anatomical=False))
    
    
    ################### Model joints #####################

    model.add_joint(
        name="rhip",
        joint_type=JointType.SPHERICAL,
        parent="PELVIS",
        child="RTHIGH",
        parent_point="right_hip",
        child_point="right_hip",
    )

    model.add_joint(
        name="lhip",
        joint_type=JointType.SPHERICAL,
        parent="PELVIS",
        child="LTHIGH",
        parent_point="left_hip",
        child_point="left_hip",
    )

    model.add_joint(
        name="rknee",
        joint_type=JointType.REVOLUTE,
        parent="RTHIGH",
        child="RSHANK",
        parent_axis=[NaturalAxis.W, NaturalAxis.W],
        child_axis=[NaturalAxis.V, NaturalAxis.U],
        theta=[np.pi / 2, np.pi / 2],
    )

    model.add_joint(
        name="lknee",
        joint_type=JointType.REVOLUTE,
        parent="LTHIGH",
        child="LSHANK",
        parent_axis=[NaturalAxis.W, NaturalAxis.W],
        child_axis=[NaturalAxis.V, NaturalAxis.U],
        theta=[np.pi / 2, np.pi / 2],
    )

    model.add_joint(
        name="rankle",
        joint_type=JointType.REVOLUTE,
        parent="RSHANK",
        child="RFOOT",
        parent_axis=[NaturalAxis.W, NaturalAxis.W],
        child_axis=[NaturalAxis.V, NaturalAxis.U],
        theta=[np.pi / 2, np.pi / 2],
    )

    model.add_joint(
        name="lankle",
        joint_type=JointType.REVOLUTE,
        parent="LSHANK",
        child="LFOOT",
        parent_axis=[NaturalAxis.W, NaturalAxis.W],
        child_axis=[NaturalAxis.V, NaturalAxis.U],
        theta=[np.pi / 2, np.pi / 2],
    )

    model.add_joint(
        name="rshoulder",
        joint_type=JointType.SPHERICAL,
        parent="TORSO",
        child="RUARM",
        parent_point="right_shoulder",
        child_point="right_shoulder",
    )

    model.add_joint(
        name="lshoulder",
        joint_type=JointType.SPHERICAL,
        parent="TORSO",
        child="LUARM",
        parent_point="left_shoulder",
        child_point="left_shoulder",
    )


    model.add_joint(
        name="neck",
        joint_type=JointType.SPHERICAL,
        parent="TORSO",
        child="HEAD",
        parent_point="neck",
        child_point="neck",
    )

    model.add_joint(
        name="relbow",
        joint_type=JointType.REVOLUTE,
        parent="RUARM",
        child="RFOREARM",
        parent_axis=[NaturalAxis.W, NaturalAxis.W],
        child_axis=[NaturalAxis.V, NaturalAxis.U],
        theta=[np.pi / 2, np.pi / 2],
    )

    model.add_joint(
        name="lelbow",
        joint_type=JointType.REVOLUTE,
        parent="LUARM",
        child="LFOREARM",
        parent_axis=[NaturalAxis.W, NaturalAxis.W],
        child_axis=[NaturalAxis.V, NaturalAxis.U],
        theta=[np.pi / 2, np.pi / 2],
    )

    c3d_data = C3dData(f"{c3d_filename}")

    # Put the model together, print it and print it to a bioMod file
    natural_model = model.update(c3d_data)

    return natural_model






def model_creation(
    c3d_filename: str, 
    is_static: bool, 
    is_init: bool 
) -> BiomechanicalModel:
    model = BiomechanicalModelTemplate()

    ######### Right and Left Foot #########
    midRToes = lambda m, bio: MarkerTemplate.middle_of(m, bio, "right_Btoe", "right_Stoe")
    midLToes = lambda m, bio: MarkerTemplate.middle_of(m, bio, "left_Btoe", "left_Stoe")

    # vrfoot = lambda m, bio: AxisTemplate.from_start_to_end(m, bio, start=midRToes, end="right_ankle")
    # vlfoot = lambda m, bio: AxisTemplate.from_start_to_end(m, bio, start=midLToes, end="left_ankle")

    vrfoot = lambda m, bio: Axis(start=midRToes, end=Marker(name='right_ankle',position=m["right_ankle"])).axis()
    vlfoot = lambda m, bio: Axis(start=midLToes, end=Marker(name='left_ankle',position=m["left_ankle"])).axis()


    model["RFOOT"] = SegmentTemplate(
        natural_segment=NaturalSegmentTemplate(
            u_axis=AxisTemplate(start="right_heel", end=midRToes),  # ici midRToes ne dépend pas de m, bio ==> problème?
            # proximal_point =  lambda m, bio: ["right_ankle"] ,
            proximal_point=lambda m, bio: m["right_ankle"],  # ==> quel est le bon formalisme?
            distal_point=lambda m, bio: MarkerTemplate.middle_of(m, bio, "right_Btoe", "right_Stoe"),
            w_axis=AxisFunctionTemplate(
                function=lambda m, bio: MarkerTemplate.normal_to(m, bio, "right_ankle", midRToes(m, bio), "right_knee")
            ),
        )
    )
    model["RFOOT"].add_marker(
        MarkerTemplate(name="right_ankle", parent_name="RFOOT", is_technical=True, is_anatomical=False)
    )
    model["RFOOT"].add_marker(
        MarkerTemplate(name="right_heel", parent_name="RFOOT", is_technical=True, is_anatomical=False)
    )

    if is_static == True:
        model["RFOOT"].add_marker(
            MarkerTemplate(name="right_Btoe", parent_name="RFOOT", is_technical=True, is_anatomical=False)
        )
        model["RFOOT"].add_marker(
            MarkerTemplate(name="right_Stoe", parent_name="RFOOT", is_technical=True, is_anatomical=False)
        )

    model["LFOOT"] = SegmentTemplate(
        natural_segment=NaturalSegmentTemplate(
            u_axis=AxisTemplate(start="left_heel", end=midLToes),  # ici midRToes ne dépend pas de m, bio ==> problème?
            proximal_point=lambda m, bio: m["left_ankle"],
            distal_point=lambda m, bio: MarkerTemplate.middle_of(m, bio, "left_Btoe", "left_Stoe"),
            w_axis=AxisFunctionTemplate(
                function=lambda m, bio: MarkerTemplate.normal_to(m, bio, "left_ankle", midLToes(m, bio), "left_knee")
            ),
        )
    )
    model["LFOOT"].add_marker(
        MarkerTemplate(name="left_ankle", parent_name="LFOOT", is_technical=True, is_anatomical=False)
    )
    model["LFOOT"].add_marker(
        MarkerTemplate(name="left_heel", parent_name="LFOOT", is_technical=True, is_anatomical=False)
    )
    
    if is_static == True:
        model["LFOOT"].add_marker(
            MarkerTemplate(name="left_Btoe", parent_name="LFOOT", is_technical=True, is_anatomical=False)
        )
        model["LFOOT"].add_marker(
            MarkerTemplate(name="left_Stoe", parent_name="LFOOT", is_technical=True, is_anatomical=False)
        )

    ######## Right and Left Shank #########

    #vrshank = lambda m, bio: AxisTemplate.from_start_to_end(m, bio, start="right_ankle", end="right_knee")  # vérifier que v va bien du distal au proximal
    wrshank = lambda m, bio: MarkerTemplate.normal_to(m, bio, "right_ankle", midRToes(m, bio), "right_knee")

    vrshank = lambda m, bio: Axis(start=Marker(name='right_ankle',position=m["right_ankle"]), end=Marker(name='right_knee',position=m["right_knee"])).axis()

    model["RSHANK"] = SegmentTemplate(
        natural_segment=NaturalSegmentTemplate(
            u_axis=AxisFunctionTemplate(
                function=lambda m, bio: AxisTemplate.normal_to_vectors(m, bio, vrshank(m, bio), wrshank(m, bio))
            ), # voir comment on remplace cette fonction par normalized_cross_product ==> pourquoi le PULL marche pas
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

    

    vlshank = lambda m, bio: AxisTemplate.from_start_to_end(
        m, bio, start="left_ankle", end="left_knee"
    ) 
    wlshank = lambda m, bio: MarkerTemplate.normal_to(m, bio, "left_ankle", midLToes(m, bio), "left_knee")

    model["LSHANK"] = SegmentTemplate(
        natural_segment=NaturalSegmentTemplate(
            u_axis=AxisFunctionTemplate(
                function=lambda m, bio: AxisTemplate.normal_to_vectors(m, bio, vlshank(m, bio), wlshank(m, bio))
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

    ############## Right and Left Thigh ##############

    vrthigh = lambda m, bio: AxisTemplate.from_start_to_end(m, bio, start="right_knee", end="right_hip")
    wrthigh = lambda m, bio: MarkerTemplate.normal_to(m, bio, "right_knee", "right_hip", "right_ankle")

    if is_init == True:
        model["RTHIGH"] = SegmentTemplate(
            natural_segment=NaturalSegmentTemplate(
                u_axis=AxisFunctionTemplate(
                    function=lambda m, bio: AxisTemplate.normal_to_vectors(m, bio, vrshank(m, bio), vrfoot(m, bio))
                ),
                proximal_point=lambda m, bio: m["right_hip"],
                distal_point=lambda m, bio: m["right_knee"],
                w_axis=AxisFunctionTemplate(function=wrthigh),
            )
        )

    else:
        model["RTHIGH"] = SegmentTemplate(
            natural_segment=NaturalSegmentTemplate(
                u_axis=AxisFunctionTemplate(
                    function=lambda m, bio: AxisTemplate.normal_to_vectors(m, bio, vrthigh(m, bio), wrthigh(m, bio))
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

    vlthigh = lambda m, bio: AxisTemplate.from_start_to_end(m, bio, start="left_knee", end="left_hip")
    wlthigh = lambda m, bio: MarkerTemplate.normal_to(m, bio, "left_knee", "left_hip", "left_ankle")

    model["LTHIGH"] = SegmentTemplate(
        natural_segment=NaturalSegmentTemplate(
            u_axis=AxisFunctionTemplate(
                function=lambda m, bio: AxisTemplate.normal_to_vectors(m, bio, vlthigh(m, bio), wlthigh(m, bio))
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

    ################## Pelvis ###########################
    midHip = lambda m, bio: MarkerTemplate.middle_of(m, bio, "right_hip", "left_hip")
    vpelvis = lambda m, bio: AxisTemplate.from_start_to_end(m, bio, start=midHip(m, bio), end="chest")

    wpelvis = lambda m, bio: AxisTemplate.from_start_to_end(m, bio, start="left_hip", end="right_hip")

    model["PELVIS"] = SegmentTemplate(
        natural_segment=NaturalSegmentTemplate(
            u_axis=AxisFunctionTemplate(
                function=lambda m, bio: AxisTemplate.normal_to_vectors(m, bio, vpelvis(m, bio), wpelvis(m, bio))
            ),
            proximal_point=lambda m, bio: m["chest"],
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


    vtorso = lambda m, bio: AxisTemplate.from_start_to_end(m, bio, start = "chest", end = "neck")
    wtorso = lambda m, bio: AxisTemplate.from_start_to_end(m, bio, start = "left_shoulder", end = "right_shoulder")
    
    model["TORSO"] = SegmentTemplate(
        natural_segment = NaturalSegmentTemplate(
            u_axis = AxisFunctionTemplate(
                function = lambda m, bio: AxisTemplate.normal_to_vectors(m, bio, vtorso(m, bio), wtorso(m, bio))
            ),
            proximal_point= lambda m, bio : m["neck"],
            distal_point= lambda m, bio : m["chest"],
            w_axis = AxisTemplate(start = "left_shoulder", end="right_shoulder"),
        )
    )
    
    model["TORSO"].add_marker(MarkerTemplate(name="right_shoulder", parent_name="TORSO", is_technical=True, is_anatomical=False))
    model["TORSO"].add_marker(MarkerTemplate(name="left_shoulder", parent_name="TORSO", is_technical=True, is_anatomical=False))
    model["TORSO"].add_marker(MarkerTemplate(name="chest", parent_name="TORSO", is_technical=True))
    model["TORSO"].add_marker(MarkerTemplate(name="neck", parent_name="TORSO", is_technical=True))
    
    ################ Head ##################
    
    vhead = lambda m, bio: AxisTemplate.from_start_to_end(m, bio, start = "neck", end = "top_head")
    whead = lambda m, bio: AxisTemplate.from_start_to_end(m, bio, start = "left_ear", end="right_ear")
    model["HEAD"] = SegmentTemplate(
        natural_segment= NaturalSegmentTemplate(
            u_axis = AxisFunctionTemplate(
                function = lambda m, bio: AxisTemplate.normal_to_vectors(m, bio, vhead(m, bio), whead(m, bio))
            ),
            proximal_point= lambda m, bio: m["top_head"],
            distal_point= lambda m, bio: m["neck"],
            w_axis=AxisTemplate(start = "left_ear", end="right_ear"),
        )
    )
    
    model["HEAD"].add_marker(MarkerTemplate(name="right_ear", parent_name="HEAD", is_technical=True, is_anatomical=False))
    model["HEAD"].add_marker(MarkerTemplate(name="left_ear", parent_name="HEAD", is_technical=True, is_anatomical=False))
    model["HEAD"].add_marker(MarkerTemplate(name="right_eye", parent_name="HEAD", is_technical=True, is_anatomical=False))
    model["HEAD"].add_marker(MarkerTemplate(name="left_eye", parent_name="HEAD", is_technical=True, is_anatomical=False))
    model["HEAD"].add_marker(MarkerTemplate(name="top_head", parent_name="HEAD", is_technical=True, is_anatomical=False))
    model["HEAD"].add_marker(MarkerTemplate(name="neck", parent_name="HEAD", is_technical=True))
    
    
    ################## Right and Left Upper arm and Forearm ######################
    
    vruarm = lambda m, bio: AxisTemplate.from_start_to_end(m, bio,start = "right_elbow", end = "right_shoulder")
    vrforearm = lambda m, bio: AxisTemplate.from_start_to_end(m, bio,start = "right_wrist", end = "right_elbow")
    wruarm = lambda m, bio: AxisTemplate.normal_to_vectors(m, bio, vruarm(m, bio), vrforearm(m, bio))
    wrforearm = lambda m, bio: AxisTemplate.normal_to_vectors(m, bio, vruarm(m, bio), vrforearm(m, bio))
    
    model["RUARM"] = SegmentTemplate( # version 1 du modèle
        natural_segment=NaturalSegmentTemplate(
            u_axis = AxisFunctionTemplate(
                function = lambda m, bio: AxisTemplate.normal_to_vectors(m, bio, vruarm(m, bio), wruarm(m, bio))
            ), 
            proximal_point= lambda m, bio: m["right_shoulder"],
            distal_point = lambda m, bio: m["right_elbow"],
            w_axis = AxisFunctionTemplate(
                function = lambda m, bio: AxisTemplate.normal_to_vectors(m, bio, vruarm(m, bio), vrforearm(m, bio))
                ),
        )
    )
    
    model["RUARM"].add_marker(MarkerTemplate(name="right_shoulder", parent_name="RUARM", is_technical=True, is_anatomical=False))
    model["RUARM"].add_marker(MarkerTemplate(name="right_elbow", parent_name="RUARM", is_technical=True, is_anatomical=False))
    

    model["RFOREARM"] = SegmentTemplate(
        natural_segment=NaturalSegmentTemplate(
            u_axis =AxisFunctionTemplate(
                function = lambda m, bio : AxisTemplate.normal_to_vectors(m, bio, vrforearm(m, bio), wrforearm(m, bio))
            ),
            proximal_point= lambda m, bio: m["right_elbow"],
            distal_point= lambda m, bio: m["right_wrist"],
            w_axis=AxisFunctionTemplate(
                function = lambda m, bio: AxisTemplate.normal_to_vectors(m, bio, vruarm(m, bio), vrforearm(m, bio))
                ),
        )
    )
    
    model["RFOREARM"].add_marker(MarkerTemplate(name="right_elbow", parent_name="RFOREARM", is_technical=True, is_anatomical=False))
    model["RFOREARM"].add_marker(MarkerTemplate(name="right_wrist", parent_name="RFOREARM", is_technical=True, is_anatomical=False))
    
    
    vluarm = lambda m, bio: AxisTemplate.from_start_to_end(m, bio,start = "left_elbow", end = "left_shoulder")
    vlforearm = lambda m, bio: AxisTemplate.from_start_to_end(m, bio,start = "left_wrist", end = "left_elbow")
    wluarm = lambda m, bio: AxisTemplate.normal_to_vectors(m, bio, vluarm(m, bio), vlforearm(m, bio))
    wlforearm = lambda m, bio: AxisTemplate.normal_to_vectors(m, bio, vluarm(m, bio), vlforearm(m, bio))
    
    model["LUARM"] = SegmentTemplate(
        natural_segment=NaturalSegmentTemplate(
            u_axis = AxisFunctionTemplate(
                function = lambda m, bio: AxisTemplate.normal_to_vectors(m, bio, vluarm(m, bio), wluarm(m, bio))
            ), 
            proximal_point= lambda m, bio: m["left_shoulder"],
            distal_point = lambda m, bio: m["left_elbow"],
            w_axis = AxisFunctionTemplate(
                function = lambda m, bio: AxisTemplate.normal_to_vectors(m, bio, vluarm(m, bio), vlforearm(m, bio))
                ),
        )
    )
    
    model["LUARM"].add_marker(MarkerTemplate(name="left_shoulder", parent_name="LUARM", is_technical=True, is_anatomical=False))
    model["LUARM"].add_marker(MarkerTemplate(name="left_elbow", parent_name="LUARM", is_technical=True, is_anatomical=False))
    
    model["LFOREARM"] = SegmentTemplate(
        natural_segment=NaturalSegmentTemplate(
            u_axis =AxisFunctionTemplate(
                function = lambda m, bio : AxisTemplate.normal_to_vectors(m, bio, vlforearm(m, bio), wlforearm(m, bio))
            ),
            proximal_point= lambda m, bio: m["left_elbow"],
            distal_point= lambda m, bio: m["left_wrist"],
            w_axis=AxisFunctionTemplate(
                function = lambda m, bio: AxisTemplate.normal_to_vectors(m, bio, vluarm(m, bio), vlforearm(m, bio))
                ),
        )
    )
    
    model["LFOREARM"].add_marker(MarkerTemplate(name="left_elbow", parent_name="LFOREARM", is_technical=True, is_anatomical=False))
    model["LFOREARM"].add_marker(MarkerTemplate(name="left_wrist", parent_name="LFOREARM", is_technical=True, is_anatomical=False))
    
    
    ################### Model joints #####################

    model.add_joint(
        name="rhip",
        joint_type=JointType.SPHERICAL,
        parent="PELVIS",
        child="RTHIGH",
        parent_point="right_hip",
        child_point="right_hip",
    )

    model.add_joint(
        name="lhip",
        joint_type=JointType.SPHERICAL,
        parent="PELVIS",
        child="LTHIGH",
        parent_point="left_hip",
        child_point="left_hip",
    )

    model.add_joint(
        name="rknee",
        joint_type=JointType.REVOLUTE,
        parent="RTHIGH",
        child="RSHANK",
        parent_axis=[NaturalAxis.W, NaturalAxis.W],
        child_axis=[NaturalAxis.V, NaturalAxis.U],
        theta=[np.pi / 2, np.pi / 2],
    )

    model.add_joint(
        name="lknee",
        joint_type=JointType.REVOLUTE,
        parent="LTHIGH",
        child="LSHANK",
        parent_axis=[NaturalAxis.W, NaturalAxis.W],
        child_axis=[NaturalAxis.V, NaturalAxis.U],
        theta=[np.pi / 2, np.pi / 2],
    )

    model.add_joint(
        name="rankle",
        joint_type=JointType.REVOLUTE,
        parent="RSHANK",
        child="RFOOT",
        parent_axis=[NaturalAxis.W, NaturalAxis.W],
        child_axis=[NaturalAxis.V, NaturalAxis.U],
        theta=[np.pi / 2, np.pi / 2],
    )

    model.add_joint(
        name="lankle",
        joint_type=JointType.REVOLUTE,
        parent="LSHANK",
        child="LFOOT",
        parent_axis=[NaturalAxis.W, NaturalAxis.W],
        child_axis=[NaturalAxis.V, NaturalAxis.U],
        theta=[np.pi / 2, np.pi / 2],
    )

    model.add_joint(
        name="rshoulder",
        joint_type=JointType.SPHERICAL,
        parent="TORSO",
        child="RUARM",
        parent_point="right_shoulder",
        child_point="right_shoulder",
    )

    model.add_joint(
        name="lshoulder",
        joint_type=JointType.SPHERICAL,
        parent="TORSO",
        child="LUARM",
        parent_point="left_shoulder",
        child_point="left_shoulder",
    )


    model.add_joint(
        name="neck",
        joint_type=JointType.SPHERICAL,
        parent="TORSO",
        child="HEAD",
        parent_point="neck",
        child_point="neck",
    )

    model.add_joint(
        name="relbow",
        joint_type=JointType.REVOLUTE,
        parent="RUARM",
        child="RFOREARM",
        parent_axis=[NaturalAxis.W, NaturalAxis.W],
        child_axis=[NaturalAxis.V, NaturalAxis.U],
        theta=[np.pi / 2, np.pi / 2],
    )

    model.add_joint(
        name="lelbow",
        joint_type=JointType.REVOLUTE,
        parent="LUARM",
        child="LFOREARM",
        parent_axis=[NaturalAxis.W, NaturalAxis.W],
        child_axis=[NaturalAxis.V, NaturalAxis.U],
        theta=[np.pi / 2, np.pi / 2],
    )

    c3d_data = C3dData(f"{c3d_filename}")

    # Put the model together, print it and print it to a bioMod file
    natural_model = model.update(c3d_data)

    return natural_model









def main():
    # create a c3d file with data
    filename_static = "D:/Users/chaumeil/these/openpose/sujet5/wDLT_results_static.c3d"
    filename_dynamic = "D:/Users/chaumeil/these/openpose/sujet5/wDLT_results.c3d"
    # Create the model from a c3d file and markers as template
    model_optim = model_creation(filename_static, False, False)
    model_static = model_creation(filename_static, True, False)
    
    # Base optim (all markers used in the model are use in the optim)
    # load experimental markers, usecols is used to select only the markers that are in the model
    # and it rearrange them in the same order as the model
    marker_xp_initialize = Markers.from_c3d(
        filename_dynamic, usecols=model_static.marker_names_technical
    ).to_numpy()
    ## If some marker that are used in the static to create the model are not used during the optim it is necessary to define
    # a model allowing to compute a initial guess.
    marker_xp_optim = Markers.from_c3d(filename_dynamic, usecols=model_optim.marker_names_technical).to_numpy()
    # compute the natural coordinates
    #Q_initialize = model_w_glob.Q_from_markers(marker_xp_initialize[:, :, :])
    Q_initialize = model_static.Q_from_markers(marker_xp_initialize[:, :, :])
    # faire un Q_initialize ici avec les méthodes alternatives de construction de la cuisse et des bras

    # No Q_init necessary if all marker are used as the
    ik_solver_base = InverseKinematics(model_optim, marker_xp_optim[0:3, :, :], solve_frame_per_frame=True,Q_init=Q_initialize)



    # Here a Q_init is necessary because some marker that are used in the static to create the model are not used during the optim
    ik_solver_optim = InverseKinematics(
        model_optim, marker_xp_optim[0:3, :, :], solve_frame_per_frame=True, Q_init=Q_initialize, active_direct_frame_constraints=False
    )

    # ik = InverseKinematics(natural_model, markers = None, heatmap_parameters, cam_parameters, solve_frame_per_frame=True) ==> objectif à atteindre

    # Different method can be used for the optim
    method_to_use = "ipopt"  # tend to find lower cost functions but may flip axis.
    # method_to_use = "sqpmethod"  # tend to be faster (with limited-memory hessian approximation)

    Qbase = ik_solver_base.solve(method=method_to_use)
    #Qoptim = ik_solver_optim.solve(method=method_to_use)
    ik_solver_base.check_segment_determinants()
    # verifier que les contraintes sont respectées
    # for i in range(len(Qbase)):
    #     print(i)
    #     print(model_optim.holonomic_constraints(NaturalCoordinates(Qbase[:,i])))

    # print(model_optim.natural_coordinates_to_joint_angles(NaturalCoordinates(Qbase[:,0])))
    # modifier les joints en rajoutant projection basis, parent basis et child basis
    from bionc import Viz

    # display the experimental markers in red and the model markers in green
    # almost superimposed because the model is well defined on the experimental data

    # à décommenter
    bionc_viz = Viz(model_static, show_center_of_mass=False)
    bionc_viz.animate(Qbase, markers_xp=marker_xp_initialize)

    #bionc_viz = Viz(model_optim, show_center_of_mass=False)
    #bionc_viz.animate(Qoptim, markers_xp=marker_xp_optim)

    # remove the c3d file
    # os.remove(filename_static)

    # dump the model in a pickle format
    # model_initialize.save("../models/full_body.nc")


if __name__ == "__main__":
    main()
