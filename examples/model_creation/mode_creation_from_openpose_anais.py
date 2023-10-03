import os
from pathlib import Path

import ezc3d
import numpy as np
from pyomeca import Markers

from bionc import (
    AxisTemplate,
    AxisFunctionTemplate,
    BiomechanicalModelTemplate,
    MarkerTemplate,
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

## TODO : Check the Buv and Bwu matrices
## maybe code FREE joint as GROUND FREE  ==> pour l'instant je décide de ne rien mettre vu que je n'ai pas de segment ground
## understand where to put m, bio

def model_creation_from_measured_data(c3d_filename: str = "D:/Users/chaumeil/these/openpose/sujet5/wDLT_results_static.c3d") -> BiomechanicalModel:
    
    model = BiomechanicalModelTemplate()

    ######### Right and Left Foot #########

    midRToes = lambda m, bio: MarkerTemplate.middle_of(m, bio, "right_Btoe", "right_Stoe")
    midLToes = lambda m, bio: MarkerTemplate.middle_of(m, bio, "left_Btoe", "left_Stoe")

    model["RFOOT"] = SegmentTemplate(
        natural_segment=NaturalSegmentTemplate(
            u_axis = AxisTemplate(start = "right_heel", end = midRToes), # ici midRToes ne dépend pas de m, bio ==> problème?
            #proximal_point =  lambda m, bio: ["right_ankle"] ,
            proximal_point = lambda m, bio: m["right_ankle"], #==> quel est le bon formalisme?
            distal_point = lambda m, bio: MarkerTemplate.middle_of(m, bio, "right_Btoe", "right_Stoe"),
            w_axis = AxisFunctionTemplate(
                function = lambda m, bio: -MarkerTemplate.normal_to_w(m, bio, midRToes(m, bio) , "right_ankle", "right_knee") 
            ),
        )
    )
    model["RFOOT"].add_marker(MarkerTemplate(name = "right_ankle", parent_name="RFOOT", is_technical=False, is_anatomical=True))
    model["RFOOT"].add_marker(MarkerTemplate(name = "right_heel", parent_name="RFOOT", is_technical=False, is_anatomical=True))

    model["LFOOT"] = SegmentTemplate(
        natural_segment=NaturalSegmentTemplate(
            u_axis = AxisTemplate(start = "left_heel", end = midLToes), # ici midRToes ne dépend pas de m, bio ==> problème?
            proximal_point =  lambda m, bio: m["left_ankle"] ,
            distal_point = lambda m, bio: MarkerTemplate.middle_of(m, bio, "left_Btoe", "left_Stoe"),
            w_axis = AxisFunctionTemplate(
                function = lambda m, bio: -MarkerTemplate.normal_to_w(m, bio, midLToes(m, bio) , "left_ankle", "left_knee") 
            ),
        )
    )
    model["LFOOT"].add_marker(MarkerTemplate(name = "left_ankle", parent_name="LFOOT", is_technical=False, is_anatomical=True))
    model["LFOOT"].add_marker(MarkerTemplate(name = "left_heel", parent_name="LFOOT", is_technical=False, is_anatomical=True))

    ######## Right and Left Shank #########

    vrshank = lambda m, bio: AxisTemplate(start = "right_ankle", end = "right_knee") # vérifier que v va bien du distal au proximal
    wrshank = lambda m, bio: AxisFunctionTemplate(function = lambda m, bio: -MarkerTemplate.normal_to_w(m, bio, midRToes(m, bio) , "right_ankle","right_knee"))

    model["RSHANK"] = SegmentTemplate(
        natural_segment=NaturalSegmentTemplate(
            u_axis= AxisFunctionTemplate(
                function = lambda m, bio: MarkerTemplate.normal_to_vectors(m, bio, vrshank(m, bio), wrshank(m, bio))
            ),
              
            proximal_point= lambda m, bio: m["right_knee"],
            distal_point= lambda m, bio: m["right_ankle"],
            w_axis = AxisFunctionTemplate(
                function = lambda m, bio: -MarkerTemplate.normal_to_w(m, bio, midRToes(m, bio), "right_ankle", "right_knee")
            ),
        )
    )
    model["RSHANK"].add_marker(MarkerTemplate(name="right_knee", parent_name="RSHANK", is_technical=False, is_anatomical=True))
    model["RSHANK"].add_marker(MarkerTemplate(name="right_ankle", parent_name="RSHANK", is_technical=False, is_anatomical=True))
       
    #######

    vlshank = lambda m, bio: AxisTemplate(start = "left_ankle", end = "left_knee") # vérifier que v va bien du distal au proximal
    wlshank = lambda m, bio: AxisFunctionTemplate(function = lambda m, bio: -MarkerTemplate.normal_to_w(m, bio, midLToes(m, bio) , "left_ankle","left_knee"))

    model["LSHANK"] = SegmentTemplate(
        natural_segment=NaturalSegmentTemplate(
            u_axis= AxisFunctionTemplate(
                function = lambda m, bio: MarkerTemplate.normal_to_vectors(m, bio, vlshank(m, bio), wlshank(m, bio))
            ),              
            proximal_point= lambda m, bio: m["left_knee"],
            distal_point= lambda m, bio: m["left_ankle"],
            w_axis = AxisFunctionTemplate(
                function = lambda m, bio: -MarkerTemplate.normal_to_w(m, bio, midLToes , "left_ankle","left_knee")
            ),
        )
    )
    model["LSHANK"].add_marker(MarkerTemplate(name="left_knee", parent_name="LSHANK", is_technical=False, is_anatomical=True))
    model["LSHANK"].add_marker(MarkerTemplate(name="left_ankle", parent_name="LSHANK", is_technical=False, is_anatomical=True))

    ############## Right and Left Thigh ##############
    
    vrthigh = lambda m, bio: AxisTemplate(start = "right_knee", end = "right_hip")
    wrthigh = lambda m, bio: AxisFunctionTemplate(function = lambda m, bio: MarkerTemplate.normal_to_w(m, bio, "right_ankle", "right_knee", "right_hip"))

    model["RTHIGH"] = SegmentTemplate(
        natural_segment = NaturalSegmentTemplate(
            u_axis = AxisFunctionTemplate(
                function = lambda m, bio: MarkerTemplate.normal_to_vectors(m, bio, vrthigh(m, bio), wrthigh(m, bio))
            ),
            proximal_point = lambda m, bio: m["right_hip"],
            distal_point = lambda m, bio: m["right_knee"],
            w_axis = AxisFunctionTemplate(
                function = lambda m, bio: MarkerTemplate.normal_to_w(m, bio, "right_ankle", "right_knee", "right_hip")
            ),
        )
    )
    model["RTHIGH"].add_marker(MarkerTemplate(name="right_knee", parent_name="RTHIGH", is_technical=False, is_anatomical=True))
    model["RTHIGH"].add_marker(MarkerTemplate(name="right_hip", parent_name="RTHIGH", is_technical=False, is_anatomical=True))

    vlthigh = lambda m, bio: AxisTemplate(start = "left_knee", end = "left_hip")
    wlthigh = lambda m, bio: AxisFunctionTemplate(function = lambda m, bio: MarkerTemplate.normal_to_w(m, bio, "left_ankle", "left_knee", "left_hip"))

    model["LTHIGH"] = SegmentTemplate(
        natural_segment = NaturalSegmentTemplate(
            u_axis = AxisFunctionTemplate(
                function = lambda m, bio: MarkerTemplate.normal_to_vectors(m, bio, vlthigh(m, bio), wlthigh(m, bio))
            ),
            proximal_point = lambda m, bio: m["left_hip"],
            distal_point = lambda m, bio: m["left_knee"],
            w_axis = AxisFunctionTemplate(
                function = lambda m, bio: MarkerTemplate.normal_to_w(m, bio, "left_ankle", "left_knee", "left_hip")
            ),
        )
    )
    model["LTHIGH"].add_marker(MarkerTemplate(name="left_knee", parent_name="LTHIGH", is_technical=False, is_anatomical=True))
    model["LTHIGH"].add_marker(MarkerTemplate(name="left_hip", parent_name="LTHIGH", is_technical=False, is_anatomical=True))

    ################## Pelvis ###########################
    midHip = lambda m, bio: MarkerTemplate.middle_of(m, bio, "right_hip", "left_hip")
    vpelvis = lambda m, bio: AxisTemplate(start = midHip, end = "chest")
    wpelvis = AxisTemplate(start = "left_hip", end = "right_hip")

    model["PELVIS"] = SegmentTemplate(
        natural_segment = NaturalSegmentTemplate(
            u_axis = AxisFunctionTemplate(
                function = lambda m, bio: MarkerTemplate.normal_to_vectors(m, bio, vpelvis(m, bio), wpelvis(m, bio))
            ),
            proximal_point= lambda m, bio: m["chest"], 
            distal_point = lambda m, bio: MarkerTemplate.middle_of(m, bio, "right_hip", "left_hip"),
            w_axis = AxisTemplate(start = "left_hip", end = "right_hip"),
        )
    )
    model["PELVIS"].add_marker(MarkerTemplate(name="right_hip", parent_name="PELVIS", is_technical=False, is_anatomical=True))
    model["PELVIS"].add_marker(MarkerTemplate(name="left_hip", parent_name="PELVIS", is_technical=False, is_anatomical=True))

    ################ TORSO #####################
    vtorso = lambda m, bio: AxisTemplate(start = "chest", end = "neck")
    wtorso = lambda m, bio: AxisTemplate(start = "left_shoulder", end = "right_shoulder")

    model["TORSO"] = SegmentTemplate(
        natural_segment = NaturalSegmentTemplate(
            u_axis = AxisFunctionTemplate(
                function = lambda m, bio: MarkerTemplate.normal_to_vectors(m, bio, vtorso(m, bio), wtorso(m, bio))
            ),
            proximal_point= lambda m, bio : m["neck"],
            distal_point= lambda m, bio : m["chest"],
            w_axis = AxisTemplate(start = "left_shoulder", end="right_shoulder"),
        )
    )

    model["TORSO"].add_marker(MarkerTemplate(name="right_hip", parent_name="TORSO", is_technical=False, is_anatomical=True))
    model["TORSO"].add_marker(MarkerTemplate(name="left_hip", parent_name="TORSO", is_technical=False, is_anatomical=True))
    model["TORSO"].add_marker(MarkerTemplate(name="chest", parent_name="TORSO", is_technical=True))

    ################ Head ##################

    vhead = lambda m, bio: AxisTemplate(start = "neck", end = "top_head")
    whead = lambda m, bio: AxisTemplate(start = "left_ear", end="right_ear")
    model["HEAD"] = SegmentTemplate(
        natural_segment= NaturalSegmentTemplate(
            u_axis = AxisFunctionTemplate(
                function = lambda m, bio: MarkerTemplate.normal_to_vectors(m, bio, vhead(m, bio), whead(m, bio))
            ), 
            proximal_point= lambda m, bio: m["top_head"], 
            distal_point= lambda m, bio: m["neck"],
            w_axis=AxisTemplate(start = "left_ear", end="right_ear"),
        )
    )
    
    model["HEAD"].add_marker(MarkerTemplate(name="right_ear", parent_name="HEAD", is_technical=False, is_anatomical=True))
    model["HEAD"].add_marker(MarkerTemplate(name="left_ear", parent_name="HEAD", is_technical=False, is_anatomical=True))
    model["HEAD"].add_marker(MarkerTemplate(name="right_eye", parent_name="HEAD", is_technical=False, is_anatomical=True))
    model["HEAD"].add_marker(MarkerTemplate(name="left_eye", parent_name="HEAD", is_technical=False, is_anatomical=True))
    model["HEAD"].add_marker(MarkerTemplate(name="top_head", parent_name="HEAD", is_technical=False, is_anatomical=True))


    ################## Right and Left Upper arm and Forearm ######################

    vruarm = lambda m, bio: AxisTemplate(start = "right_elbow", end = "right_shoulder")
    vrforearm = lambda m, bio: AxisTemplate(start = "right_wrist", end = "right_elbow")  
    wruarm = lambda m, bio: AxisFunctionTemplate(function = lambda m, bio: MarkerTemplate.normal_to_vectors(m, bio, vruarm(m, bio), vrforearm(m, bio)))
    wrforearm = lambda m, bio: AxisFunctionTemplate(function = lambda m, bio: MarkerTemplate.normal_to_vectors(m, bio, vruarm(m, bio), vrforearm(m, bio)))

    model["RUARM"] = SegmentTemplate(
        natural_segment=NaturalSegmentTemplate(
            u_axis = AxisFunctionTemplate(
                function = lambda m, bio: MarkerTemplate.normal_to_vectors(m, bio, vruarm(m, bio), wruarm(m, bio))
            ), ## est ce qu'on est sûrs ici que wruarm va avoir le même comportement que le w_axis associé au segment RUARM??
            proximal_point= lambda m, bio: m["right_shoulder"],
            distal_point = lambda m, bio: m["right_elbow"],
            w_axis = AxisFunctionTemplate(
                function = lambda m, bio: MarkerTemplate.normal_to_vectors(m, bio, vruarm(m, bio), vrforearm(m, bio))
                ),
        )
    )

    model["RUARM"].add_marker(MarkerTemplate(name="right_shoulder", parent_name="RUARM", is_technical=False, is_anatomical=True))
    model["RUARM"].add_marker(MarkerTemplate(name="right_elbow", parent_name="RUARM", is_technical=False, is_anatomical=True))

    model["RFOREARM"] = SegmentTemplate(
        natural_segment=NaturalSegmentTemplate(
            u_axis =AxisFunctionTemplate(
                function = lambda m, bio : MarkerTemplate.normal_to_vectors(m, bio, vrforearm(m, bio), wrforearm(m, bio))
            ),
            proximal_point= lambda m, bio: m["right_elbow"],
            distal_point= lambda m, bio: m["right_wrist"],
            w_axis=AxisFunctionTemplate(
                function = lambda m, bio: MarkerTemplate.normal_to_vectors(m, bio, vruarm(m, bio), vrforearm(m, bio))
                ),
        )
    )

    model["RFOREARM"].add_marker(MarkerTemplate(name="right_elbow", parent_name="RFOREARM", is_technical=False, is_anatomical=True))
    model["RFOREARM"].add_marker(MarkerTemplate(name="right_wrist", parent_name="RFOREARM", is_technical=False, is_anatomical=True))
    

    vluarm = lambda m, bio: AxisTemplate(start = "left_elbow", end = "left_shoulder")
    vlforearm = lambda m, bio: AxisTemplate(start = "left_wrist", end = "left_elbow")  
    wluarm = lambda m, bio: AxisFunctionTemplate(function = lambda m, bio: MarkerTemplate.normal_to_vectors(m, bio, vluarm(m, bio), vlforearm(m, bio)))
    wlforearm = lambda m, bio: AxisFunctionTemplate(function = lambda m, bio: MarkerTemplate.normal_to_vectors(m, bio, vluarm(m, bio), vlforearm(m, bio)))

    model["LUARM"] = SegmentTemplate(
        natural_segment=NaturalSegmentTemplate(
            u_axis = AxisFunctionTemplate(
                function = lambda m, bio: MarkerTemplate.normal_to_vectors(m, bio, vluarm(m, bio), wluarm(m, bio))
            ), ## est ce qu'on est sûrs ici que wruarm va avoir le même comportement que le w_axis associé au segment RUARM??
            proximal_point= lambda m, bio: m["left_shoulder"],
            distal_point = lambda m, bio: m["left_elbow"],
            w_axis = AxisFunctionTemplate(
                function = lambda m, bio: MarkerTemplate.normal_to_vectors(m, bio, vluarm(m, bio), vlforearm(m, bio))
                ),
        )
    )

    model["LUARM"].add_marker(MarkerTemplate(name="left_shoulder", parent_name="LUARM", is_technical=False, is_anatomical=True))
    model["LUARM"].add_marker(MarkerTemplate(name="left_elbow", parent_name="LUARM", is_technical=False, is_anatomical=True))

    model["LFOREARM"] = SegmentTemplate(
        natural_segment=NaturalSegmentTemplate(
            u_axis =AxisFunctionTemplate(
                function = lambda m, bio : MarkerTemplate.normal_to_vectors(m, bio, vlforearm(m, bio), wlforearm(m, bio))
            ),
            proximal_point= lambda m, bio: m["left_elbow"],
            distal_point= lambda m, bio: m["left_wrist"],
            w_axis=AxisFunctionTemplate(
                function = lambda m, bio: MarkerTemplate.normal_to_vectors(m, bio, vluarm(m, bio), vlforearm(m, bio))
                ),
        )
    )

    model["LFOREARM"].add_marker(MarkerTemplate(name="left_elbow", parent_name="LFOREARM", is_technical=False, is_anatomical=True))
    model["LFOREARM"].add_marker(MarkerTemplate(name="left_wrist", parent_name="LFOREARM", is_technical=False, is_anatomical=True))
    


    ################### Model joints #####################

    model.add_joint(
        name="rhip",
        joint_type=JointType.SPHERICAL,
        parent="PELVIS",
        child="RTHIGH",
        projection_basis=EulerSequence.ZXY,  # to either project joint torque or joint angle
        # we need to define the parent and child basis
        parent_basis=TransformationMatrixUtil(
            # defining the segment coordinate system
            plane=(NaturalAxis.W, NaturalAxis.U),  # the plane to define the cross product
            axis_to_keep=NaturalAxis.W,  # it means W = Z
        ).to_enum(),
        child_basis=TransformationMatrixType.Bvu,
    )   

    model.add_joint(
        name="lhip",
        joint_type=JointType.SPHERICAL,
        parent="PELVIS",
        child="LTHIGH",
        projection_basis=EulerSequence.ZXY,  # to either project joint torque or joint angle
        # we need to define the parent and child basis
        parent_basis=TransformationMatrixUtil(
            # defining the segment coordinate system
            plane=(NaturalAxis.W, NaturalAxis.U),  # the plane to define the cross product
            axis_to_keep=NaturalAxis.W,  # it means W = Z
        ).to_enum(),
        child_basis=TransformationMatrixType.Bvu,
    )   


    model.add_joint(
        name="rknee",
        joint_type=JointType.REVOLUTE,
        parent="RTHIGH",
        child="RSHANK",
        projection_basis=EulerSequence.ZXY,  # to either project joint torque or joint angle
        # we need to define the parent and child basis
        parent_basis=TransformationMatrixUtil(
            # defining the segment coordinate system
            plane=(NaturalAxis.W, NaturalAxis.U),  # the plane to define the cross product
            axis_to_keep=NaturalAxis.W,  # it means W = Z
        ).to_enum(),
        child_basis=TransformationMatrixType.Bvu,
    )


    model.add_joint(
        name="lknee",
        joint_type=JointType.REVOLUTE,
        parent="LTHIGH",
        child="LSHANK",
        projection_basis=EulerSequence.ZXY,  # to either project joint torque or joint angle
        # we need to define the parent and child basis
        parent_basis=TransformationMatrixUtil(
            # defining the segment coordinate system
            plane=(NaturalAxis.W, NaturalAxis.U),  # the plane to define the cross product
            axis_to_keep=NaturalAxis.W,  # it means W = Z
        ).to_enum(),
        child_basis=TransformationMatrixType.Bvu,
    )


    model.add_joint(
        name="rankle",
        joint_type=JointType.REVOLUTE,
        parent="RSHANK",
        child="RFOOT",
        projection_basis=EulerSequence.ZXY,  # to either project joint torque or joint angle
        # we need to define the parent and child basis
        parent_basis=TransformationMatrixUtil(
            # defining the segment coordinate system
            plane=(NaturalAxis.W, NaturalAxis.U),  # the plane to define the cross product
            axis_to_keep=NaturalAxis.W,  # it means W = Z
        ).to_enum(),
        child_basis=TransformationMatrixType.Bvu,
    )

    model.add_joint(
        name="lankle",
        joint_type=JointType.REVOLUTE,
        parent="LSHANK",
        child="LFOOT",
        projection_basis=EulerSequence.ZXY,  # to either project joint torque or joint angle
        # we need to define the parent and child basis
        parent_basis=TransformationMatrixUtil(
            # defining the segment coordinate system
            plane=(NaturalAxis.W, NaturalAxis.U),  # the plane to define the cross product
            axis_to_keep=NaturalAxis.W,  # it means W = Z
        ).to_enum(),
        child_basis=TransformationMatrixType.Bvu,
    )


    model.add_joint(
        name="rshoulder",
        joint_type=JointType.SPHERICAL,
        parent="TORSO",
        child="RUARM",
        projection_basis=EulerSequence.ZXY,  # to either project joint torque or joint angle
        # we need to define the parent and child basis
        parent_basis=TransformationMatrixUtil(
            # defining the segment coordinate system
            plane=(NaturalAxis.W, NaturalAxis.U),  # the plane to define the cross product
            axis_to_keep=NaturalAxis.W,  # it means W = Z
        ).to_enum(),
        child_basis=TransformationMatrixType.Bvu,
    )

    
    model.add_joint(
        name="lshoulder",
        joint_type=JointType.SPHERICAL,
        parent="TORSO",
        child="LUARM",
        projection_basis=EulerSequence.ZXY,  # to either project joint torque or joint angle
        # we need to define the parent and child basis
        parent_basis=TransformationMatrixUtil(
            # defining the segment coordinate system
            plane=(NaturalAxis.W, NaturalAxis.U),  # the plane to define the cross product
            axis_to_keep=NaturalAxis.W,  # it means W = Z
        ).to_enum(),
        child_basis=TransformationMatrixType.Bvu,
    )

    model.add_joint(
        name="relbow",
        joint_type=JointType.REVOLUTE,
        parent="RUARM",
        child="RFOREARM",
        projection_basis=EulerSequence.ZXY,  # to either project joint torque or joint angle
        # we need to define the parent and child basis
        parent_basis=TransformationMatrixUtil(
            # defining the segment coordinate system
            plane=(NaturalAxis.W, NaturalAxis.U),  # the plane to define the cross product
            axis_to_keep=NaturalAxis.W,  # it means W = Z
        ).to_enum(),
        child_basis=TransformationMatrixType.Bvu,
    )

    model.add_joint(
        name="lelbow",
        joint_type=JointType.REVOLUTE,
        parent="LUARM",
        child="LFOREARM",
        projection_basis=EulerSequence.ZXY,  # to either project joint torque or joint angle
        # we need to define the parent and child basis
        parent_basis=TransformationMatrixUtil(
            # defining the segment coordinate system
            plane=(NaturalAxis.W, NaturalAxis.U),  # the plane to define the cross product
            axis_to_keep=NaturalAxis.W,  # it means W = Z
        ).to_enum(),
        child_basis=TransformationMatrixType.Bvu,
    )

    model.add_joint(
        name="neck",
        joint_type=JointType.REVOLUTE,
        parent="TORSO",
        child="HEAD",
        projection_basis=EulerSequence.ZXY,  # to either project joint torque or joint angle
        # we need to define the parent and child basis
        parent_basis=TransformationMatrixUtil(
            # defining the segment coordinate system
            plane=(NaturalAxis.W, NaturalAxis.U),  # the plane to define the cross product
            axis_to_keep=NaturalAxis.W,  # it means W = Z
        ).to_enum(),
        child_basis=TransformationMatrixType.Bvu,
    )




    c3d_data = C3dData(f"{c3d_filename}")

    # Put the model together, print it and print it to a bioMod file
    natural_model = model.update(c3d_data)

    return natural_model





def main():
    # create a c3d file with data
    filename_static = 'D:/Users/chaumeil/these/openpose/sujet5/wDLT_results_static.c3d'
    filename_dynamic = 'D:/Users/chaumeil/these/openpose/sujet5/wDLT_results.c3d'
    # Create the model from a c3d file and markers as template
    model = model_creation_from_measured_data(filename_static)

    # load experimental markers
    markers_xp = Markers.from_c3d(filename_dynamic).to_numpy()

    # compute the natural coordinates
    Qxp = model.Q_from_markers(markers_xp[:, :, 0:2])

    from bionc import Viz

    # display the experimental markers in red and the model markers in green
    # almost superimposed because the model is well defined on the experimental data
    bionc_viz = Viz(model, show_center_of_mass=False)
    bionc_viz.animate(Qxp, markers_xp=markers_xp)

    # remove the c3d file
    os.remove(filename_static)

    # dump the model in a pickle format
    model.save("../models/full_body.nc")


if __name__ == "__main__":
    main()
