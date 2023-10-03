import os
from pathlib import Path

import ezc3d
import numpy as np
from pyomeca import Markers
from bionc import InverseKinematics, Viz
import time


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
    NaturalCoordinates
)


def model_creation(
        c3d_filename: str 
) -> BiomechanicalModel:
    model = BiomechanicalModelTemplate()

    midRToes = lambda m, bio: MarkerTemplate.middle_of(m, bio, "right_Btoe", "right_Stoe")

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
    model["RFOOT"].add_marker(
        MarkerTemplate(name="right_Btoe", parent_name="RFOOT", is_technical=True, is_anatomical=False)
    )
    model["RFOOT"].add_marker(
        MarkerTemplate(name="right_Stoe", parent_name="RFOOT", is_technical=True, is_anatomical=False)
    )

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
        name="rknee",
        joint_type=JointType.REVOLUTE,
        parent="RTHIGH",
        child="RSHANK",
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
    model_optim = model_creation(filename_static)

    # Base optim (all markers used in the model are use in the optim)
    # load experimental markers, usecols is used to select only the markers that are in the model
    # and it rearrange them in the same order as the model
    marker_xp_initialize = Markers.from_c3d(
        filename_dynamic, usecols=model_optim.marker_names_technical
    ).to_numpy()
    ## If some marker that are used in the static to create the model are not used during the optim it is necessary to define
    # a model allowing to compute a initial guess.
    marker_xp_optim = Markers.from_c3d(filename_dynamic, usecols=model_optim.marker_names_technical).to_numpy()
    # compute the natural coordinates
    Q_initialize = model_optim.Q_from_markers(marker_xp_initialize[:, :, 0:5])

    # No Q_init necessary if all marker are used as the
    ik_solver_base = InverseKinematics(model_optim, marker_xp_optim[0:3, :, 0:5], solve_frame_per_frame=True,Q_init=Q_initialize)

    # ik = InverseKinematics(natural_model, markers = None, heatmap_parameters, cam_parameters, solve_frame_per_frame=True) ==> objectif à atteindre

    # Different method can be used for the optim
    method_to_use = "ipopt"  # tend to find lower cost functions but may flip axis.
    # method_to_use = "sqpmethod"  # tend to be faster (with limited-memory hessian approximation)

    Qbase = ik_solver_base.solve(method=method_to_use)

    from bionc import Viz

    # display the experimental markers in red and the model markers in green
    # almost superimposed because the model is well defined on the experimental data

    # à décommenter
    bionc_viz = Viz(model_optim, show_center_of_mass=False)
    bionc_viz.animate(Qbase, markers_xp=marker_xp_initialize[:, :, 0:5])



if __name__ == "__main__":
    main()



















































