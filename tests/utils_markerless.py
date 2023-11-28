import os
from pathlib import Path

import ezc3d
import numpy as np
from pyomeca import Markers
from bionc import InverseKinematics, Viz
import time

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
    NaturalCoordinates,
)


def model_creation(c3d_filename: str, is_static: bool, is_init: bool) -> BiomechanicalModel:
    model = BiomechanicalModelTemplate()

    ######### Right and Left Foot #########
    midRToes = lambda m, bio: MarkerTemplate.middle_of(m, bio, "right_Btoe", "right_Stoe")
    midLToes = lambda m, bio: MarkerTemplate.middle_of(m, bio, "left_Btoe", "left_Stoe")

    vrfoot = lambda m, bio: Axis(start=midRToes, end=Marker(name="right_ankle", position=m["right_ankle"])).axis()
    vlfoot = lambda m, bio: Axis(start=midLToes, end=Marker(name="left_ankle", position=m["left_ankle"])).axis()

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

    if is_static == True:
        model["RFOOT"].add_marker(
            MarkerTemplate(name="right_Btoe", parent_name="RFOOT", is_technical=True, is_anatomical=False)
        )
        model["RFOOT"].add_marker(
            MarkerTemplate(name="right_Stoe", parent_name="RFOOT", is_technical=True, is_anatomical=False)
        )

    wrshank = lambda m, bio: MarkerTemplate.normal_to(m, bio, "right_ankle", midRToes(m, bio), "right_knee")

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

    model.add_joint(
        name="rankle",
        joint_type=JointType.REVOLUTE,
        parent="RSHANK",
        child="RFOOT",
        parent_axis=[NaturalAxis.W, NaturalAxis.W],
        child_axis=[NaturalAxis.V, NaturalAxis.U],
        theta=[np.pi / 2, np.pi / 2],
        projection_basis=EulerSequence.ZXY,
        parent_basis=TransformationMatrixUtil(
            plane=(NaturalAxis.W, NaturalAxis.U),
            axis_to_keep=NaturalAxis.W,
        ).to_enum(),
        # child_basis = TransformationMatrixType.Buw, # pas encore implémentée
    )

    c3d_data = C3dData(f"{c3d_filename}")

    # Put the model together, print it and print it to a bioMod file
    natural_model = model.update(c3d_data)

    return natural_model
