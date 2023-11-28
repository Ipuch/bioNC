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
    NaturalAxis,
)


def model_creation_markerless(c3d_filename: str, is_static: bool) -> BiomechanicalModel:
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
        "nose",
        "left_eye",
        "right_eye",
        "left_ear",
        "right_ear",
        "left_shoulder",
        "right_shoulder",
        "left_elbow",
        "right_elbow",
        "left_wrist",
        "right_wrist",
        "left_hip",
        "right_hip",
        "left_knee",
        "right_knee",
        "left_ankle",
        "right_ankle",
        "neck",
        "top_head",
        "left_Btoe",
        "left_Stoe",
        "left_heel",
        "right_Btoe",
        "right_Stoe",
        "right_heel",
        "chest",
    )

    # Fill it with random data
    c3d["parameters"]["POINT"]["RATE"]["value"] = [60]
    c3d["parameters"]["POINT"]["LABELS"]["value"] = marker_tuple
    c3d["parameters"]["POINT"]["UNITS"]["value"] = ["m"]

    c3d["data"]["points"] = np.ones((4, len(marker_tuple), 2))
    c3d["data"]["points"][:3, 0, :] = np.array(
        [[-0.91854227, -0.91902238], [0.05200319, 0.05374093], [1.49332643, 1.4971571]]
    )

    c3d["data"]["points"][:3, 1, :] = np.array(
        [[-0.95271301, -0.94939047], [0.08060213, 0.08279321], [1.52161992, 1.52492511]]
    )

    c3d["data"]["points"][:3, 2, :] = np.array(
        [[-0.94741046, -0.94711775], [0.01856505, 0.0200668], [1.51977348, 1.52575552]]
    )

    c3d["data"]["points"][:3, 3, :] = np.array(
        [[-1.03818119, -1.03827941], [0.11691833, 0.11831527], [1.49800241, 1.49932575]]
    )

    c3d["data"]["points"][:3, 4, :] = np.array(
        [[-1.02148211, -1.02131343], [-0.03525349, -0.03630647], [1.49858201, 1.50062203]]
    )

    c3d["data"]["points"][:3, 5, :] = np.array(
        [[-1.09745896, -1.09394026], [0.18494068, 0.18574314], [1.31930518, 1.32184243]]
    )

    c3d["data"]["points"][:3, 6, :] = np.array(
        [[-1.03047395, -1.03167784], [-0.11858987, -0.11587628], [1.32264996, 1.32356834]]
    )

    c3d["data"]["points"][:3, 7, :] = np.array(
        [[-1.19339979, -1.18560421], [0.18626621, 0.18993977], [1.07780051, 1.07937753]]
    )

    c3d["data"]["points"][:3, 8, :] = np.array(
        [[-1.02185977, -1.0232619], [-0.17527503, -0.17106456], [1.05751991, 1.05676305]]
    )

    c3d["data"]["points"][:3, 9, :] = np.array(
        [[-1.2452333, -1.23288262], [0.20704316, 0.21474415], [0.83132827, 0.83130342]]
    )

    c3d["data"]["points"][:3, 10, :] = np.array(
        [[-0.8866607, -0.89089376], [-0.20218505, -0.20022821], [0.86211771, 0.86090463]]
    )

    c3d["data"]["points"][:3, 11, :] = np.array(
        [[-1.07936883, -1.08186245], [0.13325864, 0.13386433], [0.8722074, 0.87232262]]
    )

    c3d["data"]["points"][:3, 12, :] = np.array(
        [[-1.06865537, -1.06281137], [-0.05479637, -0.05288993], [0.87060797, 0.86970109]]
    )

    c3d["data"]["points"][:3, 13, :] = np.array(
        [[-0.98763353, -0.99402088], [0.12318356, 0.1263728], [0.4743509, 0.47215393]]
    )

    c3d["data"]["points"][:3, 14, :] = np.array(
        [[-1.08386636, -1.07111251], [-0.0476904, -0.04517972], [0.45526105, 0.45666268]]
    )

    c3d["data"]["points"][:3, 15, :] = np.array(
        [[-0.96583068, -0.97522831], [0.13564242, 0.13411123], [0.06141762, 0.06047921]]
    )

    c3d["data"]["points"][:3, 16, :] = np.array(
        [[-1.3368181, -1.33367074], [-0.10489886, -0.10492831], [0.12287701, 0.14064722]]
    )

    c3d["data"]["points"][:3, 17, :] = np.array(
        [[-1.04488289, -1.04401088], [0.03717564, 0.03940626], [1.39459002, 1.39639688]]
    )

    c3d["data"]["points"][:3, 18, :] = np.array(
        [[-0.99522835, -0.9940474], [0.04511438, 0.0478076], [1.62753665, 1.63064623]]
    )

    c3d["data"]["points"][:3, 19, :] = np.array(
        [[-0.80465186, -0.81480652], [0.14636216, 0.14638965], [0.02420488, 0.02074622]]
    )

    c3d["data"]["points"][:3, 20, :] = np.array(
        [[-0.83566433, -0.84988207], [0.17215359, 0.17253058], [0.02334609, 0.02162735]]
    )

    c3d["data"]["points"][:3, 21, :] = np.array(
        [[-0.99721444, -1.01125038], [0.1285318, 0.1268564], [0.02685708, 0.02434235]]
    )

    c3d["data"]["points"][:3, 22, :] = np.array(
        [[-1.21956623, -1.22231615], [-0.12216686, -0.12088177], [0.02139223, 0.02307803]]
    )

    c3d["data"]["points"][:3, 23, :] = np.array(
        [[-1.24555492, -1.24877322], [-0.13943221, -0.13825899], [0.0291789, 0.03187862]]
    )

    c3d["data"]["points"][:3, 24, :] = np.array(
        [[-1.38447905, -1.38308716], [-0.10356064, -0.10016831], [0.11137947, 0.12152971]]
    )

    c3d["data"]["points"][:3, 25, :] = np.array(
        [[-1.06898928, -1.06757295], [0.03620327, 0.03771031], [1.0961926, 1.09685862]]
    )

    # Write the c3d file
    filename = f"{Path(__file__).parent.resolve()}/statref.c3d"
    c3d.write(filename)

    return filename


def main():
    # create a c3d file with data
    filename = generate_c3d_file()
    # Create the model from a c3d file and markers as template
    model = model_creation_markerless(filename, is_static=False)

    # load experimental markers
    markers_xp = Markers.from_c3d(filename).to_numpy()

    # remove the c3d file
    os.remove(filename)

    # dump the model in a pickle format
    model.save("../models/lower_limb.nc")


if __name__ == "__main__":
    main()
