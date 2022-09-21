import os

import numpy as np
import biorbd
from bioNC import (
    AxisTemplate,
    BiomechanicalModelTemplate,
    C3dData,
    MarkerTemplate,
    SegmentTemplate,
    NaturalSegmentCoordinateSystemTemplate,
)
import ezc3d

# from .de_leva import DeLevaTable todo: add this to the example
#
# This examples shows how to
#     1. Create a model from scratch using a template with marker names (model_creation_from_data)

#


def model_creation_from_measured_data():
    """
    Create a model from a data file and we build the biomechanical model as a template using the marker names
    """

    # Fill the kinematic chain model
    model = BiomechanicalModelTemplate()
    # de_leva = DeLevaTable(total_mass=100, sex="female")

    model["THIGH"] = SegmentTemplate(
        segment_coordinate_system=NaturalSegmentCoordinateSystemTemplate(
            u_axis=AxisTemplate(
                start="HIP_CENTER",
                # u_axis is defined from the normal of the plane formed by the hip center, the medial epicondyle and the
                # lateral epicondyle
                end=lambda m, bio: np.cross(m["HIP_CENTER"] - m["LFE"], m["HIP_CENTER"] - m["MFE"])
                / np.linalg.norm(np.cross(m["HIP_CENTER"] - m["LFE"], m["HIP_CENTER"] - m["MFE"])),
            ),
            proximal_point="HIP_CENTER",
            # the knee joint computed from the medial femoral epicondyle and the lateral femoral epicondyle
            distal_point=lambda m, bio: (m["MFE"] + m["LFE"]) / 2,
            w_axis=AxisTemplate(start="MFE", end="LFE"),
        )
    )

    model["THIGH"].add_marker(MarkerTemplate("HIP_CENTER", parent_name="THIGH"))
    model["THIGH"].add_marker(MarkerTemplate("MFE", parent_name="THIGH"))
    model["THIGH"].add_marker(MarkerTemplate("LFE", parent_name="THIGH"))
    model["THIGH"].add_marker(
        MarkerTemplate("KNEE_JOINT", function=lambda m, bio: (m["MFE"] + m["LFE"]) / 2, parent_name="THIGH")
    )

    model["SHANK"] = SegmentTemplate(
        segment_coordinate_system=NaturalSegmentCoordinateSystemTemplate(
            u_axis=AxisTemplate(
                start="KNEE_CENTER",
                # u_axis is defined from the normal of the plane formed by the hip center, the medial epicondyle and the
                # lateral epicondyle
                end=lambda m, bio: (
                    np.cross(m["KNEE_CENTER"] - m["LM"], m["KNEE_CENTER"] - m["MM"])
                    / np.linalg.norm(np.cross(m["KNEE_CENTER"] - m["LM"], m["KNEE_CENTER"] - m["MM"]))
                ),
            ),
            proximal_point="KNEE_CENTER",
            # the knee joint computed from the medial femoral epicondyle and the lateral femoral epicondyle
            distal_point=lambda m, bio: (m["LM"] + m["MM"]) / 2,
            w_axis=AxisTemplate(start="LM", end="MM"),
        )
    )
    model["SHANK"].add_marker(
        MarkerTemplate("KNEE_JOINT", function=lambda m, bio: (m["MFE"] + m["LFE"]) / 2, parent_name="KNEE_JOINT")
    )
    model["SHANK"].add_marker(MarkerTemplate("LM", parent_name="SHANK"))
    model["SHANK"].add_marker(MarkerTemplate("MM", parent_name="SHANK"))
    model["SHANK"].add_marker(
        MarkerTemplate("ANKLE_JOINT", function=lambda m, bio: (m["LM"] + m["MM"]) / 2, parent_name="SHANK")
    )

    model["FOOT"] = SegmentTemplate(
        segment_coordinate_system=NaturalSegmentCoordinateSystemTemplate(
            u_axis=AxisTemplate(
                start="ANKLE_JOINT",
                # u_axis is defined from calcaneous (CAL) to the middle of M1 and M5
                end=lambda m, bio: (m["CAL"] - (m["M1"] + m["M5"]) / 2)
                / np.linalg.norm(m["CAL"] - (m["M1"] + m["M5"]) / 2),
            ),
            proximal_point="ANKLE_JOINT",
            #  middle of M1 and M5
            distal_point=lambda m, bio: (m["M1"] + m["M5"]) / 2,
            w_axis=AxisTemplate(start="M1", end="M5"),
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
