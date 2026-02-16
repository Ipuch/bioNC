import numpy as np
import os
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
)
from tests.utils import TestUtils

# special load to test the script
# otherwise one could have use standard import
bionc = TestUtils.bionc_folder()
module = TestUtils.load_module(bionc + "/examples/model_creation/right_side_lower_limb.py")


def model_creation_from_measured_data(c3d_filename: str = "statref.c3d") -> BiomechanicalModel:
    """
    Create a model from a data file and we build the biomechanical model as a template using the marker names
    """

    # Fill the kinematic chain model
    model = BiomechanicalModelTemplate()
    # de_leva = DeLevaTable(total_mass=100, sex="female")

    right_hip_joint = lambda m, bio: module.harrington2007(m["RFWT"], m["LFWT"], m["RBWT"], m["LBWT"])[0]
    left_hip_joint = lambda m, bio: module.harrington2007(m["RFWT"], m["LFWT"], m["RBWT"], m["LBWT"])[1]
    right_knee_joint = lambda m, bio: MarkerTemplate.middle_of(m, bio, "RKNI", "RKNE")
    right_ankle_joint = lambda m, bio: MarkerTemplate.middle_of(m, bio, "RANE", "RANI")
    left_knee_joint = lambda m, bio: MarkerTemplate.middle_of(m, bio, "LKNI", "LKNE")
    left_ankle_joint = lambda m, bio: MarkerTemplate.middle_of(m, bio, "LANE", "LANI")

    model["PELVIS"] = SegmentTemplate(
        natural_segment=NaturalSegmentTemplate(
            u_axis=AxisTemplate(
                # from the middle of posterior illiac spine to the middle of anterior illiac spine
                start=lambda m, bio: MarkerTemplate.middle_of(m, bio, "RBWT", "LBWT"),
                end=lambda m, bio: MarkerTemplate.middle_of(m, bio, "RFWT", "LFWT"),
            ),
            # middle of the right and left posterior superior iliac spine
            # or sacroiliac joint
            proximal_point=lambda m, bio: MarkerTemplate.middle_of(m, bio, "RBWT", "LBWT"),
            # Hip joint center projected in the sagittal plane of the pelvis
            # middle of the right and left hip joint center
            distal_point=lambda m, bio: right_hip_joint(m, bio),
            # normal to the sagittal plane of the pelvis
            w_axis=AxisTemplate(start=left_hip_joint, end=right_hip_joint),
        )
    )

    model["PELVIS"].add_marker(MarkerTemplate(name="RFWT", parent_name="PELVIS", is_technical=True))
    model["PELVIS"].add_marker(MarkerTemplate(name="LFWT", parent_name="PELVIS", is_technical=True))
    model["PELVIS"].add_marker(MarkerTemplate(name="RBWT", parent_name="PELVIS", is_technical=True))
    model["PELVIS"].add_marker(MarkerTemplate(name="LBWT", parent_name="PELVIS", is_technical=True))
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
                function=lambda m, bio: MarkerTemplate.normal_to(m, bio, right_hip_joint(m, bio), "RKNE", "RKNI")
            ),
            proximal_point=right_hip_joint,
            # the knee joint computed from the medial femoral epicondyle and the lateral femoral epicondyle
            distal_point=lambda m, bio: MarkerTemplate.middle_of(m, bio, "RKNE", "RKNI"),
            w_axis=AxisTemplate(start="RKNI", end="RKNE"),
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
    model["RTHIGH"].add_marker(MarkerTemplate("RKNE", parent_name="RTHIGH", is_technical=True))
    model["RTHIGH"].add_marker(MarkerTemplate("RKNI", parent_name="RTHIGH", is_technical=True))
    model["RTHIGH"].add_marker(
        MarkerTemplate(
            "RIGHT_KNEE_JOINT", function=right_knee_joint, parent_name="RTHIGH", is_technical=False, is_anatomical=True
        )
    )

    model["LTHIGH"] = SegmentTemplate(
        natural_segment=NaturalSegmentTemplate(
            u_axis=AxisFunctionTemplate(
                function=lambda m, bio: MarkerTemplate.normal_to(m, bio, left_hip_joint(m, bio), "LKNE", "LKNI")
            ),
            proximal_point=left_hip_joint,
            # the knee joint computed from the medial femoral epicondyle and the lateral femoral epicondyle
            distal_point=lambda m, bio: MarkerTemplate.middle_of(m, bio, "LKNE", "LKNI"),
            w_axis=AxisTemplate(start="LKNE", end="LKNI"),
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
    model["LTHIGH"].add_marker(MarkerTemplate("LKNE", parent_name="LTHIGH", is_technical=True))
    model["LTHIGH"].add_marker(MarkerTemplate("LKNI", parent_name="LTHIGH", is_technical=True))
    model["LTHIGH"].add_marker(
        MarkerTemplate(
            "LEFT_KNEE_JOINT", function=left_knee_joint, parent_name="LTHIGH", is_technical=False, is_anatomical=True
        )
    )

    model["RSHANK"] = SegmentTemplate(
        natural_segment=NaturalSegmentTemplate(
            u_axis=AxisFunctionTemplate(
                function=lambda m, bio: MarkerTemplate.normal_to(m, bio, right_knee_joint(m, bio), "RANE", "RANI")
            ),
            proximal_point=right_knee_joint,
            # the knee joint computed from the medial femoral epicondyle and the lateral femoral epicondyle
            distal_point=lambda m, bio: MarkerTemplate.middle_of(m, bio, "RANE", "RANI"),
            w_axis=AxisTemplate(start="RANI", end="RANE"),
        )
    )
    model["RSHANK"].add_marker(
        MarkerTemplate(
            "RIGHT_KNEE_JOINT", right_knee_joint, parent_name="RSHANK", is_technical=False, is_anatomical=True
        )
    )
    model["RSHANK"].add_marker(MarkerTemplate("RANE", parent_name="RSHANK", is_technical=True))
    model["RSHANK"].add_marker(MarkerTemplate("RANI", parent_name="RSHANK", is_technical=True))
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
                function=lambda m, bio: MarkerTemplate.normal_to(m, bio, left_knee_joint(m, bio), "LANE", "LANI")
            ),
            proximal_point=left_knee_joint,
            # the knee joint computed from the medial femoral epicondyle and the lateral femoral epicondyle
            distal_point=lambda m, bio: MarkerTemplate.middle_of(m, bio, "LANE", "LANI"),
            w_axis=AxisTemplate(start="LANE", end="LANI"),
        )
    )
    model["LSHANK"].add_marker(
        MarkerTemplate("LEFT_KNEE_JOINT", left_knee_joint, parent_name="LSHANK", is_technical=False, is_anatomical=True)
    )
    model["LSHANK"].add_marker(MarkerTemplate("LANE", parent_name="LSHANK", is_technical=True))
    model["LSHANK"].add_marker(MarkerTemplate("LANI", parent_name="LSHANK", is_technical=True))
    model["LSHANK"].add_marker(
        MarkerTemplate(
            "LEFT_ANKLE_JOINT", function=left_ankle_joint, parent_name="LSHANK", is_technical=False, is_anatomical=True
        )
    )

    model["RFOOT"] = SegmentTemplate(
        natural_segment=NaturalSegmentTemplate(
            # u_axis is defined from calcaneous (CAL) to the middle of M1 and M5
            u_axis=AxisTemplate(
                start="RHEE",
                end=lambda m, bio: MarkerTemplate.middle_of(m, bio, "RTARI", "RTAR"),
            ),
            proximal_point=right_ankle_joint,
            #  middle of M1 and M5
            distal_point=lambda m, bio: MarkerTemplate.middle_of(m, bio, "RTARI", "RTAR"),
            w_axis=AxisTemplate(start="RTARI", end="RTAR"),
        )
    )

    model["RFOOT"].add_marker(MarkerTemplate("RHEE", parent_name="RFOOT", is_technical=True))
    model["RFOOT"].add_marker(MarkerTemplate("RTARI", parent_name="RFOOT", is_technical=True))
    model["RFOOT"].add_marker(MarkerTemplate("RTAR", parent_name="RFOOT", is_technical=True))
    model["RFOOT"].add_marker(
        MarkerTemplate(
            "RIGHT_ANKLE_JOINT", function=right_ankle_joint, parent_name="RFOOT", is_technical=False, is_anatomical=True
        )
    )

    model["LFOOT"] = SegmentTemplate(
        natural_segment=NaturalSegmentTemplate(
            # u_axis is defined from calcaneous (CAL) to the middle of M1 and M5
            u_axis=AxisTemplate(
                start="LHEE",
                end=lambda m, bio: MarkerTemplate.middle_of(m, bio, "LTARI", "LTAR"),
            ),
            proximal_point=left_ankle_joint,
            #  middle of M1 and M5
            distal_point=lambda m, bio: MarkerTemplate.middle_of(m, bio, "LTARI", "LTAR"),
            w_axis=AxisTemplate(start="LTAR", end="LTARI"),
        )
    )
    #
    model["LFOOT"].add_marker(MarkerTemplate("LHEE", parent_name="LFOOT", is_technical=True))
    model["LFOOT"].add_marker(MarkerTemplate("LTARI", parent_name="LFOOT", is_technical=True))
    model["LFOOT"].add_marker(MarkerTemplate("LTAR", parent_name="LFOOT", is_technical=True))
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
    )

    model.add_joint(
        name="left_hip",
        joint_type=JointType.SPHERICAL,
        parent="PELVIS",
        child="LTHIGH",
    )

    model.add_joint(
        name="right_knee",
        joint_type=JointType.SPHERICAL,
        parent="RTHIGH",
        child="RSHANK",
    )

    model.add_joint(
        name="left_knee",
        joint_type=JointType.SPHERICAL,
        parent="LTHIGH",
        child="LSHANK",
    )

    model.add_joint(
        name="right_ankle",
        joint_type=JointType.SPHERICAL,
        parent="RSHANK",
        child="RFOOT",
    )

    model.add_joint(
        name="left_ankle",
        joint_type=JointType.SPHERICAL,
        parent="LSHANK",
        child="LFOOT",
    )

    c3d_data = C3dData(f"{c3d_filename}")

    # Put the model together, print it and print it to a bioMod file
    natural_model = model.update(c3d_data)

    return natural_model


def main():
    # create a c3d file with data
    filename = module.generate_c3d_file(two_side=True)
    # Create the model from a c3d file and markers as template
    model = model_creation_from_measured_data(filename)

    # load experimental markers, usecols is used to select only the markers that are in the model
    # and it rearrange them in the same order as the model
    markers_xp = Markers.from_c3d(filename, usecols=model.marker_names_technical).to_numpy()

    # compute the natural coordinates
    Qxp = model.Q_from_markers(markers_xp[:, :, 0:2])

    from bionc.vizualization.pyorerun_interface import BioncModelNoMesh
    from pyorerun import PhaseRerun, PyoMarkers

    # display the experimental markers in blue and the model in white
    # almost superimposed because the model is well defined on the experimental data
    prr = PhaseRerun(t_span=np.linspace(0, 1, markers_xp.shape[2]))
    model_interface = BioncModelNoMesh(model)

    pyomarkers = PyoMarkers.from_c3d(filename)
    prr.add_animated_model(model_interface, Qxp, pyomarkers)
    prr.rerun()

    # remove the c3d file
    os.remove(filename)

    # dump the model in a pickle format
    model.save("../models/lower_limbs.nc")


if __name__ == "__main__":
    main()
