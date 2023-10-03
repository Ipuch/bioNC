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
from bionc import InverseKinematics, Viz
from tests.utils import TestUtils
import time

# special load to test the script
# otherwise one could have use standard import
bionc = TestUtils.bionc_folder()
module = TestUtils.load_module(bionc + "/examples/model_creation/right_side_lower_limb.py")


def model_creation_from_measured_data(
    c3d_filename: str = "statref.c3d",
) -> BiomechanicalModel:
    """
    Create a model from a data file and we build the biomechanical model as a template using the marker names
    """

    # Fill the kinematic chain model
    model = BiomechanicalModelTemplate()
    # de_leva = DeLevaTable(total_mass=100, sex="female")

    right_hip_joint = lambda m, bio: module.harrington2007(m["RASI"], m["LASI"], m["RPSI"], m["LPSI"])[0]
    left_hip_joint = lambda m, bio: module.harrington2007(m["RASI"], m["LASI"], m["RPSI"], m["LPSI"])[1]
    mid_hip_joint = lambda m, bio: (left_hip_joint(m, bio) + right_hip_joint(m, bio)) / 2
    right_knee_joint = lambda m, bio: MarkerTemplate.middle_of(m, bio, "RFLE", "RFME")
    right_ankle_joint = lambda m, bio: MarkerTemplate.middle_of(m, bio, "RLM", "RMM")
    left_knee_joint = lambda m, bio: MarkerTemplate.middle_of(m, bio, "LFLE", "LFME")
    left_ankle_joint = lambda m, bio: MarkerTemplate.middle_of(m, bio, "LLM", "LMM")

    model["PELVIS"] = SegmentTemplate(
        natural_segment=NaturalSegmentTemplate(
            u_axis=AxisTemplate(
                # from the middle of posterior illiac spine to the middle of anterior illiac spine
                start=lambda m, bio: MarkerTemplate.middle_of(m, bio, "RPSI", "LPSI"),
                end=lambda m, bio: MarkerTemplate.middle_of(m, bio, "RASI", "LASI"),
            ),
            # middle of the right and left posterior superior iliac spine
            # or sacroiliac joint
            proximal_point=lambda m, bio: MarkerTemplate.middle_of(m, bio, "RPSI", "LPSI"),
            # Hip joint center projected in the sagittal plane of the pelvis
            # middle of the right and left hip joint center
            distal_point=lambda m, bio: mid_hip_joint(m, bio),
            # normal to the sagittal plane of the pelvis
            w_axis=AxisTemplate(start=left_hip_joint, end=right_hip_joint),
        )
    )

    model["PELVIS"].add_marker(MarkerTemplate(name="RASI", parent_name="PELVIS", is_technical=True))
    model["PELVIS"].add_marker(MarkerTemplate(name="LASI", parent_name="PELVIS", is_technical=True))
    model["PELVIS"].add_marker(MarkerTemplate(name="RPSI", parent_name="PELVIS", is_technical=True))
    model["PELVIS"].add_marker(MarkerTemplate(name="LPSI", parent_name="PELVIS", is_technical=True))
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
            name="LEFT_HIP_JOINT",
            function=left_hip_joint,
            parent_name="PELVIS",
            is_technical=False,
            is_anatomical=True,
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
            name="RIGHT_HIP_JOINT",
            function=right_hip_joint,
            parent_name="RTHIGH",
            is_technical=False,
            is_anatomical=True,
        )
    )
    model["RTHIGH"].add_marker(MarkerTemplate("RGT", parent_name="RTHIGH", is_technical=True))
    model["RTHIGH"].add_marker(MarkerTemplate("RFLE", parent_name="RTHIGH", is_technical=True))
    model["RTHIGH"].add_marker(MarkerTemplate("RFME", parent_name="RTHIGH", is_technical=True))
    model["RTHIGH"].add_marker(
        MarkerTemplate(
            "RIGHT_KNEE_JOINT",
            function=right_knee_joint,
            parent_name="RTHIGH",
            is_technical=False,
            is_anatomical=True,
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
            name="LEFT_HIP_JOINT",
            function=left_hip_joint,
            parent_name="LTHIGH",
            is_technical=False,
            is_anatomical=True,
        )
    )
    model["LTHIGH"].add_marker(MarkerTemplate("LGT", parent_name="LTHIGH", is_technical=True))
    model["LTHIGH"].add_marker(MarkerTemplate("LFLE", parent_name="LTHIGH", is_technical=True))
    model["LTHIGH"].add_marker(MarkerTemplate("LFME", parent_name="LTHIGH", is_technical=True))
    model["LTHIGH"].add_marker(
        MarkerTemplate(
            "LEFT_KNEE_JOINT",
            function=left_knee_joint,
            parent_name="LTHIGH",
            is_technical=False,
            is_anatomical=True,
        )
    )

    model["RSHANK"] = SegmentTemplate(
        natural_segment=NaturalSegmentTemplate(
            u_axis=AxisFunctionTemplate(
                function=lambda m, bio: MarkerTemplate.normal_to(m, bio, right_knee_joint(m, bio), "RLM", "RMM")
            ),
            proximal_point=right_knee_joint,
            # the knee joint computed from the medial femoral epicondyle and the lateral femoral epicondyle
            distal_point=lambda m, bio: MarkerTemplate.middle_of(m, bio, "RLM", "RMM"),
            w_axis=AxisTemplate(start="RMM", end="RLM"),
        )
    )
    model["RSHANK"].add_marker(
        MarkerTemplate(
            "RIGHT_KNEE_JOINT",
            right_knee_joint,
            parent_name="RSHANK",
            is_technical=False,
            is_anatomical=True,
        )
    )
    model["RSHANK"].add_marker(MarkerTemplate("RLM", parent_name="RSHANK", is_technical=True))
    model["RSHANK"].add_marker(MarkerTemplate("RMM", parent_name="RSHANK", is_technical=True))
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
                function=lambda m, bio: MarkerTemplate.normal_to(m, bio, left_knee_joint(m, bio), "LMM", "LLM")
            ),
            proximal_point=left_knee_joint,
            # the knee joint computed from the medial femoral epicondyle and the lateral femoral epicondyle
            distal_point=lambda m, bio: MarkerTemplate.middle_of(m, bio, "LLM", "LMM"),
            w_axis=AxisTemplate(start="LLM", end="LMM"),
        )
    )
    model["LSHANK"].add_marker(
        MarkerTemplate(
            "LEFT_KNEE_JOINT",
            left_knee_joint,
            parent_name="LSHANK",
            is_technical=False,
            is_anatomical=True,
        )
    )
    model["LSHANK"].add_marker(MarkerTemplate("LLM", parent_name="LSHANK", is_technical=True))
    model["LSHANK"].add_marker(MarkerTemplate("LMM", parent_name="LSHANK", is_technical=True))
    model["LSHANK"].add_marker(
        MarkerTemplate(
            "LEFT_ANKLE_JOINT",
            function=left_ankle_joint,
            parent_name="LSHANK",
            is_technical=False,
            is_anatomical=True,
        )
    )

    model["RFOOT"] = SegmentTemplate(
        natural_segment=NaturalSegmentTemplate(
            # u_axis is defined from calcaneous (CAL) to the middle of M1 and M5
            u_axis=AxisTemplate(
                start="RHEE",
                end=lambda m, bio: MarkerTemplate.middle_of(m, bio, "RFM1", "RFM5"),
            ),
            proximal_point=right_ankle_joint,
            #  middle of M1 and M5
            distal_point=lambda m, bio: MarkerTemplate.middle_of(m, bio, "RFM1", "RFM5"),
            w_axis=AxisTemplate(start="RFM1", end="RFM5"),
        )
    )

    model["RFOOT"].add_marker(MarkerTemplate("RHEE", parent_name="RFOOT", is_technical=True))
    model["RFOOT"].add_marker(MarkerTemplate("RFM1", parent_name="RFOOT", is_technical=True))
    model["RFOOT"].add_marker(MarkerTemplate("RFM5", parent_name="RFOOT", is_technical=True))
    model["RFOOT"].add_marker(
        MarkerTemplate(
            "RIGHT_ANKLE_JOINT",
            function=right_ankle_joint,
            parent_name="RFOOT",
            is_technical=False,
            is_anatomical=True,
        )
    )

    model["LFOOT"] = SegmentTemplate(
        natural_segment=NaturalSegmentTemplate(
            # u_axis is defined from calcaneous (CAL) to the middle of M1 and M5
            u_axis=AxisTemplate(
                start="LHEE",
                end=lambda m, bio: MarkerTemplate.middle_of(m, bio, "LFM1", "LFM5"),
            ),
            proximal_point=left_ankle_joint,
            #  middle of M1 and M5
            distal_point=lambda m, bio: MarkerTemplate.middle_of(m, bio, "LFM1", "LFM5"),
            w_axis=AxisTemplate(start="LFM5", end="LFM1"),
        )
    )
    #
    model["LFOOT"].add_marker(MarkerTemplate("LHEE", parent_name="LFOOT", is_technical=True))
    model["LFOOT"].add_marker(MarkerTemplate("LFM1", parent_name="LFOOT", is_technical=True))
    model["LFOOT"].add_marker(MarkerTemplate("LFM5", parent_name="LFOOT", is_technical=True))
    model["LFOOT"].add_marker(
        MarkerTemplate(
            "LEFT_ANKLE_JOINT",
            function=left_ankle_joint,
            parent_name="LFOOT",
            is_technical=False,
            is_anatomical=True,
        )
    )

    model.add_joint(
        name="right_hip",
        joint_type=JointType.SPHERICAL,
        parent="PELVIS",
        child="RTHIGH",
        parent_point="RIGHT_HIP_JOINT",
        child_point="RIGHT_HIP_JOINT",
    )

    model.add_joint(
        name="left_hip",
        joint_type=JointType.CONSTANT_LENGTH,
        parent="PELVIS",
        child="LTHIGH",
        parent_point="LEFT_HIP_JOINT",
        child_point="LEFT_HIP_JOINT",
        length=0.05,
    )
    # model.add_joint(
    #     name="right_hip",
    #     joint_type=JointType.GENERAL_SPHERICAL,
    #     parent="PELVIS",
    #     child="RTHIGH",
    #     parent_point="RIGHT_HIP_JOINT",
    #     child_point="RIGHT_HIP_JOINT",
    # )
    #
    # model.add_joint(
    #     name="left_hip",
    #     joint_type=JointType.GENERAL_SPHERICAL,
    #     parent="PELVIS",
    #     child="LTHIGH",
    #     parent_point="LEFT_HIP_JOINT",
    #     child_point="LEFT_HIP_JOINT",
    # )

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
    # model.add_joint(
    #     name="left_knee",
    #     joint_type=JointType.HINGE,
    #     parent="LTHIGH",
    #     child="LSHANK",
    #     parent_axis=[
    #         model["LTHIGH"].natural_segment.u_axis,
    #         model["LTHIGH"].natural_segment.w_axis,
    #     ],
    #     child_axis=[model["LSHANK"].natural_segment.w_axis, model["LSHANK"].natural_segment.u_axis],
    #     theta=[90, 90],
    # )

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
    filename_static = "Statique_en_m.c3d"
    filename_dynamic = "Marche_en_m.c3d"

    # Create the model from a c3d file and markers as template
    model = model_creation_from_measured_data(filename_static)

    # load experimental markers, usecols is used to select only the markers that are in the model
    # and it rearrange them in the same order as the model
    markers = Markers.from_c3d(filename_dynamic, usecols=model.marker_names_technical).to_numpy()

    # compute the natural coordinates ==> is not necessary if you want to use the inverse kinematics
    Qxp = model.Q_from_markers(markers[:, :, :])

    # you can import the class from bionc
    ik_solver = InverseKinematics(model, markers[0:3, :, :], solve_frame_per_frame=True)
    # or you can use the model method
    ik_solver = model.inverse_kinematics(markers[0:3, :, :], solve_frame_per_frame=True)

    tic0 = time.time()
    Qopt_sqp = ik_solver.solve(method="sqpmethod")  # tend to be faster (with limited-memory hessian approximation)
    toc0 = time.time()

    tic1 = time.time()
    Qopt_ipopt = ik_solver.solve(method="ipopt")  # tend to find lower cost functions but may flip axis.
    toc1 = time.time()

    print(f"Time to solve {markers.shape[2]} frames with sqpmethod: {toc0 - tic0}")
    # print(f"time to solve {markers.shape[2]} frames with ipopt: {toc1 - tic1}")

    # display the experimental markers in red and the model markers in green
    # almost superimposed because the model is well defined on the experimental data
    bionc_viz = Viz(model, show_center_of_mass=False, show_model_markers=True, show_frames=True)
    # bionc_viz.animate(Qxp, markers_xp=markers)
    bionc_viz.animate(Qxp, markers_xp=markers)
    bionc_viz.animate(Qopt_sqp, markers_xp=markers)
    bionc_viz.animate(Qopt_ipopt, markers_xp=markers)

    # dump the model in a pickle format
    model.save("../models/lower_limbs.nc")


if __name__ == "__main__":
    main()
