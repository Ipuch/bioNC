from pyomeca import Markers
from bionc import (
    AxisTemplate,
    AxisFunctionTemplate,
    BiomechanicalModelTemplate,
    ExternalForceList,
    MarkerTemplate,
    SegmentTemplate,
    NaturalSegmentTemplate,
    SegmentNaturalCoordinates,
    NaturalAccelerations,
    NaturalCoordinates,
    SegmentNaturalAccelerations,
    InertiaParametersTemplate,
    C3dData,
    BiomechanicalModel,
    JointType,
    ExternalForce,
)

import ezc3d
import PF_processing as PF_proc
import signal_processing as signal_proc
from bionc import InverseKinematics, Viz
from tests.utils import TestUtils
import time
import numpy as np
from scipy import signal

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

    right_hip_joint = lambda m, bio: module.harrington2007(m["RIAS"], m["LIAS"], m["RIPS"], m["LIPS"])[0]
    left_hip_joint = lambda m, bio: module.harrington2007(m["RIAS"], m["LIAS"], m["RIPS"], m["LIPS"])[1]
    mid_hip_joint = lambda m, bio: (left_hip_joint(m, bio) + right_hip_joint(m, bio)) / 2
    right_knee_joint = lambda m, bio: MarkerTemplate.middle_of(m, bio, "RFLE", "RFME")
    right_ankle_joint = lambda m, bio: MarkerTemplate.middle_of(m, bio, "RFAL", "RTAM")

    model["PELVIS"] = SegmentTemplate(
        natural_segment=NaturalSegmentTemplate(
            u_axis=AxisTemplate(
                # from the middle of posterior illiac spine to the middle of anterior illiac spine
                start=lambda m, bio: MarkerTemplate.middle_of(m, bio, "RIPS", "LIPS"),
                end=lambda m, bio: MarkerTemplate.middle_of(m, bio, "RIAS", "LIAS"),
            ),
            # middle of the right and left posterior superior iliac spine
            # or sacroiliac joint
            proximal_point=lambda m, bio: MarkerTemplate.middle_of(m, bio, "RIPS", "LIPS"),
            # Hip joint center projected in the sagittal plane of the pelvis
            # middle of the right and left hip joint center
            distal_point=lambda m, bio: mid_hip_joint(m, bio),
            # normal to the sagittal plane of the pelvis
            w_axis=AxisTemplate(start=left_hip_joint, end=right_hip_joint),
        ),
        inertia_parameters=InertiaParametersTemplate(
            mass=1,
            center_of_mass=np.array([0, -0.5, 0]),  # in segment coordinates system
            inertia=np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
        ),
    )

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
        ),
        inertia_parameters=InertiaParametersTemplate(
            mass=1,
            center_of_mass=np.array([0, -0.5, 0]),  # in segment coordinates system
            inertia=np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
        ),
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
    model["RTHIGH"].add_marker(MarkerTemplate("RThigh", parent_name="RTHIGH", is_technical=True))
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

    model["RSHANK"] = SegmentTemplate(
        natural_segment=NaturalSegmentTemplate(
            u_axis=AxisFunctionTemplate(
                function=lambda m, bio: MarkerTemplate.normal_to(m, bio, right_knee_joint(m, bio), "RFAL", "RTAM")
            ),
            proximal_point=right_knee_joint,
            # the knee joint computed from the medial femoral epicondyle and the lateral femoral epicondyle
            distal_point=lambda m, bio: MarkerTemplate.middle_of(m, bio, "RFAL", "RTAM"),
            w_axis=AxisTemplate(start="RTAM", end="RFAL"),
        ),
        inertia_parameters=InertiaParametersTemplate(
            mass=1,
            center_of_mass=np.array([0, -0.5, 0]),  # in segment coordinates system
            inertia=np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
        ),
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
        ),
        inertia_parameters=InertiaParametersTemplate(
            mass=1,
            center_of_mass=np.array([0, -0.5, 0]),  # in segment coordinates system
            inertia=np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
        ),
    )

    # Update natural inertia parameters

    model["RFOOT"].add_marker(MarkerTemplate("RFCC", parent_name="RFOOT", is_technical=True))
    model["RFOOT"].add_marker(MarkerTemplate("RFM5", parent_name="RFOOT", is_technical=True))
    model["RFOOT"].add_marker(MarkerTemplate("RFM1", parent_name="RFOOT", is_technical=True))

    model["RFOOT"].add_marker(
        MarkerTemplate(
            "RIGHT_ANKLE_JOINT",
            function=right_ankle_joint,
            parent_name="RFOOT",
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
        name="right_knee",
        joint_type=JointType.SPHERICAL,
        parent="RTHIGH",
        child="RSHANK",
    )

    model.add_joint(
        name="right_ankle",
        joint_type=JointType.SPHERICAL,
        parent="RSHANK",
        child="RFOOT",
    )

    c3d_data = C3dData(f"{c3d_filename}")

    # Put the model together, print it and print it to a bioMod file
    natural_model = model.update(c3d_data)

    return natural_model


def main():
    # create a c3d file with data
    filename_static = "../../sandbox/subject03_walk_c3d_m_short.c3d"
    filename_dynamic = "../../sandbox/subject03_walk_c3d_m_short.c3d"

    # ---------Inverse kinematics----------------
    # Create the model from a c3d file and markers as template
    model = model_creation_from_measured_data(filename_static)

    # load experimental markers, usecols is used to select only the markers that are in the model
    # and it rearrange them in the same order as the model
    markers = Markers.from_c3d(filename_dynamic, usecols=model.marker_names_technical).to_numpy()

    # compute the natural coordinates ==> is not necessary if you want to use the inverse kinematics
    Qxp = model.Q_from_markers(markers[:, :, :])

    # you can import the class from bionc
    # model_casadi = model._model_mx
    # model_casadi.segments_no_ground['RTHIGH'].values()
    # phim = model_casadi.markers_constraints(markers, Qxp, only_technical=True)

    ik_solver = InverseKinematics(model, markers[0:3, :, :], solve_frame_per_frame=True)
    toto = ik_solver._model_mx
    # or you can use the model method
    ik_solver = model.inverse_kinematics(markers[0:3, :, :], solve_frame_per_frame=True)

    tic0 = time.time()
    Qopt_sqp = ik_solver.solve(method="sqpmethod")  # tend to be faster (with limited-memory hessian approximation)
    toc0 = time.time()

    # ---------Dynamics----------------
    # Extract the forces plateforms
    c3d = ezc3d.c3d(filename_dynamic, extract_forceplat_data=True)

    # Calculation of the qdot and qddot
    freq = c3d["parameters"]["POINT"]["RATE"]["value"][0]  # frequency
    nb_frame = c3d["parameters"]["POINT"]["FRAMES"]["value"][0]

    cut_freq = 6

    w = cut_freq / (freq / 2)  # Normalize the frequency
    b, a = signal.butter(4, w, "low")

    Qdopt_sqp = np.gradient(Qopt_sqp, 1 / freq, axis=1)
    Qdopt_sqp_filt = signal.filtfilt(b, a, Qdopt_sqp, axis=1)
    Qddopt_sqp = np.gradient(Qdopt_sqp_filt, 1 / freq, axis=1)
    Qddopt_sqp_filt = signal.filtfilt(b, a, Qddopt_sqp, axis=1)

    PF1_force, PF1_moment, PF1_origin, PF1_freq = PF_proc.extract_data(c3d, 0)
    PF2_force, PF2_moment, PF2_origin, PF2_freq = PF_proc.extract_data(c3d, 1)

    # # resampling
    PF1_force, PF1_moment = PF_proc.resample_data(PF1_force, PF1_moment, freq, PF1_freq, nb_frame)
    PF2_force, PF2_moment = PF_proc.resample_data(PF2_force, PF2_moment, freq, PF2_freq, nb_frame)

    # compute moments at the origin of the global reference frame
    PF1_moment0 = PF_proc.moment_at_origin(PF1_force, PF1_moment, PF1_origin)
    PF2_moment0 = PF_proc.moment_at_origin(PF2_force, PF2_moment, PF2_origin)

    # compute CoP (for the visualisation)
    PF1_CoP = PF_proc.compute_CoP(PF1_force, PF1_moment0)
    PF2_CoP = PF_proc.compute_CoP(PF2_force, PF2_moment0)

    # So now I have my forces express in the global frame :
    # How to express it in the segment frame
    center_of_plateform = np.repeat(PF1_origin[:, np.newaxis], Qopt_sqp.shape[1], axis=1)
    # Calculation of point - rp
    u_foot = Qopt_sqp[36:39, :]
    rp_foot = Qopt_sqp[39:42, :]
    rd_foot = Qopt_sqp[42:45, :]
    w_foot = Qopt_sqp[45:48, :]

    from bionc.bionc_numpy.cartesian_vector import vector_projection_in_non_orthogonal_basis

    point = vector_projection_in_non_orthogonal_basis(center_of_plateform - rp_foot, u_foot, rp_foot - rd_foot, w_foot)

    # Projection of the force on the segment frame
    position_point_application = rp_foot - np.repeat(PF1_origin[:, np.newaxis], Qopt_sqp.shape[1], axis=1)

    GRF = ExternalForce.from_components(
        application_point_in_local=position_point_application, force=PF1_force, torque=PF1_moment0
    )
    list_external_force = ExternalForceList.empty_from_nb_segment(model.nb_segments)
    # Integration of the force in the list
    list_external_force.add_external_force(3, GRF)
    # How to do a list of forces :
    # Should it be expressed in the global reference frame or in the segment frame ?
    for ind_frame in range(0, Qopt_sqp.shape[1]):
        tuple_of_Q = [
            SegmentNaturalCoordinates.from_components(
                u=Qopt_sqp[i * 12 : i * 12 + 3, ind_frame],
                rp=Qopt_sqp[i * 12 + 3 : i * 12 + 6, ind_frame],
                rd=Qopt_sqp[i * 12 + 6 : i * 12 + 9, ind_frame],
                w=Qopt_sqp[i * 12 + 9 : i * 12 + 12, ind_frame],
            )
            for i in range(0, model.nb_segments)
        ]
        tuple_of_Qddot = [
            SegmentNaturalAccelerations.from_components(
                uddot=Qddopt_sqp_filt[i * 12 : i * 12 + 3, ind_frame],
                rpddot=Qddopt_sqp_filt[i * 12 + 3 : i * 12 + 6, ind_frame],
                rdddot=Qddopt_sqp_filt[i * 12 + 6 : i * 12 + 9, ind_frame],
                wddot=Qddopt_sqp_filt[i * 12 + 9 : i * 12 + 12, ind_frame],
            )
            for i in range(0, model.nb_segments)
        ]
        list_temp_force = ExternalForceList.empty_from_nb_segment(model.nb_segments)
        for ind_segment in range(0, model.nb_segments):
            if len(list_external_force.segment_external_forces(ind_segment)) > 0:
                point_temp = list_external_force.segment_external_forces(3)[0].application_point_in_local[:, ind_frame]
                force_temp = list_external_force.segment_external_forces(3)[0].force[:, ind_frame]
                torque_temp = list_external_force.segment_external_forces(3)[0].torque[:, ind_frame]
                list_temp_force.add_external_force(
                    ind_segment,
                    ExternalForce.from_components(
                        application_point_in_local=point_temp, force=force_temp, torque=torque_temp
                    ),
                )

        Q = NaturalCoordinates.from_qi(tuple(tuple_of_Q))
        Qddot = NaturalAccelerations.from_qddoti(tuple(tuple_of_Qddot))
        torques, forces, lambdas = model.inverse_dynamics(Q=Q, Qddot=Qddot, external_forces=list_temp_force)

    # tic1 = time.time()
    # Qopt_ipopt = ik_solver.solve(method="ipopt")  # tend to find lower cost functions but may flip axis.
    # toc1 = time.time()

    # print(f"Time to solve {markers.shape[2]} frames with sqpmethod: {toc0 - tic0}")
    # print(f"time to solve {markers.shape[2]} frames with ipopt: {toc1 - tic1}")

    # display the experimental markers in red and the model markers in green
    # almost superimposed because the model is well defined on the experimental data
    # bionc_viz = Viz(model, show_center_of_mass=False, show_model_markers=True, show_frames=True)
    # # bionc_viz.animate(Qxp, markers_xp=markers)
    # #bionc_viz.animate(Qxp, markers_xp=markers)
    # bionc_viz.animate(Qopt_sqp, markers_xp=markers)
    # bionc_viz.animate(Qopt_ipopt, markers_xp=markers)

    # dump the model in a pickle format
    # model.save("../models/lower_limbs.nc")


if __name__ == "__main__":
    main()
