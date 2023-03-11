import numpy as np

from bionc import JointType

# from bionc.bionc_casadi import (
#     BiomechanicalModel,
#     NaturalSegment,
# )
from bionc.bionc_numpy import (
    BiomechanicalModel,
    NaturalSegment,
    NaturalMarker,
)


def create_knee_model() -> BiomechanicalModel:
    """
    This function creates a biomechanical model of a knee with a parallel mechanism.

    Data from
    ----------
    Sancisi, N., Gasparutto, X., Parenti-Castelli, V., & Dumas, R. (2017).
    A multi-body optimization framework with a knee kinematic model including articular contacts and ligaments.
    Meccanica, 52, 695-711.

    """
    model = BiomechanicalModel()
    # fill the biomechanical model with the segment
    model["THIGH"] = NaturalSegment(
        name="THIGH",
        alpha=89.15807479515183 * np.pi / 180,
        beta=89.99767608318231 * np.pi / 180,
        gamma=89.99418352863022 * np.pi / 180,
        length=0.3460518602753062,
        mass=6,
        center_of_mass=np.array([0, -0.1, 0]),  # in segment coordinates system
        inertia=np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),  # in segment coordinates system
    )

    # added markers
    model["THIGH"].add_natural_marker(
        NaturalMarker(
            name="hip_center",
            parent_name="THIGH",
            position=np.array([0, 0, 0]),
            is_technical=True,
            is_anatomical=False,
        )
    )
    model["THIGH"].add_natural_marker(
        NaturalMarker(
            name="condyle_center",
            parent_name="THIGH",
            position=np.array([0, -1, 0]),
            is_technical=True,
            is_anatomical=False,
        )
    )

    model["THIGH"].add_natural_marker_from_segment_coordinates(
        name="medial_centre_femur",
        location=np.array([0.0002458, 0.0034071, -0.0232019]),
        is_distal_location=True,
        is_technical=True,
        is_anatomical=True,
    )
    model["THIGH"].add_natural_marker_from_segment_coordinates(
        name="lateral_centre_femur",
        location=np.array([-0.0032853, 0.0021225, 0.0262054]),
        is_distal_location=True,
        is_technical=True,
        is_anatomical=True,
    )
    model["THIGH"].add_natural_marker_from_segment_coordinates(
        name="ACL_femur",
        location=np.array([-0.0067712, 0.0075255, 0.0091575]),
        is_distal_location=True,
        is_technical=False,
        is_anatomical=True,
    )
    model["THIGH"].add_natural_marker_from_segment_coordinates(
        name="PCL_femur",
        location=np.array([-0.0026610, -0.0010906, -0.0021857]),
        is_distal_location=True,
        is_technical=False,
        is_anatomical=True,
    )
    model["THIGH"].add_natural_marker_from_segment_coordinates(
        name="MCL_femur",
        location=np.array([0.0027608, 0.0057798, -0.0476279]),
        is_distal_location=True,
        is_technical=False,
        is_anatomical=True,
    )
    model["THIGH"].add_natural_marker_from_segment_coordinates(
        name="LCL_femur",
        location=np.array([0.0032800, 0.0022812, -0.0361895]),
        is_distal_location=True,
        is_technical=False,
        is_anatomical=True,
    )

    model["SHANK"] = NaturalSegment(
        name="SHANK",
        alpha=103.42850151901055 * np.pi / 180,
        beta=101.68512120130346 * np.pi / 180,
        gamma=89.99891614467779 * np.pi / 180,
        length=0.3964720545006924,
        mass=4,
        center_of_mass=np.array([0, -0.1, 0]),  # in segment coordinates system
        inertia=np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),  # in segment coordinates system
    )

    model["SHANK"].add_natural_marker(
        NaturalMarker(
            name="condyle_center",
            parent_name="SHANK",
            position=np.array([0, 0, 0]),
            is_technical=True,
            is_anatomical=False,
        )
    )
    model["SHANK"].add_natural_marker(
        NaturalMarker(
            name="ankle_center",
            parent_name="SHANK",
            position=np.array([0, -1, 0]),
            is_technical=True,
            is_anatomical=False,
        )
    )

    model["SHANK"].add_natural_marker_from_segment_coordinates(
        name="medial_contact_shank",
        location=np.array([-0.0021344, -0.0286241, -0.0191308]),
        is_technical=False,
        is_anatomical=True,
    )
    model["SHANK"].add_natural_marker_from_segment_coordinates(
        name="lateral_contact_shank",
        location=np.array([-0.0027946, -0.0260861, 0.0243679]),
        is_technical=False,
        is_anatomical=True,
    )
    model["SHANK"].add_natural_marker_from_segment_coordinates(
        name="ACL_shank",
        location=np.array([0.0127709, -0.0261454, -0.0000617]),
        is_technical=False,
        is_anatomical=True,
    )
    model["SHANK"].add_natural_marker_from_segment_coordinates(
        name="PCL_shank",
        location=np.array([-0.0258519, -0.0381449, -0.003521]),
        is_technical=False,
        is_anatomical=True,
    )
    model["SHANK"].add_natural_marker_from_segment_coordinates(
        name="MCL_shank",
        location=np.array([0.0021345, -0.11710682, -0.0057872]),
        is_technical=False,
        is_anatomical=True,
    )
    model["SHANK"].add_natural_marker_from_segment_coordinates(
        name="LCL_shank",
        location=np.array([0.0242639, -0.0479992, 0.0371213]),
        is_technical=False,
        is_anatomical=True,
    )
    model["SHANK"].add_natural_vector_from_segment_coordinates(
        name="medial_normal_shank",
        direction=np.array([0.0000675, 0.0009896, -0.0001273]),
        normalize=True,
    )
    model["SHANK"].add_natural_vector_from_segment_coordinates(
        name="lateral_normal_shank",
        direction=np.array([-0.0000881, 0.0009942, 0.0000617]),
        normalize=True,
    )

    # model._add_joint(
    #     dict(
    #         name="WELD",
    #         joint_type=JointType.GROUND_WELD,
    #         parent="GROUND",
    #         child="THIGH",
    #         Q_child_ref=np.array([0.9339, 0.3453, 0.0927, -0.0632, 0.7578, 0.1445, 0.0564, 0.4331, 0.1487, -0.0962, -0.0069, 0.9953])
    #     )
    # )

    model._add_joint(
        dict(
            name="HIP",
            joint_type=JointType.GROUND_SPHERICAL,
            parent="GROUND",
            child="THIGH",
            ground_application_point=np.array([-0.0632, 0.7578, 0.1445]),
        )
    )


    model._add_joint(
        dict(
            name="SPHERICAL_KNEE",
            joint_type=JointType.SPHERICAL,
            parent="THIGH",
            child="SHANK",
        )
    )

    # Parallel Mechanism joint definition
    # model._add_joint(
    #     dict(
    #         name="medial_knee",
    #         joint_type=JointType.SPHERE_ON_PLANE,
    #         parent="THIGH",
    #         child="SHANK",
    #         sphere_radius=0.03232,
    #         sphere_center="medial_centre_femur",
    #         plane_normal="medial_normal_shank",
    #         plane_point="medial_contact_shank",
    #     )
    # )
    # model._add_joint(
    #     dict(
    #         name="lateral_knee",
    #         joint_type=JointType.SPHERE_ON_PLANE,
    #         parent="THIGH",
    #         child="SHANK",
    #         sphere_radius=0.02834,
    #         sphere_center="lateral_centre_femur",
    #         plane_normal="lateral_normal_shank",
    #         plane_point="lateral_contact_shank",
    #     )
    # )
    # model._add_joint(
    #     dict(
    #         name="ACL",
    #         joint_type=JointType.CONSTANT_LENGTH,
    #         parent="THIGH",
    #         child="SHANK",
    #         # length=0.04053,  # from article
    #         length=0.04356820117764876,  # from data
    #         parent_point="ACL_femur",
    #         child_point="ACL_shank",
    #     )
    # )
    # model._add_joint(
    #     dict(
    #         name="PCL",
    #         joint_type=JointType.CONSTANT_LENGTH,
    #         parent="THIGH",
    #         child="SHANK",
    #         # length=0.04326, # from article
    #         length=0.042833461279363376,  # from data
    #         parent_point="PCL_femur",
    #         child_point="PCL_shank",
    #     )
    # )
    # model._add_joint(
    #     dict(
    #         name="MCL",
    #         joint_type=JointType.CONSTANT_LENGTH,
    #         parent="THIGH",
    #         child="SHANK",
    #         # length=0.12970, # from article
    #         length=0.10828123262317323,
    #         parent_point="MCL_femur",
    #         child_point="MCL_shank",
    #     )
    # )

    return model
