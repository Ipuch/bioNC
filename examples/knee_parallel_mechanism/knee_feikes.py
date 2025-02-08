import numpy as np

from bionc import JointType
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
    Generic model of the right knee
    J.D. Feikes et al. J Biomech 36 (2003) 125-129
    In tibia SCS = femur SCS at full extension
    Origin at midpoint between medial and lateral centers
    X = -X ISB, Y = Z ISB, Z = Y ISB

    """
    model = BiomechanicalModel()
    # fill the biomechanical model with the segment
    model["THIGH"] = NaturalSegment.with_cartesian_inertial_parameters(
        name="THIGH",
        length=0.4,
        mass=12,
        center_of_mass=np.array([0, -0.1, 0]),  # in segment coordinates system
        inertia=np.array([[0.1, 0, 0], [0, 0.1, 0], [0, 0, 0.1]]),  # in segment coordinates system
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
        location=np.array([-5.0, 0.6, -22.3]) / 1000,
        is_distal_location=True,
        is_technical=True,
        is_anatomical=True,
    )
    model["THIGH"].add_natural_marker_from_segment_coordinates(
        name="lateral_centre_femur",
        location=np.array([5.0, -0.6, 22.3]) / 1000,
        is_distal_location=True,
        is_technical=True,
        is_anatomical=True,
    )
    model["THIGH"].add_natural_marker_from_segment_coordinates(
        name="ACL_femur",
        location=np.array([-27.8, 12.7, 5.0]) / 1000,
        is_distal_location=True,
        is_technical=False,
        is_anatomical=True,
    )
    model["THIGH"].add_natural_marker_from_segment_coordinates(
        name="PCL_femur",
        location=np.array([-20.6, -4.3, -15.7]) / 1000,
        is_distal_location=True,
        is_technical=False,
        is_anatomical=True,
    )
    model["THIGH"].add_natural_marker_from_segment_coordinates(
        name="MCL_femur",
        location=np.array([-9.7, 10.2, -42.3]) / 1000,
        is_distal_location=True,
        is_technical=False,
        is_anatomical=True,
    )

    model["SHANK"] = NaturalSegment.with_cartesian_inertial_parameters(
        name="SHANK",
        length=0.4,
        mass=8,
        center_of_mass=np.array([0, -0.1, 0]),  # in segment coordinates system
        inertia=np.array([[0.1, 0, 0], [0, 0.1, 0], [0, 0, 0.1]]),  # in segment coordinates system
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
        location=np.array([-0.005, -0.0202, -0.0223]),
        is_technical=False,
        is_anatomical=True,
    )
    model["SHANK"].add_natural_marker_from_segment_coordinates(
        name="lateral_contact_shank",
        location=np.array([0.005, -0.0202, 0.0223]),
        is_technical=False,
        is_anatomical=True,
    )
    model["SHANK"].add_natural_marker_from_segment_coordinates(
        name="ACL_shank",
        location=np.array([-0.0, -0.0152, 0.0]),
        is_technical=False,
        is_anatomical=True,
    )
    model["SHANK"].add_natural_marker_from_segment_coordinates(
        name="PCL_shank",
        location=np.array([-0.0303, -0.0237, -0.0024]),
        is_technical=False,
        is_anatomical=True,
    )
    model["SHANK"].add_natural_marker_from_segment_coordinates(
        name="MCL_shank",
        location=np.array([0.0012, -0.0672, -0.0036]),
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
        direction=np.array([0, 1, 0]),
        normalize=True,
    )
    model["SHANK"].add_natural_vector_from_segment_coordinates(
        name="lateral_normal_shank",
        direction=np.array([0, 1, 0]),
        normalize=True,
    )

    model._add_joint(
        dict(
            name="HIP",
            joint_type=JointType.GROUND_SPHERICAL,
            parent="GROUND",
            child="THIGH",
            ground_application_point=np.array([0, 0, 0]),
        )
    )

    # model._add_joint(
    #     dict(
    #         name="SPHERICAL_KNEE",
    #         joint_type=JointType.SPHERICAL,
    #         parent="THIGH",
    #         child="SHANK",
    #     )
    # )

    # Parallel Mechanism joint definition
    model._add_joint(
        dict(
            name="medial_knee",
            joint_type=JointType.SPHERE_ON_PLANE,
            parent="THIGH",
            child="SHANK",
            sphere_radius=np.linalg.norm(np.array([-5.0, -20.2, -22.3]) / 1000 - np.array([-5.0, 0.6, -22.3]) / 1000),
            sphere_center="medial_centre_femur",
            plane_normal="medial_normal_shank",
            plane_point="medial_contact_shank",
        )
    )
    model._add_joint(
        dict(
            name="lateral_knee",
            joint_type=JointType.SPHERE_ON_PLANE,
            parent="THIGH",
            child="SHANK",
            sphere_radius=np.linalg.norm(np.array([5.0, -20.2, 22.3]) / 1000 - np.array([5.0, -0.6, 22.3]) / 1000),
            sphere_center="lateral_centre_femur",
            plane_normal="lateral_normal_shank",
            plane_point="lateral_contact_shank",
        )
    )
    model._add_joint(
        dict(
            name="ACL",
            joint_type=JointType.CONSTANT_LENGTH,
            parent="THIGH",
            child="SHANK",
            length=np.linalg.norm(np.array([-0.0, -15.2, 0.0]) / 1000 - np.array([-27.8, 12.7, 5.0]) / 1000),
            parent_point="ACL_femur",
            child_point="ACL_shank",
        )
    )
    model._add_joint(
        dict(
            name="PCL",
            joint_type=JointType.CONSTANT_LENGTH,
            parent="THIGH",
            child="SHANK",
            length=np.linalg.norm(np.array([-20.6, -4.3, -15.7]) / 1000 - np.array([-30.3, -23.7, -2.4]) / 1000),
            parent_point="PCL_femur",
            child_point="PCL_shank",
        )
    )
    model._add_joint(
        dict(
            name="MCL",
            joint_type=JointType.CONSTANT_LENGTH,
            parent="THIGH",
            child="SHANK",
            length=np.linalg.norm(np.array([-9.7, 10.2, -42.3]) / 1000 - np.array([1.2, -67.2, -3.6]) / 1000),
            parent_point="MCL_femur",
            child_point="MCL_shank",
        )
    )

    return model
