import numpy as np

from bionc.bionc_numpy import (
    BiomechanicalModel,
    NaturalSegment,
    JointType,
)
from bionc.bionc_casadi import (
    BiomechanicalModel,
    NaturalSegment,
    # JointType,
)
from bionc import NaturalAxis, CartesianAxis

# Let's create a model
model = BiomechanicalModel()
# fill the biomechanical model with the segment
model["THIGH"] = NaturalSegment(
    name="THIGH",
    alpha=103.42850151901055 * np.pi / 180,
    beta=101.68512120130346 * np.pi / 180,
    gamma=89.99891614467779 * np.pi / 180,
    length=0.3964720545006924,
    mass=6,
    center_of_mass=np.array([0, 0.1, 0]),  # in segment coordinates system
    inertia=np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),  # in segment coordinates system
    is_ground=True,
)

# added markers
model["THIGH"].add_natural_marker_from_segment_coordinates(
    name="medial_centre_femur",
    location=np.array([0.0002, 0.0034, -0.0232]),
    is_distal_location=True,
    is_technical=False,
    is_anatomical=True,
)
model["THIGH"].add_natural_marker_from_segment_coordinates(
    name="lateral_centre_femur",
    location=np.array([-0.0033, 0.0021, 0.0262]),
    is_distal_location=True,
    is_technical=False,
    is_anatomical=True,
)
model["THIGH"].add_natural_marker_from_segment_coordinates(
    name="ACL_femur",
    location=np.array([-0.0068, 0.0075, 0.0092]),
    is_distal_location=True,
    is_technical=False,
    is_anatomical=True,
)
model["THIGH"].add_natural_marker_from_segment_coordinates(
    name="PCL_femur",
    location=np.array([-0.0027, -0.0011, -0.0022]),
    is_distal_location=True,
    is_technical=False,
    is_anatomical=True,
)
model["THIGH"].add_natural_marker_from_segment_coordinates(
    name="MCL_femur",
    location=np.array([0.0028, 0.0058, -0.0476]),
    is_distal_location=True,
    is_technical=False,
    is_anatomical=True,
)

model["SHANK"] = NaturalSegment(
    name="SHANK",
    alpha=89.15807479515183 * np.pi / 180,
    beta=89.99767608318231 * np.pi / 180,
    gamma=89.99418352863022 * np.pi / 180,
    length=0.3460518602753062,
    mass=4,
    center_of_mass=np.array([0, 0.1, 0]),  # in segment coordinates system
    inertia=np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),  # in segment coordinates system
)
model["SHANK"].add_natural_marker_from_segment_coordinates(
    name="medial_contact_shank",
    location=np.array([-0.0021, -0.0286, -0.0191]),
    is_technical=False,
    is_anatomical=True,
)
model["SHANK"].add_natural_marker_from_segment_coordinates(
    name="lateral_contact_shank",
    location=np.array([-0.0028, -0.0261, 0.0244]),
    is_technical=False,
    is_anatomical=True,
)
model["SHANK"].add_natural_marker_from_segment_coordinates(
    name="ACL_shank",
    location=np.array([0.0128, -0.0261, -0.0001]),
    is_technical=False,
    is_anatomical=True,
)
model["SHANK"].add_natural_marker_from_segment_coordinates(
    name="PCL_shank",
    location=np.array([-0.0259, -0.0381, -0.0035]),
    is_technical=False,
    is_anatomical=True,
)
model["SHANK"].add_natural_marker_from_segment_coordinates(
    name="MCL_shank",
    location=np.array([0.0021, -0.1171, -0.0058]),
    is_technical=False,
    is_anatomical=True,
)
model["SHANK"].add_natural_vector_from_segment_coordinates(
    name="medial_normal_shank",
    direction=np.array([0.0128, -0.0261, -0.0001]),
    normalize=True,
)
model["SHANK"].add_natural_vector_from_segment_coordinates(
    name="lateral_normal_shank",
    direction=np.array([0.0675, 0.9896, -0.1273]),
    normalize=True,
)

print(model)
# # add a revolute joint (still experimental)
# # if you want to add a revolute joint,
# # you need to ensure that x is always orthogonal to u and v
# model._add_joint(
#     dict(
#         name="hinge",
#         joint_type=JointType.GROUND_REVOLUTE,
#         parent="GROUND",
#         child="pendulum",
#         parent_axis=[CartesianAxis.X, CartesianAxis.X],
#         child_axis=[NaturalAxis.V, NaturalAxis.W],  # meaning we pivot around the cartesian x-axis
#         theta=[np.pi / 2, np.pi / 2],
#     )
# )