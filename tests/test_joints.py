import numpy as np
import pytest

from bionc import JointType, NaturalAxis, CartesianAxis
from .utils import TestUtils


@pytest.mark.parametrize(
    "joint_type",
    JointType,
)
@pytest.mark.parametrize(
    "bionc_type",
    ["numpy", "casadi"],
)
def test_joints(bionc_type, joint_type: JointType):
    if bionc_type == "casadi":
        from bionc.bionc_casadi import (
            BiomechanicalModel,
            NaturalSegment,
            SegmentNaturalCoordinates,
            Joint,
            GroundJoint,
        )
    else:
        from bionc.bionc_numpy import (
            BiomechanicalModel,
            NaturalSegment,
            SegmentNaturalCoordinates,
            Joint,
            GroundJoint,
        )

    box = NaturalSegment(
        name="box",
        alpha=np.pi / 2,
        beta=np.pi / 2,
        gamma=np.pi / 2,
        length=1,
        mass=1,
        center_of_mass=np.array([0, 0, 0]),  # scs
        inertia=np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),  # scs
    )

    bbox = NaturalSegment(
        name="bbox",
        alpha=np.pi / 1.9,
        beta=np.pi / 2.3,
        gamma=np.pi / 2.1,
        length=1.5,
        mass=1.1,
        center_of_mass=np.array([0.1, 0.11, 0.111]),  # scs
        inertia=np.array([[1.1, 0, 0], [0, 1.2, 0], [0, 0, 1.3]]),  # scs
    )

    model = BiomechanicalModel()
    model["box"] = box
    model["bbox"] = bbox
    if joint_type == JointType.REVOLUTE:
        parent_axis = NaturalAxis.U, NaturalAxis.V
        child_axis = NaturalAxis.V, NaturalAxis.W
        theta = np.pi / 3, 3 * np.pi / 4
        joint = Joint.Hinge(
            name="hinge", parent=box, child=bbox, index=0, parent_axis=parent_axis, child_axis=child_axis, theta=theta
        )

    elif joint_type == JointType.UNIVERSAL:
        joint = Joint.Universal(
            name="universal",
            parent=box,
            child=bbox,
            index=0,
            parent_axis=NaturalAxis.U,
            child_axis=NaturalAxis.W,
            theta=0.4,
        )
    elif joint_type == JointType.SPHERICAL:
        joint = Joint.Spherical(
            name="spherical",
            parent=box,
            child=bbox,
            index=0,
        )
    elif joint_type == JointType.GROUND_REVOLUTE:
        joint = GroundJoint.Hinge(
            name="hinge",
            child="box",
            index=0,
            parent_axis=[CartesianAxis.X, CartesianAxis.X],
            child_axis=[NaturalAxis.V, NaturalAxis.W],  # meaning we pivot around the cartesian x-axis
            theta=[np.pi / 2, np.pi / 2],
        )
    elif joint_type == JointType.CONSTANT_LENGTH:
        box.add_natural_marker_from_segment_coordinates(
            name="P1",
            location=[0.1, 0.2, 0.3],
            is_anatomical=True,
        )
        bbox.add_natural_marker_from_segment_coordinates(
            name="P2",
            location=[0.2, 0.04, 0.05],
            is_anatomical=True,
        )
        joint = Joint.ConstantLength(
            name="constant_length",
            parent=box,
            child=bbox,
            parent_point="P1",
            child_point="P2",
            length=1.5,
            index=0,
        )
    elif joint_type == JointType.SPHERE_ON_PLANE:
        box.add_natural_marker_from_segment_coordinates(
            name="SPHERE_CENTER",
            location=np.array([0.1, 0.2, 0.3]),
            is_anatomical=True,
        )
        bbox.add_natural_marker_from_segment_coordinates(
            name="PLANE_POINT",
            location=np.array([0.2, 0.04, 0.05]),
            is_anatomical=True,
        )
        bbox.add_natural_vector_from_segment_coordinates(
            name="PLANE_NORMAL",
            direction=np.array([0.2, 0.04, 0.05]),
            normalize=True,
        )
        joint = Joint.SphereOnPlane(
            name="constant_length",
            parent=box,
            child=bbox,
            sphere_radius=0.02,
            sphere_center="SPHERE_CENTER",
            plane_point="PLANE_POINT",
            plane_normal="PLANE_NORMAL",
            index=0,
        )
    else:
        raise ValueError("Joint type not tested yet")

    Q1 = SegmentNaturalCoordinates.from_components(
        u=[1, 2, 3.05],
        rp=[1.1, 1, 3.1],
        rd=[1.2, 2, 4.1],
        w=[1.3, 2, 5.1],
    )
    Q2 = SegmentNaturalCoordinates.from_components(
        u=[1.4, 2.1, 3.2],
        rp=[1.5, 1.1, 3.2],
        rd=[1.6, 2.2, 4.2],
        w=[1.7, 2, 5.3],
    )

    if joint_type == JointType.REVOLUTE:
        TestUtils.assert_equal(
            joint.constraint(Q1, Q2),
            np.array([-0.3, 0.9, 0.9, -5.85, -6.762893]),
            decimal=6,
        )
        parent_jacobian, child_jacobian = joint.constraint_jacobian(Q1, Q2)
        TestUtils.assert_equal(
            parent_jacobian,
            np.array(
                [
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                    [-0.1, -1.1, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.7, 2.0, 5.3, -1.7, -2.0, -5.3, 0.0, 0.0, 0.0],
                ]
            ),
            decimal=6,
        )
        TestUtils.assert_equal(
            child_jacobian,
            np.array(
                [
                    [0.0, 0.0, 0.0, -1.0, -0.0, -0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, -0.0, -1.0, -0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, -0.0, -0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0, 2.0, 3.05, -1.0, -2.0, -3.05, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.1, -1.0, -1.0],
                ]
            ),
            decimal=6,
        )
        parent_jacobian_dot, child_jacobian_dot = joint.constraint_jacobian_derivative(Q1, Q2)
        TestUtils.assert_equal(
            parent_jacobian_dot,
            np.vstack(
                (
                    np.zeros((3, 12)),
                    np.array(
                        [
                            [-0.1, -1.1, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 1.7, 2.0, 5.3, -1.7, -2.0, -5.3, 0.0, 0.0, 0.0],
                        ]
                    ),
                )
            ),
            decimal=6,
        )
        TestUtils.assert_equal(
            child_jacobian_dot,
            np.vstack(
                (
                    np.zeros((3, 12)),
                    np.array(
                        [
                            [0.0, 0.0, 0.0, 1.0, 2.0, 3.05, -1.0, -2.0, -3.05, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.1, -1.0, -1.0],
                        ]
                    ),
                )
            ),
            decimal=6,
        )

    elif joint_type == JointType.UNIVERSAL:
        TestUtils.assert_equal(
            joint.constraint(Q1, Q2),
            np.array([-0.3, 0.9, 0.9, 20.943939]),
            decimal=6,
        )
        parent_jacobian, child_jacobian = joint.constraint_jacobian(Q1, Q2)
        TestUtils.assert_equal(
            parent_jacobian,
            np.array(
                [
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                    [1.7, 2.0, 5.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                ]
            ),
            decimal=6,
        )
        TestUtils.assert_equal(
            child_jacobian,
            np.array(
                [
                    [0.0, 0.0, 0.0, -1.0, -0.0, -0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, -0.0, -1.0, -0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, -0.0, -0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.05],
                ]
            ),
            decimal=6,
        )

        parent_jacobian_dot, child_jacobian_dot = joint.constraint_jacobian_derivative(Q1, Q2)
        TestUtils.assert_equal(
            parent_jacobian_dot,
            np.vstack(
                (
                    np.zeros((3, 12)),
                    np.array(
                        [
                            [1.7, 2.0, 5.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        ]
                    ),
                )
            ),
            decimal=6,
        )
        TestUtils.assert_equal(
            child_jacobian_dot,
            np.vstack(
                (
                    np.zeros((3, 12)),
                    np.array(
                        [
                            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.05],
                        ]
                    ),
                )
            ),
            decimal=6,
        )

    elif joint_type == JointType.SPHERICAL:
        TestUtils.assert_equal(
            joint.constraint(Q1, Q2),
            np.array([-0.3, 0.9, 0.9]),
            decimal=6,
        )
        parent_jacobian, child_jacobian = joint.constraint_jacobian(Q1, Q2)
        TestUtils.assert_equal(
            parent_jacobian,
            np.array(
                [
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                ]
            ),
            decimal=6,
        )
        TestUtils.assert_equal(
            child_jacobian,
            np.array(
                [
                    [0.0, 0.0, 0.0, -1.0, -0.0, -0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, -0.0, -1.0, -0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, -0.0, -0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                ]
            ),
            decimal=6,
        )

        parent_jacobian_dot, child_jacobian_dot = joint.constraint_jacobian_derivative(Q1, Q2)
        TestUtils.assert_equal(
            parent_jacobian_dot,
            np.zeros((3, 12)),
            decimal=6,
        )
        TestUtils.assert_equal(
            child_jacobian_dot,
            np.zeros((3, 12)),
            decimal=6,
        )

    elif joint_type == JointType.CONSTANT_LENGTH:
        TestUtils.assert_equal(
            joint.constraint(Q1, Q2),
            -2.2268202329241826,
            decimal=6,
        )
        parent_jacobian, child_jacobian = joint.constraint_jacobian(Q1, Q2)
        parent_jacobian_res = np.array(
            [
                [
                    0.14471581,
                    0.03917453,
                    0.16649716,
                    1.73658967,
                    0.47009436,
                    1.99796594,
                    -0.28943161,
                    -0.07834906,
                    -0.33299432,
                    0.43414742,
                    0.11752359,
                    0.49949148,
                ]
            ]
        )

        child_jacobian_res = np.array(
            [
                [
                    1.40678676e-01,
                    2.36089035e-03,
                    -2.03939179e-01,
                    7.61248370e-01,
                    1.27753827e-02,
                    -1.10356716e00,
                    -5.78549890e-02,
                    -9.70930982e-04,
                    8.38712678e-02,
                    6.20317725e-02,
                    1.04102638e-03,
                    -8.99262708e-02,
                ]
            ]
        )

        TestUtils.assert_equal(
            parent_jacobian,
            parent_jacobian_res,
            decimal=6,
            squeeze=False,
        )
        TestUtils.assert_equal(
            child_jacobian,
            child_jacobian_res,
            decimal=6,
            squeeze=False,
        )

        parent_jacobian_dot, child_jacobian_dot = joint.constraint_jacobian_derivative(Q1, Q2)
        TestUtils.assert_equal(
            parent_jacobian_dot,
            parent_jacobian_res,
            decimal=6,
            squeeze=False,
        )
        TestUtils.assert_equal(
            child_jacobian_dot,
            child_jacobian_res,
            decimal=6,
            squeeze=False,
        )

    elif joint_type == JointType.SPHERE_ON_PLANE:
        TestUtils.assert_equal(
            joint.constraint(Q1, Q2),
            1.7484589974450258,
            decimal=6,
        )
        parent_jacobian, child_jacobian = joint.constraint_jacobian(Q1, Q2)
        parent_jacobian_res = np.array(
            [
                [
                    -0.73656513,
                    -0.48743281,
                    -0.49076582,
                    -2.31099651,
                    -2.60951802,
                    -5.08350697,
                    0.30291703,
                    0.2004598,
                    0.20183052,
                    -0.32478583,
                    -0.2149318,
                    -0.21640148,
                ]
            ]
        )

        child_jacobian_res = np.array(
            [
                [
                    0.20080795,
                    0.24090582,
                    0.48816764,
                    2.40969538,
                    2.89086986,
                    5.85801174,
                    -0.4016159,
                    -0.48181164,
                    -0.97633529,
                    0.60242384,
                    0.72271747,
                    1.46450293,
                ]
            ]
        )

        TestUtils.assert_equal(
            parent_jacobian,
            parent_jacobian_res,
            decimal=6,
            squeeze=False,
        )
        TestUtils.assert_equal(
            child_jacobian,
            child_jacobian_res,
            decimal=6,
            squeeze=False,
        )

        parent_jacobian_dot, child_jacobian_dot = joint.constraint_jacobian_derivative(Q1, Q2)
        TestUtils.assert_equal(
            parent_jacobian_dot,
            parent_jacobian_res,
            decimal=6,
            squeeze=False,
        )
        TestUtils.assert_equal(
            child_jacobian_dot,
            child_jacobian_res,
            decimal=6,
            squeeze=False,
        )

    elif joint_type == JointType.GROUND_REVOLUTE:
        TestUtils.assert_equal(
            joint.constraint(Q1, Q2),
            np.array([-1.5, -1.1, -3.2, -0.1, 1.7]),
            decimal=6,
        )
        child_jacobian = joint.constraint_jacobian(Q1, Q2)
        TestUtils.assert_equal(
            child_jacobian,
            np.array(
                [
                    [0.0, 0.0, 0.0, -1.0, -0.0, -0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, -0.0, -1.0, -0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, -0.0, -0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                ]
            ),
            decimal=6,
        )

        child_jacobian_dot = joint.constraint_jacobian_derivative(Q1, Q2)
        TestUtils.assert_equal(
            child_jacobian_dot,
            np.zeros((5, 12)),
            decimal=6,
        )
