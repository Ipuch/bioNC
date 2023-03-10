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
    elif joint_type == JointType.GROUND_SPHERICAL:
        joint = GroundJoint.Spherical(
            name="spherical",
            child="box",
            index=0,
            ground_application_point=np.array([0.1, 0.2, 0.3]),
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
            -1.466320668046882,
            decimal=6,
        )
        parent_jacobian, child_jacobian = joint.constraint_jacobian(Q1, Q2)
        parent_jacobian_res = np.array(
            [
                [
                    0.163589,
                    0.056279,
                    0.231968,
                    1.963063,
                    0.675351,
                    2.783621,
                    -0.327177,
                    -0.112558,
                    -0.463937,
                    0.490766,
                    0.168838,
                    0.695905,
                ]
            ]
        )

        child_jacobian_res = np.array(
            [
                [
                    9.04513848e-02,
                    -2.90285748e-02,
                    -3.41122532e-01,
                    4.57572532e-01,
                    -1.46848813e-01,
                    -1.72565960e00,
                    -5.31560771e-03,
                    1.70593868e-03,
                    2.00469408e-02,
                    4.60339376e-03,
                    -1.47736776e-03,
                    -1.73609430e-02,
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
            2.5792148159608956,
            decimal=6,
        )
        parent_jacobian, child_jacobian = joint.constraint_jacobian(Q1, Q2)

        parent_jacobian_res = np.array(
            [
                [
                    -0.497388,
                    -0.337959,
                    0.162488,
                    -1.439366,
                    -2.055235,
                    -3.238992,
                    0.02923,
                    0.019861,
                    -0.009549,
                    -0.025314,
                    -0.0172,
                    0.00827,
                ]
            ]
        )

        child_jacobian_res = np.array(
            [
                [
                    0.141014,
                    0.203537,
                    0.324854,
                    1.692163,
                    2.442449,
                    3.89825,
                    -0.282027,
                    -0.407075,
                    -0.649708,
                    0.423041,
                    0.610612,
                    0.974562,
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

    elif joint_type == JointType.GROUND_SPHERICAL:
        TestUtils.assert_equal(
            joint.constraint(Q1, Q2),
            np.array([-1.4, -0.9, -2.9]),
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
                ]
            ),
            decimal=6,
        )

        child_jacobian_dot = joint.constraint_jacobian_derivative(Q1, Q2)
        TestUtils.assert_equal(
            child_jacobian_dot,
            np.zeros((3, 12)),
            decimal=6,
        )

        assert joint.parent_constraint_jacobian(Q1, Q2) is None
        assert joint.parent_constraint_jacobian_derivative(Q1, Q2) is None
