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
        assert joint.nb_joint_dof == 1
        assert joint.nb_constraints == 5
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
        assert joint.nb_joint_dof == 2
        assert joint.nb_constraints == 4
    elif joint_type == JointType.SPHERICAL:
        joint = Joint.Spherical(
            name="spherical",
            parent=box,
            child=bbox,
            index=0,
        )
        assert joint.nb_joint_dof == 3
        assert joint.nb_constraints == 3
    elif joint_type == JointType.GROUND_REVOLUTE:
        joint = GroundJoint.Hinge(
            name="hinge",
            child="box",
            index=0,
            parent_axis=[CartesianAxis.X, CartesianAxis.X],
            child_axis=[NaturalAxis.V, NaturalAxis.W],  # meaning we pivot around the cartesian x-axis
            theta=[np.pi / 2, np.pi / 2],
        )
        assert joint.nb_joint_dof == 1
        assert joint.nb_constraints == 5
    elif joint_type == JointType.GROUND_SPHERICAL:
        joint = GroundJoint.Spherical(
            name="spherical",
            child="box",
            index=0,
            ground_application_point=np.array([0.1, 0.2, 0.3]),
        )
        assert joint.nb_joint_dof == 3
        assert joint.nb_constraints == 3
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
        assert joint.nb_constraints == 1
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
        assert joint.nb_constraints == 1
        assert joint.nb_joint_dof == 5
    elif joint_type == JointType.GROUND_WELD:
        joint = GroundJoint.Weld(
            name="Weld",
            child=bbox,
            rp_child_ref=np.array([0.1, 0.2, 0.3]),
            rd_child_ref=np.array([0.2, 0.04, 0.05]),
            index=0,
        )
        assert joint.nb_constraints == 6
        assert joint.nb_joint_dof == 0
    elif joint_type == JointType.GROUND_FREE:
        joint = GroundJoint.Free(
            name="Free",
            child=bbox,
            index=0,
        )
        assert joint.nb_constraints == 0
        assert joint.nb_joint_dof == 6
    elif joint_type == JointType.GROUND_UNIVERSAL:
        joint = GroundJoint.Universal(
            name="Universal",
            child=bbox,
            index=0,
            parent_axis=CartesianAxis.X,
            child_axis=NaturalAxis.V,
            theta=np.pi / 2,
        )
        assert joint.nb_constraints == 4
        assert joint.nb_joint_dof == 2
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
            -1.471104681468824,
            decimal=6,
        )
        parent_jacobian, child_jacobian = joint.constraint_jacobian(Q1, Q2)
        parent_jacobian_res = np.array(
            [
                [
                    -0.045416,
                    0.01429,
                    0.169968,
                    -0.544993,
                    0.171483,
                    2.039611,
                    0.090832,
                    -0.028581,
                    -0.339935,
                    -0.136248,
                    0.042871,
                    0.509903,
                ]
            ]
        )

        child_jacobian_res = np.array(
            [
                [
                    9.083224e-02,
                    -2.858051e-02,
                    -3.399352e-01,
                    4.594992e-01,
                    -1.445821e-01,
                    -1.719653e00,
                    -5.337990e-03,
                    1.679607e-03,
                    1.997716e-02,
                    4.877145e-03,
                    -1.534601e-03,
                    -1.825247e-02,
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
            2.57731347673652,
            decimal=6,
        )
        parent_jacobian, child_jacobian = joint.constraint_jacobian(Q1, Q2)

        parent_jacobian_res = np.array(
            [
                [
                    -0.499201,
                    -0.340093,
                    0.156834,
                    -1.444006,
                    -2.060695,
                    -3.25346,
                    0.029337,
                    0.019986,
                    -0.009217,
                    -0.026804,
                    -0.018261,
                    0.008421,
                ]
            ]
        )

        child_jacobian_res = np.array(
            [
                [
                    0.141467,
                    0.204071,
                    0.326268,
                    1.697603,
                    2.44885,
                    3.915212,
                    -0.282934,
                    -0.408142,
                    -0.652535,
                    0.424401,
                    0.612212,
                    0.978803,
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

    elif joint_type == JointType.GROUND_WELD:
        TestUtils.assert_equal(
            joint.constraint(Q1, Q2),
            np.array([-1.4, -0.9, -2.9, -1.4, -2.16, -4.15]),
            decimal=6,
        )
        child_jacobian = joint.constraint_jacobian(Q1, Q2)
        TestUtils.assert_equal(
            child_jacobian,
            -np.eye(12)[3:9, :],
            decimal=6,
        )

        child_jacobian_dot = joint.constraint_jacobian_derivative(Q1, Q2)
        TestUtils.assert_equal(
            child_jacobian_dot,
            np.zeros((6, 12)),
            decimal=6,
        )

        assert joint.parent_constraint_jacobian(Q1, Q2) is None
        assert joint.parent_constraint_jacobian_derivative(Q1, Q2) is None

    elif joint_type == JointType.GROUND_UNIVERSAL:
        TestUtils.assert_equal(
            joint.constraint(Q1, Q2),
            np.array([-1.5, -1.1, -3.2, -0.1]),
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
                ]
            ),
            decimal=6,
        )

        child_jacobian_dot = joint.constraint_jacobian_derivative(Q1, Q2)
        TestUtils.assert_equal(
            child_jacobian_dot,
            np.zeros((4, 12)),
            decimal=6,
        )

        assert joint.parent_constraint_jacobian(Q1, Q2) is None
        assert joint.parent_constraint_jacobian_derivative(Q1, Q2) is None


# def test_numpy_jacobian_from_casadi_derivatives()
# todo:
# # numpy version
# Q_test = NaturalCoordinates(np.arange(24))
# jacobian_numpy = model.joint_constraints_jacobian(Q_test)
#
# model_mx = model.to_mx()
# sym = NaturalCoordinatesMX.sym(2)
# j_constraints_sym = model_mx.joint_constraints(sym)
# # jacobian
# j_jacobian_sym = jacobian(j_constraints_sym, sym)
# j_jacobian_func = Function("j_jacobian_func", [sym], [j_jacobian_sym])
#
# jacobian_mx = j_jacobian_func(np.arange(24)).toarray()
