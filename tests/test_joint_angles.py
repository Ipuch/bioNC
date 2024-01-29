import numpy as np
import pytest

from bionc import NaturalCoordinates, TransformationMatrixType, EulerSequence, NaturalAxis, JointType
from .utils import TestUtils


@pytest.mark.parametrize(
    "bionc_type",
    [
        "numpy",
        "casadi"
    ],
)
def test_joint_angles(bionc_type):
    if bionc_type == "casadi":
        from bionc.bionc_casadi import (
            BiomechanicalModel,
            NaturalSegment,
        )
    else:
        from bionc.bionc_numpy import (
            BiomechanicalModel,
            NaturalSegment,
        )

    model = BiomechanicalModel()

    model["PELVIS"] = NaturalSegment.with_cartesian_inertial_parameters(
        name="PELVIS",
        alpha=1.5460,
        beta=1.5708,
        gamma=1.5708,
        length=0.2207,
        mass=1,
        center_of_mass=np.array([0, 0, 0]),  # scs
        inertia=np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),  # scs
        inertial_transformation_matrix=TransformationMatrixType.Buv,
    )

    model["RTHIGH"] = NaturalSegment.with_cartesian_inertial_parameters(
        name="RTHIGH",
        alpha=1.4017,
        beta=1.5708,
        gamma=1.5708,
        length=0.4167,
        mass=1,
        center_of_mass=np.array([0, 0, 0]),  # scs
        inertia=np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),  # scs
        inertial_transformation_matrix=TransformationMatrixType.Buv,
    )

    model["RSHANK"] = NaturalSegment.with_cartesian_inertial_parameters(
        name="RSHANK",
        alpha=1.5708,
        beta=1.5708,
        gamma=1.5708,
        length=0.4297,
        mass=1,
        center_of_mass=np.array([0, 0, 0]),  # scs
        inertia=np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),  # scs
        inertial_transformation_matrix=TransformationMatrixType.Buv,
    )

    model["RFOOT"] = NaturalSegment.with_cartesian_inertial_parameters(
        name="RFOOT",
        alpha=1.5708,
        beta=1.5239,
        gamma=3.0042,
        length=0.1601,
        mass=1,
        center_of_mass=np.array([0, 0, 0]),  # scs
        inertia=np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),  # scs
        inertial_transformation_matrix=TransformationMatrixType.Buv,
    )

    # model._add_joint(
    #     dict(
    #         name="free_joint_PELVIS",
    #         joint_type=JointType.GROUND_FREE,
    #         parent="GROUND",
    #         child="PELVIS",
    #         projection_basis=EulerSequence.XYZ,
    #     )
    # )

    model._add_joint(
        dict(
            name="rhip",
            joint_type=JointType.SPHERICAL,
            parent="PELVIS",
            child="RTHIGH",
            projection_basis=EulerSequence.ZXY,
            parent_basis=TransformationMatrixType.Bwu,
            child_basis=TransformationMatrixType.Buv,
        )
    )

    model._add_joint(
        dict(
            name="rknee",
            joint_type=JointType.REVOLUTE,
            parent="RTHIGH",
            child="RSHANK",
            parent_axis=[NaturalAxis.W, NaturalAxis.W],
            child_axis=[NaturalAxis.V, NaturalAxis.U],
            theta=[np.pi / 2, np.pi / 2],
            projection_basis=EulerSequence.ZXY,
            parent_basis=TransformationMatrixType.Bwu,
            child_basis=TransformationMatrixType.Buv,
        )
    )

    model._add_joint(
        dict(
            name="rankle",
            joint_type=JointType.REVOLUTE,
            parent="RSHANK",
            child="RFOOT",
            parent_axis=[NaturalAxis.W, NaturalAxis.W],
            child_axis=[NaturalAxis.V, NaturalAxis.U],
            theta=[np.pi / 2, np.pi / 2],
            projection_basis=EulerSequence.ZXY,
            parent_basis=TransformationMatrixType.Bwu,
            child_basis=TransformationMatrixType.Buw,
        )
    )

    Q = NaturalCoordinates(np.array(
        [[0.9965052576],
         [-0.0568879923],
         [-0.0611639426],
         [-1.0926996469],
         [0.0317322835],
         [1.10625422],
         [-1.1062166095],
         [0.031825874],
         [0.8859438896],
         [-0.0552734198],
         [-0.9980848821],
         [0.0277743976],
         [0.9997517101],
         [-0.0047420645],
         [0.0217722535],
         [-1.1113491058],
         [-0.0608527958],
         [0.888522923],
         [-1.1022518873],
         [-0.0553245507],
         [0.4719953239],
         [-0.0080620584],
         [-0.9878759079],
         [0.155036105],
         [0.8272720653],
         [-0.0936852912],
         [-0.5539350107],
         [-1.1022518873],
         [-0.0553245507],
         [0.4719953239],
         [-1.3436225653],
         [-0.1085152924],
         [0.1205172166],
         [-0.0080620584],
         [-0.9878759079],
         [0.155036105],
         [0.8115181535],
         [-0.1429465021],
         [-0.5665726644],
         [-1.3436225653],
         [-0.1085152924],
         [0.1205172166],
         [-1.2266210318],
         [-0.1263815612],
         [0.0127591994],
         [-0.0080620584],
         [-0.9878759079],
         [0.155036105]]))

    test_angles = model.natural_coordinates_to_joint_angles(
        NaturalCoordinates(Q)
    )

    assert model.joint_names == ['free_joint_PELVIS', 'rhip', 'rknee', 'rankle']

    TestUtils.assert_equal(test_angles[:, 0], np.array([1.56776633, -0.05683643, -0.06138343]))
    TestUtils.assert_equal(test_angles[:, 1], np.array([0.08229968, 0.04222329, 0.04998718]))
    TestUtils.assert_equal(test_angles[:, 2], np.array([-6.17331215e-01, 4.40146090e-18, -2.56368559e-18]))
    TestUtils.assert_equal(test_angles[:, 3], np.array([-0.02708583, -0.00033724, -0.04684882]))
