import os
import numpy as np
import pytest

from bionc import JointType

from .utils import TestUtils


@pytest.mark.parametrize(
    "joint_type",
    JointType,
)
@pytest.mark.parametrize(
    "bionc_type",
    ["numpy", "casadi"],
)
def test_biomech_model(bionc_type, joint_type: JointType):

    if bionc_type == "casadi":
        from bionc.bionc_casadi import (
            BiomechanicalModel,
            NaturalSegment,
            SegmentNaturalVelocities,
            NaturalVelocities,
            SegmentNaturalCoordinates,
            NaturalCoordinates,
            Joint,
        )
    else:
        from bionc.bionc_numpy import (
            BiomechanicalModel,
            NaturalSegment,
            SegmentNaturalVelocities,
            NaturalVelocities,
            SegmentNaturalCoordinates,
            NaturalCoordinates,
            Joint,
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
        alpha=np.pi / 5,
        beta=np.pi / 3,
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
        joint = Joint.Hinge(joint_name="hinge", parent=box, child=bbox, theta_1=0.2, theta_2=0.3)
    elif joint_type == JointType.UNIVERSAL:
        joint = Joint.Universal(joint_name="universal", parent=box, child=bbox, theta=0.4)
    elif joint_type == JointType.SPHERICAL:
        joint = Joint.Spherical(joint_name="spherical", parent=box, child=bbox)
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
            np.array([-0.3, 0.9, 0.9, -8.9001, 21.384664]),
            decimal=6,
        )
        with pytest.raises(NotImplementedError, match="This function is not implemented yet"):
            joint.constraint_jacobian(Q1, Q2)

    elif joint_type == JointType.UNIVERSAL:
        TestUtils.assert_equal(
            joint.constraint(Q1, Q2),
            np.array([-0.3, 0.9, 0.9, -0.921061]),
            decimal=6,
        )
        with pytest.raises(NotImplementedError, match="This function is not implemented yet"):
            joint.constraint_jacobian(Q1, Q2)

    elif joint_type == JointType.SPHERICAL:
        TestUtils.assert_equal(
            joint.constraint(Q1, Q2),
            np.array([-0.3, 0.9, 0.9]),
            decimal=6,
        )
        with pytest.raises(NotImplementedError, match="This function is not implemented yet"):
            joint.constraint_jacobian(Q1, Q2)
