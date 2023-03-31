import pytest
from .utils import TestUtils
import numpy as np


@pytest.mark.parametrize(
    "bionc_type",
    [
        "numpy",
        # "casadi",
    ],
)
def test_joints(bionc_type):
    if bionc_type == "casadi":
        from bionc.bionc_casadi import (
            BiomechanicalModel,
            NaturalSegment,
            SegmentNaturalCoordinates, SegmentNaturalVelocities,
            Joint,
            GroundJoint,
            ExternalForceList, ExternalForce,
        )
    else:
        from bionc.bionc_numpy import (
            BiomechanicalModel,
            NaturalSegment,
            SegmentNaturalCoordinates,SegmentNaturalVelocities,
            ExternalForceList, ExternalForce,
        )

    if bionc_type == "numpy":
        from bionc.bionc_numpy import NaturalCoordinates, NaturalVelocities
    else:
        from bionc.bionc_casadi import NaturalCoordinates, NaturalVelocities

    bionc = TestUtils.bionc_folder()
    module = TestUtils.load_module(bionc + "/examples/forward_dynamics/n_link_pendulum.py")

    nb_segments = 1
    model = module.build_n_link_pendulum(nb_segments=nb_segments)
    if bionc_type == "casadi":
        model = model.to_mx()

    tuple_of_Q = [
        SegmentNaturalCoordinates.from_components(u=[1, 0, 0], rp=[0, -i, 0], rd=[0, -i - 1, 0], w=[0, 0, 1])
        for i in range(0, nb_segments)
    ]
    Q = NaturalCoordinates.from_qi(tuple(tuple_of_Q))

    tuple_of_Qdot = [
        SegmentNaturalVelocities.from_components(udot=[0, 0, 0], rpdot=[0, 0, 0], rddot=[0, 0, 0], wdot=[0, 0, 0])
        for i in range(0, nb_segments)
    ]
    Qdot = NaturalVelocities.from_qdoti(tuple(tuple_of_Qdot))

    fext = ExternalForceList.empty_from_nb_segment(nb_segments)
    force1 = ExternalForce.from_components(
        # force=np.array([0, 0, 1 * 9.81]),
        force=np.array([0, 0, 0]),
        torque=np.array([0, 0, 0]),
        application_point_in_global=np.array([0, 0.1, 0]),
    )
    fext.add_external_force(external_force=force1, segment_index=0)

    Qddot, lagrange_multipliers = model.forward_dynamics(Q, Qdot, external_forces=fext)
    print(Qddot)
    print(lagrange_multipliers)



