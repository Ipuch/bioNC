import pytest
from .utils import TestUtils
import numpy as np


@pytest.mark.parametrize(
    "bionc_type",
    [
        "numpy",
    ],
)
@pytest.mark.parametrize(
    "external_force_tuple",
    [
        (np.array([0, 0, 1 * 9.81]), np.array([0, 0, 0]), np.array([0, -0.5, 0])),
        (np.array([0, 0, 0]), np.array([-1 * 9.81 * 0.50, 0, 0]), np.array([0, 0, 0])),
    ],
)
def test_external_force(bionc_type, external_force_tuple):
    from bionc.bionc_numpy import (
        ExternalForceList, ExternalForce,
        SegmentNaturalCoordinates, NaturalCoordinates,
        SegmentNaturalVelocities, NaturalVelocities,
    )

    bionc = TestUtils.bionc_folder()
    module = TestUtils.load_module(bionc + "/examples/forward_dynamics/pendulum_with_force.py")

    fext = ExternalForceList.empty_from_nb_segment(1)
    force1 = ExternalForce.from_components(
        force=external_force_tuple[0],
        torque=external_force_tuple[1],
        application_point_in_local=external_force_tuple[2],
    )
    fext.add_external_force(external_force=force1, segment_index=0)

    _, _, all_states, _ = module.apply_force_and_drop_pendulum(t_final=1, external_forces=fext)

    # check that the pendulum is not moving
    Qi = SegmentNaturalCoordinates.from_components(u=[1, 0, 0], rp=[0, 0, 0], rd=[0, -1, 0], w=[0, 0, 1])
    Q = NaturalCoordinates(Qi)
    Qdoti = SegmentNaturalVelocities.from_components(udot=[0, 0, 0], rpdot=[0, 0, 0], rddot=[0, 0, 0], wdot=[0, 0, 0])
    Qdot = NaturalVelocities(Qdoti)

    np.testing.assert_allclose(all_states[:12, 0], Q.to_array(), atol=1e-6)
    np.testing.assert_allclose(all_states[:12, -1], Q.to_array(), atol=1e-6)

    np.testing.assert_allclose(all_states[12:, 0], Qdot.to_array(), atol=1e-6)
    np.testing.assert_allclose(all_states[12:, -1], Qdot.to_array(), atol=1e-6)









