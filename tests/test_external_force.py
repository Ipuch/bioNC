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
        ExternalForceList,
        ExternalForce,
        SegmentNaturalCoordinates,
        NaturalCoordinates,
        SegmentNaturalVelocities,
        NaturalVelocities,
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


@pytest.mark.parametrize(
    "bionc_type",
    [
        "numpy",
        # "casadi",
    ],
)
def test_external_force(bionc_type):
    if bionc_type == "numpy":
        from bionc.bionc_numpy import (
            ExternalForceList,
            ExternalForce,
            SegmentNaturalCoordinates,
            NaturalCoordinates,
            SegmentNaturalVelocities,
            NaturalVelocities,
        )
    else:
        from bionc.bionc_casadi import (
            ExternalForceList,
            ExternalForce,
            SegmentNaturalCoordinates,
            NaturalCoordinates,
            SegmentNaturalVelocities,
            NaturalVelocities,
        )

    force1 = ExternalForce.from_components(
        force=np.array([0.01, 0.02, 0.03]),
        torque=np.array([0.04, 0.05, 0.06]),
        application_point_in_local=np.array([0.07, 0.08, 0.09]),
    )
    force2 = ExternalForce.from_components(
        force=np.array([0.11, 0.12, 0.13]),
        torque=np.array([0.14, 0.15, 0.16]),
        application_point_in_local=np.array([0.17, 0.18, 0.19]),
    )

    TestUtils.assert_equal(force1.torque, np.array([0.04, 0.05, 0.06]))
    TestUtils.assert_equal(force1.force, np.array([0.01, 0.02, 0.03]))
    TestUtils.assert_equal(force2.torque, np.array([0.14, 0.15, 0.16]))
    TestUtils.assert_equal(force2.force, np.array([0.11, 0.12, 0.13]))

    fext = ExternalForceList.empty_from_nb_segment(3)
    fext.add_external_force(external_force=force1, segment_index=0)
    fext.add_external_force(external_force=force2, segment_index=2)

    # check that the pendulum is not moving
    Q0 = SegmentNaturalCoordinates.from_components(
        u=[1, 0, 0],
        rp=[0, 0, 0],
        rd=[0, -1, 0],
        w=[0, 0, 1],
    )
    Q1 = Q0 + 0.1
    Q2 = Q1 + 0.1
    Q = NaturalCoordinates.from_qi((Q0, Q1, Q2))

    pseudo_interpolation_matrix = force2.compute_pseudo_interpolation_matrix(Q2)
    natural_force = force2.to_natural_force(Q2)
    natural_forces = fext.to_natural_external_forces(Q)

    TestUtils.assert_equal(
        pseudo_interpolation_matrix,
        np.array(
            [
                [
                    0.0,
                    0.14285714,
                    0.0,
                    0.17142857,
                    0.17142857,
                    1.02857143,
                    -0.17142857,
                    -0.17142857,
                    -1.02857143,
                    0.0,
                    0.0,
                    0.0,
                ],
                [
                    0.0,
                    0.14285714,
                    0.0,
                    0.02857143,
                    0.02857143,
                    0.17142857,
                    -0.02857143,
                    -0.02857143,
                    -0.17142857,
                    0.85714286,
                    0.14285714,
                    0.14285714,
                ],
                [
                    0.0,
                    0.85714286,
                    0.0,
                    0.02857143,
                    0.02857143,
                    0.17142857,
                    -0.02857143,
                    -0.02857143,
                    -0.17142857,
                    0.0,
                    0.0,
                    0.0,
                ],
            ]
        ),
    )

    TestUtils.assert_equal(
        natural_force,
        np.array(
            [
                0.0187,
                0.19897143,
                0.0221,
                0.16265714,
                0.17445714,
                0.35054286,
                -0.05265714,
                -0.05445714,
                -0.22054286,
                0.14947143,
                0.04422857,
                0.04612857,
            ]
        ),
    )

    TestUtils.assert_equal(
        np.array(
            [
                [0.0007],
                [0.0614],
                [0.0021],
                [0.0108],
                [0.0216],
                [0.0724],
                [-0.0008],
                [-0.0016],
                [-0.0424],
                [0.0509],
                [0.0018],
                [0.0027],
                [0.0],
                [0.0],
                [0.0],
                [0.0],
                [0.0],
                [0.0],
                [0.0],
                [0.0],
                [0.0],
                [0.0],
                [0.0],
                [0.0],
                [0.0187],
                [0.19897143],
                [0.0221],
                [0.16265714],
                [0.17445714],
                [0.35054286],
                [-0.05265714],
                [-0.05445714],
                [-0.22054286],
                [0.14947143],
                [0.04422857],
                [0.04612857],
            ]
        ),
        natural_forces,
    )
