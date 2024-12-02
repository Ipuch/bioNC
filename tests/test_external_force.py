import numpy as np
import pytest
from casadi import vertcat

from .utils import TestUtils


@pytest.mark.parametrize(
    "bionc_type",
    [
        "numpy",
    ],
)
@pytest.mark.parametrize(
    "external_force_tuple",
    [
        (np.array([0, 0, 1 * 9.81]), np.array([0, 0, 0]), np.array([0, 0.5, 0])),
        (np.array([0, 0, 0]), np.array([-1 * 9.81 * 0.50, 0, 0]), np.array([0, 0, 0])),
    ],
)
def test_external_force(bionc_type, external_force_tuple):
    from bionc.bionc_numpy import (
        ExternalForceSet,
        SegmentNaturalCoordinates,
        NaturalCoordinates,
        SegmentNaturalVelocities,
        NaturalVelocities,
    )

    bionc = TestUtils.bionc_folder()
    module = TestUtils.load_module(bionc + "/examples/forward_dynamics/pendulum_with_force.py")

    fext = ExternalForceSet.empty_from_nb_segment(1)
    external_force = np.concatenate([external_force_tuple[1], external_force_tuple[0]])
    fext.add_in_global_local_point(
        external_force=external_force, segment_index=0, point_in_local=external_force_tuple[2]
    )

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
        "casadi",
    ],
)
def test_external_force_local_point(bionc_type):
    if bionc_type == "numpy":
        from bionc.bionc_numpy import (
            ExternalForceSet,
            ExternalForceInGlobalLocalPoint,
            SegmentNaturalCoordinates,
            NaturalCoordinates,
            SegmentNaturalVelocities,
            NaturalVelocities,
        )
    else:
        from bionc.bionc_casadi import (
            ExternalForceSet,
            ExternalForceInGlobalLocalPoint,
            SegmentNaturalCoordinates,
            NaturalCoordinates,
            SegmentNaturalVelocities,
            NaturalVelocities,
        )

    force1 = ExternalForceInGlobalLocalPoint.from_components(
        force=np.array([0.01, 0.02, 0.03]),
        torque=np.array([0.04, 0.05, 0.06]),
        application_point_in_local=np.array([0.07, 0.08, 0.09]),
    )
    force2 = ExternalForceInGlobalLocalPoint.from_components(
        force=np.array([0.11, 0.12, 0.13]),
        torque=np.array([0.14, 0.15, 0.16]),
        application_point_in_local=np.array([0.17, 0.18, 0.19]),
    )

    TestUtils.assert_equal(force1.torque, np.array([0.04, 0.05, 0.06]), squeeze=True, expand=True)
    TestUtils.assert_equal(force1.force, np.array([0.01, 0.02, 0.03]), squeeze=True)
    TestUtils.assert_equal(force2.torque, np.array([0.14, 0.15, 0.16]), squeeze=True)
    TestUtils.assert_equal(force2.force, np.array([0.11, 0.12, 0.13]), squeeze=True)

    fext = ExternalForceSet.empty_from_nb_segment(3)
    fext.add_in_global_local_point(
        external_force=(
            np.concatenate([force1.torque, force1.force])
            if bionc_type == "numpy"
            else vertcat(force1.torque, force1.force)
        ),
        segment_index=0,
        point_in_local=np.array([0.07, 0.08, 0.09]),
    )
    fext.add_in_global_local_point(
        external_force=(
            np.concatenate([force2.torque, force2.force])
            if bionc_type == "numpy"
            else vertcat(force2.torque, force2.force)
        ),
        segment_index=2,
        point_in_local=np.array([0.17, 0.18, 0.19]),
    )

    # check that the pendulum is not moving
    Q0 = SegmentNaturalCoordinates.from_components(
        u=[1, 0, 0],
        rp=[0, 0, 0],
        rd=[0, -1, 0],
        w=[0, 0, 1],
    )
    Q1 = SegmentNaturalCoordinates(Q0 + 0.1)
    Q2 = SegmentNaturalCoordinates(Q1 + 0.1)
    Q = NaturalCoordinates.from_qi((Q0, Q1, Q2))

    pseudo_interpolation_matrix = Q2.compute_pseudo_interpolation_matrix()

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
        expand=False,
    )

    natural_force = force2.transport_on_proximal(Q2).to_generalized_natural_forces(Q2)
    natural_force_2_expected = np.array(
        [
            0.0,
            0.17762857,
            0.0,
            0.14266857,
            0.15266857,
            0.32601143,
            -0.03266857,
            -0.03266857,
            -0.19601143,
            0.13083429,
            0.02180571,
            0.02180571,
        ]
    )
    TestUtils.assert_equal(natural_force, natural_force_2_expected, expand=False, squeeze=True)

    natural_forces = fext.to_natural_external_forces(Q)
    complete_natural_force_expected = np.concatenate(
        (
            np.array(
                [
                    [0.0],
                    [0.0594],
                    [0.0],
                    [0.01],
                    [0.02],
                    [0.0694],
                    [0.0],
                    [0.0],
                    [-0.0394],
                    [0.0512],
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
                    [0.0],
                    [0.0],
                ]
            ),
            natural_force_2_expected[:, np.newaxis],
        )
    )
    TestUtils.assert_equal(
        natural_forces,
        complete_natural_force_expected,
        expand=False,
        squeeze=False,
    )

    transported_force = force2.transport_on_proximal(Q2)
    TestUtils.assert_equal(transported_force.force, np.array([0.11, 0.12, 0.13]), squeeze=True)
    TestUtils.assert_equal(transported_force.torque, np.array([0.13868, 0.15264, 0.15868]), squeeze=True)

    new_natural_force = (
        force2.transport_on_proximal(Q2)
        .transport_to_another_segment(
            Qfrom=Q2,
            Qto=Q1,
        )
        .to_generalized_natural_forces(Q1)
    )

    TestUtils.assert_equal(
        new_natural_force,
        np.array(
            [
                0.0,
                0.1689,
                0.0,
                0.12522333,
                0.13522333,
                0.29745667,
                -0.01522333,
                -0.01522333,
                -0.16745667,
                0.14175333,
                0.01288667,
                0.01288667,
            ]
        ),
        expand=False,
        squeeze=True,
    )

    fext.add_in_global_local_point(
        segment_index=2,
        external_force=(
            np.concatenate([force2.torque, force2.force])
            if bionc_type == "numpy"
            else vertcat(force2.torque, force2.force)
        ),
        point_in_local=np.array([0.17, 0.18, 0.19]),
    )
    segment_force_2 = fext.to_segment_natural_external_forces(Q=Q, segment_idx=2)
    TestUtils.assert_equal(
        np.squeeze(segment_force_2) if bionc_type == "numpy" else segment_force_2,
        np.squeeze(natural_force_2_expected * 2) if bionc_type == "numpy" else natural_force_2_expected * 2.0,
        expand=False,
        squeeze=True,
    )


@pytest.mark.parametrize(
    "bionc_type",
    [
        "numpy",
        "casadi",
    ],
)
def test_external_force_in_global(bionc_type):
    if bionc_type == "numpy":
        from bionc.bionc_numpy import (
            ExternalForceSet,
            ExternalForceInGlobal,
            SegmentNaturalCoordinates,
            NaturalCoordinates,
            SegmentNaturalVelocities,
            NaturalVelocities,
        )
    else:
        from bionc.bionc_casadi import (
            ExternalForceSet,
            ExternalForceInGlobal,
            SegmentNaturalCoordinates,
            NaturalCoordinates,
            SegmentNaturalVelocities,
            NaturalVelocities,
        )

    force1 = ExternalForceInGlobal.from_components(
        force=np.array([0.01, 0.02, 0.03]),
        torque=np.array([0.04, 0.05, 0.06]),
        application_point_in_global=np.array([0.07, 0.08, 0.09]),
    )
    force2 = ExternalForceInGlobal.from_components(
        force=np.array([0.11, 0.12, 0.13]),
        torque=np.array([0.14, 0.15, 0.16]),
        application_point_in_global=np.array([0.17, 0.18, 0.19]),
    )

    TestUtils.assert_equal(force1.torque, np.array([0.04, 0.05, 0.06]), squeeze=True)
    TestUtils.assert_equal(force1.force, np.array([0.01, 0.02, 0.03]), squeeze=True)
    TestUtils.assert_equal(force2.torque, np.array([0.14, 0.15, 0.16]), squeeze=True)
    TestUtils.assert_equal(force2.force, np.array([0.11, 0.12, 0.13]), squeeze=True)

    fext = ExternalForceSet.empty_from_nb_segment(3)
    fext.add_in_global(
        external_force=(
            np.concatenate([force1.torque, force1.force])
            if bionc_type == "numpy"
            else vertcat(force1.torque, force1.force)
        ),
        segment_index=0,
        point_in_global=np.array([0.07, 0.08, 0.09]),
    )
    fext.add_in_global(
        external_force=(
            np.concatenate([force2.torque, force2.force])
            if bionc_type == "numpy"
            else vertcat(force2.torque, force2.force)
        ),
        segment_index=2,
        point_in_global=np.array([0.17, 0.18, 0.19]),
    )

    # check that the pendulum is not moving
    Q0 = SegmentNaturalCoordinates.from_components(
        u=[1, 0, 0],
        rp=[0, 0, 0],
        rd=[0, -1, 0],
        w=[0, 0, 1],
    )
    Q1 = SegmentNaturalCoordinates(Q0 + 0.1)
    Q2 = SegmentNaturalCoordinates(Q1 + 0.1)
    Q = NaturalCoordinates.from_qi((Q0, Q1, Q2))

    natural_force = force2.transport_on_proximal(Q2).to_generalized_natural_forces(Q2)
    natural_force_2_expected = np.array(
        [
            0.0,
            0.17957143,
            0.0,
            0.14305714,
            0.15305714,
            0.32834286,
            -0.03305714,
            -0.03305714,
            -0.19834286,
            0.12617143,
            0.02102857,
            0.02102857,
        ]
    )
    TestUtils.assert_equal(natural_force, natural_force_2_expected, expand=False, squeeze=True)

    natural_forces = fext.to_natural_external_forces(Q)
    complete_natural_force_expected = np.concatenate(
        (
            np.array(
                [
                    [0.0],
                    [0.0594],
                    [0.0],
                    [0.01],
                    [0.02],
                    [0.0694],
                    [0.0],
                    [0.0],
                    [-0.0394],
                    [0.0512],
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
                    [0.0],
                    [0.0],
                ]
            ),
            natural_force_2_expected[:, np.newaxis],
        )
    )
    TestUtils.assert_equal(
        natural_forces,
        complete_natural_force_expected,
        expand=False,
        squeeze=False,
    )

    new_natural_force = (
        force2.transport_on_proximal(Q2)
        .transport_to_another_segment(
            Qfrom=Q2,
            Qto=Q1,
        )
        .to_generalized_natural_forces(Q1)
    )

    TestUtils.assert_equal(
        new_natural_force,
        np.array(
            [
                0.0,
                0.17116667,
                0.0,
                0.12545,
                0.13545,
                0.29995,
                -0.01545,
                -0.01545,
                -0.16995,
                0.13676667,
                0.01243333,
                0.01243333,
            ]
        ),
        expand=False,
        squeeze=True,
    )

    fext.add_in_global(
        segment_index=2,
        external_force=(
            np.concatenate([force2.torque, force2.force])
            if bionc_type == "numpy"
            else vertcat(force2.torque, force2.force)
        ),
        point_in_global=np.array([0.17, 0.18, 0.19]),
    )
    segment_force_2 = fext.to_segment_natural_external_forces(Q=Q, segment_idx=2)
    TestUtils.assert_equal(
        np.squeeze(segment_force_2) if bionc_type == "numpy" else segment_force_2,
        np.squeeze(natural_force_2_expected * 2) if bionc_type == "numpy" else natural_force_2_expected * 2.0,
        expand=False,
        squeeze=True,
    )


@pytest.mark.parametrize(
    "bionc_type",
    [
        "numpy",
        "casadi",
    ],
)
def test_external_force_in_local(bionc_type):
    if bionc_type == "numpy":
        from bionc.bionc_numpy import (
            ExternalForceSet,
            ExternalForceInLocal,
            SegmentNaturalCoordinates,
            NaturalCoordinates,
            SegmentNaturalVelocities,
            NaturalVelocities,
        )
    else:
        from bionc.bionc_casadi import (
            ExternalForceSet,
            ExternalForceInLocal,
            SegmentNaturalCoordinates,
            NaturalCoordinates,
            SegmentNaturalVelocities,
            NaturalVelocities,
        )
    dummy_transformation_matrix = np.eye(3) + np.ones((3, 3)) * 0.001
    dummy_transformation_matrix[2, 1] = 0.2
    force1 = ExternalForceInLocal.from_components(
        force=np.array([0.01, 0.02, 0.03]),
        torque=np.array([0.04, 0.05, 0.06]),
        application_point_in_local=np.array([0.07, 0.08, 0.09]),
        transformation_matrix=dummy_transformation_matrix,
    )
    force2 = ExternalForceInLocal.from_components(
        force=np.array([0.11, 0.12, 0.13]),
        torque=np.array([0.14, 0.15, 0.16]),
        application_point_in_local=np.array([0.17, 0.18, 0.19]),
        transformation_matrix=dummy_transformation_matrix,
    )

    TestUtils.assert_equal(force1.torque, np.array([0.04, 0.05, 0.06]), squeeze=True)
    TestUtils.assert_equal(force1.force, np.array([0.01, 0.02, 0.03]), squeeze=True)
    TestUtils.assert_equal(force2.torque, np.array([0.14, 0.15, 0.16]), squeeze=True)
    TestUtils.assert_equal(force2.force, np.array([0.11, 0.12, 0.13]), squeeze=True)

    fext = ExternalForceSet.empty_from_nb_segment(3)
    fext.add_in_local(
        external_force=(
            np.concatenate([force1.torque, force1.force])
            if bionc_type == "numpy"
            else vertcat(force1.torque, force1.force)
        ),
        segment_index=0,
        point_in_local=np.array([0.07, 0.08, 0.09]),
        transformation_matrix=dummy_transformation_matrix,
    )
    fext.add_in_local(
        external_force=(
            np.concatenate([force2.torque, force2.force])
            if bionc_type == "numpy"
            else vertcat(force2.torque, force2.force)
        ),
        segment_index=2,
        point_in_local=np.array([0.17, 0.18, 0.19]),
        transformation_matrix=dummy_transformation_matrix,
    )

    # check that the pendulum is not moving
    Q0 = SegmentNaturalCoordinates.from_components(
        u=[1, 0, 0],
        rp=[0, 0, 0],
        rd=[0, -1, 0],
        w=[0, 0, 1],
    )
    Q1 = SegmentNaturalCoordinates(Q0 + 0.1)
    Q2 = SegmentNaturalCoordinates(Q1 + 0.1)
    Q = NaturalCoordinates.from_qi((Q0, Q1, Q2))

    natural_force = force2.transport_on_proximal(Q2).to_generalized_natural_forces(Q2)
    natural_force_2_expected = np.array(
        [
            0.0,
            0.24572225,
            0.0,
            0.20196297,
            0.18615927,
            0.44411017,
            -0.04442944,
            -0.04442944,
            -0.26657665,
            0.1537273,
            0.02562122,
            0.02562122,
        ]
    )
    TestUtils.assert_equal(
        natural_force,
        natural_force_2_expected,
        expand=False,
        squeeze=True,
    )

    natural_forces = fext.to_natural_external_forces(Q)
    complete_natural_force_expected = np.concatenate(
        (
            np.array(
                [
                    [0.0],
                    [0.05967894],
                    [0.0],
                    [0.00994612],
                    [0.01398684],
                    [0.06867157],
                    [0.0],
                    [0.0],
                    [-0.03872545],
                    [0.0391508],
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
                    [0.0],
                    [0.0],
                ]
            ),
            natural_force_2_expected[:, np.newaxis],
        )
    )
    TestUtils.assert_equal(
        natural_forces,
        complete_natural_force_expected,
        expand=False,
        squeeze=False,
    )

    new_natural_force = (
        force2.transport_on_proximal(Q2)
        .transport_to_another_segment(
            Qfrom=Q2,
            Qto=Q1,
        )
        .to_generalized_natural_forces(Q1)
    )

    TestUtils.assert_equal(
        new_natural_force,
        np.array(
            [
                0.0,
                0.2383283,
                0.0,
                0.17818587,
                0.16238218,
                0.40470934,
                -0.02065235,
                -0.02065235,
                -0.22717582,
                0.16623615,
                0.01511238,
                0.01511238,
            ]
        ),
        expand=False,
        squeeze=True,
    )

    fext.add_in_local(
        segment_index=2,
        external_force=(
            np.concatenate([force2.torque, force2.force])
            if bionc_type == "numpy"
            else vertcat(force2.torque, force2.force)
        ),
        point_in_local=np.array([0.17, 0.18, 0.19]),
        transformation_matrix=dummy_transformation_matrix,
    )
    segment_force_2 = fext.to_segment_natural_external_forces(Q=Q, segment_idx=2)
    TestUtils.assert_equal(
        np.squeeze(segment_force_2) if bionc_type == "numpy" else segment_force_2,
        np.squeeze(natural_force_2_expected * 2) if bionc_type == "numpy" else natural_force_2_expected * 2.0,
        expand=False,
        squeeze=True,
    )
