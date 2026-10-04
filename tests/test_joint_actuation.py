import numpy as np
import pytest
from casadi import Function

from bionc import JointType, EulerSequence, CartesianAxis, NaturalAxis
from .utils import TestUtils

GRAVITY = 9.81


def _segment(name, center_of_mass=(0.02, -0.25, 0.03)):
    from bionc.bionc_numpy import NaturalSegment

    return NaturalSegment.with_cartesian_inertial_parameters(
        name=name,
        length=0.5,
        mass=2.0,
        center_of_mass=np.array(center_of_mass),
        inertia=np.diag([0.02, 0.005, 0.03]),
    )


def _rot(axis, angle):
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K


def _pose(R, rp, length=0.5):
    return np.concatenate((R[:, 0], rp, rp - length * R[:, 1], R[:, 2]))


def _rigid_velocity(q, omega, vp):
    u, rp, rd, w = q[0:3], q[3:6], q[6:9], q[9:12]
    return np.concatenate((np.cross(omega, u), vp, vp + np.cross(omega, rd - rp), np.cross(omega, w)))


def _two_segment_model(child_joint_type, **child_joint_kwargs):
    from bionc.bionc_numpy import BiomechanicalModel

    model = BiomechanicalModel()
    model["A"] = _segment("A")
    model["B"] = _segment("B")
    return model


def _spherical_chain():
    from bionc.bionc_numpy import BiomechanicalModel

    model = BiomechanicalModel()
    model["A"] = _segment("A")
    model["B"] = _segment("B")
    model._add_joint(
        dict(
            name="hip",
            joint_type=JointType.GROUND_SPHERICAL,
            parent="GROUND",
            child="A",
            projection_basis=EulerSequence.ZXY,
        )
    )
    model._add_joint(
        dict(name="knee", joint_type=JointType.SPHERICAL, parent="A", child="B", projection_basis=EulerSequence.ZXY)
    )
    return model


def _spherical_chain_pose(rng):
    RA, RB = _rot([1, 2, 0.3], 0.7), _rot([0.2, -1, 1], 1.1)
    qA = _pose(RA, np.zeros(3))
    qB = _pose(RB, qA[6:9])
    return qA, qB, RA, RB


def _chain_cases(kind, rng):
    """Returns model, qA, qB, omega_A, omega_B (rigid angular velocities satisfying the constraints)."""
    from bionc.bionc_numpy import BiomechanicalModel

    if kind == "spherical":
        model = _spherical_chain()
        qA, qB, _, _ = _spherical_chain_pose(rng)
        return model, qA, qB, rng.normal(size=3), rng.normal(size=3)

    model = BiomechanicalModel()
    model["A"] = _segment("A")
    model["B"] = _segment("B")
    if kind == "hinge":
        model._add_joint(
            dict(
                name="hip",
                joint_type=JointType.GROUND_REVOLUTE,
                parent="GROUND",
                child="A",
                parent_axis=[CartesianAxis.X, CartesianAxis.X],
                child_axis=[NaturalAxis.V, NaturalAxis.W],
                theta=[np.pi / 2, np.pi / 2],
            )
        )
        model._add_joint(
            dict(
                name="knee",
                joint_type=JointType.REVOLUTE,
                parent="A",
                child="B",
                parent_axis=[NaturalAxis.U, NaturalAxis.U],
                child_axis=[NaturalAxis.V, NaturalAxis.W],
                theta=[np.pi / 2, np.pi / 2],
            )
        )
        RA = _rot([1, 0, 0], 0.6)
        RB = _rot(RA[:, 0], -0.9) @ RA
        qA, qB = _pose(RA, np.zeros(3)), None
        qB = _pose(RB, qA[6:9])
        omega_A = 0.7 * np.array([1.0, 0, 0])
        omega_B = omega_A + 1.3 * RA[:, 0]
        return model, qA, qB, omega_A, omega_B
    if kind == "universal":
        model._add_joint(
            dict(
                name="hip",
                joint_type=JointType.GROUND_UNIVERSAL,
                parent="GROUND",
                child="A",
                parent_axis=CartesianAxis.X,
                child_axis=NaturalAxis.V,
                theta=np.pi / 2,
            )
        )
        model._add_joint(
            dict(
                name="knee",
                joint_type=JointType.SPHERICAL,
                parent="A",
                child="B",
                projection_basis=EulerSequence.ZXY,
            )
        )
        RA = _rot([1, 0, 0], 0.5) @ _rot([0, 1, 0], 0.8)
        RB = _rot([0.2, -1, 1], 1.1)
        qA = _pose(RA, np.zeros(3))
        qB = _pose(RB, qA[6:9])
        # universal relative velocity lies in the span of the parent axis x and the child axis v
        omega_A = 0.7 * np.array([1.0, 0, 0]) - 0.4 * RA[:, 1]
        return model, qA, qB, omega_A, rng.normal(size=3)
    raise ValueError(kind)


def _joint_rates(model, Q_np, omegas, ground_omega=np.zeros(3)):
    """Joint rates conjugate to the joint generalized forces, thetadot = lstsq(E, omega_child - omega_parent)."""
    from bionc.bionc_numpy import SegmentNaturalCoordinates

    rates = []
    for joint in sorted(model.joints.values(), key=lambda j: j.index):
        Q_child = SegmentNaturalCoordinates(Q_np[12 * joint.child.index : 12 * joint.child.index + 12])
        Q_parent = (
            None
            if joint.parent is None
            else SegmentNaturalCoordinates(Q_np[12 * joint.parent.index : 12 * joint.parent.index + 12])
        )
        translations, rotations = joint.dof_axes(Q_parent, Q_child)
        assert translations is None
        omega_parent = ground_omega if joint.parent is None else omegas[joint.parent.index]
        rates.append(np.linalg.lstsq(rotations, omegas[joint.child.index] - omega_parent, rcond=None)[0])
    return np.concatenate(rates)


def _natural_joint_forces(bionc_type, model, Q_np, q):
    from bionc.bionc_numpy import NaturalCoordinates
    from bionc.bionc_numpy.generalized_force import JointGeneralizedForcesList

    if bionc_type == "numpy":
        forces = JointGeneralizedForcesList.empty_from_nb_joint(model.nb_joints)
        forces.add_all_joint_generalized_forces(model, q, NaturalCoordinates(Q_np))
        return np.asarray(forces.to_natural_joint_forces(model, NaturalCoordinates(Q_np))).ravel()

    from bionc.bionc_casadi import NaturalCoordinates as NaturalCoordinatesMX
    from bionc.bionc_casadi.generalized_force import JointGeneralizedForcesList as JointGeneralizedForcesListMX

    model_mx = model.to_mx()
    Q_mx = NaturalCoordinatesMX(Q_np)
    forces = JointGeneralizedForcesListMX.empty_from_nb_joint(model_mx.nb_joints)
    forces.add_all_joint_generalized_forces(model_mx, q, Q_mx)
    f = Function("f", [], [forces.to_natural_joint_forces(model_mx, Q_mx)])()["o0"]
    return np.asarray(f.toarray()).ravel()


@pytest.mark.parametrize("bionc_type", ["numpy", "casadi"])
@pytest.mark.parametrize("kind", ["spherical", "hinge", "universal"])
def test_joint_actuation_power_consistency(bionc_type, kind):
    rng = np.random.default_rng(1)
    model, qA, qB, omega_A, omega_B = _chain_cases(kind, rng)
    Q_np = np.concatenate((qA, qB))
    (
        TestUtils.assert_equal(
            model.holonomic_constraints(__import__("bionc").bionc_numpy.NaturalCoordinates(Q_np)),
            np.zeros(model.nb_holonomic_constraints),
            expand=False,
        )
        if False
        else None
    )

    from bionc.bionc_numpy import NaturalCoordinates

    np.testing.assert_allclose(model.holonomic_constraints(NaturalCoordinates(Q_np)), 0, atol=1e-12)

    Qdot = np.concatenate((_rigid_velocity(qA, omega_A, np.zeros(3)), None if False else np.zeros(12)))
    vB = Qdot[6:9]  # proximal point of B is the distal point of A
    Qdot[12:] = _rigid_velocity(qB, omega_B, vB)
    np.testing.assert_allclose(model.holonomic_constraints_jacobian(NaturalCoordinates(Q_np)) @ Qdot, 0, atol=1e-12)

    q = rng.normal(size=model.nb_joint_dof)
    f = _natural_joint_forces(bionc_type, model, Q_np, q)
    rates = _joint_rates(model, Q_np, [omega_A, omega_B])
    assert rates.shape == q.shape
    np.testing.assert_allclose(Qdot @ f, q @ rates, atol=1e-10)
    assert abs(q @ rates) > 1e-3


@pytest.mark.parametrize("bionc_type", ["numpy", "casadi"])
def test_hinge_torque_holds_pendulum(bionc_type):
    from bionc.bionc_numpy import BiomechanicalModel, NaturalCoordinates, NaturalVelocities

    model = BiomechanicalModel()
    model["P"] = _segment("P", center_of_mass=(0, -0.25, 0))
    model._add_joint(
        dict(
            name="hinge",
            joint_type=JointType.GROUND_REVOLUTE,
            parent="GROUND",
            child="P",
            parent_axis=[CartesianAxis.X, CartesianAxis.X],
            child_axis=[NaturalAxis.V, NaturalAxis.W],
            theta=[np.pi / 2, np.pi / 2],
        )
    )
    segment = model.segments["P"]
    R = _rot([1, 0, 0], np.radians(30))
    q = _pose(R, np.zeros(3))
    Q = NaturalCoordinates(q)
    Qdot = NaturalVelocities(np.zeros(12))
    np.testing.assert_allclose(model.holonomic_constraints(Q), 0, atol=1e-12)

    interpolation = segment.natural_center_of_mass.interpolate().to_array()
    com = interpolation @ q
    gravity_moment_x = np.cross(com, segment.mass * np.array([0, 0, -GRAVITY]))[0]
    assert abs(gravity_moment_x) > 1e-2

    def forward(torque):
        if bionc_type == "numpy":
            kwargs = {} if torque is None else dict(joint_generalized_forces=np.array([torque]))
            qddot, _ = model.forward_dynamics(Q, Qdot, **kwargs)
            return np.asarray(qddot).ravel()
        from bionc.bionc_casadi import NaturalCoordinates as NCmx, NaturalVelocities as NVmx

        model_mx = model.to_mx()
        kwargs = {} if torque is None else dict(joint_generalized_forces=np.array([torque]))
        qddot = model_mx.forward_dynamics(NCmx(q), NVmx(np.zeros(12)), **kwargs)[0]
        return np.asarray(Function("f", [], [qddot])()["o0"].toarray()).ravel()

    # the actuator balances the gravity moment
    assert np.max(np.abs(forward(-gravity_moment_x))) < 1e-10

    # without torque: angular acceleration about x from the moment of gravity about the pivot (Huygens)
    inertia_com = R @ np.diag([0.02, 0.005, 0.03]) @ R.T
    d2 = com[1] ** 2 + com[2] ** 2
    inertia_pivot = inertia_com[0, 0] + segment.mass * d2
    alpha = gravity_moment_x / inertia_pivot
    com_acceleration = interpolation @ forward(None)
    np.testing.assert_allclose(com_acceleration, np.cross([alpha, 0, 0], com), atol=1e-8)


def test_spherical_actuation_matches_inverse_dynamics():
    from bionc.bionc_numpy import NaturalAccelerations, NaturalCoordinates, NaturalVelocities, SegmentNaturalCoordinates

    rng = np.random.default_rng(1)
    model = _spherical_chain()
    qA, qB, _, _ = _spherical_chain_pose(rng)
    Q_np = np.concatenate((qA, qB))
    Q = NaturalCoordinates(Q_np)

    torques, _, _ = model.inverse_dynamics(Q, NaturalAccelerations(np.zeros(24)))
    torques = np.asarray(torques)
    assert np.max(np.abs(torques)) > 1e-2  # gravity has to be held

    q = []
    for joint in sorted(model.joints.values(), key=lambda j: j.index):
        i = joint.child.index
        Q_child = SegmentNaturalCoordinates(Q_np[12 * i : 12 * i + 12])
        Q_parent = None if joint.parent is None else SegmentNaturalCoordinates(Q_np[12 * joint.parent.index :][:12])
        _, axes = joint.dof_axes(Q_parent, Q_child)
        q.append(axes.T @ torques[:, i])
    q = np.concatenate(q)

    qddot, _ = model.forward_dynamics(Q, NaturalVelocities(np.zeros(24)), joint_generalized_forces=q)
    assert np.max(np.abs(np.asarray(qddot))) < 1e-8
    # and without the actuators the chain falls
    qddot_free, _ = model.forward_dynamics(Q, NaturalVelocities(np.zeros(24)))
    assert np.max(np.abs(np.asarray(qddot_free))) > 1e-2


def test_joint_actuation_errors():
    from bionc.bionc_numpy import NaturalCoordinates, NaturalVelocities

    model = _spherical_chain()
    rng = np.random.default_rng(1)
    qA, qB, _, _ = _spherical_chain_pose(rng)
    Q = NaturalCoordinates(np.concatenate((qA, qB)))
    Qdot = NaturalVelocities(np.zeros(24))
    with pytest.raises(ValueError):
        model.forward_dynamics(Q, Qdot, joint_generalized_forces=np.zeros(model.nb_joint_dof + 1))

    bionc = TestUtils.bionc_folder()
    module = TestUtils.load_module(bionc + "/examples/knee_parallel_mechanism/knee_feikes.py")
    knee = module.create_knee_model()
    Q = NaturalCoordinates(np.zeros(12 * knee.nb_segments))
    Qdot = NaturalVelocities(np.zeros(12 * knee.nb_segments))
    q = np.zeros(knee.nb_joint_dof)
    q[-1] = 1.0  # a dof of a sphere on plane or constant length joint
    with pytest.raises(NotImplementedError):
        knee.forward_dynamics(Q, Qdot, joint_generalized_forces=q)
