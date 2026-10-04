import numpy as np

from bionc import (
    BiomechanicalModel,
    SegmentNaturalCoordinates,
    SegmentNaturalVelocities,
    NaturalCoordinates,
    NaturalVelocities,
    NaturalAccelerations,
    NaturalSegment,
    JointType,
    EulerSequence,
    TransformationMatrixType,
    RK4,
)


def build_3d_pendulum():
    # Let's create a model
    model = BiomechanicalModel()
    # fill the biomechanical model with the segment
    model["pendulum"] = NaturalSegment.with_cartesian_inertial_parameters(
        name="pendulum",
        alpha=np.pi / 2,  # setting alpha, beta, gamma to pi/2 creates a orthogonal coordinate system
        beta=np.pi / 2,
        gamma=np.pi / 2,
        length=1,
        mass=1,
        center_of_mass=np.array([0, -1, -0.5]),  # in segment coordinates system
        inertia=np.array([[0.01, 0, 0], [0, 0.001, 0], [0, 0, 0.01]]),  # in segment coordinates system
    )
    # add a spherical joint (still experimental)

    model._add_joint(
        dict(
            name="spherical",
            joint_type=JointType.GROUND_SPHERICAL,
            parent="GROUND",
            child="pendulum",
            projection_basis=EulerSequence.XYZ,
            child_basis=TransformationMatrixType.Buv,
        )
    )

    model.save("pendulum_3d.nmod")

    return model


def apply_force_and_drop_pendulum(t_final: float = 10, Q_init: np.ndarray = None, joint_generalized_forces=None):
    """
    This function is used to test the external force

    Parameters
    ----------
    t_final: float
        The final time of the simulation
    Q_init: np.ndarray
        The initial natural coordinates [12], hanging along -y by default
    joint_generalized_forces: np.ndarray
        Constant joint torques about the XYZ Euler axes of the spherical joint [3], None for a passive pendulum

    Returns
    -------
    tuple[BiomechanicalModel, np.ndarray, np.ndarray, Callable]:
        model : BiomechanicalModel
            The model to be simulated
        time_steps : np.ndarray
            The time steps of the simulation
        all_states : np.ndarray
            The states of the system at each time step X = [Q, Qdot]
        dynamics : Callable
            The dynamics of the system, f(t, X) = [Xdot, lambdas]

    """
    model = build_3d_pendulum()

    if Q_init is None:
        Q_init = SegmentNaturalCoordinates.from_components(u=[1, 0, 0], rp=[0, 0, 0], rd=[0, -1, 0], w=[0, 0, 1])
    Q = NaturalCoordinates(np.asarray(Q_init).reshape(12))
    Qdoti = SegmentNaturalVelocities.from_components(udot=[0, 0, 0], rpdot=[0, 0, 0], rddot=[0, 0, 0], wdot=[0, 0, 0])
    Qdot = NaturalVelocities(Qdoti)

    time_steps, all_states, dynamics = drop_the_pendulum(
        model=model,
        Q_init=Q,
        Qdot_init=Qdot,
        joint_generalized_forces=joint_generalized_forces,
        t_final=t_final,
        steps_per_second=200,
    )

    return model, time_steps, all_states, dynamics


def drop_the_pendulum(
    model: BiomechanicalModel,
    Q_init: NaturalCoordinates,
    Qdot_init: NaturalVelocities,
    joint_generalized_forces: np.ndarray = None,
    t_final: float = 2,
    steps_per_second: int = 50,
):
    """
    This function simulates the dynamics of a natural segment falling from 0m during 2s

    Parameters
    ----------
    model : BiomechanicalModel
        The model to be simulated
    Q_init : SegmentNaturalCoordinates
        The initial natural coordinates of the segment
    Qdot_init : SegmentNaturalVelocities
        The initial natural velocities of the segment
    joint_generalized_forces : np.ndarray
        Constant joint torques, one per degree of freedom of the spherical joint (about its XYZ Euler axes)
    t_final : float, optional
        The final time of the simulation, by default 2
    steps_per_second : int, optional
        The number of steps per second, by default 50

    Returns
    -------
    tuple:
        time_steps : np.ndarray
            The time steps of the simulation
        all_states : np.ndarray
            The states of the system at each time step X = [Q, Qdot]
        dynamics : Callable
            The dynamics of the system, f(t, X) = [Xdot, lambdas]
    """

    print("Evaluate Rigid Body Constraints:")
    print(model.rigid_body_constraints(Q_init))

    if (model.rigid_body_constraints(Q_init) > 1e-6).any():
        print(model.rigid_body_constraints(Q_init))
        raise ValueError(
            "The segment natural coordinates don't satisfy the rigid body constraint, at initial conditions."
        )

    t_final = t_final  # [s]
    steps_per_second = steps_per_second
    time_steps = np.linspace(0, t_final, steps_per_second * t_final + 1)

    # initial conditions, x0 = [Qi, Qidot]
    states_0 = np.concatenate((Q_init.to_array(), Qdot_init.to_array()), axis=0)

    # Create the forward dynamics function Callable (f(t, x) -> xdot)
    def dynamics(t, states):
        idx_coordinates = slice(0, model.nb_Q)
        idx_velocities = slice(model.nb_Q, model.nb_Q + model.nb_Qdot)

        qddot, lambdas = model.forward_dynamics(
            NaturalCoordinates(states[idx_coordinates]),
            NaturalVelocities(states[idx_velocities]),
            joint_generalized_forces=joint_generalized_forces,
            stabilization=dict(alpha=50, beta=20),
        )
        return np.concatenate((states[idx_velocities], qddot.to_array()), axis=0), lambdas

    # Solve the Initial Value Problem (IVP) for each time step
    # normalize_idx = model.normalized_coordinates
    all_states = RK4(
        t=time_steps,
        f=lambda t, states: dynamics(t, states)[0],
        y0=states_0,
        # normalize_idx=normalize_idx
    )

    return time_steps, all_states, dynamics


def rotated_pose(rotation: np.ndarray) -> np.ndarray:
    """Natural coordinates of the pendulum, rotated about the pivot from the pose hanging along -y"""
    u, rd, w = rotation @ np.array([1.0, 0, 0]), rotation @ np.array([0, -1.0, 0]), rotation @ np.array([0, 0, 1.0])
    return np.concatenate((u, np.zeros(3), rd, w))


def holding_joint_torques(model: BiomechanicalModel, Q: np.ndarray) -> np.ndarray:
    """
    Joint torques that hold the pendulum still at Q, as minimal coordinate generalized forces: one torque per
    degree of freedom of the spherical joint, about its XYZ Euler axes.

    The inverse dynamics gives the moment M applied by the joint on the pendulum (a spherical joint transmits no
    reaction moment, so M is the actuator torque). The generalized force of the Euler angle i is its power
    conjugate, q_i = e_i . M, with e_i the Euler axes of the joint (joint.dof_axes).
    """
    Q = NaturalCoordinates(np.asarray(Q).reshape(12))
    torques, _, _ = model.inverse_dynamics(Q=Q, Qddot=NaturalAccelerations(np.zeros(12)))
    joint = model.joints["spherical"]
    _, euler_axes = joint.dof_axes(None, Q.vector(0))
    return euler_axes.T @ torques[:, 0]


def post_computations(model: BiomechanicalModel, time_steps: np.ndarray, all_states: np.ndarray, dynamics):
    """
    This function computes:
     - the rigid body constraint error
     - the rigid body constraint jacobian derivative error
     - the joint constraint error
     - the lagrange multipliers of the rigid body constraint

    Parameters
    ----------
    model : NaturalSegment
        The segment to be simulated
    time_steps : np.ndarray
        The time steps of the simulation
    all_states : np.ndarray
        The states of the system at each time step X = [Q, Qdot]
    dynamics : Callable
        The dynamics of the system, f(t, X) = [Xdot, lambdas]

    Returns
    -------
    tuple:
        rigid_body_constraint_error : np.ndarray
            The rigid body constraint error at each time step
        rigid_body_constraint_jacobian_derivative_error : np.ndarray
            The rigid body constraint jacobian derivative error at each time step
        joint_constraints: np.ndarray
            The joint constraints at each time step
        lambdas : np.ndarray
            The lagrange multipliers of the rigid body constraint at each time step
    """
    idx_coordinates = slice(0, model.nb_Q)
    idx_velocities = slice(model.nb_Q, model.nb_Q + model.nb_Qdot)

    # compute the quantities of interest after the integration
    all_lambdas = np.zeros((model.nb_holonomic_constraints, len(time_steps)))
    defects = np.zeros((model.nb_rigid_body_constraints, len(time_steps)))
    defects_dot = np.zeros((model.nb_rigid_body_constraints, len(time_steps)))
    joint_defects = np.zeros((model.nb_joint_constraints, len(time_steps)))
    joint_defects_dot = np.zeros((model.nb_joint_constraints, len(time_steps)))

    for i in range(len(time_steps)):
        defects[:, i] = model.rigid_body_constraints(NaturalCoordinates(all_states[idx_coordinates, i]))
        defects_dot[:, i] = model.rigid_body_constraints_derivative(
            NaturalCoordinates(all_states[idx_coordinates, i]), NaturalVelocities(all_states[idx_velocities, i])
        )

        joint_defects[:, i] = model.joint_constraints(NaturalCoordinates(all_states[idx_coordinates, i]))
        # todo : to be implemented
        # joint_defects_dot = model.joint_constraints_derivative(
        #     NaturalCoordinates(all_states[idx_coordinates, i]),
        #     NaturalVelocities(all_states[idx_velocities, i]))
        # )

        all_lambdas[:, i : i + 1] = dynamics(time_steps[i], all_states[:, i])[1]

    return defects, defects_dot, joint_defects, all_lambdas


def main(mode: str = "constant_torque", show_results: bool = True):
    """
    mode
        "passive": the pendulum falls from its pose hanging along -y
        "hold": tilted pendulum held still by the joint torques of the inverse dynamics, projected on the Euler axes
        "constant_torque": from the passive equilibrium (center of mass below the pivot), a constant torque about the
            X Euler axis makes the pendulum swing about the angle phi* with m g d sin(phi*) = torque
    """
    model = build_3d_pendulum()
    segment = model.segments["pendulum"]
    center_of_mass = np.asarray(segment.natural_center_of_mass.interpolate().to_array() @ rotated_pose(np.eye(3)))
    distance = np.linalg.norm(center_of_mass)  # pivot to center of mass

    # passive equilibrium: rotation about x that brings the center of mass right below the pivot
    equilibrium_angle = np.arctan2(-center_of_mass[1], -center_of_mass[2])

    if mode == "passive":
        Q_init, joint_generalized_forces = None, None
    elif mode == "hold":
        # tilted from the passive equilibrium, keeping the center of mass below the pivot: with constant torques,
        # the equilibria with the center of mass above the horizontal are unstable
        Q_init = rotated_pose(rotation_x(np.radians(25)) @ rotation_y(np.radians(15)) @ rotation_x(equilibrium_angle))
        joint_generalized_forces = holding_joint_torques(model, Q_init)
        print("holding torques about the XYZ Euler axes [N.m]:", joint_generalized_forces.round(4))
    elif mode == "constant_torque":
        Q_init = rotated_pose(rotation_x(equilibrium_angle))
        torque = 0.5 * segment.mass * 9.81 * distance  # equilibrium at phi* = 30 deg
        joint_generalized_forces = np.array([torque, 0, 0])
        # from rest, the work of the torque equals the gain of potential energy at the turning point:
        # torque * phi = m g d (1 - cos(phi)), solved by Newton from 2 phi*
        phi_max = np.radians(60)
        for _ in range(20):
            residual = torque * phi_max - segment.mass * 9.81 * distance * (1 - np.cos(phi_max))
            phi_max -= residual / (torque - segment.mass * 9.81 * distance * np.sin(phi_max))
        print(
            f"torque about X {torque:.3f} N.m: from {np.degrees(equilibrium_angle):.2f} deg, the Euler angle about X "
            f"swings about {np.degrees(equilibrium_angle) + 30:.2f} deg, up to {np.degrees(equilibrium_angle + phi_max):.2f} deg"
        )
    else:
        raise ValueError("mode must be 'passive', 'hold' or 'constant_torque'")

    model, time_steps, all_states, dynamics = apply_force_and_drop_pendulum(
        t_final=5, Q_init=Q_init, joint_generalized_forces=joint_generalized_forces
    )

    joint_angles = np.degrees(
        np.array([model.natural_coordinates_to_joint_angles(NaturalCoordinates(q))[:, 0] for q in all_states[:12].T])
    )
    print(
        "joint angles (XYZ) [deg] min / mean / max:",
        joint_angles.min(axis=0).round(2),
        joint_angles.mean(axis=0).round(2),
        joint_angles.max(axis=0).round(2),
    )

    if show_results:
        import matplotlib.pyplot as plt

        plt.figure()
        for i, axis in enumerate("XYZ"):
            plt.plot(time_steps, joint_angles[:, i], label=f"Euler angle about {axis}")
        plt.xlabel("time [s]")
        plt.ylabel("[deg]")
        plt.title(f"Spherical joint angles, mode {mode!r}")
        plt.legend()
        plt.show()

    return model, all_states, time_steps


def rotation_x(angle: float) -> np.ndarray:
    return np.array([[1, 0, 0], [0, np.cos(angle), -np.sin(angle)], [0, np.sin(angle), np.cos(angle)]])


def rotation_y(angle: float) -> np.ndarray:
    return np.array([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]])


if __name__ == "__main__":
    model, all_states, time_steps = main(mode="constant_torque", show_results=True)

    # animate the motion
    from pyorerun import PhaseRerun
    from bionc.vizualization.pyorerun_interface import BioncModelNoMesh

    prr = PhaseRerun(t_span=time_steps[:200])
    model_interface = BioncModelNoMesh(model)
    prr.add_animated_model(model_interface, all_states[:12, :200])
    prr.rerun()
