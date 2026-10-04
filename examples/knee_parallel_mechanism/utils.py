from bionc import (
    BiomechanicalModel,
    NaturalCoordinates,
    NaturalVelocities,
    RK4,
)
import numpy as np


def forward_integration(
    model: BiomechanicalModel,
    Q_init: NaturalCoordinates,
    Qdot_init: NaturalVelocities,
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

    if (model.rigid_body_constraints(Q_init) > 1e-4).any():
        print(model.rigid_body_constraints(Q_init))
        raise ValueError(
            "The segment natural coordinates don't satisfy the rigid body constraint, at initial conditions."
        )

    t_final = t_final  # [s]
    steps_per_second = steps_per_second
    time_steps = np.linspace(0, t_final, int(steps_per_second * t_final + 1))

    # initial conditions, x0 = [Qi, Qidot]
    states_0 = np.concatenate((Q_init.to_array(), Qdot_init.to_array()), axis=0)

    # Create the forward dynamics function Callable (f(t, x) -> xdot)
    def dynamics(t, states):
        idx_coordinates = slice(0, model.nb_Q)
        idx_velocities = slice(model.nb_Q, model.nb_Q + model.nb_Qdot)

        qddot, lambdas = model.forward_dynamics(
            NaturalCoordinates(states[idx_coordinates]),
            NaturalVelocities(states[idx_velocities]),
            # stabilization=dict(alpha=0.5, beta=0.5),
        )
        return np.concatenate((states[idx_velocities], qddot.to_array()), axis=0), lambdas

    # Solve the Initial Value Problem (IVP) for each time step
    normalize_idx = model.normalized_coordinates
    all_states = RK4(t=time_steps, f=lambda t, states: dynamics(t, states)[0], y0=states_0, normalize_idx=normalize_idx)

    return time_steps, all_states, dynamics


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


def knee_angles(model: BiomechanicalModel, Q: NaturalCoordinates, femur: str = "THIGH", tibia: str = "SHANK"):
    """
    Flexion (positive), abduction and axial rotation of the tibia relative to the femur [deg],
    as a ZXY sequence of the femur segment coordinate system (X anterior, Y proximal, Z lateral)
    """
    femur_segment, tibia_segment = model.segments[femur], model.segments[tibia]
    R = (
        femur_segment.segment_coordinates_system(Q.vector(femur_segment.index)).rot.T
        @ tibia_segment.segment_coordinates_system(Q.vector(tibia_segment.index)).rot
    )
    z = np.arctan2(-R[0, 1], R[1, 1])
    x = np.arcsin(np.clip(R[2, 1], -1, 1))
    y = np.arctan2(-R[2, 0], R[2, 2])
    return np.degrees([-z, x, y])


def knee_pose_at_flexion(
    model: BiomechanicalModel,
    Q_femur: np.ndarray,
    Q_tibia: np.ndarray,
    flexion: float,
    femur: str = "THIGH",
    tibia: str = "SHANK",
    step: float = 0.002,
) -> np.ndarray:
    """
    Follow the one degree of freedom path of the knee mechanism, femur fixed, from Q_tibia (a pose satisfying the
    knee constraints, e.g. full extension) in the direction of flexion, up to the given flexion [deg].

    Returns
    -------
    np.ndarray
        The tibia natural coordinates [12]
    """
    from bionc.bionc_numpy import SegmentNaturalCoordinates

    femur_segment, tibia_segment = model.segments[femur], model.segments[tibia]
    joints = [j for j in model.joints.values() if j.parent is femur_segment and j.child is tibia_segment]
    Q_femur = SegmentNaturalCoordinates(np.asarray(Q_femur, dtype=float).reshape(12))
    q_tibia = np.asarray(Q_tibia, dtype=float).reshape(12).copy()

    def constraints(q):
        Q_t = SegmentNaturalCoordinates(q)
        return np.concatenate(
            [np.atleast_1d(j.constraint(Q_femur, Q_t)).ravel() for j in joints]
            + [tibia_segment.rigid_body_constraint(Q_t)]
        )

    def jacobian(q):
        Q_t = SegmentNaturalCoordinates(q)
        return np.vstack(
            [np.atleast_2d(j.child_constraint_jacobian(Q_femur, Q_t)) for j in joints]
            + [tibia_segment.rigid_body_constraint_jacobian(Q_t)]
        )

    def flexion_of(q):
        Q = np.zeros(12 * model.nb_segments)
        Q[12 * femur_segment.index : 12 * femur_segment.index + 12] = Q_femur.to_array()
        Q[12 * tibia_segment.index : 12 * tibia_segment.index + 12] = q
        return knee_angles(model, NaturalCoordinates(Q), femur, tibia)[0]

    previous_direction = None
    while flexion_of(q_tibia) < flexion:
        # the knee constraints leave one degree of freedom: the null space of their jacobian
        direction = np.linalg.svd(jacobian(q_tibia))[2][-1]
        if previous_direction is None:
            if flexion_of(q_tibia + 1e-4 * direction) < flexion_of(q_tibia):
                direction = -direction
        elif direction @ previous_direction < 0:
            direction = -direction
        previous_direction = direction
        q_tibia = q_tibia + step * direction
        # Newton projection back on the constraints
        for _ in range(20):
            residual = constraints(q_tibia)
            if np.abs(residual).max() < 1e-13:
                break
            q_tibia = q_tibia - np.linalg.lstsq(jacobian(q_tibia), residual, rcond=None)[0]

    return q_tibia


def joint_lambda_slices(model: BiomechanicalModel) -> dict:
    """
    Position of each joint in the vector of Lagrange multipliers, which follows the holonomic constraints:
    for each segment, the constraints of the joints of which it is the child, then its 6 rigid body constraints.
    """
    slices, start = {}, 0
    for i, segment in enumerate(model.segments_no_ground.values()):
        for joint in model.joints_from_child_index(i, remove_free_joints=True):
            slices[joint.name] = slice(start, start + joint.nb_constraints)
            start += joint.nb_constraints
        start += 6
    return slices


def knee_joint_loads(
    model: BiomechanicalModel, Q: NaturalCoordinates, lambdas: np.ndarray, femur: str = "THIGH", tibia: str = "SHANK"
) -> dict:
    """
    Physical loads of the joints between the femur and the tibia, from the Lagrange multipliers.
    With G Qddot + K^T lambda = f, the constraint forces are -K^T lambda:
    - SphereOnPlane, phi = (P - A).n - r [m], lambda [N]: the femur pushes the tibia with lambda n at the contact,
      the contact force is -lambda, positive in compression (a negative value would mean the sphere pulls the plane).
    - ConstantLength, phi = |P_f - P_t|^2 - L^2 [m^2], lambda [N/m]: the ligament pulls the tibia with
      2 lambda (P_f - P_t), its tension is 2 lambda L, positive in tension (a negative value would mean it pushes).

    Returns
    -------
    dict
        joint name -> dict(kind, load [N], force_on_tibia [3], point [3]), force and point in the global frame
    """
    from bionc.bionc_numpy.joints import Joint

    femur_segment, tibia_segment = model.segments[femur], model.segments[tibia]
    Q_femur, Q_tibia = Q.vector(femur_segment.index), Q.vector(tibia_segment.index)
    lambdas = np.asarray(lambdas).reshape(-1)
    slices = joint_lambda_slices(model)
    loads = {}
    for name, joint in model.joints.items():
        if joint.parent is not femur_segment or joint.child is not tibia_segment:
            continue
        lam = float(lambdas[slices[name]][0])
        if isinstance(joint, Joint.SphereOnPlane):
            center = joint.sphere_center.position_in_global(Q_femur).reshape(3)
            normal = np.asarray(joint.plane_normal.position_in_global(Q_tibia)).reshape(3)
            normal = normal / np.linalg.norm(normal)
            loads[name] = dict(
                kind="contact",
                load=-lam,
                force_on_tibia=lam * normal,
                point=center - joint.sphere_radius * normal,
            )
        elif isinstance(joint, Joint.ConstantLength):
            femoral = joint.parent_point.position_in_global(Q_femur).reshape(3)
            tibial = joint.child_point.position_in_global(Q_tibia).reshape(3)
            loads[name] = dict(
                kind="ligament",
                load=2 * lam * joint.length,
                force_on_tibia=2 * lam * (femoral - tibial),
                point=tibial,
            )
    return loads


def knee_translation(model: BiomechanicalModel, Q: NaturalCoordinates, femur: str = "THIGH", tibia: str = "SHANK"):
    """
    Position of the tibia origin (rp of the tibia) relative to the femur knee centre (rd of the femur),
    in the femur segment coordinate system (X anterior, Y proximal, Z lateral) [m]
    """
    femur_segment, tibia_segment = model.segments[femur], model.segments[tibia]
    Q_femur, Q_tibia = Q.vector(femur_segment.index), Q.vector(tibia_segment.index)
    R = femur_segment.segment_coordinates_system(Q_femur).rot
    return R.T @ (np.asarray(Q_tibia.rp).reshape(3) - np.asarray(Q_femur.rd).reshape(3))
