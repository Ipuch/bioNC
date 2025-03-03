import numpy as np

from bionc import NaturalAxis, CartesianAxis, RK4, TransformationMatrixType
from bionc.bionc_numpy import (
    BiomechanicalModel,
    NaturalSegment,
    JointType,
    SegmentNaturalCoordinates,
    NaturalCoordinates,
    SegmentNaturalVelocities,
    NaturalVelocities,
)


def drop_the_pendulum(
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
    print("Evaluate Rigid Body Constraints Jacobian Derivative:")
    print(model.rigid_body_constraint_jacobian_derivative(Qdot_init))

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
        )
        return np.concatenate((states[idx_velocities], qddot.to_array()), axis=0), lambdas

    # Solve the Initial Value Problem (IVP) for each time step
    all_states = RK4(t=time_steps, f=lambda t, states: dynamics(t, states)[0], y0=states_0)

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


def build_n_link_pendulum(nb_segments: int = 1) -> BiomechanicalModel:
    """Build a n-link pendulum model"""
    if nb_segments < 1:
        raise ValueError("The number of segment must be greater than 1")
    # Let's create a model
    model = BiomechanicalModel()
    # number of segments
    # fill the biomechanical model with the segment
    for i in range(nb_segments):
        name = f"pendulum_{i}"
        model[name] = NaturalSegment.with_cartesian_inertial_parameters(
            name=name,
            alpha=np.pi / 2,  # setting alpha, beta, gamma to pi/2 creates an orthogonal coordinate system
            beta=np.pi / 2,
            gamma=np.pi / 2,
            length=1,
            mass=1,
            center_of_mass=np.array([0, 0.1, 0]),  # in segment coordinates system
            inertia=np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),  # in segment coordinates system
            inertial_transformation_matrix=TransformationMatrixType.Buv,
        )
    # add a revolute joint (still experimental)
    # if you want to add a revolute joint,
    # you need to ensure that x is always orthogonal to u and v
    model._add_joint(
        dict(
            name="hinge_0",
            joint_type=JointType.GROUND_REVOLUTE,
            parent="GROUND",
            child="pendulum_0",
            parent_axis=[CartesianAxis.X, CartesianAxis.X],
            child_axis=[NaturalAxis.V, NaturalAxis.W],  # meaning we pivot around the cartesian x-axis
            theta=[np.pi / 2, np.pi / 2],
        )
    )
    for i in range(1, nb_segments):
        model._add_joint(
            dict(
                name=f"hinge_{i}",
                joint_type=JointType.REVOLUTE,
                parent=f"pendulum_{i - 1}",
                child=f"pendulum_{i}",
                parent_axis=[NaturalAxis.U, NaturalAxis.U],
                child_axis=[NaturalAxis.V, NaturalAxis.W],
                theta=[np.pi / 2, np.pi / 2],
            )
        )
    return model


if __name__ == "__main__":
    display_model_matrices = False
    display_post_computation = False
    # Let's create a model
    nb_segments = 20
    model = build_n_link_pendulum(nb_segments=nb_segments)

    print(model.joints)
    print(model.nb_joints)
    print(model.nb_joint_constraints)

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

    print(model.joint_constraints(Q))
    print(model.joint_constraints_jacobian(Q))
    print(model.holonomic_constraints(Q))
    print(model.holonomic_constraints_jacobian(Q))

    # one can comment this section if he doesn't to display the matrices
    if display_model_matrices:
        from matplotlib import pyplot as plt

        K = model.joint_constraints_jacobian(Q)
        # subplot with 3 columns and 1 row
        fig, axs = plt.subplots(1, 3)
        # spy(K) on axis (0,0)
        axs[0].spy(K)
        # title of the axis (0,0)
        axs[0].set_title("Constraint Jacobian K")
        #
        axs[1].spy(model.mass_matrix)
        axs[1].set_title("Mass matrix M")
        #
        G = model.mass_matrix
        K = model.rigid_body_constraints_jacobian(Q)
        Kdot = model.rigid_body_constraint_jacobian_derivative(Qdot)
        upper_KKT_matrix = np.concatenate((G, K.T), axis=1)
        lower_KKT_matrix = np.concatenate((K, np.zeros((K.shape[0], K.shape[0]))), axis=1)
        KKT_matrix = np.concatenate((upper_KKT_matrix, lower_KKT_matrix), axis=0)

        axs[2].spy(KKT_matrix)
        axs[2].set_title("KKT matrix")
        plt.show()

    # actual simulation
    t_final = 10  # seconds
    steps_per_second = 50
    time_steps, all_states, dynamics = drop_the_pendulum(
        model=model,
        Q_init=Q,
        Qdot_init=Qdot,
        t_final=t_final,
        steps_per_second=steps_per_second,
    )

    if display_post_computation:
        defects, defects_dot, joint_defects, all_lambdas = post_computations(
            model=model,
            time_steps=time_steps,
            all_states=all_states,
            dynamics=dynamics,
        )

        from viz import plot_series

        # Plot the results
        # the following graphs have to be near zero the more the simulation is long, the more constraints drift from zero
        plot_series(time_steps, defects, legend="rigid_constraint")  # Phi_r
        plot_series(time_steps, defects_dot, legend="rigid_constraint_derivative")  # Phi_r_dot
        plot_series(time_steps, joint_defects, legend="joint_constraint")  # Phi_j
        # the lagrange multipliers are the forces applied to maintain the system (rigidbody and joint constraints)
        plot_series(time_steps, all_lambdas, legend="lagrange_multipliers")  # lambda

    # animate the motion
    from pyorerun import PhaseRerun
    from bionc.vizualization.pyorerun_interface import BioncModelNoMesh

    prr = PhaseRerun(t_span=time_steps)
    model_interface = BioncModelNoMesh(model)
    prr.add_animated_model(model_interface, all_states[: (nb_segments * 12), :])
    prr.rerun()
