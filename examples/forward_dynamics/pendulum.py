import numpy as np

from bionc.bionc_numpy import (
    BiomechanicalModel,
    NaturalSegment,
    JointType,
    SegmentNaturalCoordinates,
    NaturalCoordinates,
    SegmentNaturalVelocities,
    NaturalVelocities,
)
from bionc import NaturalAxis, CartesianAxis, RK4


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


def main(mode: str = "x_revolute", show_structure: bool = False, show_results: bool = True):
    # Let's create a model
    model = BiomechanicalModel()
    # fill the biomechanical model with the segment
    model["pendulum"] = NaturalSegment(
        name="pendulum",
        alpha=np.pi / 2,  # setting alpha, beta, gamma to pi/2 creates a orthogonal coordinate system
        beta=np.pi / 2,
        gamma=np.pi / 2,
        length=1,
        mass=1,
        center_of_mass=np.array([0.1, 0.1, -0.1]),  # in segment coordinates system
        inertia=np.array([[0.05, 0, 0], [0, 0.05, 0], [0, 0, 0.05]]),  # in segment coordinates system
    )

    if mode == "x_revolute":
        # meaning we pivot around the cartesian x-axis
        # if you want to add a revolute joint,
        # you need to ensure that x is always orthogonal to u and v
        parent_axis = [CartesianAxis.X, CartesianAxis.X]
        child_axis = [NaturalAxis.V, NaturalAxis.W]
    elif mode == "y_revolute":
        # meaning we pivot around the cartesian y-axis
        parent_axis = [CartesianAxis.Y, CartesianAxis.Y]
        child_axis = [NaturalAxis.U, NaturalAxis.W]
    elif mode == "z_revolute":
        # meaning we pivot around the cartesian z-axis
        # turn the pendulum around to make it turn along w-axis
        parent_axis = [CartesianAxis.X, CartesianAxis.X]
        child_axis = [NaturalAxis.U, NaturalAxis.V]
    else:
        raise ValueError("Unknown mode. please choose between x_revolute, y_revolute or z_revolute")

    model._add_joint(
        dict(
            name="hinge",
            joint_type=JointType.GROUND_REVOLUTE,
            parent="GROUND",
            child="pendulum",
            parent_axis=parent_axis,
            child_axis=child_axis,
            theta=[np.pi / 2, np.pi / 2],
        )
    )

    model.save("pendulum.nmod")

    print(model.joints)
    print(model.nb_joints)
    print(model.nb_joint_constraints)

    if mode in ("x_revolute", "y_revolute"):
        Qi = SegmentNaturalCoordinates.from_components(u=[1, 0, 0], rp=[0, 0, 0], rd=[0, -1, 0], w=[0, 0, 1])
    elif mode == "z_revolute":
        Qi = SegmentNaturalCoordinates.from_components(u=[0, 0, -1], rp=[0, 0, 0], rd=[0, -1, 0], w=[1, 0, 0])
    else:
        raise ValueError("Unknown mode. please choose between x_revolute, y_revolute or z_revolute")

    Q = NaturalCoordinates(Qi)
    Qdoti = SegmentNaturalVelocities.from_components(udot=[0, 0, 0], rpdot=[0, 0, 0], rddot=[0, 0, 0], wdot=[0, 0, 0])
    Qdot = NaturalVelocities(Qdoti)

    print(model.joint_constraints(Q))
    print(model.joint_constraints_jacobian(Q))
    print(model.holonomic_constraints(Q))
    print(model.holonomic_constraints_jacobian(Q))

    if show_structure:
        from matplotlib import pyplot as plt

        Qi_random = SegmentNaturalCoordinates.from_components(
            u=[0.1, 0.2, 0.3],
            rp=[0.4, 0.5, 0.6],
            rd=[0.7, 0.8, 0.9],
            w=[1.0, 1.1, 1.2],
        )
        Q_random = NaturalCoordinates(Qi_random)
        plt.figure()
        plt.spy(model.mass_matrix)
        plt.title("Mass matrix")

        plt.figure()
        plt.spy(model.rigid_body_constraints_jacobian(Q_random))
        plt.title("Rigid body constraints jacobian")
        plt.ylabel("Rigid body constraints")
        plt.xlabel("Natural coordinates")

        plt.figure()
        plt.spy(model.holonomic_constraints_jacobian(Q_random))
        plt.title("Holonomic constraints jacobian")
        plt.ylabel("Holonomic constraints")
        plt.xlabel("Natural coordinates")

        plt.figure()
        G = model.mass_matrix
        K = model.holonomic_constraints_jacobian(Q_random)

        upper_KKT_matrix = np.concatenate((G, K.T), axis=1)
        lower_KKT_matrix = np.concatenate((K, np.zeros((K.shape[0], K.shape[0]))), axis=1)
        KKT_matrix = np.concatenate((upper_KKT_matrix, lower_KKT_matrix), axis=0)
        plt.spy(KKT_matrix)
        plt.title("KKT matrix")

        plt.show()

    # The actual simulation
    t_final = 10
    time_steps, all_states, dynamics = drop_the_pendulum(
        model=model,
        Q_init=Q,
        Qdot_init=Qdot,
        t_final=t_final,
    )

    if show_results:
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

    return model, all_states


if __name__ == "__main__":
    # model, all_states = main("x_revolute", show_results=False)
    # model, all_states = main("y_revolute", show_results=False)
    model, all_states = main("z_revolute", show_structure=True, show_results=True)

    # animate the motion
    from bionc import Viz

    viz = Viz(model)
    viz.animate(all_states[:12, :], None, frame_rate=50)
