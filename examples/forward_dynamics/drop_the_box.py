import numpy as np
from typing import Callable

from bionc import RK4
from bionc.bionc_numpy import (
    NaturalSegment,
    SegmentNaturalCoordinates,
    SegmentNaturalVelocities,
)


def drop_the_box(
    my_segment: NaturalSegment,
    Q_init: SegmentNaturalCoordinates,
    Qdot_init: SegmentNaturalVelocities,
    t_final: float = 2,
    steps_per_second: int = 50,
):
    """
    This function simulates the dynamics of a natural segment falling from 0m during 2s

    Parameters
    ----------
    my_segment : NaturalSegment
        The segment to be simulated
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
    print(my_segment.rigid_body_constraint(Q_init))

    if (my_segment.rigid_body_constraint(Q_init) > 1e-6).any():
        print(my_segment.rigid_body_constraint(Q_init))
        raise ValueError(
            "The segment natural coordinates don't satisfy the rigid body constraint, at initial conditions."
        )

    t_final = t_final  # [s]
    steps_per_second = steps_per_second
    time_steps = np.linspace(0, t_final, steps_per_second * t_final + 1)

    # initial conditions, x0 = [Qi, Qidot]
    states_0 = np.concatenate((Q_init.vector, Qdot_init.vector), axis=0)

    # Create the forward dynamics function Callable (f(t, x) -> xdot)
    def dynamics(t, states):
        qddot, lambdas = my_segment.differential_algebraic_equation(
            states[0:12],
            states[12:24],
        )
        return np.concatenate((states[12:24], qddot), axis=0), lambdas

    # Solve the Initial Value Problem (IVP) for each time step
    all_states = RK4(t=time_steps, f=lambda t, states: dynamics(t, states)[0], y0=states_0)

    return time_steps, all_states, dynamics


def post_computations(segment: NaturalSegment, time_steps: np.ndarray, all_states: np.ndarray, dynamics: Callable):
    """
    This function computes:
     - the rigid body constraint error
     - the rigid body constraint jacobian derivative error
     - the lagrange multipliers of the rigid body constraint
     - the center of mass position

    Parameters
    ----------
    segment : NaturalSegment
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
        lambdas : np.ndarray
            The lagrange multipliers of the rigid body constraint at each time step
        center_of_mass : np.ndarray
            The center of mass position at each time step
    """
    # compute the quantities of interest after the integration
    all_lambdas = np.zeros((6, len(time_steps)))
    defects = np.zeros((6, len(time_steps)))
    defects_dot = np.zeros((6, len(time_steps)))
    center_of_mass = np.zeros((3, len(time_steps)))
    for i in range(len(time_steps)):
        defects[:, i] = segment.rigid_body_constraint(SegmentNaturalCoordinates(all_states[0:12, i]))
        defects_dot[:, i] = segment.rigid_body_constraint_derivative(
            SegmentNaturalCoordinates(all_states[0:12, i]),
            SegmentNaturalVelocities(all_states[12:24, i]),
        )
        all_lambdas[:, i] = dynamics(time_steps[i], all_states[:, i])[1]
        center_of_mass[:, i] = segment.natural_center_of_mass.interpolate() @ all_states[0:12, i]

    return defects, defects_dot, all_lambdas, center_of_mass


if __name__ == "__main__":
    # Let's create a segment
    my_segment = NaturalSegment.with_cartesian_inertial_parameters(
        name="box",
        alpha=np.pi / 2,  # setting alpha, beta, gamma to pi/2 creates a orthogonal coordinate system
        beta=np.pi / 2,
        gamma=np.pi / 2,
        length=1,
        mass=1,
        center_of_mass=np.array([0, 0, 0]),  # in segment coordinates system
        inertia=np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),  # in segment coordinates system
    )

    # Let's create a motion now
    # One can comment, one of the following Qi to pick, one initial condition
    # u as x-axis, w as z axis
    Qi = SegmentNaturalCoordinates.from_components(u=[1, 0, 0], rp=[0, 0, 0], rd=[0, -1, 0], w=[0, 0, 1])
    # u as y-axis
    # Qi = SegmentNaturalCoordinates.from_components(
    #     u=[0, 1, 0], rp=[0, 0, 0], rd=[0, 0, -1], w=[1, 0, 0]
    #     )
    # u as z-axis
    # Qi = SegmentNaturalCoordinates.from_components(u=[0, 0, 1], rp=[0, 0, 0], rd=[-1, 0, 0], w=[0, 1, 0])

    # Velocities are set to zero at t=0
    Qidot = SegmentNaturalVelocities.from_components(
        udot=np.array([0, 0, 0]), rpdot=np.array([0, 0, 0]), rddot=np.array([0, 0, 0]), wdot=np.array([0, 0, 0])
    )

    t_final = 2
    time_steps, all_states, dynamics = drop_the_box(
        my_segment=my_segment,
        Q_init=Qi,
        Qdot_init=Qidot,
        t_final=t_final,
    )

    defects, defects_dot, all_lambdas, center_of_mass = post_computations(my_segment, time_steps, all_states, dynamics)

    from viz import plot_series, animate_natural_segment

    # Plot the results
    # plot_series(time_steps, defects, legend="rigid_constraint")  # Phi_r
    # plot_series(time_steps, defects_dot, legend="rigid_constraint_derivative")  # Phi_r_dot
    # plot_series(time_steps, all_lambdas, legend="lagrange_multipliers")  # lambda

    # animate the motion
    animate_natural_segment(time_steps, all_states, center_of_mass, t_final)
