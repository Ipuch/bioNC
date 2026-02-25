from typing import Callable

import numpy as np

from ..bionc_numpy import NaturalCoordinates, NaturalVelocities, BiomechanicalModel


def RK4(
    t: np.ndarray,
    f: Callable,
    y0: np.ndarray,
    normalize_idx: tuple[tuple[int, ...]] = None,
    args=(),
) -> np.ndarray:
    """
    Runge-Kutta 4th order method

    Parameters
    ----------
    t : array_like
        time steps
    f : Callable
        function to be integrated in the form f(t, y, *args)
    y0 : np.ndarray
        initial conditions of states
    normalize_idx : tuple(tuple)
        indices of states to be normalized together
    args : tuple
        additional arguments to be passed to the function f

    Returns
    -------
    y : array_like
        states for each time step

    """
    n = len(t)
    y = np.zeros((len(y0), n))
    y[:, 0] = y0
    for i in range(n - 1):
        h = t[i + 1] - t[i]
        yi = np.squeeze(y[:, i])
        k1 = f(t[i], yi, *args)
        k2 = f(t[i] + h / 2.0, yi + k1 * h / 2.0, *args)
        k3 = f(t[i] + h / 2.0, yi + k2 * h / 2.0, *args)
        k4 = f(t[i] + h, yi + k3 * h, *args)
        y[:, i + 1] = yi + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        # verify after each time step the normalization of the states
        if normalize_idx is not None:
            for idx in normalize_idx:
                y[idx, i + 1] = y[idx, i + 1] / np.linalg.norm(y[idx, i + 1])
    return y


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
