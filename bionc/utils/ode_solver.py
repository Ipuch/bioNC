from typing import Callable

import numpy as np


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
