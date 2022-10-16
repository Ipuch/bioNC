import numpy as np


def vnop_array(V: np.ndarray, e1: np.ndarray, e2: np.ndarray, e3: np.ndarray) -> np.ndarray:
    """
    This function converts a vector expressed in the global coordinate system
    to a vector expressed in a non-orthogonal coordinate system.

    Parameters
    ----------
    V: np.ndarray
        The vector expressed in the global coordinate system
    e1: np.ndarray
        The first vector of the non-orthogonal coordinate system, usually the u-axis
    e2: np.ndarray
        The second vector of the non-orthogonal coordinate system, usually the v-axis
    e3: np.ndarray
        The third vector of the non-orthogonal coordinate system, usually the w-axis

    Returns
    -------
    vnop: np.ndarray
        The vector expressed in the non-orthogonal coordinate system

    """

    if V.shape[0] != 3:
        raise ValueError("The vector must be expressed in 3D.")
    if len(V.shape) == 1:
        V = V[:, np.newaxis]

    if e1.shape[0] != 3:
        raise ValueError("The first vector of the non-orthogonal coordinate system must be expressed in 3D.")
    if e2.shape[0] != 3:
        raise ValueError("The second vector of the non-orthogonal coordinate system must be expressed in 3D.")
    if e3.shape[0] != 3:
        raise ValueError("The third vector of the non-orthogonal coordinate system must be expressed in 3D.")

    vnop = np.zeros(V.shape)

    vnop[0, :] = np.sum(np.cross(e2, e3, axis=0) * V, 0) / np.sum(np.cross(e1, e2, axis=0) * e3, 0)
    vnop[1, :] = np.sum(np.cross(e3, e1, axis=0) * V, 0) / np.sum(np.cross(e1, e2, axis=0) * e3, 0)
    vnop[2, :] = np.sum(np.cross(e1, e2, axis=0) * V, 0) / np.sum(np.cross(e1, e2, axis=0) * e3, 0)

    return vnop
