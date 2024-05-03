import numpy as np
from pyomeca import Markers


def validate_numpy_markers(experimental_markers: np.ndarray) -> np.ndarray:
    if experimental_markers.shape[0] != 3 or experimental_markers.shape[1] < 1 or len(experimental_markers.shape) < 3:
        raise ValueError(f"experimental_markers must be a 3xNxM numpy array. Got {experimental_markers.shape} instead.")
    return experimental_markers


def load_and_validate_markers(experimental_markers: str | np.ndarray) -> np.ndarray:
    """
    Process the experimental markers input.

    Parameters
    ----------
    experimental_markers : str or np.ndarray
        The experimental markers (3xNxM numpy array), or a path to a c3d file.

    Returns
    -------
    np.ndarray
        The checked experimental markers (3xNxM numpy array)
    """
    if isinstance(experimental_markers, str):
        return Markers.from_c3d(experimental_markers).to_numpy()
    elif isinstance(experimental_markers, np.ndarray):
        return validate_numpy_markers(experimental_markers)
    else:
        raise TypeError(
            f"experimental_markers must be a path as a string or a numpy array of size 3xNxM. Got {type(experimental_markers)} instead."
        )
