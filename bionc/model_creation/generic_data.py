import numpy as np


class GenericData:
    """
    Implementation of the `Data` protocol from model_creation

    Attributes
    ----------
    values : dict[str, np.ndarray]
        The values of the markers
    """

    def __init__(self, markers: np.ndarray, markers_names: tuple[str, ...]):

        if markers.shape[0] == 3:
            markers = np.concatenate((markers, np.ones((1, markers.shape[1], markers.shape[2]))), axis=0)

        self.values = {}
        for i, marker_name in enumerate(markers_names):
            self.values[marker_name] = markers[:, i, :]
