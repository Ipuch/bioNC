import numpy as np

from .natural_marker import NaturalMarker


class Axis:
    def __init__(self, start: NaturalMarker, end: NaturalMarker):
        """
        Parameters
        ----------
        start:
            The initial NaturalMarker
        end:
            The final NaturalMarker
        """
        self.start_point = start
        self.end_point = end

    def axis(self) -> np.ndarray:
        """
        Returns the axis vector
        """
        start = self.start_point.position
        end = self.end_point.position
        return end - start
