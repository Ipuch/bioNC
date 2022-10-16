import numpy as np

from .natural_marker import Marker


class Axis:
    def __init__(self, start: Marker, end: Marker):
        """
        Parameters
        ----------
        start:
            The initial SegmentMarker
        end:
            The final SegmentMarker
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
