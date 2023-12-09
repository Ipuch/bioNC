from casadi import MX, norm_1

from ..natural_marker import Marker


class Axis:
    def __init__(self, start: Marker, end: Marker):
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

    def axis(self) -> MX:
        """
        Returns the axis vector
        """
        start = self.start_point.position
        end = self.end_point.position
        return (end - start) / norm_1(end[:3] - start[:3])
