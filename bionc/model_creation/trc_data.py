import numpy as np
import trc


class TRCData:
    """
    Implementation of TRCdata following the `Data` protocol from model_creation
    """

    def __init__(self, trc_path, first_frame: int = 0, last_frame: int = None):
        self.trc_data = trc.TRCData()
        self.trc_data.load(trc_path)

        self.first_frame = first_frame
        self.last_frame = last_frame if last_frame is not None else self.trc_data["Frame#"][-1]

        self.values = {}
        for marker_name in self.trc_data["Markers"]:
            self.values[marker_name] = self._get_position((marker_name,)).squeeze()

    def mean_marker_positions(self, marker_names: tuple[str, ...]) -> np.ndarray:
        return np.mean(np.nanmean(self._get_position(marker_names), axis=2), axis=1)

    def _indices_in_trc(self, from_markers: tuple[str, ...]) -> tuple[int, ...]:
        return tuple(self.trc_data["Markers"].index(n) for n in from_markers)

    def _get_position(self, marker_names: tuple[str, ...]):
        nb_frames = self.last_frame - self.first_frame + 1
        positions = np.zeros((4, len(marker_names), nb_frames ))
        for frame in range(self.first_frame, self.last_frame + 1):
            positions[:3, :, frame] = np.array(self.trc_data[frame][1]).T[:,  self._indices_in_trc(marker_names)]
        positions[-1, :, :] = 1

        return self._to_meter(positions)

    def _to_meter(self, data: np.array) -> np.array:
        units = self.trc_data["Units"]
        factor = 1000 if units == "mm" else 1
        data /= factor
        data[-1, :, :] = 1

        return data
