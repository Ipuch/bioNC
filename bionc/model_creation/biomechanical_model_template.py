from .protocols import Data

from .segment_template import SegmentTemplate
from bionc.bionc_numpy.biomechanical_model import BiomechanicalModel


class BiomechanicalModelTemplate:
    def __init__(self):
        self.segments = {}

    def __getitem__(self, name: str):
        return self.segments[name]

    def __setitem__(self, name: str, segment: SegmentTemplate):
        if segment.name is not None and segment.name != name:
            raise ValueError(
                "The segment name should be the same as the 'key'. Alternatively, segment.name can be left undefined"
            )
        segment.name = name  # Make sure the name of the segment fits the internal one
        self.segments[name] = segment

    def update(self, data: Data) -> BiomechanicalModel:
        """
        Collapse the model to an actual personalized biomechanical model based on the generic model and the data
        file (usually a static trial)

        Parameters
        ----------
        data
            The data to collapse the model from
        """
        model = BiomechanicalModel()
        for name in self.segments:
            s = self.segments[name]

            Q_xp = s.natural_segment.experimental_Q(data, model)

            natural_segment = s.natural_segment.update()
            natural_segment.set_name(name)

            # inertia_parameters = None
            # if s.inertia_parameters is not None:
            # todo: this is not working yet
            # natural_segment.set_inertia_parameters(s.inertia_parameters)

            model[s.name] = natural_segment

            for marker in s.markers:
                model.segments[name].add_marker(marker.to_segment_marker(data, model, Q_xp))

        return model
