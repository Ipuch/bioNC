from .protocols import Data
from ..model_computations.segment import Segment
from .segment_template import SegmentTemplate
from ..model_computations.natural_segment import NaturalSegment
from ..model_computations.biomechanical_model import BiomechanicalModel


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

            scs = NaturalSegment()
            if s.segment_coordinate_system is not None:
                scs = s.segment_coordinate_system.to_scs(
                    data,
                    model,
                    model.segments[s.parent_name].segment_coordinate_system if s.parent_name else None,
                )

            inertia_parameters = None
            if s.inertia_parameters is not None:
                inertia_parameters = s.inertia_parameters.to_real(data, model, scs)

            mesh = None
            if s.mesh is not None:
                mesh = s.mesh.to_mesh(data, model, scs)

            model[s.name] = Segment(
                name=s.name,
                parent_name=s.parent_name,
                segment_coordinate_system=scs,
                translations=s.translations,
                rotations=s.rotations,
                inertia_parameters=inertia_parameters,
            )

            for marker in s.markers:
                model.segments[name].add_marker(marker.to_marker(data, model, scs))

        return model
