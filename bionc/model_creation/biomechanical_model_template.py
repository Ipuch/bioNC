from .protocols import Data

from .segment_template import SegmentTemplate
from ..bionc_numpy.biomechanical_model import BiomechanicalModel
from ..bionc_numpy.enums import JointType
from ..utils.enums import NaturalAxis

class BiomechanicalModelTemplate:
    def __init__(self):
        self.segments = {}
        self.joints = {}

    def __getitem__(self, name: str):
        return self.segments[name]

    def __setitem__(self, name: str, segment: SegmentTemplate):
        if segment.name is not None and segment.name != name:
            raise ValueError(
                "The segment name should be the same as the 'key'. Alternatively, segment.name can be left undefined"
            )
        segment.name = name  # Make sure the name of the segment fits the internal one
        self.segments[name] = segment

    def add_joint(self,
                  name: str,
                  joint_type: JointType,
                  parent: str,
                  child: str,
                  parent_axis : NaturalAxis | tuple[NaturalAxis] | list[NaturalAxis] = None,
                  child_axis : NaturalAxis | tuple[NaturalAxis] | list[NaturalAxis] = None,
                  theta: float | tuple[float] | list[float] = None,
                  ):
        """
        This method adds a joint to the model

        Parameters
        ----------
        name: str
            The name of the joint
        joint_type : JointType
            The joint to add
        parent : str
            The name of the parent segment
        child : str
            The name of the child segment
        parent_axis : NaturalAxis | tuple[NaturalAxis] | list[NaturalAxis]
            The axis of the parent segment, zero, one or two element but not more.
        child_axis : NaturalAxis | tuple[NaturalAxis] | list[NaturalAxis]
            The axis of the child segment, zero, one or two element but not more.
        theta : float | tuple[float] | list[float]
            The angle of axis constraints, zero, one or two element but not more.

        Returns
        -------
        None
        """
        if name is None:
            name = f"{parent}_{child}"
        self.joints[name] = dict(
            name=name,
            joint_type=joint_type,
            parent=parent,
            child=child,
            parent_axis=parent_axis,
            child_axis=child_axis,
            theta=theta,
        )

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
                model.segments[name].add_natural_marker(marker.to_natural_marker(data, model, Q_xp))

        for key, joint in self.joints.items():
            model._add_joint(joint)

        return model
