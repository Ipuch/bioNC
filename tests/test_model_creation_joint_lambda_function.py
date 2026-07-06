from types import SimpleNamespace

import numpy as np

from bionc.model_creation.biomechanical_model_template import BiomechanicalModelTemplate


class FakeData:
    def __init__(self):
        self.values = {"marker": np.ones((4, 1))}


class FakeMarkerTemplate:
    def __init__(self, name):
        self.name = name

    def to_natural_marker(self, data, model, Q_xp):
        return {
            "name": self.name,
            "Q_xp": np.asarray(Q_xp).copy(),
            "segments": tuple(model.segments.keys()),
        }


class FakeUpdatedNaturalSegment:
    def __init__(self):
        self.name = None
        self.experimental_Q_function = None
        self.added_markers = []

    def set_name(self, name):
        self.name = name

    def set_experimental_Q_function(self, experimental_Q_function):
        self.experimental_Q_function = experimental_Q_function

    def add_natural_marker(self, marker):
        self.added_markers.append(marker)


class FakeNaturalSegmentTemplate:
    def __init__(self, q_xp):
        self.q_xp = np.asarray(q_xp)

    def experimental_Q(self, data, model):
        return self.q_xp

    def update(self):
        return FakeUpdatedNaturalSegment()


class FakeSegmentTemplate:
    def __init__(self, name, natural_segment, markers):
        self.name = name
        self.natural_segment = natural_segment
        self.markers = markers


class FakeBiomechanicalModel:
    def __init__(self):
        self.segments = {}
        self.added_joints = []

    def __setitem__(self, key, segment):
        self.segments[key] = segment

    def _add_joint(self, joint):
        self.added_joints.append(joint)


def test_update_resolves_callable_joint_fields_and_keeps_segment_data(monkeypatch):
    import bionc.bionc_numpy.biomechanical_model as biomechanical_model_module

    monkeypatch.setattr(biomechanical_model_module, "BiomechanicalModel", FakeBiomechanicalModel)

    template = BiomechanicalModelTemplate()
    template.segments = {
        "SEGMENT": FakeSegmentTemplate(
            name="SEGMENT",
            natural_segment=FakeNaturalSegmentTemplate(q_xp=[1.0, 2.0, 3.0]),
            markers=[FakeMarkerTemplate("M0")],
        )
    }
    template.joints = {
        "JOINT": {
            "name": "JOINT",
            "joint_type": object(),
            "parent": "GROUND",
            "child": "SEGMENT",
            "theta": lambda data, model: np.array([1.0, 3.0, 5.0]),
            "length": lambda data, model: [2.0, 4.0, 6.0],
        }
    }

    data = FakeData()
    model = template.update(data)

    assert isinstance(model, FakeBiomechanicalModel)
    assert list(model.segments) == ["SEGMENT"]

    updated_segment = model.segments["SEGMENT"]
    assert updated_segment.name == "SEGMENT"
    assert updated_segment.experimental_Q_function.__self__ is template.segments["SEGMENT"].natural_segment
    assert updated_segment.experimental_Q_function.__func__ is template.segments["SEGMENT"].natural_segment.experimental_Q.__func__
    assert len(updated_segment.added_markers) == 1
    assert updated_segment.added_markers[0]["name"] == "M0"
    np.testing.assert_array_equal(updated_segment.added_markers[0]["Q_xp"], np.array([1.0, 2.0, 3.0]))
    assert updated_segment.added_markers[0]["segments"] == ("SEGMENT",)

    assert len(model.added_joints) == 1
    resolved_joint = model.added_joints[0]
    assert resolved_joint["theta"] == 3.0
    assert resolved_joint["length"] == 4.0
    assert resolved_joint["name"] == "JOINT"
    assert resolved_joint["parent"] == "GROUND"
    assert resolved_joint["child"] == "SEGMENT"


def test_resolve_joint_callables_reduces_callable_values_to_means():
    template = BiomechanicalModelTemplate()
    joint = {
        "name": "JOINT",
        "joint_type": SimpleNamespace(),
        "parent": "GROUND",
        "child": "SEGMENT",
        "theta": lambda data, model: np.array([2.0, 4.0, 6.0]),
        "length": lambda data, model: [1.0, 5.0],
        "projection_basis": None,
    }

    resolved = template._resolve_joint_callables(joint, SimpleNamespace(values={}), SimpleNamespace())

    assert resolved["theta"] == 4.0
    assert resolved["length"] == 3.0
    assert resolved["name"] == "JOINT"
    assert resolved["parent"] == "GROUND"
    assert resolved["child"] == "SEGMENT"