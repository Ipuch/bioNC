import os

import numpy as np

from bionc import (
    AxisTemplate,
    SegmentTemplate,
    NaturalSegmentTemplate,
    MarkerTemplate,
    BiomechanicalModelTemplate,
    NaturalSegment,
)


def test_segment_template():
    model = BiomechanicalModelTemplate()
    right_ankle_joint = lambda m, bio: MarkerTemplate.middle_of(m, bio, "RANE", "RANI")

    model["FOOT"] = SegmentTemplate(
        natural_segment=NaturalSegmentTemplate(
            u_axis=AxisTemplate(
                start=right_ankle_joint,
                # u_axis is defined from calcaneous (CAL) to the middle of M1 and M5
                end=lambda m, bio: (m["RHEE"] - (m["RTARI"] + m["RTAR"]) / 2)
                / np.linalg.norm(m["RHEE"] - (m["RTARI"] + m["RTAR"]) / 2),
            ),
            proximal_point=right_ankle_joint,
            #  middle of M1 and M5
            distal_point=lambda m, bio: MarkerTemplate.middle_of(m, bio, "RTARI", "RTAR"),
            w_axis=AxisTemplate(start="RTARI", end="RTAR"),
        )
    )

    # verify stuff in model
    assert model["FOOT"].name == "FOOT"

    assert model["FOOT"].natural_segment.distal_point.name is None
    assert model["FOOT"].natural_segment.distal_point.is_anatomical == False
    assert model["FOOT"].natural_segment.distal_point.is_technical == True
    assert model["FOOT"].natural_segment.distal_point.marker_type == "Marker"

    assert model["FOOT"].natural_segment.proximal_point.name is None
    assert model["FOOT"].natural_segment.proximal_point.is_anatomical == False
    assert model["FOOT"].natural_segment.proximal_point.is_technical == True
    assert model["FOOT"].natural_segment.proximal_point.marker_type == "Marker"

    assert model["FOOT"].natural_segment.u_axis.end.name is None
    assert model["FOOT"].natural_segment.u_axis.end.is_anatomical == False
    assert model["FOOT"].natural_segment.u_axis.end.is_technical == True
    assert model["FOOT"].natural_segment.u_axis.end.marker_type == "Marker"

    assert model["FOOT"].natural_segment.u_axis.start.name is None
    assert model["FOOT"].natural_segment.u_axis.start.is_anatomical == False
    assert model["FOOT"].natural_segment.u_axis.start.is_technical == True
    assert model["FOOT"].natural_segment.u_axis.start.marker_type == "Marker"

    assert model["FOOT"].natural_segment.w_axis.end.name is None
    assert model["FOOT"].natural_segment.w_axis.end.is_anatomical == False
    assert model["FOOT"].natural_segment.w_axis.end.is_technical == True
    assert model["FOOT"].natural_segment.w_axis.end.marker_type == "Marker"

    assert model["FOOT"].natural_segment.w_axis.start.name is None
    assert model["FOOT"].natural_segment.w_axis.start.is_anatomical == False
    assert model["FOOT"].natural_segment.w_axis.start.is_technical == True
    assert model["FOOT"].natural_segment.w_axis.start.marker_type == "Marker"

    assert model["FOOT"].inertia_parameters is None
    assert model["FOOT"].markers == []


def test_model_creation():
    from examples.model_creation import model_creation_from_measured_data, generate_c3d_file

    # create a c3d file with data
    filename = generate_c3d_file()
    # Create the model from a c3d file
    model = model_creation_from_measured_data(filename)

    assert isinstance(model.segments, dict)
    assert len(model.segments) == 3
    assert model.segments["FOOT"].name == "FOOT"
    assert model.segments["SHANK"].name == "SHANK"
    assert model.segments["THIGH"].name == "THIGH"

    prop = dict(
        FOOT=dict(
            length=0.13890177648456653, gamma=0.8567148572956443, beta=1.7556031976569564, alpha=1.591624071977332
        ),
        SHANK=dict(
            length=0.3531381284447682, gamma=1.7873340587846938, beta=1.4399616784423517, alpha=1.54503182793888
        ),
        THIGH=dict(
            length=0.39649708859834826, gamma=2.3049927484597585, beta=1.7463313954337523, alpha=1.5377930856781998
        ),
    )

    # verify segments values are NaturalSegment objects
    for s, key in zip(model.segments.values(), model.segments.keys()):
        assert isinstance(s, NaturalSegment)

        np.testing.assert_almost_equal(s.length, prop[key]["length"])
        np.testing.assert_almost_equal(s.gamma, prop[key]["gamma"])
        np.testing.assert_almost_equal(s.beta, prop[key]["beta"])
        np.testing.assert_almost_equal(s.alpha, prop[key]["alpha"])

    # todo: test the markers and global matrices of the model
    # rigidbody constraints
    # joint constraints
    # markers constraints

    # remove the c3d file
    os.remove(filename)
