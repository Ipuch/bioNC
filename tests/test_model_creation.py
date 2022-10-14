import numpy as np

from bionc import (
    AxisTemplate,
    SegmentTemplate,
    NaturalSegmentTemplate,
    MarkerTemplate,
    BiomechanicalModelTemplate,
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
