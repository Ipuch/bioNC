"""
Tests for __repr__ and __str__ methods of core bionc classes.
"""
import numpy as np
import pytest


@pytest.mark.parametrize(
    "bionc_type",
    ["numpy", "casadi"],
)
def test_natural_segment_repr(bionc_type):
    if bionc_type == "casadi":
        from bionc.bionc_casadi import NaturalSegment
    else:
        from bionc.bionc_numpy import NaturalSegment

    seg = NaturalSegment(name="test_segment", length=0.4, mass=1.5, index=0)
    result = repr(seg)
    assert "NaturalSegment" in result
    assert "test_segment" in result
    assert "index=0" in result


@pytest.mark.parametrize(
    "bionc_type",
    ["numpy", "casadi"],
)
def test_natural_segment_str(bionc_type):
    if bionc_type == "casadi":
        from bionc.bionc_casadi import NaturalSegment
    else:
        from bionc.bionc_numpy import NaturalSegment

    seg = NaturalSegment(
        name="thigh",
        length=0.4,
        alpha=np.pi / 2,
        beta=np.pi / 2,
        gamma=np.pi / 2,
        mass=7.5,
        index=1,
    )
    result = str(seg)
    assert "NaturalSegment: thigh" in result
    assert "index: 1" in result
    assert "length: 0.4000 m" in result
    assert "mass: 7.5000 kg" in result
    assert "90.00°" in result  # alpha, beta, gamma should be 90 degrees


@pytest.mark.parametrize(
    "bionc_type",
    ["numpy", "casadi"],
)
def test_segment_natural_coordinates_repr(bionc_type):
    if bionc_type == "casadi":
        from bionc.bionc_casadi import SegmentNaturalCoordinates
    else:
        from bionc.bionc_numpy import SegmentNaturalCoordinates

    Q = SegmentNaturalCoordinates.from_components(
        u=[1, 0, 0], rp=[0, 0, 1], rd=[0, 0, 0], w=[0, 1, 0]
    )
    result = repr(Q)
    assert "SegmentNaturalCoordinates" in result


@pytest.mark.parametrize(
    "bionc_type",
    ["numpy", "casadi"],
)
def test_segment_natural_coordinates_str(bionc_type):
    if bionc_type == "casadi":
        from bionc.bionc_casadi import SegmentNaturalCoordinates
    else:
        from bionc.bionc_numpy import SegmentNaturalCoordinates

    Q = SegmentNaturalCoordinates.from_components(
        u=[1, 0, 0], rp=[0, 0, 1], rd=[0, 0, 0], w=[0, 1, 0]
    )
    result = str(Q)
    assert "SegmentNaturalCoordinates:" in result
    assert "u  =" in result
    assert "rp =" in result
    assert "rd =" in result
    assert "w  =" in result


@pytest.mark.parametrize(
    "bionc_type",
    ["numpy", "casadi"],
)
def test_natural_coordinates_repr(bionc_type):
    if bionc_type == "casadi":
        from bionc.bionc_casadi import SegmentNaturalCoordinates, NaturalCoordinates
    else:
        from bionc.bionc_numpy import SegmentNaturalCoordinates, NaturalCoordinates

    Q = SegmentNaturalCoordinates.from_components(
        u=[1, 0, 0], rp=[0, 0, 1], rd=[0, 0, 0], w=[0, 1, 0]
    )
    Q_full = NaturalCoordinates.from_qi((Q,))
    result = repr(Q_full)
    assert "NaturalCoordinates" in result
    assert "nb_segments=1" in result


@pytest.mark.parametrize(
    "bionc_type",
    ["numpy", "casadi"],
)
def test_natural_coordinates_str(bionc_type):
    if bionc_type == "casadi":
        from bionc.bionc_casadi import SegmentNaturalCoordinates, NaturalCoordinates
    else:
        from bionc.bionc_numpy import SegmentNaturalCoordinates, NaturalCoordinates

    Q = SegmentNaturalCoordinates.from_components(
        u=[1, 0, 0], rp=[0, 0, 1], rd=[0, 0, 0], w=[0, 1, 0]
    )
    Q_full = NaturalCoordinates.from_qi((Q,))
    result = str(Q_full)
    assert "NaturalCoordinates with 1 segment(s)" in result


@pytest.mark.parametrize(
    "bionc_type",
    ["numpy", "casadi"],
)
def test_segment_natural_velocities_repr_str(bionc_type):
    if bionc_type == "casadi":
        from bionc.bionc_casadi.natural_velocities import SegmentNaturalVelocities
    else:
        from bionc.bionc_numpy.natural_velocities import SegmentNaturalVelocities

    Qdot = SegmentNaturalVelocities.from_components(
        udot=[0.1, 0, 0], rpdot=[0, 0, 0.2], rddot=[0, 0, 0], wdot=[0, 0.1, 0]
    )
    assert "SegmentNaturalVelocities" in repr(Qdot)
    result = str(Qdot)
    assert "udot  =" in result
    assert "rpdot =" in result


@pytest.mark.parametrize(
    "bionc_type",
    ["numpy", "casadi"],
)
def test_segment_natural_accelerations_repr_str(bionc_type):
    if bionc_type == "casadi":
        from bionc.bionc_casadi.natural_accelerations import SegmentNaturalAccelerations
    else:
        from bionc.bionc_numpy.natural_accelerations import SegmentNaturalAccelerations

    Qddot = SegmentNaturalAccelerations.from_components(
        uddot=[0.01, 0, 0], rpddot=[0, 0, 0.02], rdddot=[0, 0, 0], wddot=[0, 0.01, 0]
    )
    assert "SegmentNaturalAccelerations" in repr(Qddot)
    result = str(Qddot)
    assert "uddot  =" in result
    assert "rpddot =" in result


@pytest.mark.parametrize(
    "bionc_type",
    ["numpy", "casadi"],
)
def test_biomechanical_model_repr(bionc_type):
    if bionc_type == "casadi":
        from bionc.bionc_casadi import NaturalSegment, BiomechanicalModel
    else:
        from bionc.bionc_numpy import NaturalSegment, BiomechanicalModel

    model = BiomechanicalModel()
    seg = NaturalSegment(name="seg1", length=0.4, mass=1.5)
    model["seg1"] = seg
    result = repr(model)
    assert "BiomechanicalModel" in result
    assert "nb_segments=1" in result


@pytest.mark.parametrize(
    "bionc_type",
    ["numpy", "casadi"],
)
def test_biomechanical_model_str(bionc_type):
    if bionc_type == "casadi":
        from bionc.bionc_casadi import NaturalSegment, BiomechanicalModel
    else:
        from bionc.bionc_numpy import NaturalSegment, BiomechanicalModel

    model = BiomechanicalModel()
    seg = NaturalSegment(name="seg1", length=0.4, mass=1.5)
    model["seg1"] = seg
    result = str(model)
    assert "BiomechanicalModel" in result
    assert "Segments: 1" in result
    assert "seg1" in result
    assert "Total mass: 1.5000 kg" in result
