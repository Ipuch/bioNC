import numpy as np
import pytest

pyorerun = pytest.importorskip("pyorerun")

from bionc import (
    BiomechanicalModel,
    CartesianAxis,
    JointType,
    NaturalAxis,
    NaturalCoordinates,
    NaturalSegment,
    SegmentNaturalCoordinates,
    TransformationMatrixType,
)
from bionc.vizualization.animations import NaturalVectorColors
from bionc.vizualization.pyorerun_natural_vectors import add_natural_vectors


def _hanging_pendulum_Q(nb_frames: int = 3) -> tuple[BiomechanicalModel, np.ndarray]:
    model = BiomechanicalModel()
    model["pendulum"] = NaturalSegment.with_cartesian_inertial_parameters(
        name="pendulum",
        alpha=np.pi / 2,
        beta=np.pi / 2,
        gamma=np.pi / 2,
        length=1.0,
        mass=1.0,
        center_of_mass=np.array([0.0, -0.5, 0.0]),
        inertia=np.diag([0.05, 0.05, 0.05]),
        inertial_transformation_matrix=TransformationMatrixType.Buv,
    )
    model._add_joint(
        dict(
            name="hinge",
            joint_type=JointType.GROUND_REVOLUTE,
            parent="GROUND",
            child="pendulum",
            parent_axis=[CartesianAxis.X, CartesianAxis.X],
            child_axis=[NaturalAxis.V, NaturalAxis.W],
            theta=[np.pi / 2, np.pi / 2],
        )
    )
    Qi = SegmentNaturalCoordinates.from_components(u=[1, 0, 0], rp=[0, 0, 0], rd=[0, -1, 0], w=[0, 0, 1])
    Q = np.asarray(NaturalCoordinates(Qi)).reshape(-1, 1)
    Q_traj = np.repeat(Q, nb_frames, axis=1)
    return model, Q_traj


def test_add_natural_vectors_registers_three_per_segment():
    model, Q = _hanging_pendulum_Q(nb_frames=4)
    prr = pyorerun.PhaseRerun(t_span=np.linspace(0.0, 1.0, Q.shape[1]))
    add_natural_vectors(prr, model, Q)

    names = prr.xp_data.component_names
    assert len(names) == 3 * model.nb_segments
    for axis in ("pendulum_u", "pendulum_v", "pendulum_w"):
        assert any(axis in n for n in names), names


def test_add_natural_vectors_origins_and_endpoints():
    """Origins must equal rp; endpoints follow rp + scale * axis."""
    model, Q = _hanging_pendulum_Q(nb_frames=2)
    scale = dict(scale_u=0.3, scale_v=0.7, scale_w=0.4)
    prr = pyorerun.PhaseRerun(t_span=np.linspace(0.0, 1.0, Q.shape[1]))
    add_natural_vectors(prr, model, Q, **scale)

    rp = Q[3:6, :]
    u = Q[0:3, :]
    rd = Q[6:9, :]
    w = Q[9:12, :]
    expected_magnitudes = {
        "pendulum_u": scale["scale_u"] * u,
        "pendulum_v": scale["scale_v"] * (rd - rp),
        "pendulum_w": scale["scale_w"] * w,
    }

    by_axis = {}
    for vec in prr.xp_data.xp_data:
        for axis_name in expected_magnitudes:
            if axis_name in vec.name:
                by_axis[axis_name] = vec
                break
    assert set(by_axis) == set(expected_magnitudes)

    for axis_name, vec in by_axis.items():
        np.testing.assert_allclose(vec.vector_origins, rp)
        np.testing.assert_allclose(vec.vector_magnitude, expected_magnitudes[axis_name])


def test_add_natural_vectors_uses_natural_vector_colors():
    """u/v/w arrows must carry the colors defined in NaturalVectorColors."""
    from pyorerun.abstract.markers import rgb255_to_hex_rgba

    model, Q = _hanging_pendulum_Q()
    prr = pyorerun.PhaseRerun(t_span=np.linspace(0.0, 1.0, Q.shape[1]))
    add_natural_vectors(prr, model, Q)

    expected = {
        "pendulum_u": rgb255_to_hex_rgba(np.asarray(NaturalVectorColors.U.value)),
        "pendulum_v": rgb255_to_hex_rgba(np.asarray(NaturalVectorColors.V.value)),
        "pendulum_w": rgb255_to_hex_rgba(np.asarray(NaturalVectorColors.W.value)),
    }
    for vec in prr.xp_data.xp_data:
        for axis_name, color in expected.items():
            if axis_name in vec.name:
                assert vec._color == color, f"{axis_name} got {vec._color} expected {color}"


def test_add_natural_vectors_rejects_wrong_Q_shape():
    model, _ = _hanging_pendulum_Q()
    prr = pyorerun.PhaseRerun(t_span=np.array([0.0, 1.0]))

    with pytest.raises(ValueError, match="must have shape"):
        add_natural_vectors(prr, model, np.zeros((11, 2)))
    with pytest.raises(ValueError, match="must have shape"):
        add_natural_vectors(prr, model, np.zeros(12))


def test_add_natural_vectors_to_chunk_runs():
    """to_chunk() should serialize without raising for our colored vectors."""
    model, Q = _hanging_pendulum_Q(nb_frames=5)
    prr = pyorerun.PhaseRerun(t_span=np.linspace(0.0, 1.0, Q.shape[1]))
    add_natural_vectors(prr, model, Q)
    chunks = prr.xp_data.to_chunk()
    assert len(chunks) == 3 * model.nb_segments
