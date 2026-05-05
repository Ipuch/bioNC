import numpy as np
import pytest

from .utils import TestUtils


def test_pendulum_with_muscle_example_runs():
    bionc = TestUtils.bionc_folder()
    module = TestUtils.load_module(bionc + "/examples/muscle/pendulum_with_muscle.py")
    result = module.main()

    assert result["model"].nb_muscles == 1
    assert result["model"].muscle_names == ["muscle1"]

    expected_length = np.sqrt(0.38)
    u_in = np.array([-0.3, -0.5, -0.2]) / expected_length
    expected_ma = np.zeros(12)
    expected_ma[3:6] = -0.5 * u_in
    expected_ma[6:9] = -0.5 * u_in

    np.testing.assert_almost_equal(float(result["length_numpy"].squeeze()), expected_length, decimal=10)
    np.testing.assert_almost_equal(float(result["length_casadi"]), expected_length, decimal=10)
    np.testing.assert_almost_equal(result["moment_arm_numpy"].squeeze(), expected_ma, decimal=10)
    np.testing.assert_almost_equal(result["moment_arm_casadi"], expected_ma, decimal=10)


def _build_pendulum_with_muscle(bionc_type: str):
    if bionc_type == "casadi":
        from bionc.bionc_casadi import (
            BiomechanicalModel,
            Muscle,
            MuscleViaPoint,
            NaturalSegment,
            SegmentNaturalCoordinates,
            NaturalCoordinates,
        )
        from bionc.bionc_casadi.enums import JointType
    else:
        from bionc.bionc_numpy import (
            BiomechanicalModel,
            Muscle,
            MuscleViaPoint,
            NaturalSegment,
            SegmentNaturalCoordinates,
            NaturalCoordinates,
        )
        from bionc.bionc_numpy.enums import JointType
    from bionc import CartesianAxis, NaturalAxis, TransformationMatrixType

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

    origin = MuscleViaPoint(name="origin", parent_name="GROUND", position=np.array([0.3, 0.0, 0.2]))
    insertion = MuscleViaPoint(name="insertion", parent_name="pendulum", position=(0.0, -0.5, 0.0))
    model.add_muscle(Muscle(name="muscle1", via_points=[origin, insertion]))

    Qi = SegmentNaturalCoordinates.from_components(
        u=[1, 0, 0], rp=[0, 0, 0], rd=[0, -1, 0], w=[0, 0, 1]
    )
    Q = NaturalCoordinates(Qi)
    return model, Q


# Pendulum hanging vertically: insertion at v=-0.5 sits at (0, -0.5, 0).
# Origin = (0.3, 0, 0.2). Length = sqrt(0.09 + 0.25 + 0.04) = sqrt(0.38).
EXPECTED_LENGTH = np.sqrt(0.38)

# u = (insertion - origin) / ||...|| = (-0.3, -0.5, -0.2) / sqrt(0.38)
# Insertion's interpolation matrix N at (0, -0.5, 0) selects 0.5*rp + 0.5*rd.
# u_at_insertion = +u_incoming (no outgoing). Contribution N^T · u puts +0.5*u into rp and +0.5*u into rd.
# moment_arm = -dL/dQ.
_u_in = np.array([-0.3, -0.5, -0.2]) / EXPECTED_LENGTH
EXPECTED_MOMENT_ARM = np.zeros(12)
EXPECTED_MOMENT_ARM[3:6] = -0.5 * _u_in
EXPECTED_MOMENT_ARM[6:9] = -0.5 * _u_in


@pytest.mark.parametrize("bionc_type", ["numpy", "casadi"])
def test_muscle_length_and_moment_arm(bionc_type):
    model, Q = _build_pendulum_with_muscle(bionc_type)

    assert model.nb_muscles == 1
    assert model.muscle_names == ["muscle1"]

    if bionc_type == "casadi":
        from casadi import Function
        from bionc.bionc_casadi import NaturalCoordinates as MXNaturalCoordinates

        Q_num = np.array([1, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 1], dtype=float)
        Q_sym = MXNaturalCoordinates.sym(model.nb_segments)
        L = float(np.array(Function("L", [Q_sym], [model.muscle_lengths(Q_sym)])(Q_num)).squeeze())
        MA = np.array(Function("MA", [Q_sym], [model.muscle_moment_arms(Q_sym)])(Q_num)).squeeze()
    else:
        L = float(model.muscle_lengths(Q).squeeze())
        MA = model.muscle_moment_arms(Q).squeeze()

    np.testing.assert_almost_equal(L, EXPECTED_LENGTH, decimal=10)
    np.testing.assert_almost_equal(MA, EXPECTED_MOMENT_ARM, decimal=10)


def test_numpy_and_casadi_agree():
    model_np, Q_np = _build_pendulum_with_muscle("numpy")
    model_mx, Q_mx = _build_pendulum_with_muscle("casadi")

    from casadi import Function
    from bionc.bionc_casadi import NaturalCoordinates as MXNaturalCoordinates

    Q_num = np.array([1, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 1], dtype=float)
    Q_sym = MXNaturalCoordinates.sym(model_mx.nb_segments)
    L_mx = float(np.array(Function("L", [Q_sym], [model_mx.muscle_lengths(Q_sym)])(Q_num)).squeeze())
    MA_mx = np.array(
        Function("MA", [Q_sym], [model_mx.muscle_moment_arms(Q_sym)])(Q_num)
    ).squeeze()

    L_np = float(model_np.muscle_lengths(Q_np).squeeze())
    MA_np = model_np.muscle_moment_arms(Q_np).squeeze()

    np.testing.assert_almost_equal(L_np, L_mx, decimal=10)
    np.testing.assert_almost_equal(MA_np, MA_mx, decimal=10)


def test_muscle_to_mx_via_model():
    """to_mx() on the BiomechanicalModel should propagate muscles."""
    model_np, _ = _build_pendulum_with_muscle("numpy")
    model_mx = model_np.to_mx()
    assert model_mx.nb_muscles == 1
    assert model_mx.muscle_names == ["muscle1"]


def test_muscle_requires_two_via_points():
    from bionc.bionc_numpy import Muscle, MuscleViaPoint

    with pytest.raises(ValueError, match="at least two via points"):
        Muscle(name="bad", via_points=[MuscleViaPoint("a", "GROUND", np.zeros(3))])


def test_finite_difference_matches_moment_arm():
    """Numerically validate moment_arm = -dL/dQ via central finite differences."""
    model, Q = _build_pendulum_with_muscle("numpy")
    from bionc.bionc_numpy import NaturalCoordinates

    Q_arr = np.array(Q.to_array(), dtype=float).reshape(-1)
    eps = 1e-6
    fd = np.zeros_like(Q_arr)
    for i in range(Q_arr.size):
        dq = np.zeros_like(Q_arr)
        dq[i] = eps
        L_plus = model.muscle_lengths(NaturalCoordinates(Q_arr + dq))[0]
        L_minus = model.muscle_lengths(NaturalCoordinates(Q_arr - dq))[0]
        fd[i] = -(L_plus - L_minus) / (2 * eps)

    MA = model.muscle_moment_arms(Q).squeeze()
    np.testing.assert_allclose(MA, fd, atol=1e-7)
