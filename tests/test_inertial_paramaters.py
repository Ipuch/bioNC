import numpy as np
import pytest

from bionc import TransformationMatrixType

from .utils import TestUtils


@pytest.mark.parametrize(
    "bionc_type",
    [
        "numpy",
        # "casadi"
    ],
)
def test_init_with_valid_parameters(bionc_type):
    if bionc_type == "numpy":
        from bionc import NaturalInertialParameters

    mass = 5.0
    natural_center_of_mass = np.array([[1.0], [2.0], [3.0]])
    natural_pseudo_inertia = np.eye(3)

    obj = NaturalInertialParameters(mass, natural_center_of_mass, natural_pseudo_inertia)

    TestUtils.assert_equal(obj.mass, mass)
    TestUtils.assert_equal(obj.natural_center_of_mass, natural_center_of_mass)
    TestUtils.assert_equal(obj.natural_pseudo_inertia, natural_pseudo_inertia)


@pytest.mark.parametrize(
    "bionc_type",
    [
        "numpy",
        # "casadi"
    ],
)
def test_init_with_valid_parameters(bionc_type):
    if bionc_type == "numpy":
        from bionc import NaturalInertialParameters

    mass = 5.0
    wrong_shape_center_of_mass = np.array([1.0, 2.0])
    wrong_shape_pseudo_inertia = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    with pytest.raises(ValueError, match="Center of mass must be 3x1"):
        NaturalInertialParameters(mass, wrong_shape_center_of_mass, np.eye(3))

    with pytest.raises(ValueError, match="Pseudo inertia matrix must be 3x3"):
        NaturalInertialParameters(mass, np.array([[1.0], [2.0], [3.0]]), wrong_shape_pseudo_inertia)


@pytest.mark.parametrize(
    "bionc_type",
    [
        "numpy",
        # "casadi"
    ],
)
def test_mass_property(bionc_type):
    if bionc_type == "numpy":
        from bionc import NaturalInertialParameters

    mass = 7.0
    obj = NaturalInertialParameters(mass, np.array([[1.0], [2.0], [3.0]]), np.eye(3))
    TestUtils.assert_equal(obj.mass, mass)


@pytest.mark.parametrize(
    "bionc_type",
    [
        "numpy",
        # "casadi"
    ],
)
def test_natural_center_of_mass_property(bionc_type):
    if bionc_type == "numpy":
        from bionc import NaturalInertialParameters

    center = np.array([[1.0], [2.0], [3.0]])
    obj = NaturalInertialParameters(5.0, center, np.eye(3))

    TestUtils.assert_equal(obj.natural_center_of_mass, center)


@pytest.mark.parametrize(
    "bionc_type",
    [
        "numpy",
        # "casadi"
    ],
)
def test_pseudo_inertia_matrix_property(bionc_type):
    if bionc_type == "numpy":
        from bionc import NaturalInertialParameters

    inertia = np.ones((3, 3))
    obj = NaturalInertialParameters(5.0, np.array([[1.0], [2.0], [3.0]]), inertia)

    TestUtils.assert_equal(obj.natural_pseudo_inertia, inertia)
    assert obj._initial_transformation_matrix is None


@pytest.mark.parametrize(
    "bionc_type",
    [
        "numpy",
        # "casadi"
    ],
)
def test_from_cartesian_inertial_parameters(bionc_type):
    if bionc_type == "numpy":
        from bionc import (
            NaturalInertialParameters,
            compute_transformation_matrix,
        )

    mass = 5.0
    center_of_mass = np.array([[1.0], [2.0], [3.0]])
    inertia = np.eye(3)
    transformation_matrix = compute_transformation_matrix(
        TransformationMatrixType.Buv,
        1.0,
        np.pi / 2 + 0.1,
        np.pi / 2 - 0.05,
        np.pi / 2 + 0.01,
    )

    obj = NaturalInertialParameters.from_cartesian_inertial_parameters(
        mass, center_of_mass, inertia, transformation_matrix
    )
    TestUtils.assert_equal(obj.mass, mass)
    TestUtils.assert_equal(obj.natural_center_of_mass, np.array([[0.87212626], [2.29999071], [3.01872294]]))
    TestUtils.assert_equal(
        obj.natural_pseudo_inertia,
        np.array(
            [
                [67.37658449, -9.66919423, -16.68507356],
                [-9.66919423, 45.26679692, -27.5734654],
                [-16.68507356, -27.5734654, 26.32554364],
            ]
        ),
    )

    TestUtils.assert_equal(obj._initial_transformation_matrix, transformation_matrix)

    TestUtils.assert_equal(obj.center_of_mass(), np.array([[1.0], [2.0], [3.0]]))
    TestUtils.assert_equal(obj.inertia(), np.eye(3))

    transformation_matrix_2 = compute_transformation_matrix(
        TransformationMatrixType.Buv,
        1.1,
        np.pi / 2 + 0.11,
        np.pi / 2 - 0.051,
        np.pi / 2 + 0.011,
    )

    TestUtils.assert_equal(
        obj.center_of_mass(transformation_matrix_2), np.array([[0.99818507], [2.20011923], [2.9967137]])
    )
    TestUtils.assert_equal(
        obj.inertia(transformation_matrix_2),
        np.array(
            [
                [-3.08559735, -0.16400776, 0.05638617],
                [-0.16400776, 11.82224066, 0.00336482],
                [0.05638617, 0.00336482, -3.24142165],
            ]
        ),
    )


def _huygens_test_data():
    mass = 2.5
    length = 0.4
    center_of_mass = np.array([0.1, -0.3, 0.05])
    inertia = np.array(
        [
            [0.30, 0.05, -0.02],
            [0.05, 0.25, 0.04],
            [-0.02, 0.04, 0.20],
        ]
    )
    return mass, length, center_of_mass, inertia


def _to_numpy(value):
    from casadi import MX, evalf

    if isinstance(value, MX):
        return np.array(evalf(value).full())
    return np.array(value)


def _build_orthogonal_segment(bionc_type):
    if bionc_type == "numpy":
        from bionc import NaturalSegment
    else:
        from bionc.bionc_casadi import NaturalSegment

    mass, length, center_of_mass, inertia = _huygens_test_data()
    return NaturalSegment.with_cartesian_inertial_parameters(
        name="huygens",
        alpha=np.pi / 2,
        beta=np.pi / 2,
        gamma=np.pi / 2,
        length=length,
        mass=mass,
        center_of_mass=center_of_mass[:, np.newaxis],
        inertia=inertia,
        inertial_transformation_matrix=TransformationMatrixType.Buv,
    )


@pytest.mark.parametrize("bionc_type", ["numpy", "casadi"])
def test_pseudo_inertia_huygens_formula(bionc_type):
    """Orthogonal segment: J = inv(B) (I_C + m ((c.c) E - c c^T)) inv(B)^T with B = diag(1, L, 1)."""
    mass, length, c, inertia = _huygens_test_data()
    segment = _build_orthogonal_segment(bionc_type)

    inertia_at_proximal_point = inertia + mass * ((c @ c) * np.eye(3) - np.outer(c, c))
    inv_b = np.linalg.inv(np.diag([1.0, length, 1.0]))
    expected = inv_b @ inertia_at_proximal_point @ inv_b.T

    TestUtils.assert_equal(_to_numpy(segment.natural_pseudo_inertia), expected, expand=False)

