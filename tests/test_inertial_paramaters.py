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
    TestUtils.assert_equal(obj.pseudo_inertia_matrix, natural_pseudo_inertia)

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
    wrong_shape_center_of_mass = np.array([1.0, 2.0, 3.0])
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

    TestUtils.assert_equal(obj.pseudo_inertia_matrix, inertia)
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

    obj = NaturalInertialParameters.from_cartesian_inertial_parameters(mass, center_of_mass, inertia,
                                                                       transformation_matrix)

    assert obj.mass == mass
    assert np.array_equal(obj.natural_center_of_mass, center_of_mass)
    assert np.array_equal(obj.pseudo_inertia_matrix, inertia)

    TestUtils.assert_equal(obj.mass, mass)
    TestUtils.assert_equal(obj.natural_center_of_mass, np.array([[0.87212626],[2.29999071],[3.01872294]]))
    TestUtils.assert_equal(
        obj.pseudo_inertia_matrix,
        np.array(
            [[ 58.25814425, -14.43488179, -17.05540064],
             [-14.43488179,  54.77616526,  -8.35459409],
             [-17.05540064,  -8.35459409,  57.71369183]]
        ))

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

    TestUtils.assert_equal(obj.center_of_mass(transformation_matrix_2),
                           np.array([[0.99818507],[2.20011923],[2.9967137 ]])
                           )
    TestUtils.assert_equal(obj.inertia(transformation_matrix_2),
                            np.array([[-2.23189053, -0.72722177,  0.90716309],
               [-0.72722177,  9.69832619, -0.5630764 ],
               [ 0.90716309, -0.5630764 , -2.39357526]])
                            )


