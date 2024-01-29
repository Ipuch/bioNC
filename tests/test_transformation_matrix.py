import numpy as np
import pytest

from bionc import NaturalAxis
from bionc.bionc_numpy.transformation_matrix import compute_transformation_matrix
from bionc.bionc_numpy.transformation_matrix_inverse import compute_transformation_matrix_inverse
from bionc.utils.enums import TransformationMatrixType
from bionc.utils.transformation_matrix import check_plane, TransformationMatrixUtil, check_axis_to_keep
from .utils import TestUtils


def test_check_plane():
    plane = (NaturalAxis.U, NaturalAxis.U)
    with pytest.raises(
            ValueError,
            match=f"Plane must be a tuple of different axis, got \(<NaturalAxis.U: 'U'>, <NaturalAxis.U: 'U'>\)"
    ):
        check_plane(plane)
    plane = (NaturalAxis.U, NaturalAxis.U, NaturalAxis.U)
    with pytest.raises(ValueError, match="Plane must be a tuple of length 2, got 3"):
        check_plane(plane)
    plane = (NaturalAxis.U, 1)
    with pytest.raises(ValueError, match=f"Plane must be a tuple of NaturalAxis, got \(<NaturalAxis.U: 'U'>, 1\)"):
        check_plane(plane)
    plane = (NaturalAxis.V, NaturalAxis.U)
    with pytest.raises(
            ValueError,
            match=f"Invert Axis in plane, because it would lead to an indirect frame, got \(<NaturalAxis.V: 'V'>, <NaturalAxis.U: 'U'>\)",
    ):
        check_plane(plane)
    plane = (NaturalAxis.U, NaturalAxis.W)
    with pytest.raises(
            ValueError,
            match=f"Invert Axis in plane, because it would lead to an indirect frame, got \(<NaturalAxis.U: 'U'>, <NaturalAxis.W: 'W'>\)",
    ):
        check_plane(plane)
    plane = (NaturalAxis.W, NaturalAxis.V)
    with pytest.raises(
            ValueError,
            match=f"Invert Axis in plane, because it would lead to an indirect frame, got \(<NaturalAxis.W: 'W'>, <NaturalAxis.V: 'V'>\)",
    ):
        check_plane(plane)


length = 2.0
alpha = 0.5
beta = 0.6
gamma = 0.7

length_i = 0.5
alpha_i = np.pi / 2 + 0.1
beta_i = np.pi / 2 + 0.1
gamma_i = np.pi / 2 - 0.1


def test_transformation_matrix_Buv():
    result = compute_transformation_matrix(TransformationMatrixType.Buv, length, alpha, beta, gamma)
    assert isinstance(result, np.ndarray)
    assert result.shape == (3, 3)
    np.testing.assert_almost_equal(
        result, np.array([[1.0, 1.5296844, 0.8253356], [0.0, 1.2884354, 0.3823724], [0.0, 0.0, 0.4154666]])
    )


def test_transformation_matrix_Buv_inverse():
    result = compute_transformation_matrix(TransformationMatrixType.Buv, length_i, alpha_i, beta_i, gamma_i)
    result_inverse = compute_transformation_matrix_inverse(TransformationMatrixType.Buv, length_i, alpha_i, beta_i,
                                                           gamma_i)

    np.finfo('float64')

    assert isinstance(result_inverse, np.ndarray)
    assert result_inverse.shape == (3, 3)
    np.testing.assert_almost_equal(
        result_inverse,
        np.linalg.inv(result)
    )

    for i in range(100):
        print(i)
        inv_num = np.linalg.inv(result)
        inv_inv_num = np.linalg.inv(np.linalg.inv(inv_num))
        np.testing.assert_almost_equal(
            inv_inv_num,
            inv_num
        )
        result = np.linalg.inv(inv_num)

    for i in range(100):
        print(i)
        inv_num = result_inverse
        inv_inv_num = np.linalg.inv(np.linalg.inv(inv_num))
        np.testing.assert_almost_equal(
            inv_inv_num,
            inv_num
        )
        result = np.linalg.inv(inv_num)


def test_transformation_matrix_Bvu():
    result = compute_transformation_matrix(TransformationMatrixType.Bvu, length, alpha, beta, gamma)
    assert isinstance(result, np.ndarray)
    assert result.shape == (3, 3)
    np.testing.assert_almost_equal(
        result,
        np.array(
            [[0.6442177, 0.0, 0.2392413], [0.7648422, 2.0, 0.8775826], [0.0, 0.0, 0.4154666]],
        ),
    )


def test_transformation_matrix_Bwu():
    result = compute_transformation_matrix(TransformationMatrixType.Bwu, length, alpha, beta, gamma)
    assert isinstance(result, np.ndarray)
    assert result.shape == (3, 3)
    np.testing.assert_almost_equal(
        result, np.array([[0.5646425, -1.0358334, 0.0], [0.0, 0.9480368, 0.0], [0.8253356, 1.7551651, 1.0]])
    )


def test_transformation_matrix_Buw():
    result = compute_transformation_matrix(TransformationMatrixType.Buw, length, alpha, beta, gamma)
    assert isinstance(result, np.ndarray)
    assert result.shape == (3, 3)
    np.testing.assert_almost_equal(
        result,
        np.array(
            [
                [1.0, 1.529684374568977, 0.8253356149096783],
                [0.0, 0.9480367617186112, 0.0],
                [0.0, -0.4807683268354297, 0.5646424733950354],
            ]
        ),
    )


def test_transformation_matrix_Bvw():
    with pytest.raises(NotImplementedError):
        compute_transformation_matrix(TransformationMatrixType.Bvw, length, alpha, beta, gamma)


def test_transformation_matrix_Bwv():
    with pytest.raises(NotImplementedError):
        compute_transformation_matrix(TransformationMatrixType.Bwv, length, alpha, beta, gamma)


def test_invalid_matrix_type():
    with pytest.raises(ValueError):
        compute_transformation_matrix("INVALID_TYPE", length, alpha, beta, gamma)


@pytest.mark.parametrize(
    "bionc_type",
    ["numpy", "casadi"],
)
def test_segment_transformation_matrix(bionc_type):
    if bionc_type == "casadi":
        from bionc.bionc_casadi import (
            NaturalSegment,
        )
    else:
        from bionc.bionc_numpy import (
            NaturalSegment,
        )

    bbox = NaturalSegment.with_cartesian_inertial_parameters(
        name="bbox",
        alpha=np.pi / 1.9,
        beta=np.pi / 2.3,
        gamma=np.pi / 2.1,
        length=1.5,
        mass=1.1,
        center_of_mass=np.array([0.1, 0.11, 0.111]),  # scs
        inertia=np.array([[1.1, 0, 0], [0, 1.2, 0], [0, 0, 1.3]]),  # scs
        inertial_transformation_matrix=TransformationMatrixType.Buv,
    )

    res_Buv = np.array([[1.0, 0.0, 0.0], [0.11209514, 1.4958057, 0.0], [0.20345601, -0.09805782, 0.97416134]])

    TestUtils.assert_equal(bbox.compute_transformation_matrix(), res_Buv)
    TestUtils.assert_equal(bbox.compute_transformation_matrix(matrix_type=TransformationMatrixType.Buv), res_Buv)
    TestUtils.assert_equal(bbox.compute_transformation_matrix(matrix_type="Buv"), res_Buv)

    res_Bvu = np.array([[0.9972038, 0.07473009, 0.0], [0.0, 1.5, 0.0], [0.21021498, -0.08257935, 0.97416134]])
    TestUtils.assert_equal(bbox.compute_transformation_matrix(matrix_type=TransformationMatrixType.Bvu), res_Bvu)
    TestUtils.assert_equal(bbox.compute_transformation_matrix(matrix_type="Bvu"), res_Bvu)

    res_Bwu = np.array([[0.97908409, 0.0, 0.20345601], [0.13783542, 1.48828492, -0.12386902], [0.0, 0.0, 1.0]])
    TestUtils.assert_equal(bbox.compute_transformation_matrix(matrix_type=TransformationMatrixType.Bwu), res_Bwu)
    TestUtils.assert_equal(bbox.compute_transformation_matrix(matrix_type="Bwu"), res_Bwu)

    res_Buw = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.11209514, 1.48828491, -0.14716265],
            [0.20345601, 0, 0.97908408],
        ]
    )
    TestUtils.assert_equal(bbox.compute_transformation_matrix(matrix_type=TransformationMatrixType.Buw), res_Buw)
    TestUtils.assert_equal(bbox.compute_transformation_matrix(matrix_type="Buw"), res_Buw)

    with pytest.raises(NotImplementedError):
        bbox.compute_transformation_matrix(matrix_type=TransformationMatrixType.Bvw)
    with pytest.raises(NotImplementedError):
        bbox.compute_transformation_matrix(matrix_type=TransformationMatrixType.Bwv)

    with pytest.raises(ValueError):
        bbox.compute_transformation_matrix(matrix_type="INVALID_TYPE")


def test_transformation_matrix_util():
    tm = TransformationMatrixUtil((NaturalAxis.U, NaturalAxis.V), NaturalAxis.U)
    assert tm.to_enum() == TransformationMatrixType.Buv

    tm = TransformationMatrixUtil((NaturalAxis.W, NaturalAxis.U), NaturalAxis.W)
    assert tm.to_enum() == TransformationMatrixType.Bwu

    tm = TransformationMatrixUtil((NaturalAxis.V, NaturalAxis.W), NaturalAxis.V)
    assert tm.to_enum() == TransformationMatrixType.Bvw

    tm = TransformationMatrixUtil((NaturalAxis.U, NaturalAxis.V), NaturalAxis.V)
    assert tm.to_enum() == TransformationMatrixType.Bvu

    tm = TransformationMatrixUtil((NaturalAxis.W, NaturalAxis.U), NaturalAxis.U)
    assert tm.to_enum() == TransformationMatrixType.Buw

    tm = TransformationMatrixUtil((NaturalAxis.V, NaturalAxis.W), NaturalAxis.W)
    assert tm.to_enum() == TransformationMatrixType.Bwv


def test_check_plane_invalid_input():
    with pytest.raises(ValueError):
        check_plane((NaturalAxis.U, NaturalAxis.U))

    with pytest.raises(ValueError):
        check_plane((NaturalAxis.V, NaturalAxis.U))

    with pytest.raises(ValueError):
        check_plane((NaturalAxis.U, NaturalAxis.W, NaturalAxis.V))


def test_check_axis_to_keep_invalid_input():
    with pytest.raises(ValueError):
        check_axis_to_keep("X")
