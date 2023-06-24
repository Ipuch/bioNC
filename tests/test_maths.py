import numpy as np
import pytest
from .utils import TestUtils

from bionc import EulerSequence


@pytest.mark.parametrize(
    "bionc_type",
    ["numpy", "casadi"],
)
def test_vnop(bionc_type):
    if bionc_type == "casadi":
        from bionc.bionc_casadi import (
            vector_projection_in_non_orthogonal_basis,
        )
    else:
        from bionc.bionc_numpy import (
            vector_projection_in_non_orthogonal_basis,
        )
    e1 = np.array([1, 0, 0])
    e2 = np.array([0, 1, 0])
    e3 = np.array([0, 0, 1])
    V = np.array([1, 1, 1])
    vnop = vector_projection_in_non_orthogonal_basis(V, e1, e2, e3)
    TestUtils.assert_equal(vnop, np.array([1, 1, 1]).T[:, np.newaxis], squeeze=False)

    e1 = np.array([1, 4, 5])
    e2 = np.array([2, -3, 6])
    e3 = np.array([3, 2, -7])
    V = np.array([-1, 3, 5])
    vnop = vector_projection_in_non_orthogonal_basis(V, e1, e2, e3)
    TestUtils.assert_equal(vnop, np.array([0.752475, -0.267327, -0.405941]).T[:, np.newaxis], squeeze=False)

    # tests errors with wrong shape
    e1 = np.array([1, 4, 5, 6])
    e2 = np.array([2, -3, 6])
    e3 = np.array([3, 2, -7])
    V = np.array([-1, 3, 5])
    with pytest.raises(ValueError):
        vector_projection_in_non_orthogonal_basis(V, e1, e2, e3)

    # e2 is wrong shape
    e1 = np.array([1, 4, 5])
    e2 = np.array([2, -3, 6, 7])

    with pytest.raises(ValueError):
        vector_projection_in_non_orthogonal_basis(V, e1, e2, e3)

    # e3 is wrong shape
    e3 = np.array([3, 2, -7, 8])
    e2 = np.array([2, -3, 6])
    with pytest.raises(ValueError):
        vector_projection_in_non_orthogonal_basis(V, e1, e2, e3)

    # V is wrong shape
    V = np.array([-1, 3, 5, 6])
    e3 = np.array([3, 2, -7])
    with pytest.raises(ValueError):
        vector_projection_in_non_orthogonal_basis(V, e1, e2, e3)

    if bionc_type == "numpy":
        e1 = np.array([[1, 4, 5], [0.5, 5, 6]]).T
        e2 = np.array([[2, -3, 6], [2, -2.3, 7]]).T
        e3 = np.array([[3, 2, -7], [1, 1.9, -8]]).T
        V = np.array([[-1, 3, 5], [-0.5, 3, 6]]).T
        vnop = vector_projection_in_non_orthogonal_basis(V, e1, e2, e3)
        TestUtils.assert_equal(
            vnop,
            np.array([[0.752475, -0.267327, -0.405941], [0.668505, -0.203698, -0.426857]]).T,
            decimal=5,
        )


@pytest.mark.parametrize(
    "bionc_type",
    ["numpy", "casadi"],
)
def test_interpolate_natural_vector(bionc_type):
    if bionc_type == "casadi":
        from bionc.bionc_casadi import (
            NaturalVector,
        )
    else:
        from bionc.bionc_numpy import (
            NaturalVector,
        )

    vector = np.array([1, 2, 3])
    interpolation_matrix = NaturalVector(vector).interpolate()
    TestUtils.assert_equal(
        interpolation_matrix,
        np.array(
            [
                [1.0, 0.0, 0.0, 3.0, 0.0, 0.0, -2.0, -0.0, -0.0, 3.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 3.0, 0.0, -0.0, -2.0, -0.0, 0.0, 3.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0, 3.0, -0.0, -0.0, -2.0, 0.0, 0.0, 3.0],
            ]
        ),
    )


@pytest.mark.parametrize(
    "bionc_type",
    [
        "numpy",
        # "casadi",
    ],
)
def test_euler_vectors(bionc_type):
    for seq in EulerSequence:
        # not implemented in biorbd
        if seq == EulerSequence.XYX:
            continue
        if seq == EulerSequence.XZX:
            continue
        if seq == EulerSequence.XYX:
            continue

        _subtest_rotations([0.1, 0.2, 0.3], [0.4, 0.5, 0.6], bionc_type=bionc_type, seq=seq)
        _subtest_rotations([-0.1, -0.2, 0.3], [0.41, 0.51, -0.61], bionc_type=bionc_type, seq=seq)
        _subtest_rotations(
            [np.pi / 3, np.pi / 4, np.pi / 5], [np.pi / 6, np.pi / 7, np.pi / 8], bionc_type=bionc_type, seq=seq
        )
        _subtest_rotations(
            [np.pi / 3, -np.pi / 4, -np.pi / 5], [np.pi / 6, -np.pi / 3, np.pi / 8], bionc_type=bionc_type, seq=seq
        )
        # more extreme angles
        _subtest_rotations(
            [np.pi / 2, -np.pi / 2, np.pi / 2], [np.pi / 2, np.pi / 2, -np.pi / 2], bionc_type=bionc_type, seq=seq
        )
        _subtest_rotations(
            [np.pi / 2, -np.pi / 2, np.pi / 2], [-np.pi / 2, np.pi / 2, -np.pi / 2], bionc_type=bionc_type, seq=seq
        )
        _subtest_rotations(
            [np.pi / 2, -2 * np.pi / 3, np.pi / 2], [np.pi / 2, -np.pi / 2, -np.pi / 2], bionc_type=bionc_type, seq=seq
        )
        # more more extreme angles
        _subtest_rotations([np.pi, -np.pi, np.pi], [np.pi, np.pi, -np.pi], bionc_type=bionc_type, seq=seq)
        _subtest_rotations([np.pi, -np.pi, np.pi], [-np.pi, np.pi, -np.pi], bionc_type=bionc_type, seq=seq)
        _subtest_rotations([np.pi, -2 * np.pi, np.pi], [np.pi, -np.pi, -np.pi], bionc_type=bionc_type, seq=seq)


def _subtest_rotations(euler_rot_angles_1: list, euler_rot_angles_2: list, bionc_type: str, seq: EulerSequence):
    if bionc_type == "casadi":
        print("todo")
        # from bionc.bionc_casadi.rotations import (
        #     euler_axes_from_rotation_matrices,
        # )
    else:
        from bionc.bionc_numpy.rotations import (
            euler_axes_from_rotation_matrices,
        )
        from biorbd import Rotation

    R_parent = Rotation.fromEulerAngles(np.array(euler_rot_angles_1), "xyz")
    R_child = Rotation.fromEulerAngles(np.array(euler_rot_angles_2), "xyz")

    if bionc_type == "casadi":
        R_parent = R_parent.to_mx()
        R_child = R_child.to_mx()
    else:
        R_parent = R_parent.to_array()
        R_child = R_child.to_array()

    me1, me2, me3 = euler_axes_from_rotation_matrices(R_parent, R_child, sequence=seq, projected_frame="mixed")

    pe1, pe2, pe3 = euler_axes_from_rotation_matrices(R_parent, R_child, sequence=seq, projected_frame="parent")

    ce1, ce2, ce3 = euler_axes_from_rotation_matrices(R_parent, R_child, sequence=seq, projected_frame="child")

    TestUtils.assert_equal(me1, pe1, decimal=7)
    TestUtils.assert_equal(me2, pe2, decimal=7)
    TestUtils.assert_equal(me3, pe3, decimal=7)

    TestUtils.assert_equal(me1, ce1, decimal=7)
    TestUtils.assert_equal(me2, ce2, decimal=7)
    TestUtils.assert_equal(me3, ce3, decimal=7)

    TestUtils.assert_equal(pe1, ce1, decimal=7)
    TestUtils.assert_equal(pe2, ce2, decimal=7)
    TestUtils.assert_equal(pe3, ce3, decimal=7)
