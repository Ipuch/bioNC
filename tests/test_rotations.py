import numpy as np
import pytest
from bionc.bionc_numpy.rotations import (
    rotation_x,
    rotation_y,
    rotation_z,
    vector_from_axis,
    rotation_matrix_from_angle_and_axis,
    rotation_matrices_from_rotation_matrix,
    euler_angles_from_rotation_matrix,
 )
from bionc import CartesianAxis, EulerSequence


def test_rotation_x():
    angle = np.pi / 4
    expected_matrix = np.array([
        [1, 0, 0],
        [0, np.cos(angle), -np.sin(angle)],
        [0, np.sin(angle), np.cos(angle)]
    ])
    np.testing.assert_array_almost_equal(rotation_x(angle), expected_matrix)


def test_rotation_y():
    angle = np.pi / 4
    expected_matrix = np.array([
        [np.cos(angle), 0, np.sin(angle)],
        [0, 1, 0],
        [-np.sin(angle), 0, np.cos(angle)]
    ])
    np.testing.assert_array_almost_equal(rotation_y(angle), expected_matrix)


def test_rotation_z():
    angle = np.pi / 4
    expected_matrix = np.array([
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle), np.cos(angle), 0],
        [0, 0, 1]
    ])
    np.testing.assert_array_almost_equal(rotation_z(angle), expected_matrix)


def test_vector_from_axis():
    assert np.array_equal(vector_from_axis('x'), np.array([1, 0, 0]))
    assert np.array_equal(vector_from_axis('y'), np.array([0, 1, 0]))
    assert np.array_equal(vector_from_axis('z'), np.array([0, 0, 1]))
    assert np.array_equal(vector_from_axis(CartesianAxis.X), np.array([1, 0, 0]))
    assert np.array_equal(vector_from_axis(CartesianAxis.Y), np.array([0, 1, 0]))
    assert np.array_equal(vector_from_axis(CartesianAxis.Z), np.array([0, 0, 1]))
    with pytest.raises(ValueError):
        vector_from_axis('a')


def test_rotation_matrices_from_rotation_matrix():
    rotation_matrix = np.concatenate((np.eye(3)[:,1:], np.eye(3)[:,0:1]), axis=1)
    sequence = 'xyz'
    R0, R1, R2 = rotation_matrices_from_rotation_matrix(rotation_matrix, sequence)
    np.testing.assert_array_almost_equal(R0, rotation_x(0))
    np.testing.assert_array_almost_equal(R1, rotation_y(np.pi/2))
    np.testing.assert_array_almost_equal(R2, rotation_z(0))


def test_rotation_matrix_from_angle_and_axis():
    angle = np.pi / 4
    assert np.array_equal(rotation_matrix_from_angle_and_axis(angle, 'x'), rotation_x(angle))
    assert np.array_equal(rotation_matrix_from_angle_and_axis(angle, 'y'), rotation_y(angle))
    assert np.array_equal(rotation_matrix_from_angle_and_axis(angle, 'z'), rotation_z(angle))
    assert np.array_equal(rotation_matrix_from_angle_and_axis(angle, CartesianAxis.X), rotation_x(angle))
    assert np.array_equal(rotation_matrix_from_angle_and_axis(angle, CartesianAxis.Y), rotation_y(angle))
    assert np.array_equal(rotation_matrix_from_angle_and_axis(angle, CartesianAxis.Z), rotation_z(angle))
    with pytest.raises(ValueError):
        rotation_matrix_from_angle_and_axis(angle, 'a')


def test_euler_angles_from_rotation_matrix():
    parent_matrix = np.eye(3)
    child_matrix = np.eye(3)
    joint_sequence = EulerSequence.XYZ
    angles = euler_angles_from_rotation_matrix(parent_matrix, child_matrix, joint_sequence)
    np.testing.assert_array_almost_equal(angles, np.array([0, 0, 0]))
