from bionc import bionc_numpy as bionc_np
from bionc import bionc_casadi as bionc_mx
import numpy as np
from casadi import MX
import pytest
from .utils import TestUtils


def test_natural_velocities_numpy():
    # -------------------------------------------------------------------------------------------------------------------
    # SegmentNaturalVelocities
    # ------------------------------------------------------------------------------------------------------------------
    # List instead of np array to test the translation from list to np.array
    correct_vector = [1, 0, 0]
    wrong_vector = [1, 0]
    # Test wrong entry
    # ------------------------------------------------------------------------------------------------------------------
    # test None udot
    with pytest.raises(ValueError, match="u must be a numpy array .* or a list of 3 elements"):
        bionc_np.SegmentNaturalVelocities.from_components(None, correct_vector, correct_vector, correct_vector)

    # test wrong vector udot
    with pytest.raises(ValueError, match="u must be a 3x1 numpy array"):
        bionc_np.SegmentNaturalVelocities.from_components(wrong_vector, correct_vector, correct_vector, correct_vector)

    # test None rpdot
    with pytest.raises(ValueError, match="rp must be a numpy array .* or a list of 3 elements"):
        bionc_np.SegmentNaturalVelocities.from_components(correct_vector, None, correct_vector, correct_vector)

    # test wrong vector rpdot
    with pytest.raises(ValueError, match="rp must be a 3x1 numpy array"):
        bionc_np.SegmentNaturalVelocities.from_components(correct_vector, wrong_vector, correct_vector, correct_vector)

    # test None rddot
    with pytest.raises(ValueError, match="rd must be a numpy array .* or a list of 3 elements"):
        bionc_np.SegmentNaturalVelocities.from_components(correct_vector, correct_vector, None, correct_vector)

    # test wrong vector rddot
    with pytest.raises(ValueError, match="rd must be a 3x1 numpy array"):
        bionc_np.SegmentNaturalVelocities.from_components(correct_vector, correct_vector, wrong_vector, correct_vector)

    # test None wdot
    with pytest.raises(ValueError, match="w must be a numpy array .* or a list of 3 elements"):
        bionc_np.SegmentNaturalVelocities.from_components(correct_vector, correct_vector, correct_vector, None)

    # test wrong vector wdot
    with pytest.raises(ValueError, match="v must be a 3x1 numpy array"):
        bionc_np.SegmentNaturalVelocities.from_components(correct_vector, correct_vector, correct_vector, wrong_vector)

    # Test concatenate + parameters
    qdot_test = bionc_np.SegmentNaturalVelocities.from_components(
        udot=[1, 0, 0],
        rpdot=[2, 0, 0],
        rddot=[3, 0, 0],
        wdot=[4, 0, 0],
    )

    assert np.all(qdot_test == np.array([1, 0, 0, 2, 0, 0, 3, 0, 0, 4, 0, 0]))
    assert np.all(qdot_test.udot == np.array([1, 0, 0]))
    assert np.all(qdot_test.rpdot == np.array([2, 0, 0]))
    assert np.all(qdot_test.rddot == np.array([3, 0, 0]))
    assert np.all(qdot_test.wdot == np.array([4, 0, 0]))

    # vdot = rpdot - rddot
    assert np.all(qdot_test.vdot == np.array([-1, 0, 0]))

    # vectors
    assert np.all(qdot_test.vector == np.array([1, 0, 0, 2, 0, 0, 3, 0, 0, 4, 0, 0]))

    for ind, component in enumerate(qdot_test.to_components):
        assert np.all(component == np.array([ind + 1, 0, 0]))

    # -------------------------------------------------------------------------------------------------------------------
    # NaturalVelocities
    # ------------------------------------------------------------------------------------------------------------------
    qdot1 = bionc_np.SegmentNaturalVelocities.from_components(
        udot=np.array([1, 2, 3]),
        wdot=np.array([4, 5, 6]),
        rddot=np.array([7, 8, 9]),
        rpdot=np.array([10, 11, 12]),
    )
    qdot2 = bionc_np.SegmentNaturalVelocities.from_components(
        udot=np.array([11, 22, 33]),
        wdot=np.array([4, 5, 6]),
        rddot=np.array([7, 82, 9]),
        rpdot=np.array([110, 11, 12]),
    )

    # Wrong entry
    with pytest.raises(ValueError, match="tuple_of_Q must be a tuple of SegmentNaturalVelocities"):
        bionc_np.NaturalVelocities.from_qdoti(1)

    # One wrong entry in the list
    with pytest.raises(ValueError, match="tuple_of_Q must be a tuple of SegmentNaturalVelocities"):
        bionc_np.NaturalVelocities.from_qdoti((qdot1, qdot2, [0, 0]))

    qdot = bionc_np.NaturalVelocities.from_qdoti((qdot1, qdot2))

    #
    np.testing.assert_equal(qdot.udot(0), np.array([1, 2, 3]))
    np.testing.assert_equal(qdot.udot(1), np.array([11, 22, 33]))

    # test nb_qdoti
    np.testing.assert_equal(qdot.nb_qdoti(), 2)

    # test of the different component of vector
    np.testing.assert_equal(qdot1.vector, qdot1)
    np.testing.assert_equal(qdot.vector(0), qdot1)
    np.testing.assert_equal(qdot.vector(1), qdot2)
    np.testing.assert_equal(qdot.vector(0).udot, np.array([1, 2, 3]))
    np.testing.assert_equal(qdot.vector(1).udot, np.array([11, 22, 33]))
    np.testing.assert_equal(qdot.vector(0).wdot, np.array([4, 5, 6]))
    np.testing.assert_equal(qdot.vector(1).wdot, np.array([4, 5, 6]))
    np.testing.assert_equal(qdot.vector(0).rddot, np.array([7, 8, 9]))
    np.testing.assert_equal(qdot.vector(1).rddot, np.array([7, 82, 9]))
    np.testing.assert_equal(qdot.vector(0).rpdot, np.array([10, 11, 12]))
    np.testing.assert_equal(qdot.vector(1).rpdot, np.array([110, 11, 12]))
    np.testing.assert_equal(qdot.vector(0).vdot, np.array([3, 3, 3]))
    np.testing.assert_equal(qdot.vector(1).vdot, np.array([103, -71, 3]))

    # test concatenate
    qdot = np.concatenate((qdot1, qdot2), axis=0)
    with pytest.raises(AttributeError, match="'numpy.ndarray' object has no attribute 'udot'"):
        qdot.udot


def test_natural_velocities_casadi():
    # -------------------------------------------------------------------------------------------------------------------
    # SegmentNaturalVelocities
    # ------------------------------------------------------------------------------------------------------------------
    # List instead of np array to test the translation from list to np.array
    correct_vector = [1, 0, 0]
    wrong_vector = [1, 0]
    # Test wrong entry
    # ------------------------------------------------------------------------------------------------------------------
    # test None udot
    with pytest.raises(ValueError, match="u must be a array .* or a list of 3 elements"):
        bionc_mx.SegmentNaturalVelocities.from_components(None, correct_vector, correct_vector, correct_vector)

    # test wrong vector udot
    with pytest.raises(ValueError, match="u must be a 3x1 array"):
        bionc_mx.SegmentNaturalVelocities.from_components(wrong_vector, correct_vector, correct_vector, correct_vector)

    # test None rpdot
    with pytest.raises(ValueError, match="rp must be a array .* or a list of 3 elements"):
        bionc_mx.SegmentNaturalVelocities.from_components(correct_vector, None, correct_vector, correct_vector)

    # test wrong vector rpdot
    with pytest.raises(ValueError, match="rp must be a 3x1 array"):
        bionc_mx.SegmentNaturalVelocities.from_components(correct_vector, wrong_vector, correct_vector, correct_vector)

    # test None rddot
    with pytest.raises(ValueError, match="rd must be a array .* or a list of 3 elements"):
        bionc_mx.SegmentNaturalVelocities.from_components(correct_vector, correct_vector, None, correct_vector)

    # test wrong vector rddot
    with pytest.raises(ValueError, match="rd must be a 3x1 array"):
        bionc_mx.SegmentNaturalVelocities.from_components(correct_vector, correct_vector, wrong_vector, correct_vector)

    # test None wdot
    with pytest.raises(ValueError, match="w must be a array .* or a list of 3 elements"):
        bionc_mx.SegmentNaturalVelocities.from_components(correct_vector, correct_vector, correct_vector, None)

    # test wrong vector wdot
    with pytest.raises(ValueError, match="v must be a 3x1 array"):
        bionc_mx.SegmentNaturalVelocities.from_components(correct_vector, correct_vector, correct_vector, wrong_vector)

    # Test concatenate + parameters
    qdot_test = bionc_mx.SegmentNaturalVelocities.from_components(
        udot=[1, 0, 0],
        rpdot=[2, 0, 0],
        rddot=[3, 0, 0],
        wdot=[4, 0, 0],
    )

    assert np.all(TestUtils.mx_to_array(qdot_test) == np.array([1, 0, 0, 2, 0, 0, 3, 0, 0, 4, 0, 0]))
    assert np.all(TestUtils.mx_to_array(qdot_test.udot) == np.array([1, 0, 0]))
    assert np.all(TestUtils.mx_to_array(qdot_test.rpdot) == np.array([2, 0, 0]))
    assert np.all(TestUtils.mx_to_array(qdot_test.rddot) == np.array([3, 0, 0]))
    assert np.all(TestUtils.mx_to_array(qdot_test.wdot) == np.array([4, 0, 0]))

    # vdot = rpdot - rddot
    assert np.all(TestUtils.mx_to_array(qdot_test.vdot) == np.array([-1, 0, 0]))

    # vectors
    assert np.all(TestUtils.mx_to_array(qdot_test.vector) == np.array([1, 0, 0, 2, 0, 0, 3, 0, 0, 4, 0, 0]))

    for ind, component in enumerate(qdot_test.to_components):
        assert np.all(TestUtils.mx_to_array(component) == np.array([ind + 1, 0, 0]))

    # -------------------------------------------------------------------------------------------------------------------
    # NaturalVelocities
    # ------------------------------------------------------------------------------------------------------------------
    qdot1 = bionc_mx.SegmentNaturalVelocities.from_components(
        udot=np.array([1, 2, 3]),
        wdot=np.array([4, 5, 6]),
        rddot=np.array([7, 8, 9]),
        rpdot=np.array([10, 11, 12]),
    )
    qdot2 = bionc_mx.SegmentNaturalVelocities.from_components(
        udot=np.array([11, 22, 33]),
        wdot=np.array([4, 5, 6]),
        rddot=np.array([7, 82, 9]),
        rpdot=np.array([110, 11, 12]),
    )

    # Wrong entry
    with pytest.raises(ValueError, match="tuple_of_Q must be a tuple of SegmentNaturalVelocities"):
        bionc_mx.NaturalVelocities.from_qdoti(1)

    # One wrong entry in the list
    with pytest.raises(ValueError, match="tuple_of_Q must be a tuple of SegmentNaturalVelocities"):
        bionc_mx.NaturalVelocities.from_qdoti((qdot1, qdot2, [0, 0]))

    qdot = bionc_mx.NaturalVelocities.from_qdoti((qdot1, qdot2))

    #
    np.testing.assert_equal(TestUtils.mx_to_array(qdot.udot(0)), np.array([1, 2, 3]))
    np.testing.assert_equal(TestUtils.mx_to_array(qdot.udot(1)), np.array([11, 22, 33]))

    # test nb_qdoti
    np.testing.assert_equal(qdot.nb_qdoti(), 2)

    # test of the different component of vector
    np.testing.assert_equal(TestUtils.mx_to_array(qdot1.vector), TestUtils.mx_to_array(qdot1))
    np.testing.assert_equal(TestUtils.mx_to_array(qdot.vector(0)), TestUtils.mx_to_array(qdot1))
    np.testing.assert_equal(TestUtils.mx_to_array(qdot.vector(1)), TestUtils.mx_to_array(qdot2))
    np.testing.assert_equal(TestUtils.mx_to_array(qdot.vector(0).udot), np.array([1, 2, 3]))
    np.testing.assert_equal(TestUtils.mx_to_array(qdot.vector(1).udot), np.array([11, 22, 33]))
    np.testing.assert_equal(TestUtils.mx_to_array(qdot.vector(0).wdot), np.array([4, 5, 6]))
    np.testing.assert_equal(TestUtils.mx_to_array(qdot.vector(1).wdot), np.array([4, 5, 6]))
    np.testing.assert_equal(TestUtils.mx_to_array(qdot.vector(0).rddot), np.array([7, 8, 9]))
    np.testing.assert_equal(TestUtils.mx_to_array(qdot.vector(1).rddot), np.array([7, 82, 9]))
    np.testing.assert_equal(TestUtils.mx_to_array(qdot.vector(0).rpdot), np.array([10, 11, 12]))
    np.testing.assert_equal(TestUtils.mx_to_array(qdot.vector(1).rpdot), np.array([110, 11, 12]))
    np.testing.assert_equal(TestUtils.mx_to_array(qdot.vector(0).vdot), np.array([3, 3, 3]))
    np.testing.assert_equal(TestUtils.mx_to_array(qdot.vector(1).vdot), np.array([103, -71, 3]))
