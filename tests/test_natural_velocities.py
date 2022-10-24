from bionc.utils import SegmentNaturalVelocities, NaturalVelocities
import numpy as np
import pytest


def test_natural_velocities():
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
        SegmentNaturalVelocities.from_components(None, correct_vector, correct_vector, correct_vector)

    # test wrong vector udot
    with pytest.raises(ValueError, match="u must be a 3x1 numpy array"):
        SegmentNaturalVelocities.from_components(wrong_vector, correct_vector, correct_vector, correct_vector)

    # test None rpdot
    with pytest.raises(ValueError, match="rp must be a numpy array .* or a list of 3 elements"):
        SegmentNaturalVelocities.from_components(correct_vector, None, correct_vector, correct_vector)

    # test wrong vector rpdot
    with pytest.raises(ValueError, match="rp must be a 3x1 numpy array"):
        SegmentNaturalVelocities.from_components(correct_vector, wrong_vector, correct_vector, correct_vector)

    # test None rddot
    with pytest.raises(ValueError, match="rd must be a numpy array .* or a list of 3 elements"):
        SegmentNaturalVelocities.from_components(correct_vector, correct_vector, None, correct_vector)

    # test wrong vector rddot
    with pytest.raises(ValueError, match="rd must be a 3x1 numpy array"):
        SegmentNaturalVelocities.from_components(correct_vector, correct_vector, wrong_vector, correct_vector)

    # test None wdot
    with pytest.raises(ValueError, match="w must be a numpy array .* or a list of 3 elements"):
        SegmentNaturalVelocities.from_components(correct_vector, correct_vector, correct_vector, None)

    # test wrong vector wdot
    with pytest.raises(ValueError, match="v must be a 3x1 numpy array"):
        SegmentNaturalVelocities.from_components(correct_vector, correct_vector, correct_vector, wrong_vector)

    # Test concatenate + parameters
    segment_natural_velocities_test = SegmentNaturalVelocities.from_components(
        [1, 0, 0], [2, 0, 0], [3, 0, 0], [4, 0, 0]
    )

    assert np.all(segment_natural_velocities_test == np.array([1, 0, 0, 2, 0, 0, 3, 0, 0, 4, 0, 0]))
    assert np.all(segment_natural_velocities_test.udot == np.array([1, 0, 0]))
    assert np.all(segment_natural_velocities_test.rpdot == np.array([2, 0, 0]))
    assert np.all(segment_natural_velocities_test.rddot == np.array([3, 0, 0]))
    assert np.all(segment_natural_velocities_test.wdot == np.array([4, 0, 0]))
    # v = rp-rd
    assert np.all(segment_natural_velocities_test.vdot == np.array([-1, 0, 0]))

    # vectors
    assert np.all(segment_natural_velocities_test.vector == np.array([1, 0, 0, 2, 0, 0, 3, 0, 0, 4, 0, 0]))

    for ind, component in enumerate(segment_natural_velocities_test.to_components):
        assert np.all(component == np.array([ind + 1, 0, 0]))

    # -------------------------------------------------------------------------------------------------------------------
    # NaturalVelocities
    # ------------------------------------------------------------------------------------------------------------------
    qdot1 = SegmentNaturalVelocities.from_components(
        udot=np.array([1, 2, 3]),
        wdot=np.array([4, 5, 6]),
        rddot=np.array([7, 8, 9]),
        rpdot=np.array([10, 11, 12]),
    )
    qdot2 = SegmentNaturalVelocities.from_components(
        udot=np.array([11, 22, 33]),
        wdot=np.array([4, 5, 6]),
        rddot=np.array([7, 82, 9]),
        rpdot=np.array([110, 11, 12]),
    )

    # Wrong entry
    with pytest.raises(ValueError, match="tuple_of_Q must be a tuple of SegmentGeneralizedCoordinates"):
        NaturalVelocities.from_Qdoti(1)

    # One wrong entry in the list
    with pytest.raises(ValueError, match="tuple_of_Q must be a tuple of SegmentGeneralizedCoordinates"):
        NaturalVelocities.from_Qdoti((qdot1, qdot2, [0, 0]))

    qdot = NaturalVelocities.from_Qdoti((qdot1, qdot2))

    #
    np.testing.assert_equal(qdot.udot(0), np.array([1, 2, 3]))
    np.testing.assert_equal(qdot.udot(1), np.array([11, 22, 33]))

    # test nb_Qdoti
    np.testing.assert_equal(qdot.nb_Qdoti(), 2)

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
