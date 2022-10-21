from bionc.utils import SegmentNaturalVelocities,NaturalVelocities
import numpy as np
import pytest
import collections


def test_natural_velocities():
    #-------------------------------------------------------------------------------------------------------------------
    ### SegmentNaturalVelocities
    # ------------------------------------------------------------------------------------------------------------------
    # Liste instead of np array to test the translation from list to np.array
    correct_vector = [1, 0, 0]
    wrong_vector = [1, 0]
    ## Test wrong entry
    # test None udot None
    with pytest.raises(ValueError, match="u must be a numpy array .* or a list of 3 elements"):
        toto = SegmentNaturalVelocities.from_components(None, correct_vector, correct_vector, correct_vector)

    # test wrong vector udot
    with pytest.raises(ValueError, match="u must be a 3x1 numpy array"):
        toto = SegmentNaturalVelocities.from_components(wrong_vector, correct_vector, correct_vector, correct_vector)

    # test None rpdot
    with pytest.raises(ValueError, match="rp must be a numpy array .* or a list of 3 elements"):
        toto = SegmentNaturalVelocities.from_components(correct_vector, None, correct_vector, correct_vector)

    # test wrong vector rpdot
    with pytest.raises(ValueError, match="rp must be a 3x1 numpy array"):
        toto = SegmentNaturalVelocities.from_components(correct_vector, wrong_vector, correct_vector, correct_vector)

    # test None rddot
    with pytest.raises(ValueError, match="rd must be a numpy array .* or a list of 3 elements"):
        toto = SegmentNaturalVelocities.from_components(correct_vector, correct_vector, None, correct_vector)

    # test wrong vector rddot
    with pytest.raises(ValueError, match="rd must be a 3x1 numpy array"):
        toto = SegmentNaturalVelocities.from_components(correct_vector, correct_vector, wrong_vector, correct_vector)

    # test None wdot
    with pytest.raises(ValueError, match="w must be a numpy array .* or a list of 3 elements"):
        toto = SegmentNaturalVelocities.from_components(correct_vector, correct_vector, correct_vector, None)

    # test wrong vector wdot
    with pytest.raises(ValueError, match="v must be a 3x1 numpy array"):
        toto = SegmentNaturalVelocities.from_components(correct_vector, correct_vector, correct_vector, wrong_vector)

    ## Test concatenate + parameters
    SegmentNaturalVelocities_test = SegmentNaturalVelocities.from_components(correct_vector, correct_vector, correct_vector, correct_vector)

    assert np.all(SegmentNaturalVelocities_test == np.array([1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0]))
    assert np.all(SegmentNaturalVelocities_test.udot == np.array([1, 0, 0]))
    assert np.all(SegmentNaturalVelocities_test.rpdot == np.array([1, 0, 0]))
    assert np.all(SegmentNaturalVelocities_test.rddot == np.array([1, 0, 0]))
    assert np.all(SegmentNaturalVelocities_test.wdot == np.array([1, 0, 0]))
    # v = rp-rd
    assert np.all(SegmentNaturalVelocities_test.vdot == np.array([0, 0, 0]))

    # why vectors ?
    assert np.all(SegmentNaturalVelocities_test.vector == np.array([1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0]))

    for component in SegmentNaturalVelocities_test.to_components:
        assert np.all(component == np.array([1, 0, 0]))

    # -------------------------------------------------------------------------------------------------------------------
    ### NaturalVelocities
    # ------------------------------------------------------------------------------------------------------------------
    ## Wrong entry

    with pytest.raises(ValueError, match="tuple_of_Q must be a tuple of SegmentGeneralizedCoordinates"):
        NaturalVelocities(1)

    ## Extraction
    Qdot1 = SegmentNaturalVelocities.from_components(
        udot=np.array([1, 2, 3]), wdot=np.array([4, 5, 6]), rddot=np.array([7, 8, 9]), rpdot=np.array([10, 11, 12])
    )
    Qdot2 = SegmentNaturalVelocities.from_components(
        udot=np.array([11, 22, 33]), wdot=np.array([4, 5, 6]), rddot=np.array([7, 82, 9]), rpdot=np.array([110, 11, 12])
    )

    Qdot = NaturalVelocities.from_Qdoti((Qdot1, Qdot2))

    np.testing.assert_equal(Qdot.udot(0), np.array([1, 2, 3]))
    np.testing.assert_equal(Qdot.udot(1), np.array([11, 22, 33]))
    np.testing.assert_equal(Qdot.vector(0), Qdot1)
    np.testing.assert_equal(Qdot.vector(1), Qdot2)
    np.testing.assert_equal(Qdot.vector(0).udot, np.array([1, 2, 3]))
    np.testing.assert_equal(Qdot.vector(1).udot, np.array([11, 22, 33]))
    np.testing.assert_equal(Qdot.nb_Qdoti(), 2)