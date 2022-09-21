import numpy as np
import pytest
from bioNC import NaturalCoordinates, SegmentNaturalCoordinates


def test_SegmentNaturalCoordinatesCreator():

    Qi = SegmentNaturalCoordinates.from_components(
        u=np.array([0, 0, 0]), rp=np.array([4, 5, 6]), rd=np.array([7, 8, 9]), w=np.array([10, 11, 12])
    )
    np.testing.assert_equal(Qi.u, np.array([0, 0, 0]))
    np.testing.assert_equal(Qi.rp, np.array([4, 5, 6]))
    np.testing.assert_equal(Qi.rd, np.array([7, 8, 9]))
    np.testing.assert_equal(Qi.w, np.array([10, 11, 12]))
    np.testing.assert_equal(Qi.v, np.array([7, 8, 9]) - np.array([4, 5, 6]))
    np.testing.assert_equal(Qi.vector, Qi)


def test_concatenate():
    Q1 = SegmentNaturalCoordinates.from_components(
        u=np.array([1, 2, 3]), rp=np.array([4, 5, 6]), rd=np.array([7, 8, 9]), w=np.array([10, 11, 12])
    )
    Q2 = SegmentNaturalCoordinates.from_components(
        u=np.array([11, 22, 33]), rp=np.array([4, 5, 6]), rd=np.array([7, 8, 9]), w=np.array([10, 11, 12])
    )

    # Methods such as u, v, w
    # of SegmentGeneralizedCoordinatesInterface are not inherited if Q1 and Q2 are concatenated with numpy method
    Q = np.concatenate((Q1, Q2), axis=0)
    # this would raise an error
    with pytest.raises(AttributeError, match="'numpy.ndarray' object has no attribute 'u'"):
        Q.u


# Build a class called GeneralizedCoordinates to handle the concatenation of SegmentGeneralizedCoordinates
def test_NaturalCoordinatesCreator():
    Q1 = SegmentNaturalCoordinates.from_components(
        u=np.array([1, 2, 3]), rp=np.array([4, 5, 6]), rd=np.array([7, 8, 9]), w=np.array([10, 11, 12])
    )
    print(Q1.v)
    Q2 = SegmentNaturalCoordinates.from_components(
        u=np.array([11, 22, 33]), rp=np.array([4, 5, 6]), rd=np.array([7, 8, 9]), w=np.array([10, 11, 12])
    )
    Q = NaturalCoordinates.from_Qi((Q1, Q2))
    np.testing.assert_equal(Q.u(0), np.array([1, 2, 3]))
    np.testing.assert_equal(Q.u(1), np.array([11, 22, 33]))
    np.testing.assert_equal(Q.v(0), np.array([7, 8, 9]) - np.array([4, 5, 6]))
    np.testing.assert_equal(Q.v(1), np.array([7, 8, 9]) - np.array([4, 5, 6]))
    np.testing.assert_equal(Q.vector(0), Q1)
    np.testing.assert_equal(Q.vector(1), Q2)
    np.testing.assert_equal(Q.vector(0).u, np.array([1, 2, 3]))
    np.testing.assert_equal(Q.vector(1).u, np.array([11, 22, 33]))
    np.testing.assert_equal(Q.vector(0).v, np.array([7, 8, 9]) - np.array([4, 5, 6]))
    np.testing.assert_equal(Q.vector(1).v, np.array([7, 8, 9]) - np.array([4, 5, 6]))
    np.testing.assert_equal(Q.nb_Qi(), 2)
