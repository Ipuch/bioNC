import numpy as np
import pytest
from bionc import (
    bionc_numpy as bionc_np,
)


def test_SegmentNaturalCoordinates():

    Qi = bionc_np.SegmentNaturalCoordinates.from_components(
        u=np.array([0, 0, 0]),
        rp=np.array([4, 5, 6]),
        rd=np.array([7, 8, 9]),
        w=np.array([10, 11, 12]),
    )
    np.testing.assert_equal(Qi.u, np.array([0, 0, 0]))
    np.testing.assert_equal(Qi.rp, np.array([4, 5, 6]))
    np.testing.assert_equal(Qi.rd, np.array([7, 8, 9]))
    np.testing.assert_equal(Qi.w, np.array([10, 11, 12]))
    np.testing.assert_equal(Qi.v, -np.array([7, 8, 9]) + np.array([4, 5, 6]))
    np.testing.assert_equal(Qi.vector[:, np.newaxis], Qi)


# accelerations
def test_NaturalAccelerationsCreator():
    Qddot1 = bionc_np.SegmentNaturalAccelerations.from_components(
        uddot=np.array([1, 2, 3]),
        wddot=np.array([4, 5, 6]),
        rdddot=np.array([7, 8, 9]),
        rpddot=np.array([10, 11, 12]),
    )
    np.testing.assert_equal(Qddot1.uddot, np.array([1, 2, 3]))
    np.testing.assert_equal(Qddot1.wddot, np.array([4, 5, 6]))
    np.testing.assert_equal(Qddot1.rdddot, np.array([7, 8, 9]))
    np.testing.assert_equal(Qddot1.rpddot, np.array([10, 11, 12]))
    np.testing.assert_equal(Qddot1.vector, Qddot1)


def test_concatenate():
    Q1 = bionc_np.SegmentNaturalCoordinates.from_components(
        u=np.array([1, 2, 3]),
        rp=np.array([4, 5, 6]),
        rd=np.array([7, 8, 9]),
        w=np.array([10, 11, 12]),
    )
    Q2 = bionc_np.SegmentNaturalCoordinates.from_components(
        u=np.array([11, 22, 33]),
        rp=np.array([4, 5, 6]),
        rd=np.array([7, 8, 9]),
        w=np.array([10, 11, 12]),
    )

    # Methods such as u, v, w
    # of SegmentNaturalCoordinatesInterface are not inherited if Q1 and Q2 are concatenated with numpy method
    Q = np.concatenate((Q1, Q2), axis=0)
    # this would raise an error
    with pytest.raises(AttributeError, match="'numpy.ndarray' object has no attribute 'u'"):
        Q.u


def test_concatenate_accelerations():
    Qddot1 = bionc_np.SegmentNaturalAccelerations.from_components(
        uddot=np.array([1, 2, 3]),
        wddot=np.array([4, 5, 6]),
        rdddot=np.array([7, 8, 9]),
        rpddot=np.array([10, 11, 12]),
    )
    Qddot2 = bionc_np.SegmentNaturalAccelerations.from_components(
        uddot=np.array([11, 22, 33]),
        wddot=np.array([4, 5, 6]),
        rdddot=np.array([7, 82, 9]),
        rpddot=np.array([110, 11, 12]),
    )
    Qddot = np.concatenate((Qddot1, Qddot2), axis=0)
    with pytest.raises(AttributeError, match="'numpy.ndarray' object has no attribute 'uddot'"):
        Qddot.uddot


# Build a class called GeneralizedCoordinates to handle the concatenation of SegmentNaturalCoordinates
def test_NaturalCoordinates():
    Q1 = bionc_np.SegmentNaturalCoordinates.from_components(
        u=np.array([1, 2, 3]),
        rp=np.array([4, 5, 6]),
        rd=np.array([7, 8, 9]),
        w=np.array([10, 11, 12]),
    )
    print(Q1.v)
    Q2 = bionc_np.SegmentNaturalCoordinates.from_components(
        u=np.array([11, 22, 33]),
        rp=np.array([4, 5, 6]),
        rd=np.array([7, 8, 9]),
        w=np.array([10, 11, 12]),
    )
    Q = bionc_np.NaturalCoordinates.from_qi((Q1, Q2))
    np.testing.assert_equal(Q.u(0), np.array([1, 2, 3]))
    np.testing.assert_equal(Q.u(1), np.array([11, 22, 33]))
    np.testing.assert_equal(Q.v(0), -np.array([7, 8, 9]) + np.array([4, 5, 6]))
    np.testing.assert_equal(Q.v(1), -np.array([7, 8, 9]) + np.array([4, 5, 6]))
    np.testing.assert_equal(Q.vector(0), Q1)
    np.testing.assert_equal(Q.vector(1), Q2)
    np.testing.assert_equal(Q.vector(0).u, np.array([1, 2, 3]))
    np.testing.assert_equal(Q.vector(1).u, np.array([11, 22, 33]))
    np.testing.assert_equal(Q.vector(0).v, -np.array([7, 8, 9]) + np.array([4, 5, 6]))
    np.testing.assert_equal(Q.vector(1).v, -np.array([7, 8, 9]) + np.array([4, 5, 6]))
    np.testing.assert_equal(Q.nb_qi(), 2)


# do the same tests for NaturalAccelerations and SegmentNaturalAccelerations
def test_NaturalAccelerations():
    Qddot1 = bionc_np.SegmentNaturalAccelerations.from_components(
        uddot=np.array([1, 2, 3]),
        wddot=np.array([4, 5, 6]),
        rdddot=np.array([7, 8, 9]),
        rpddot=np.array([10, 11, 12]),
    )
    Qddot2 = bionc_np.SegmentNaturalAccelerations.from_components(
        uddot=np.array([11, 22, 33]),
        wddot=np.array([4, 5, 6]),
        rdddot=np.array([7, 82, 9]),
        rpddot=np.array([110, 11, 12]),
    )
    Qddot = bionc_np.NaturalAccelerations.from_qddoti((Qddot1, Qddot2))
    np.testing.assert_equal(Qddot.uddot(0), np.array([1, 2, 3]))
    np.testing.assert_equal(Qddot.uddot(1), np.array([11, 22, 33]))
    np.testing.assert_equal(Qddot.vector(0), Qddot1)
    np.testing.assert_equal(Qddot.vector(1), Qddot2)
    np.testing.assert_equal(Qddot.vector(0).uddot, np.array([1, 2, 3]))
    np.testing.assert_equal(Qddot.vector(1).uddot, np.array([11, 22, 33]))
    np.testing.assert_equal(Qddot.nb_qddoti(), 2)
