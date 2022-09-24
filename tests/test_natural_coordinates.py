import numpy as np
import pytest
from bioNC import NaturalCoordinates, SegmentNaturalCoordinates, \
    NaturalVelocities, SegmentNaturalVelocities,\
    NaturalAccelerations, SegmentNaturalAccelerations


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


# do the for SegmentNaturalVelocities
def test_SegmentNaturalVelocitiesCreator():
    Qdoti = SegmentNaturalVelocities.from_components(
        udot=np.array([1, 2, 3]), wdot=np.array([4, 5, 6]), rddot=np.array([7, 8, 9]), rpdot=np.array([10, 11, 12])
    )
    np.testing.assert_equal(Qdoti.udot, np.array([1, 2, 3]))
    np.testing.assert_equal(Qdoti.wdot, np.array([4, 5, 6]))
    np.testing.assert_equal(Qdoti.rddot, np.array([7, 8, 9]))
    np.testing.assert_equal(Qdoti.rpdot, np.array([10, 11, 12]))
    np.testing.assert_equal(Qdoti.vector, Qdoti)


# accelerations
def test_NaturalAccelerationsCreator():
    Qddot1 = SegmentNaturalAccelerations.from_components(
        uddot=np.array([1, 2, 3]), wddot=np.array([4, 5, 6]), rdddot=np.array([7, 8, 9]), rpddot=np.array([10, 11, 12])
    )
    np.testing.assert_equal(Qddot1.uddot, np.array([1, 2, 3]))
    np.testing.assert_equal(Qddot1.wddot, np.array([4, 5, 6]))
    np.testing.assert_equal(Qddot1.rdddot, np.array([7, 8, 9]))
    np.testing.assert_equal(Qddot1.rpddot, np.array([10, 11, 12]))
    np.testing.assert_equal(Qddot1.vector, Qddot1)

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

def test_concatenate_velocities():
    Qdot1 = SegmentNaturalVelocities.from_components(
        udot=np.array([1, 2, 3]), wdot=np.array([4, 5, 6]), rddot=np.array([7, 8, 9]), rpdot=np.array([10, 11, 12])
    )
    Qdot2 = SegmentNaturalVelocities.from_components(
        udot=np.array([11, 22, 33]), wdot=np.array([4, 5, 6]), rddot=np.array([7, 82, 9]), rpdot=np.array([110, 11, 12])
    )
    Qdot = np.concatenate((Qdot1, Qdot2), axis=0)
    with pytest.raises(AttributeError, match="'numpy.ndarray' object has no attribute 'udot'"):
        Qdot.udot

def test_concatenate_accelerations():
    Qddot1 = SegmentNaturalAccelerations.from_components(
        uddot=np.array([1, 2, 3]), wddot=np.array([4, 5, 6]), rdddot=np.array([7, 8, 9]), rpddot=np.array([10, 11, 12])
    )
    Qddot2 = SegmentNaturalAccelerations.from_components(
        uddot=np.array([11, 22, 33]), wddot=np.array([4, 5, 6]), rdddot=np.array([7, 82, 9]), rpddot=np.array([110, 11, 12])
    )
    Qddot = np.concatenate((Qddot1, Qddot2), axis=0)
    with pytest.raises(AttributeError, match="'numpy.ndarray' object has no attribute 'uddot'"):
        Qddot.uddot


# Build a class called GeneralizedCoordinates to handle the concatenation of SegmentGeneralizedCoordinates
def test_NaturalCoordinatesConstructor():
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


# do the same tests for NaturalVelocities and SegmentNaturalVelocities
def test_NaturalVelocitiesConstructor():
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


# do the same tests for NaturalAccelerations and SegmentNaturalAccelerations
def test_NaturalAccelerationsConstructor():
    Qddot1 = SegmentNaturalAccelerations.from_components(
        uddot=np.array([1, 2, 3]), wddot=np.array([4, 5, 6]), rdddot=np.array([7, 8, 9]), rpddot=np.array([10, 11, 12])
    )
    Qddot2 = SegmentNaturalAccelerations.from_components(
        uddot=np.array([11, 22, 33]), wddot=np.array([4, 5, 6]), rdddot=np.array([7, 82, 9]), rpddot=np.array([110, 11, 12])
    )
    Qddot = NaturalAccelerations.from_Qddoti((Qddot1, Qddot2))
    np.testing.assert_equal(Qddot.uddot(0), np.array([1, 2, 3]))
    np.testing.assert_equal(Qddot.uddot(1), np.array([11, 22, 33]))
    np.testing.assert_equal(Qddot.vector(0), Qddot1)
    np.testing.assert_equal(Qddot.vector(1), Qddot2)
    np.testing.assert_equal(Qddot.vector(0).uddot, np.array([1, 2, 3]))
    np.testing.assert_equal(Qddot.vector(1).uddot, np.array([11, 22, 33]))
    np.testing.assert_equal(Qddot.nb_Qddoti(), 2)

