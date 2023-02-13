import numpy as np
from bionc import (
    bionc_casadi as bionc_mx,
)

from .utils import TestUtils


def test_SegmentNaturalCoordinates_casadi():
    Qi = bionc_mx.SegmentNaturalCoordinates.from_components(
        u=np.array([0, 0, 0]),
        rp=np.array([4, 5, 6]),
        rd=np.array([7, 8, 9]),
        w=np.array([10, 11, 12]),
    )
    np.testing.assert_equal(TestUtils.mx_to_array(Qi.u), np.array([0, 0, 0]))
    np.testing.assert_equal(TestUtils.mx_to_array(Qi.rp), np.array([4, 5, 6]))
    np.testing.assert_equal(TestUtils.mx_to_array(Qi.rd), np.array([7, 8, 9]))
    np.testing.assert_equal(TestUtils.mx_to_array(Qi.w), np.array([10, 11, 12]))
    np.testing.assert_equal(TestUtils.mx_to_array(Qi.v), -np.array([7, 8, 9]) + np.array([4, 5, 6]))
    np.testing.assert_equal(TestUtils.mx_to_array(Qi.vector), TestUtils.mx_to_array(Qi))


def test_NaturalAccelerationsCreator_casadi():
    Qddot1 = bionc_mx.SegmentNaturalAccelerations.from_components(
        uddot=np.array([1, 2, 3]),
        wddot=np.array([4, 5, 6]),
        rdddot=np.array([7, 8, 9]),
        rpddot=np.array([10, 11, 12]),
    )
    np.testing.assert_equal(TestUtils.mx_to_array(Qddot1.uddot), np.array([1, 2, 3]))
    np.testing.assert_equal(TestUtils.mx_to_array(Qddot1.wddot), np.array([4, 5, 6]))
    np.testing.assert_equal(TestUtils.mx_to_array(Qddot1.rdddot), np.array([7, 8, 9]))
    np.testing.assert_equal(TestUtils.mx_to_array(Qddot1.rpddot), np.array([10, 11, 12]))
    np.testing.assert_equal(TestUtils.mx_to_array(Qddot1.vector), TestUtils.mx_to_array(Qddot1))


def test_NaturalCoordinates_casadi():
    Q1 = bionc_mx.SegmentNaturalCoordinates.from_components(
        u=np.array([1, 2, 3]),
        rp=np.array([4, 5, 6]),
        rd=np.array([7, 8, 9]),
        w=np.array([10, 11, 12]),
    )
    Q2 = bionc_mx.SegmentNaturalCoordinates.from_components(
        u=np.array([11, 22, 33]),
        rp=np.array([4, 5, 6]),
        rd=np.array([7, 8, 9]),
        w=np.array([10, 11, 12]),
    )
    Q = bionc_mx.NaturalCoordinates.from_qi((Q1, Q2))
    np.testing.assert_equal(TestUtils.mx_to_array(Q.u(0)), np.array([1, 2, 3]))
    np.testing.assert_equal(TestUtils.mx_to_array(Q.u(1)), np.array([11, 22, 33]))
    np.testing.assert_equal(TestUtils.mx_to_array(Q.v(0)), -np.array([7, 8, 9]) + np.array([4, 5, 6]))
    np.testing.assert_equal(TestUtils.mx_to_array(Q.v(1)), -np.array([7, 8, 9]) + np.array([4, 5, 6]))
    np.testing.assert_equal(TestUtils.mx_to_array(Q.vector(0)), TestUtils.mx_to_array(Q1))
    np.testing.assert_equal(TestUtils.mx_to_array(Q.vector(1)), TestUtils.mx_to_array(Q2))
    np.testing.assert_equal(TestUtils.mx_to_array(Q.vector(0).u), np.array([1, 2, 3]))
    np.testing.assert_equal(TestUtils.mx_to_array(Q.vector(1).u), np.array([11, 22, 33]))
    np.testing.assert_equal(TestUtils.mx_to_array(Q.vector(0).v), -np.array([7, 8, 9]) + np.array([4, 5, 6]))
    np.testing.assert_equal(TestUtils.mx_to_array(Q.vector(1).v), -np.array([7, 8, 9]) + np.array([4, 5, 6]))
    np.testing.assert_equal(Q.nb_qi(), 2)


def test_NaturalAccelerations_casadi():
    Qddot1 = bionc_mx.SegmentNaturalAccelerations.from_components(
        uddot=np.array([1, 2, 3]),
        wddot=np.array([4, 5, 6]),
        rdddot=np.array([7, 8, 9]),
        rpddot=np.array([10, 11, 12]),
    )
    Qddot2 = bionc_mx.SegmentNaturalAccelerations.from_components(
        uddot=np.array([11, 22, 33]),
        wddot=np.array([4, 5, 6]),
        rdddot=np.array([7, 82, 9]),
        rpddot=np.array([110, 11, 12]),
    )
    Qddot = bionc_mx.NaturalAccelerations.from_qddoti((Qddot1, Qddot2))
    np.testing.assert_equal(TestUtils.mx_to_array(Qddot.uddot(0)), np.array([1, 2, 3]))
    np.testing.assert_equal(TestUtils.mx_to_array(Qddot.uddot(1)), np.array([11, 22, 33]))
    np.testing.assert_equal(TestUtils.mx_to_array(Qddot.vector(0)), TestUtils.mx_to_array(Qddot1))
    np.testing.assert_equal(TestUtils.mx_to_array(Qddot.vector(1)), TestUtils.mx_to_array(Qddot2))
    np.testing.assert_equal(TestUtils.mx_to_array(Qddot.vector(0).uddot), np.array([1, 2, 3]))
    np.testing.assert_equal(TestUtils.mx_to_array(Qddot.vector(1).uddot), np.array([11, 22, 33]))
    np.testing.assert_equal(Qddot.nb_qddoti(), 2)


def test_segment_natural_vectors_sym():
    Q = bionc_mx.SegmentNaturalCoordinates.sym()
    Qdot = bionc_mx.SegmentNaturalVelocities.sym()
    Qddot = bionc_mx.SegmentNaturalAccelerations.sym()

    from casadi import Function

    f = Function("f", [Q, Qdot, Qddot], [Q, Qdot, Qddot])
    q_num = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    qdot_num = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]) * 2
    qddot_num = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]) * 3

    TestUtils.assert_equal(f(q_num, qdot_num, qddot_num)[0], q_num[:, np.newaxis])
    TestUtils.assert_equal(f(q_num, qdot_num, qddot_num)[1], qdot_num[:, np.newaxis])
    TestUtils.assert_equal(f(q_num, qdot_num, qddot_num)[2], qddot_num[:, np.newaxis])


def test_natural_vectors_sym():
    Q = bionc_mx.NaturalCoordinates.sym(2)
    Qdot = bionc_mx.NaturalVelocities.sym(2)
    Qddot = bionc_mx.NaturalAccelerations.sym(2)

    from casadi import Function

    f = Function("f", [Q, Qdot, Qddot], [Q, Qdot, Qddot])
    q_num = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 11, 22, 33, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    qdot_num = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 11, 22, 33, 4, 5, 6, 7, 8, 9, 10, 11, 12]) * 2
    qddot_num = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 11, 22, 33, 4, 5, 6, 7, 8, 9, 10, 11, 12]) * 3

    TestUtils.assert_equal(f(q_num, qdot_num, qddot_num)[0], q_num[:, np.newaxis])
    TestUtils.assert_equal(f(q_num, qdot_num, qddot_num)[1], qdot_num[:, np.newaxis])
    TestUtils.assert_equal(f(q_num, qdot_num, qddot_num)[2], qddot_num[:, np.newaxis])
