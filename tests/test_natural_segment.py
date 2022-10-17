from bionc import (
    NaturalSegment,
)
import numpy as np
import pytest


def test_natural_segment():
    # Let's create a segment
    my_segment = NaturalSegment(
        name="box",
        alpha=np.pi / 2,
        beta=np.pi / 2,
        gamma=np.pi / 2,
        length=1,
        mass=1,
        center_of_mass=np.array([0, 0.01, 0]),
        inertia=np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
    )
    # test the name of the segment
    assert my_segment.name == "box"
    np.testing.assert_equal(my_segment.alpha, np.pi / 2)
    np.testing.assert_equal(my_segment.beta, np.pi / 2)
    np.testing.assert_equal(my_segment.gamma, np.pi / 2)
    np.testing.assert_equal(my_segment.length, 1)
    np.testing.assert_equal(my_segment.mass, 1)
    np.testing.assert_equal(my_segment.center_of_mass, np.array([0, 0.01, 0]))
    np.testing.assert_equal(my_segment.inertia, np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]))

    # todo: test marker constraints
