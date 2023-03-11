from sys import platform
from casadi import sum1
import numpy as np
import pytest

from .utils import TestUtils


@pytest.mark.parametrize(
    "bionc_type",
    [
        "numpy",
        "casadi",
    ],
)
def test_ground_segment(bionc_type):
    if bionc_type == "numpy":
        from bionc.bionc_numpy import NaturalCoordinates, NaturalVelocities
    else:
        from bionc.bionc_casadi import NaturalCoordinates, NaturalVelocities

    bionc = TestUtils.bionc_folder()
    module = TestUtils.load_module(bionc + "/examples/forward_dynamics/n_link_pendulum.py")

    nb_segments = 4
    model = module.build_n_link_pendulum(nb_segments=nb_segments)
    if bionc_type == "casadi":
        model = model.to_mx()

    assert model.has_ground_segment == False
    assert len(model.segments_no_ground) == nb_segments
    assert model.nb_segments == nb_segments
    assert model.segments["pendulum_0"].index == 0
    assert model.segments["pendulum_1"].index == 1
    assert model.segments["pendulum_2"].index == 2
    assert model.segments["pendulum_3"].index == 3
    assert model.nb_joints == 4

    with pytest.raises(ValueError, match="The ground segment cannot be a parent or a child of a joint"):
        model.set_ground_segment(name="pendulum_0")

    # delete the joint
    model.remove_joint(name="hinge_0")
    assert model.nb_joints == 3
    with pytest.raises(ValueError, match="The ground segment cannot be a parent or a child of a joint"):
        model.set_ground_segment(name="pendulum_0")
    model.remove_joint(name="hinge_1")

    assert model.nb_joints == 2
    model.set_ground_segment(name="pendulum_0")

    assert model.has_ground_segment == True
    assert len(model.segments_no_ground) == nb_segments - 1
    assert model.nb_segments == nb_segments - 1
    assert model.segments["pendulum_0"].index == -1
    assert model.segments["pendulum_1"].index == 0
    assert model.segments["pendulum_2"].index == 1
    assert model.segments["pendulum_3"].index == 2

    Q = NaturalCoordinates(np.linspace(0, 0.24, 3 * 12))
    Qdot = NaturalVelocities(np.linspace(0, 0.02, 3 * 12))

    assert model.rigid_body_constraints(Q).shape[0] == 6 * 3
    assert model.rigid_body_constraints_derivative(Q, Qdot).shape[0] == 6 * 3
    assert model.rigid_body_constraints_jacobian(Q).shape == (6 * 3, 12 * 3)
    assert model.rigid_body_constraint_jacobian_derivative(Qdot).shape == (6 * 3, 12 * 3)

    assert model.joint_constraints(Q).shape[0] == 2 * 5  # (five constraints for a hinge)
    assert model.joint_constraints_jacobian_derivative(Qdot).shape == (2 * 5, 12 * 3)
    assert model.joint_constraints_jacobian(Q).shape == (2 * 5, 12 * 3)

    assert model.mass_matrix.shape == (12 * 3, 12 * 3)

    assert model.markers(Q).shape[0] == 3

    model["pendulum_0"].add_natural_marker_from_segment_coordinates("m0", np.array([0, 0, 0]))
    model["pendulum_1"].add_natural_marker_from_segment_coordinates("m1", np.array([0, 0, 0]))
    model["pendulum_2"].add_natural_marker_from_segment_coordinates("m2", np.array([0, 0, 0]))

    assert model.markers(Q).shape[0] == 3
    assert model.markers(Q).shape[1] == 2
    TestUtils.assert_equal(model.potential_energy(Q), 0.34354285714285715)
    assert model.center_of_mass_position(Q).shape[0:2] == (3, 3)
    assert model.markers_constraints(markers=np.ones((3, 2)), Q=Q).shape[0] == 6
    assert model.markers_constraints_jacobian().shape == (6, 12 * 3)
    assert model.holonomic_constraints(Q=Q).shape == (6 * 3 + 10, 1)
    assert model.holonomic_constraints_jacobian(Q=Q).shape == (6 * 3 + 10, 12 * 3)
    assert model.holonomic_constraints_jacobian_derivative(Qdot=Qdot).shape == (6 * 3 + 10, 12 * 3)
    if bionc_type == "casadi":
        TestUtils.assert_equal(sum1(model.weight()), np.array([-29.43]))
    else:
        TestUtils.assert_equal(sum(model.weight()), np.array([-29.43]))
