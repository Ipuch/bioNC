from bionc.bionc_numpy import (
    NaturalCoordinates,
    NaturalVelocities,
    NaturalAccelerations,
    ExternalForceList,
    BiomechanicalModel,
    SegmentNaturalCoordinates,
    SegmentNaturalAccelerations,
    NaturalVector,
    ExternalForce,
    NaturalSegment,
    JointType,
)

from bionc import NaturalAxis, CartesianAxis

from bionc import Viz
import numpy as np
from tests.utils import TestUtils

# import the lower limb model
bionc = TestUtils.bionc_folder()
module = TestUtils.load_module(bionc + "/examples/model_creation/right_side_lower_limb.py")

# Generate c3d file
filename = module.generate_c3d_file()
# Generate model
model = module.model_creation_from_measured_data(filename)
model_numpy = model

trees = model.segment_subtrees()
print(trees)


def inverse_dynamics(
    model: BiomechanicalModel,
    Q: NaturalCoordinates,
    Qddot: NaturalAccelerations,
    external_forces: ExternalForceList = None,
):
    """
    This function returns the forces, torques and lambdas computes through recursive Newton-Euler algorithm

    Parameters
    ----------
    model: BiomechanicalModel
        The model to be simulated
    Q: NaturalCoordinates
        The generalized coordinates of the model
    Qddot: NaturalAccelerations
        The generalized accelerations of the model
    external_forces: ExternalForceList
        The external forces applied to the model

    Returns
    -------
    torques: np.ndarray
        The intersegmental torques
    forces: np.ndarray
        The intersegmental forces
    lambdas: np.ndarray
        The lagrange multipliers due to rigid contacts constraints
    """

    if external_forces is None:
        external_forces = ExternalForceList.empty_from_nb_segment(model.nb_segments)
    else:
        if external_forces.nb_segments != model.nb_segments:
            raise ValueError(
                f"The number of segments in the model and the external forces must be the same:"
                f" segment number = {model.nb_segments}"
                f" external force size = {external_forces.nb_segments}"
            )

    if Q is None:
        raise ValueError(f"The generalized coordinates must be provided")
    if Q.nb_qi() != model.nb_segments:
        raise ValueError(
            f"The number of generalized coordinates in the model and the generalized coordinates must be the same:"
            f" model number = {model.nb_segments}"
            f" generalized coordinates size = {Q.nb_qi()}"
        )
    if Qddot is None:
        raise ValueError(f"The generalized accelerations must be provided")
    if Qddot.nb_qddoti() != model.nb_segments:
        raise ValueError(
            f"The number of generalized accelerations in the model and the generalized accelerations must be the same:"
            f" model number = {model.nb_segments}"
            f" generalized accelerations size = {Qddot.nb_qddoti()}"
        )

    # last check to verify that the model doesn't contain any closed loop
    _depth_first_search(model, 0, visited_segments=None)

    # NOTE: This won't work with two independent tree in the same model
    visited_segments = [False for _ in range(model.nb_segments)]
    torques = np.zeros((3, model.nb_segments))
    forces = np.zeros((3, model.nb_segments))
    lambdas = np.zeros((6, model.nb_segments))
    _, forces, torques, lambdas = inverse_dynamics_recursive_step(
        model=model,
        Q=Q,
        Qddot=Qddot,
        external_forces=external_forces,
        segment_index=0,
        visited_segments=visited_segments,
        torques=torques,
        forces=forces,
        lambdas=lambdas,
    )

    return torques, forces, lambdas


def _depth_first_search(model, segment_index, visited_segments=None):
    """
    This function returns the segments in a depth first search order.

    Parameters
    ----------
    model: BiomechanicalModel
        The model to be simulated
    segment_index: int
        The index of the segment to start the search from
    visited_segments: list[Segment]
        The segments already visited

    Returns
    -------
    list[Segment]
        The segments in a depth first search order
    """
    if visited_segments is None:
        visited_segments = [False for _ in range(model.nb_segments)]

    visited_segments[segment_index] = True
    for child_index in model.children(segment_index):
        if visited_segments[child_index]:
            raise RuntimeError("The model contain closed loops, we cannot use this algorithm")
        if not visited_segments[child_index]:
            visited_segments = _depth_first_search(model, child_index, visited_segments)

    return visited_segments


def inverse_dynamics_recursive_step(
    model: BiomechanicalModel,
    Q: NaturalCoordinates,
    Qddot: NaturalAccelerations,
    external_forces: ExternalForceList,
    segment_index: int = 0,
    visited_segments: list[bool, ...] = None,
    torques: np.ndarray = None,
    forces: np.ndarray = None,
    lambdas: np.ndarray = None,
):
    """
    This function returns the segments in a depth first search order.

    Parameters
    ----------
    model: BiomechanicalModel
        The model to be simulated
    Q: NaturalCoordinates
        The generalized coordinates of the model
    Qddot: NaturalAccelerations
        The generalized accelerations of the model
    external_forces: ExternalForceList
        The external forces applied to the model
    segment_index: int
        The index of the segment to start the search from
    visited_segments: list[bool]
        The segments already visited
    torques: np.ndarray
        The intersegmental torques applied to the segments
    forces: np.ndarray
        The intersegmental forces applied to the segments
    lambdas: np.ndarray
        The lagrange multipliers applied to the segments

    Returns
    -------
    list[Segment]
        The segments in a depth first search order
    """
    visited_segments[segment_index] = True

    Qi = Q.vector(segment_index)
    Qddoti = Qddot.vector(segment_index)
    external_forces_i = external_forces.to_segment_natural_external_forces(segment_index=segment_index, Q=Q)

    subtree_intersegmental_generalized_forces = np.zeros((12, 1))
    for child_index in model.children(segment_index):
        if visited_segments[child_index]:
            raise RuntimeError("The model contains closed loops, we cannot use this algorithm")
        if not visited_segments[child_index]:
            visited_segments, torques, forces, lambdas = inverse_dynamics_recursive_step(
                model,
                Q,
                Qddot,
                external_forces,
                child_index,
                visited_segments=visited_segments,
                torques=torques,
                forces=forces,
                lambdas=lambdas,
            )
        # sum the generalized forces of each subsegment and transport them to the parent proximal point
        intersegmental_generalized_forces = ExternalForce.from_components(
            application_point_in_local=[0, 0, 0], force=forces[:, child_index], torque=torques[:, child_index]
        )
        subtree_intersegmental_generalized_forces += intersegmental_generalized_forces.transport_to(
            to_segment_index=segment_index,
            new_application_point_in_local=[0, 0, 0],  # proximal point
            from_segment_index=child_index,
            Q=Q,
        )[:, np.newaxis]

    force_i, torque_i, lambda_i = _one_segment_inverse_dynamics(
        segment=model.segment_from_index(segment_index),
        Qi=Qi,
        Qddoti=Qddoti,
        subtree_intersegmental_generalized_forces=subtree_intersegmental_generalized_forces,
        segment_external_forces=external_forces_i,
    )
    # re-assigned the computed values to the output arrays
    torques[:, segment_index] = torque_i
    forces[:, segment_index] = force_i
    lambdas[:, segment_index] = lambda_i

    return visited_segments, torques, forces, lambdas


def _one_segment_inverse_dynamics(
    segment,
    Qi: SegmentNaturalCoordinates,
    Qddoti: SegmentNaturalAccelerations,
    subtree_intersegmental_generalized_forces: np.ndarray,
    segment_external_forces: np.ndarray,
):
    """
    This function computes the inverse dynamics of a segment.

    Parameters
    ----------
    segment: NaturalSegment
        The segment considered for the inverse dynamics
    Qi: SegmentNaturalCoordinates
        The generalized coordinates of the segment
    Qddoti: SegmentNaturalAccelerations
        The generalized accelerations of the segment
    subtree_intersegmental_generalized_forces : np.ndarray
        The generalized forces applied to the segment by its children
    segment_external_forces : np.ndarray
        The generalized forces applied to the segment by the external forces

    Returns
    -------
    force: np.ndarray
        The force generated by the segment
    torque: np.ndarray
        The torque generated by the segment
    lambdas: np.ndarray
        The forces generated by the rigid body constraints
    """

    proximal_interpolation_matrix = NaturalVector.proximal().interpolate()
    pseudo_interpolation_matrix = Qi.compute_pseudo_interpolation_matrix()
    rigid_body_constraints_jacobian = segment.rigid_body_constraint_jacobian(Qi=Qi)

    # make a matrix out of it
    front_matrix = np.hstack((proximal_interpolation_matrix.T, pseudo_interpolation_matrix.T, -rigid_body_constraints_jacobian.T))

    # compute the generalized forces
    generalized_forces = np.linalg.inv(front_matrix) @ (
            (segment.mass_matrix @ Qddoti)[:, np.newaxis]
        - segment.gravity_force()[:, np.newaxis]
        - segment_external_forces
        - subtree_intersegmental_generalized_forces
    )

    return generalized_forces[:3, 0], generalized_forces[3:6,0], generalized_forces[6:,0]


def build_n_link_pendulum(nb_segments: int = 1) -> BiomechanicalModel:
    """Build a n-link pendulum model"""
    if nb_segments < 1:
        raise ValueError("The number of segment must be greater than 1")
    # Let's create a model
    model = BiomechanicalModel()
    # number of segments
    # fill the biomechanical model with the segment
    for i in range(nb_segments):
        name = f"pendulum_{i}"
        model[name] = NaturalSegment(
            name=name,
            alpha=np.pi / 2,  # setting alpha, beta, gamma to pi/2 creates a orthogonal coordinate system
            beta=np.pi / 2,
            gamma=np.pi / 2,
            length=1,
            mass=1,
            center_of_mass=np.array([0, -0.5, 0]),  # in segment coordinates system
            inertia=np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),  # in segment coordinates system
        )
    # add a revolute joint (still experimental)
    # if you want to add a revolute joint,
    # you need to ensure that x is always orthogonal to u and v
    model._add_joint(
        dict(
            name="hinge_0",
            joint_type=JointType.GROUND_REVOLUTE,
            parent="GROUND",
            child="pendulum_0",
            parent_axis=[CartesianAxis.X, CartesianAxis.X],
            child_axis=[NaturalAxis.V, NaturalAxis.W],  # meaning we pivot around the cartesian x-axis
            theta=[np.pi / 2, np.pi / 2],
        )
    )
    for i in range(1, nb_segments):
        model._add_joint(
            dict(
                name=f"hinge_{i}",
                joint_type=JointType.REVOLUTE,
                parent=f"pendulum_{0}",
                child=f"pendulum_{i}",
                parent_axis=[NaturalAxis.U, NaturalAxis.U],
                child_axis=[NaturalAxis.V, NaturalAxis.W],
                theta=[np.pi / 2, np.pi / 2],
            )
        )

    return model


def main():

    nb_segments = 2

    model = build_n_link_pendulum(nb_segments=2)

    tuple_of_Q = [
        SegmentNaturalCoordinates.from_components(u=[1, 0, 0], rp=[0, -i, 0], rd=[0, -i - 1, 0], w=[0, 0, 1])
        for i in range(0, nb_segments)
    ]
    Q = NaturalCoordinates.from_qi(tuple(tuple_of_Q))

    tuple_of_Qddot = [
        SegmentNaturalAccelerations.from_components(uddot=[0, 0, 0], rpddot=[0, 0, 0], rdddot=[0, 0, 0], wddot=[0, 0, 0])
        for i in range(0, nb_segments)
    ]
    Qddot = NaturalAccelerations.from_qddoti(tuple(tuple_of_Qddot))

    torques, forces, lambdas = inverse_dynamics(model=model, Q=Q, Qddot=Qddot)

    print(torques)
    print(forces)
    print(lambdas)


if __name__ == "__main__":
    main()
