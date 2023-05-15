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
)

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


def inverse_dynamics(model, Q, Qdot, Qddot, external_forces=None):
    """
    This function computes the inverse dynamics of a model.

    Parameters
    ----------
    model: BiomechanicalModel
        The model to be simulated
    Q: NaturalCoordinates
        The generalized coordinates of the model
    Qdot: NaturalVelocities
        The generalized velocities of the model
    Qddot: NaturalAccelerations
        The generalized accelerations of the model
    external_forces: ExternalForceList
        The external forces applied to the model

    Returns
    -------
    GeneralizedForces
        The generalized forces that makes the model move like this
    """

    _, intersegmental_generalized_forces = inverse_dynamics_recursive_step(model, Q, Qdot, Qddot, external_forces, 0)


def depth_first_search(model, segment_index, visited_segments=None):
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
    for child in model.segments[segment_index].childs:
        if visited_segments[child]:
            raise RuntimeError("The model contain closed loops, we cannot use this algorithm")
        if not visited_segments[child]:
            visited_segments = (model, child, visited_segments)

    return visited_segments


def inverse_dynamics_recursive_step(
    model: BiomechanicalModel,
    Q: NaturalCoordinates,
    Qdot: NaturalVelocities,
    Qddot: NaturalAccelerations,
    external_forces: ExternalForceList,
    segment_index: int,
    visited_segments: list[bool, ...] = None,
    intersegmental_generalized_forces: np.ndarray = None,
):
    """
    This function returns the segments in a depth first search order.

    Parameters
    ----------
    model: BiomechanicalModel
        The model to be simulated
    Q: NaturalCoordinates
        The generalized coordinates of the model
    Qdot: NaturalVelocities
        The generalized velocities of the model
    Qddot: NaturalAccelerations
        The generalized accelerations of the model
    external_forces: ExternalForceList
        The external forces applied to the model
    segment_index: int
        The index of the segment to start the search from
    visited_segments: list[bool]
        The segments already visited

    Returns
    -------
    list[Segment]
        The segments in a depth first search order
    """
    if visited_segments is None:
        visited_segments = [False for _ in range(model.nb_segments)]
        intersegmental_generalized_forces = np.zeros((6, model.nb_joints))

    Qi = Q.vector(segment_index)
    Qdoti = Qdot.vector(segment_index)
    Qddoti = Qddot.vector(segment_index)
    external_forces_i = external_forces.segment_external_forces(segment_index)

    visited_segments[segment_index] = True
    subtree_intersegmental_generalized_forces = np.zeros((6, model.nb_joints))
    for child in model.segments[segment_index].childs:
        if visited_segments[child]:
            raise RuntimeError("The model contain closed loops, we cannot use this algorithm")
        if not visited_segments[child]:
            visited_segments, intersegmental_generalized_forces = inverse_dynamics_recursive_step(
                model,
                Q,
                Qdot,
                Qddot,
                external_forces,
                child,
                visited_segments,
                intersegmental_generalized_forces,
            )
        # transport the generalized forces from the child to the parent proximal point
        intersegmental_generalized_forces[:, segment_index] = intersegmental_generalized_forces.transport_to(
            segment_index=segment_index,
            # application_point=TODO:DO IT
        )
        # sum the generalized forces of each subsegments
        intersegmental_generalized_forces[:, segment_index] += intersegmental_generalized_forces[:, child]
    # add the external forces applied to this segment
    if external_forces is not None:
        intersegmental_generalized_forces[:, segment_index] += external_forces[segment_index]

    # model.segments(segment_index).gravity_force(Qi)
    # intersegmental_generalized_forces[:, segment_index]
    #  = interpolation_matrix_proximal_transpose

    # todo: to pursue

    return visited_segments, intersegmental_generalized_forces


def _one_segment_inverse_dynamics(
    segment,
    Qi: SegmentNaturalCoordinates,
    Qddoti: SegmentNaturalAccelerations,
    segment_external_forces: ExternalForce,
):
    """
    This function computes the inverse dynamics of a segment.

    Parameters
    ----------
    segment
    Qi
    Qddoti
    segment_external_forces

    Returns
    -------

    """

    proximal_interpolation_matrix = NaturalVector.proximal().interpolate()
    pseudo_interpolation_matrix = Qi.compute_pseudo_interpolation_matrix()
    rigid_body_constraints = segment.rigid_body_constraints(Q=Qi)

    # make a matrix out of it
    front_matrix = np.hstack((proximal_interpolation_matrix, pseudo_interpolation_matrix, -rigid_body_constraints))

    # compute the generalized forces
    generalized_forces = np.linalg.inv(front_matrix) @ (
        segment.mass_matrix @ Qddoti - segment.gravity_force() - segment_external_forces
    )
