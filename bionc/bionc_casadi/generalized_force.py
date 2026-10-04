from casadi import MX, horzcat, norm_2, solve, transpose
import numpy as np

from .external_force_global_local_point import ExternalForceInGlobalLocalPoint
from .natural_coordinates import SegmentNaturalCoordinates, NaturalCoordinates
from ..utils.enums import CartesianAxis, EulerSequence
from .rotations import euler_axes_from_rotation_matrices
from ..protocols.joint import JointBase as Joint


class JointGeneralizedForces(ExternalForceInGlobalLocalPoint):
    # todo: this class need to be rethinked, the inheritance is not optimal.
    #  but preserved for now as refactoring externalforces
    """
    Made to handle joint generalized forces, it inherits from ExternalForce

    Attributes
    ----------
    external_forces : np.ndarray
        The external forces
    application_point_in_local : np.ndarray
        The application point in local coordinates

    Methods
    -------
    from_joint_generalized_forces(forces, torques, translation_dof, rotation_dof, joint, Q_parent, Q_child)
        This function creates a JointGeneralizedForces from the forces and torques

    Notes
    -----
    The application point of torques is set to the proximal point of the child.
    """

    def __init__(
        self,
        external_forces: MX,
        application_point_in_local: MX,
    ):
        super().__init__(external_forces=external_forces, application_point_in_local=application_point_in_local)

        # TODO : Not impletemented at all, but only for numpy, need a revision first.


def _unit(vector: MX) -> MX:
    vector = MX(vector)
    return vector / norm_2(vector)


def joint_dof_axes(joint: Joint, Q_parent: SegmentNaturalCoordinates, Q_child: SegmentNaturalCoordinates):
    """
    Global axes of the degrees of freedom of a joint, translations first, then rotations,
    see bionc_numpy.generalized_force.joint_dof_axes.

    Returns
    -------
    tuple[MX | None, MX]
        translation axes [3 x nb_translations] (None if none), rotation axes [3 x nb_rotations]
    """
    from .joints import Joint as TwoSegmentJoint
    from .joints_with_ground import GroundJoint
    from .cartesian_vector import CartesianVector

    def parent_rotation():
        if joint.parent is None:
            return MX.eye(3)
        return joint.parent.segment_coordinates_system(Q_parent, joint.parent_basis).rot

    def euler_axes():
        if joint.projection_basis is None:
            raise ValueError(f"The joint {joint.name} needs a projection_basis to be actuated")
        child_rotation = joint.child.segment_coordinates_system(Q_child, joint.child_basis).rot
        axes = euler_axes_from_rotation_matrices(parent_rotation(), child_rotation, sequence=joint.projection_basis)
        return horzcat(*[_unit(axis) for axis in axes])

    def parent_axis(axis):
        if joint.parent is None:
            return _unit(np.asarray(CartesianVector.axis(axis)).reshape(3))
        return _unit(Q_parent.axis(axis))

    if isinstance(joint, (TwoSegmentJoint.Hinge, GroundJoint.Hinge)):
        if joint.parent_axis[0] != joint.parent_axis[1]:
            raise NotImplementedError(f"The hinge {joint.name} must use the same parent axis in both constraints")
        return None, parent_axis(joint.parent_axis[0])
    if isinstance(joint, (TwoSegmentJoint.Universal, GroundJoint.Universal)):
        return None, horzcat(parent_axis(joint.parent_axis), _unit(Q_child.axis(joint.child_axis)))
    if isinstance(joint, (TwoSegmentJoint.Spherical, GroundJoint.Spherical)):
        return None, euler_axes()
    if isinstance(joint, (TwoSegmentJoint.Free, GroundJoint.Free)):
        return parent_rotation(), euler_axes()
    raise NotImplementedError(f"Joint generalized forces are not implemented for the joint {joint.name}")


def _dual(axes: MX, values: MX) -> MX:
    """The vector x such that axes^T x = values with the least norm: x = axes (axes^T axes)^-1 values"""
    return axes @ solve(transpose(axes) @ axes, values)


def natural_joint_forces(model, Q: NaturalCoordinates, joint_generalized_forces) -> MX:
    """
    Natural generalized forces of the joint actuators [12 * nb_segments, 1],
    see bionc_numpy.generalized_force.natural_joint_forces.
    """
    from .external_force_global_on_proximal import ExternalForceInGlobalOnProximal

    joint_generalized_forces = MX(joint_generalized_forces)
    if joint_generalized_forces.shape[0] != model.nb_joint_dof:
        raise ValueError(
            f"joint_generalized_forces must have {model.nb_joint_dof} values (one per joint dof), "
            f"got {joint_generalized_forces.shape[0]}"
        )

    natural_forces = MX.zeros(12 * Q.nb_qi(), 1)
    first_dof = 0
    for joint in sorted(model.joints.values(), key=lambda j: j.index):
        q = joint_generalized_forces[first_dof : first_dof + joint.nb_joint_dof]
        first_dof += joint.nb_joint_dof
        if joint.nb_joint_dof == 0:
            continue

        child_index = joint.child.index
        Q_child = Q.vector(child_index)
        Q_parent = None if joint.parent is None else Q.vector(joint.parent.index)
        translation_axes, rotation_axes = joint_dof_axes(joint, Q_parent, Q_child)
        nb_translations = 0 if translation_axes is None else translation_axes.shape[1]
        force = MX.zeros(3, 1) if nb_translations == 0 else _dual(translation_axes, q[:nb_translations])
        torque = _dual(rotation_axes, q[nb_translations:])

        on_child = ExternalForceInGlobalOnProximal.from_components(force=force, torque=torque)
        natural_forces[12 * child_index : 12 * child_index + 12] += on_child.to_generalized_natural_forces(Q_child)
        if joint.parent is not None:
            parent_index = joint.parent.index
            on_parent = ExternalForceInGlobalOnProximal.from_components(force=-force, torque=-torque)
            natural_forces[12 * parent_index : 12 * parent_index + 12] += on_parent.transport_to_another_segment(
                Qfrom=Q_child, Qto=Q_parent
            ).to_generalized_natural_forces(Q_parent)

    return natural_forces
