from casadi import MX, solve, transpose
import numpy as np

from .external_force_global_on_proximal import ExternalForceInGlobalOnProximal
from .natural_coordinates import SegmentNaturalCoordinates, NaturalCoordinates
from ..protocols.joint import JointBase as Joint


def _dual(axes: MX, values: MX) -> MX:
    """The vector x such that axes^T x = values with the least norm, x = axes (axes^T axes)^-1 values"""
    return axes @ solve(transpose(axes) @ axes, values)


class JointGeneralizedForces(ExternalForceInGlobalOnProximal):
    """
    The force and torque of a joint actuator, in the global frame, see bionc_numpy.generalized_force.
    """

    @classmethod
    def from_joint_generalized_forces(
        cls,
        joint: Joint,
        generalized_forces: MX,
        Q_parent: SegmentNaturalCoordinates,
        Q_child: SegmentNaturalCoordinates,
    ):
        """
        Create the actuator force and torque from one value per joint degree of freedom, forces along the
        translation axes then torques about the rotation axes of the joint (see joint.dof_axes), dual basis.
        """
        generalized_forces = MX(generalized_forces)
        translation_axes, rotation_axes = joint.dof_axes(Q_parent, Q_child)
        nb_translations = 0 if translation_axes is None else translation_axes.shape[1]

        force = (
            MX.zeros(3, 1) if nb_translations == 0 else _dual(translation_axes, generalized_forces[:nb_translations])
        )
        torque = _dual(rotation_axes, generalized_forces[nb_translations:])
        return cls.from_components(force=force, torque=torque)

    def to_natural_joint_forces(self, Q_parent: SegmentNaturalCoordinates, Q_child: SegmentNaturalCoordinates):
        """
        Returns
        -------
        tuple[MX, MX]
            The natural generalized forces on the parent (None for a joint with the ground) and on the child [12 x 1]
        """
        on_child = self.to_generalized_natural_forces(Q_child)
        if Q_parent is None:
            return None, on_child

        reaction = ExternalForceInGlobalOnProximal.from_components(force=-self.force, torque=-self.torque)
        on_parent = reaction.transport_to_another_segment(Qfrom=Q_child, Qto=Q_parent).to_generalized_natural_forces(
            Q_parent
        )
        return on_parent, on_child


class JointGeneralizedForcesList:
    """
    The joint actuators of a model, one list of JointGeneralizedForces per joint,
    see bionc_numpy.generalized_force.JointGeneralizedForcesList.
    """

    def __init__(self, joint_generalized_forces: list[list[JointGeneralizedForces, ...]] = None):
        if joint_generalized_forces is None:
            raise ValueError(
                "joint_generalized_forces must be a list of JointGeneralizedForces, or use the classmethod"
                "JointGeneralizedForcesList.empty_from_nb_joint(nb_joint)"
            )
        self.joint_generalized_forces = joint_generalized_forces

    @property
    def nb_joint(self) -> int:
        """Returns the number of joints"""
        return len(self.joint_generalized_forces)

    @classmethod
    def empty_from_nb_joint(cls, nb_joint: int):
        """Create an empty JointGeneralizedForcesList from the number of joints"""
        return cls(joint_generalized_forces=[[] for _ in range(nb_joint)])

    def joint_generalized_force(self, joint_index: int) -> list[JointGeneralizedForces]:
        """Returns the generalized forces of the joint"""
        return self.joint_generalized_forces[joint_index]

    def add_generalized_force(self, joint_index: int, joint_generalized_force: JointGeneralizedForces):
        """Add a generalized force to the joint"""
        self.joint_generalized_forces[joint_index].append(joint_generalized_force)

    def add_all_joint_generalized_forces(
        self, model: "BiomechanicalModel", joint_generalized_forces: MX | np.ndarray, Q: NaturalCoordinates
    ):
        """
        Split the generalized forces of the model, one value per joint degree of freedom, joint after joint
        (ordered by joint index), into the generalized forces of each joint.
        """
        is_numeric = not isinstance(joint_generalized_forces, MX)
        if is_numeric:
            joint_generalized_forces = np.asarray(joint_generalized_forces, dtype=float).reshape(-1)
        if joint_generalized_forces.shape[0] != model.nb_joint_dof:
            raise ValueError(
                f"joint_generalized_forces must have {model.nb_joint_dof} values (one per joint dof), "
                f"got {joint_generalized_forces.shape[0]}"
            )

        first_dof = 0
        for joint in sorted(model.joints.values(), key=lambda j: j.index):
            generalized_forces = joint_generalized_forces[first_dof : first_dof + joint.nb_joint_dof]
            first_dof += joint.nb_joint_dof
            if joint.nb_joint_dof == 0 or (is_numeric and not np.any(generalized_forces)):
                continue
            Q_parent = None if joint.parent is None else Q.vector(joint.parent.index)
            self.add_generalized_force(
                joint.index,
                JointGeneralizedForces.from_joint_generalized_forces(
                    joint, generalized_forces, Q_parent, Q.vector(joint.child.index)
                ),
            )

    def to_natural_joint_forces(self, model: "BiomechanicalModel", Q: NaturalCoordinates) -> MX:
        """
        Converts and sums the joint generalized forces into the natural generalized forces of the model
        [12 * nb_segments, 1]
        """
        if self.nb_joint != model.nb_joints:
            raise ValueError(f"Got {self.nb_joint} joint generalized forces for {model.nb_joints} joints in the model")

        natural_joint_forces = MX.zeros(12 * Q.nb_qi(), 1)
        for joint_index, joint_generalized_forces in enumerate(self.joint_generalized_forces):
            joint = model.joint_from_index(joint_index)
            child_index = joint.child.index
            Q_parent = None if joint.parent is None else Q.vector(joint.parent.index)
            for joint_generalized_force in joint_generalized_forces:
                on_parent, on_child = joint_generalized_force.to_natural_joint_forces(Q_parent, Q.vector(child_index))
                natural_joint_forces[12 * child_index : 12 * (child_index + 1)] += on_child
                if on_parent is not None:
                    parent_index = joint.parent.index
                    natural_joint_forces[12 * parent_index : 12 * (parent_index + 1)] += on_parent

        return natural_joint_forces
