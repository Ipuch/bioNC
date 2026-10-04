import numpy as np

from .external_force_global_on_proximal import ExternalForceInGlobalOnProximal
from .natural_coordinates import SegmentNaturalCoordinates, NaturalCoordinates
from ..protocols.joint import JointBase as Joint


def _dual(axes: np.ndarray, values: np.ndarray) -> np.ndarray:
    """The vector x such that axes^T x = values with the least norm, x = axes (axes^T axes)^-1 values"""
    return axes @ np.linalg.solve(axes.T @ axes, values)


class JointGeneralizedForces(ExternalForceInGlobalOnProximal):
    """
    The force and torque of a joint actuator, in the global frame. The child segment receives them at its proximal
    point, the parent receives the opposite (action-reaction).
    """

    @classmethod
    def from_joint_generalized_forces(
        cls,
        joint: Joint,
        generalized_forces: np.ndarray,
        Q_parent: SegmentNaturalCoordinates,
        Q_child: SegmentNaturalCoordinates,
    ):
        """
        Create the actuator force and torque from one value per joint degree of freedom, forces along the
        translation axes then torques about the rotation axes of the joint (see joint.dof_axes).

        With E the dof axes, the torque solves E^T tau = q (dual basis), so that tau . omega_relative = sum(q_i
        dtheta_i/dt) even for non orthogonal Euler axes. Same for the force.

        Parameters
        ----------
        joint: Joint
            The joint
        generalized_forces: np.ndarray
            The generalized forces of the joint [nb_joint_dof]
        Q_parent: SegmentNaturalCoordinates
            The natural coordinates of the parent segment, None for a joint with the ground
        Q_child: SegmentNaturalCoordinates
            The natural coordinates of the child segment
        """
        generalized_forces = np.asarray(generalized_forces, dtype=float).reshape(joint.nb_joint_dof)
        translation_axes, rotation_axes = joint.dof_axes(Q_parent, Q_child)
        nb_translations = 0 if translation_axes is None else translation_axes.shape[1]

        force = np.zeros(3) if nb_translations == 0 else _dual(translation_axes, generalized_forces[:nb_translations])
        torque = _dual(rotation_axes, generalized_forces[nb_translations:])
        return cls.from_components(force=force, torque=torque)

    def to_natural_joint_forces(
        self, Q_parent: SegmentNaturalCoordinates, Q_child: SegmentNaturalCoordinates
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Converts the actuator force and torque to natural generalized forces

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            The natural generalized forces on the parent (None for a joint with the ground) and on the child [12]
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
    The joint actuators of a model, one list of JointGeneralizedForces per joint. Their natural generalized forces
    enter the equation of motion as:

    G @ Qddot + K^T @ lambda = gravity_forces + f_ext + joint_forces

    joint_forces being [nb_segments x 12, 1]

    Attributes
    ----------
    joint_generalized_forces : list
        List of JointGeneralizedForces for each joint.
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
        """
        Create an empty JointGeneralizedForcesList from the number of joints
        """
        return cls(joint_generalized_forces=[[] for _ in range(nb_joint)])

    def joint_generalized_force(self, joint_index: int) -> list[JointGeneralizedForces]:
        """Returns the generalized forces of the joint"""
        return self.joint_generalized_forces[joint_index]

    def add_generalized_force(self, joint_index: int, joint_generalized_force: JointGeneralizedForces):
        """
        Add a generalized force to the joint

        Parameters
        ----------
        joint_index: int
            The index of the joint
        joint_generalized_force:
            The joint_generalized_force to add
        """
        self.joint_generalized_forces[joint_index].append(joint_generalized_force)

    def add_all_joint_generalized_forces(
        self, model: "BiomechanicalModel", joint_generalized_forces: np.ndarray, Q: NaturalCoordinates
    ):
        """
        Split the generalized forces of the model into the generalized forces of each joint.

        Parameters
        ----------
        model: BiomechanicalModel
            The model of the system
        joint_generalized_forces: np.ndarray
            One value per joint degree of freedom, joint after joint, see model.joint_dof_indexes [nb_joint_dof]
        Q: NaturalCoordinates
            The natural coordinates of the model
        """
        joint_generalized_forces = np.asarray(joint_generalized_forces, dtype=float).reshape(-1)
        if joint_generalized_forces.shape[0] != model.nb_joint_dof:
            raise ValueError(
                f"joint_generalized_forces must have {model.nb_joint_dof} values (one per joint dof), "
                f"got {joint_generalized_forces.shape[0]}"
            )

        for joint in model.joints.values():
            dof_indexes = model.joint_dof_indexes(joint.index)
            generalized_forces = joint_generalized_forces[dof_indexes[0] : dof_indexes[-1] + 1] if dof_indexes else []
            if not np.any(generalized_forces):
                continue
            Q_parent = None if joint.parent is None else Q.vector(joint.parent.index)
            self.add_generalized_force(
                joint.index,
                JointGeneralizedForces.from_joint_generalized_forces(
                    joint, generalized_forces, Q_parent, Q.vector(joint.child.index)
                ),
            )

    def to_natural_joint_forces(self, model: "BiomechanicalModel", Q: NaturalCoordinates) -> np.ndarray:
        """
        Converts and sums the joint generalized forces into the natural generalized forces of the model

        Parameters
        ----------
        model: BiomechanicalModel
            The biomechanical model
        Q : NaturalCoordinates
            The natural coordinates of the model

        Returns
        -------
        np.ndarray
            The natural generalized forces [12 * nb_segments, 1]
        """
        if self.nb_joint != model.nb_joints:
            raise ValueError(f"Got {self.nb_joint} joint generalized forces for {model.nb_joints} joints in the model")

        natural_joint_forces = np.zeros((12 * Q.nb_qi(), 1))
        for joint_index, joint_generalized_forces in enumerate(self.joint_generalized_forces):
            joint = model.joint_from_index(joint_index)
            child_index = joint.child.index
            Q_parent = None if joint.parent is None else Q.vector(joint.parent.index)
            for joint_generalized_force in joint_generalized_forces:
                on_parent, on_child = joint_generalized_force.to_natural_joint_forces(Q_parent, Q.vector(child_index))
                natural_joint_forces[12 * child_index : 12 * (child_index + 1), 0] += on_child
                if on_parent is not None:
                    parent_index = joint.parent.index
                    natural_joint_forces[12 * parent_index : 12 * (parent_index + 1), 0] += on_parent

        return natural_joint_forces
