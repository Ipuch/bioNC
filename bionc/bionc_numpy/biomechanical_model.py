import numpy as np
from numpy import transpose
from typing import Any

from .biomechanical_model_joints import BiomechanicalModelJoints
from .biomechanical_model_markers import BiomechanicalModelMarkers
from .biomechanical_model_segments import BiomechanicalModelSegments
from .cartesian_vector import vector_projection_in_non_orthogonal_basis
from .external_force import ExternalForceSet, ExternalForce
from .generalized_force import JointGeneralizedForcesList
from .inverse_kinematics import InverseKinematics
from .natural_accelerations import NaturalAccelerations
from .natural_coordinates import NaturalCoordinates
from .natural_velocities import NaturalVelocities
from .rotations import euler_axes_from_rotation_matrices, euler_angles_from_rotation_matrix
from ..protocols.biomechanical_model import GenericBiomechanicalModel


class BiomechanicalModel(GenericBiomechanicalModel):
    """

    Methods
    -------
    to_mx()
        This function returns the equivalent of the current Casadi BiomechanicalModel compatible with CasADi
    express_joint_torques_in_euler_basis
        This function returns the joint torques expressed in the euler basis
    """

    def __init__(
        self,
        segments: dict[str, Any] | BiomechanicalModelSegments = None,
        joints: dict[str, Any] | BiomechanicalModelJoints = None,
    ):
        segments = BiomechanicalModelSegments() if segments is None else segments
        joints = BiomechanicalModelJoints() if joints is None else joints
        markers = BiomechanicalModelMarkers(segments)
        super().__init__(segments=segments, joints=joints, markers=markers)

    def to_mx(self):
        """
        This function returns the equivalent of the current BiomechanicalModel with casadi MX variables

        Returns
        -------
        BiomechanicalModel
            The equivalent of the current BiomechanicalModel with casadi MX variables
        """
        from ..bionc_casadi.biomechanical_model import BiomechanicalModel as CasadiBiomechanicalModel
        from ..bionc_casadi.biomechanical_model_segments import (
            BiomechanicalModelSegments as CasadiBiomechanicalModelSegments,
        )
        from ..bionc_casadi.biomechanical_model_joints import BiomechanicalModelJoints as CasadiBiomechanicalModelJoints

        casadi_segments = CasadiBiomechanicalModelSegments(
            segments={key: segment.to_mx() for key, segment in self.segments.items()}
        )
        casadi_joints = CasadiBiomechanicalModelJoints(
            joints={key: joint.to_mx() for key, joint in self.joints.items()}
        )
        biomechanical_model = CasadiBiomechanicalModel(
            segments=casadi_segments,
            joints=casadi_joints,
        )

        biomechanical_model._update_mass_matrix()
        biomechanical_model.set_numpy_model(self)

        return biomechanical_model

    def _update_mass_matrix(self):
        """
        This function computes the generalized mass matrix of the system, denoted G

        Returns
        -------
        np.ndarray
            generalized mass matrix of the segment [12 * nbSegment x 12 * * nbSegment]
        """
        G = np.zeros((12 * self.nb_segments, 12 * self.nb_segments))
        for i, segment in enumerate(self.segments_no_ground.values()):
            Gi = segment.mass_matrix
            if Gi is None:
                # mass matrix is None if one the segment doesn't have any inertial properties
                self._mass_matrix = None
                return
            idx = slice(12 * i, 12 * (i + 1))
            G[idx, idx] = segment.mass_matrix

        self._mass_matrix = G

    def kinetic_energy(self, Qdot: NaturalVelocities) -> float:
        """
        This function returns the kinetic energy of the system as a function of the natural coordinates Q and Qdot

        Parameters
        ----------
        Qdot : NaturalVelocities
            The natural velocities of the segment [12 x n, 1]

        Returns
        -------
        float
            The kinetic energy of the system
        """

        return 0.5 * transpose(Qdot.to_array()) @ self._mass_matrix @ Qdot.to_array()

    def potential_energy(self, Q: NaturalCoordinates) -> np.ndarray | float:
        """
        This function returns the potential energy of the system as a function of the natural coordinates Q

        Parameters
        ----------
        Q : NaturalCoordinates
            The natural coordinates of the segment [12 x n, 1]

        Returns
        -------
        float
            The potential energy of the system
        """
        E = 0
        for i, segment in enumerate(self.segments_no_ground.values()):
            E += segment.potential_energy(Q.vector(i))

        return E

    def lagrangian(self, Q: NaturalCoordinates, Qdot: NaturalVelocities) -> np.ndarray | float:
        """
        This function returns the lagrangian of the system as a function of the natural coordinates Q and Qdot

        Parameters
        ----------
        Q : NaturalCoordinates
            The natural coordinates of the segment [12 x n, 1]
        Qdot : NaturalVelocities
            The natural velocities of the segment [12 x n, 1]

        Returns
        -------
        float
            The lagrangian of the system
        """

        return self.kinetic_energy(Qdot) - self.potential_energy(Q)

    def holonomic_constraints(self, Q: NaturalCoordinates) -> np.ndarray:
        """
        This function returns the holonomic constraints of the system, denoted Phi_h
        as a function of the natural coordinates Q. They are organized as follow, for each segment:
            [Phi_k_0, Phi_r_0, Phi_k_1, Phi_r_1, ..., Phi_k_n, Phi_r_n]

        Parameters
        ----------
        Q : NaturalCoordinates
            The natural coordinates of the segment [12 * nb_segments, 1]

        Returns
        -------
            Holonomic constraints of the segment [nb_holonomic_constraints, 1]
        """
        rigid_body_constraints = self.rigid_body_constraints(Q)

        phi = np.zeros((self.nb_holonomic_constraints, 1))
        nb_constraints = 0
        # two steps in order to get a jacobian as diagonal as possible
        # it follows the order of the segments
        for i, segment in enumerate(self.segments_no_ground.values()):
            # add the joint constraints first
            joints = self.joints_from_child_index(i, remove_free_joints=True)
            if len(joints) != 0:
                for j in joints:
                    idx = slice(nb_constraints, nb_constraints + j.nb_constraints)

                    Q_parent = (
                        None if j.parent is None else Q.vector(self.segments[j.parent.name].index)
                    )  # if the joint is a joint with the ground, the parent is None
                    Q_child = Q.vector(self.segments[j.child.name].index)
                    phi[idx, 0] = j.constraint(Q_parent, Q_child)

                    nb_constraints += j.nb_constraints

            # add the rigid body constraint
            idx = slice(nb_constraints, nb_constraints + 6)
            idx_segment = slice(6 * i, 6 * (i + 1))
            phi[idx, 0] = rigid_body_constraints[idx_segment]

            nb_constraints += 6

        return phi

    def holonomic_constraints_jacobian(self, Q: NaturalCoordinates) -> np.ndarray:
        """
        This function returns the Jacobian matrix the holonomic constraints, denoted K.
        They are organized as follow, for each segmen, the rows of the matrix are:
        [Phi_k_0, Phi_r_0, Phi_k_1, Phi_r_1, ..., Phi_k_n, Phi_r_n]

        Parameters
        ----------
        Q : NaturalCoordinates
            The natural coordinates of the segment [12 * nb_segments, 1]

        Returns
        -------
            Joint constraints of the holonomic constraints [nb_holonomic_constraints, 12 * nb_segments]
        """

        # first we compute the rigid body constraints jacobian
        rigid_body_constraints_jacobian = self.rigid_body_constraints_jacobian(Q)

        # then we compute the holonomic constraints jacobian
        nb_constraints = 0
        K = np.zeros((self.nb_holonomic_constraints, 12 * self.nb_segments))
        for i, segment in enumerate(self.segments_no_ground.values()):
            # add the joint constraints first
            joints = self.joints_from_child_index(i, remove_free_joints=True)
            if len(joints) != 0:
                for j in joints:
                    idx_row = slice(nb_constraints, nb_constraints + j.nb_constraints)

                    idx_col_child = slice(
                        12 * self.segments[j.child.name].index, 12 * (self.segments[j.child.name].index + 1)
                    )
                    idx_col_parent = (
                        slice(12 * self.segments[j.parent.name].index, 12 * (self.segments[j.parent.name].index + 1))
                        if j.parent is not None
                        else None
                    )

                    Q_parent = (
                        None if j.parent is None else Q.vector(self.segments[j.parent.name].index)
                    )  # if the joint is a joint with the ground, the parent is None
                    Q_child = Q.vector(self.segments[j.child.name].index)

                    K[idx_row, idx_col_child] = j.child_constraint_jacobian(Q_parent, Q_child)

                    if j.parent is not None:  # If the joint is not a ground joint
                        K[idx_row, idx_col_parent] = j.parent_constraint_jacobian(Q_parent, Q_child)

                    nb_constraints += j.nb_constraints

            # add the rigid body constraint
            idx_row = slice(nb_constraints, nb_constraints + 6)
            idx_rigid_body_constraint = slice(6 * i, 6 * (i + 1))
            idx_segment = slice(12 * i, 12 * (i + 1))

            K[idx_row, idx_segment] = rigid_body_constraints_jacobian[idx_rigid_body_constraint, idx_segment]

            nb_constraints += 6

        return K

    def holonomic_constraints_jacobian_derivative(self, Qdot: NaturalVelocities) -> np.ndarray:
        """
        This function returns the Jacobian matrix the holonomic constraints, denoted Kdot.
        They are organized as follow, for each segment, the rows of the matrix are:
        [Phi_k_0, Phi_r_0, Phi_k_1, Phi_r_1, ..., Phi_k_n, Phi_r_n]
        [joint constraint 0, rigid body constraint 0, joint constraint 1, rigid body constraint 1, ...]

        ```math
        \begin{equation}
        \frac{d}{dt} \frac{\partial \Phi^k}{\partial Q} =
        \begin{bmatrix}
        \frac{d}{dt} \frac{\partial \Phi^k_0}{\partial Q} \\
        \frac{d}{dt} \frac{\partial \Phi^r_0}{\partial Q} \\
        \frac{d}{dt} \frac{\partial \Phi^k_1}{\partial Q} \\
        \frac{d}{dt} \frac{\partial \Phi^r_1}{\partial Q} \\
        \vdots \\
        \frac{d}{dt} \frac{\partial \Phi^k_n}{\partial Q} \\
        \frac{d}{dt} \frac{\partial \Phi^r_n}{\partial Q} \\
        \end{bmatrix}
        \end{equation}
        ```

        Parameters
        ----------
        Qdot : NaturalVelocities
            The natural velocities of the segment [12 * nb_segments, 1]

        Returns
        -------
            Holonomic constraints jacobian derivative [nb_holonomic_constraints, 12 * nb_segments]
        """

        # first we compute the rigid body constraints jacobian
        rigid_body_constraints_jacobian_dot = self.rigid_body_constraint_jacobian_derivative(Qdot)

        # then we compute the holonomic constraints jacobian
        nb_constraints = 0
        Kdot = np.zeros((self.nb_holonomic_constraints, 12 * self.nb_segments))
        for i in range(self.nb_segments):
            # add the joint constraints first
            joints = self.joints_from_child_index(i, remove_free_joints=True)
            if len(joints) != 0:
                for j in joints:
                    idx_row = slice(nb_constraints, nb_constraints + j.nb_constraints)

                    idx_col_child = slice(
                        12 * self.segments[j.child.name].index, 12 * (self.segments[j.child.name].index + 1)
                    )
                    idx_col_parent = (
                        slice(12 * self.segments[j.parent.name].index, 12 * (self.segments[j.parent.name].index + 1))
                        if j.parent is not None
                        else None
                    )

                    Qdot_parent = (
                        None if j.parent is None else Qdot.vector(self.segments[j.parent.name].index)
                    )  # if the joint is a joint with the ground, the parent is None
                    Qdot_child = Qdot.vector(self.segments[j.child.name].index)

                    Kdot[idx_row, idx_col_child] = j.child_constraint_jacobian_derivative(Qdot_parent, Qdot_child)

                    if j.parent is not None:  # If the joint is not a ground joint
                        Kdot[idx_row, idx_col_parent] = j.parent_constraint_jacobian_derivative(Qdot_parent, Qdot_child)

                    nb_constraints += j.nb_constraints

            # add the rigid body constraint
            idx_row = slice(nb_constraints, nb_constraints + 6)
            idx_rigid_body_constraint = slice(6 * i, 6 * (i + 1))
            idx_segment = slice(12 * i, 12 * (i + 1))

            Kdot[idx_row, idx_segment] = rigid_body_constraints_jacobian_dot[idx_rigid_body_constraint, idx_segment]

            nb_constraints += 6

        return Kdot

    def gravity_forces(self) -> np.ndarray:
        """
        This function returns the weights caused by the gravity forces on each segment

        Returns
        -------
            The gravity_force of each segment [12 * nb_segments, 1]
        """
        weight_vector = np.zeros((self.nb_segments * 12, 1))

        for i, segment in enumerate(self.segments_no_ground.values()):
            idx = slice(12 * i, 12 * (i + 1))
            weight_vector[idx, 0] = segment.gravity_force()

        return weight_vector

    def forward_dynamics(
        self,
        Q: NaturalCoordinates,
        Qdot: NaturalCoordinates,
        joint_generalized_forces: np.ndarray = None,
        external_forces: ExternalForceSet = None,
        stabilization: dict = None,
    ) -> np.ndarray:
        """
        This function computes the forward dynamics of the system, i.e. the acceleration of the segments

        Parameters
        ----------
        Q : NaturalCoordinates
            The natural coordinates of the segment [12 * nb_segments, 1]
        Qdot : NaturalCoordinates
            The natural coordinates time derivative of the segment [12 * nb_segments, 1]
        joint_generalized_forces : np.ndarray
            The joint generalized forces in joint euler-basis, and forces in parent basis, like in minimal coordinates,
            one per dof of the system. If None, the joint generalized forces are set to 0
        external_forces : ExternalForceSet
            The list of external forces applied on the system
        stabilization: dict
            Dictionary containing the Baumgarte's stabilization parameters:
            * alpha: float
                Stabilization parameter for the constraint
            * beta: float
                Stabilization parameter for the constraint derivative

        Returns
        -------
            Qddot : NaturalAccelerations
                The natural accelerations [12 * nb_segments, 1]
            lagrange_multipliers : np.ndarray
                The lagrange multipliers [nb_holonomic_constraints, 1]
        """
        G = self.mass_matrix
        K = self.holonomic_constraints_jacobian(Q)
        Kdot = self.holonomic_constraints_jacobian_derivative(Qdot)

        external_forces = self.external_force_set() if external_forces is None else external_forces
        fext = external_forces.to_natural_external_forces(Q)

        joint_generalized_forces_object = JointGeneralizedForcesList.empty_from_nb_joint(self.nb_segments)
        # each segment is actuated from its parent segment (assuming tree-like structure)
        # if joint_generalized_forces is not None:
        #     joint_generalized_forces_object.add_all_joint_generalized_forces(
        #         model=self,
        #         joint_generalized_forces=joint_generalized_forces,
        #         Q=Q,
        #     )
        # natural_joint_forces = joint_generalized_forces_object.to_natural_joint_forces(
        #     model=self,
        #     Q=Q,
        # )

        # KKT system
        # [G, K.T] [Qddot]  = [forces]
        # [K, 0  ] [lambda] = [biais]
        upper_KKT_matrix = np.concatenate((G, K.T), axis=1)
        lower_KKT_matrix = np.concatenate((K, np.zeros((K.shape[0], K.shape[0]))), axis=1)
        KKT_matrix = np.concatenate((upper_KKT_matrix, lower_KKT_matrix), axis=0)

        forces = (
            self.gravity_forces()
            + fext
            # + natural_joint_forces
        )
        biais = -Kdot @ Qdot

        if stabilization is not None:
            # raise NotImplementedError("Stabilization is not implemented yet")
            biais -= (
                stabilization["alpha"] * self.holonomic_constraints(Q)
                + stabilization["beta"] * self.holonomic_constraints_jacobian(Q) @ Qdot
            )

        B = np.concatenate([forces, biais], axis=0)

        # solve the linear system Ax = B with numpy
        x = np.linalg.solve(KKT_matrix, B)
        Qddoti = x[0 : self.nb_Qddot]
        lambda_i = x[self.nb_Qddot :]
        return NaturalAccelerations(Qddoti), lambda_i

    def inverse_kinematics(
        self,
        experimental_markers: np.ndarray | str,
        Q_init: NaturalCoordinates = None,
        solve_frame_per_frame: bool = True,
    ) -> InverseKinematics:
        """
        This is an interface to the inverse kinematics class. It allows to build an inverse kinematics object with the current model.

        Parameters
        ----------
        experimental_markers : np.ndarray | str
            The experimental markers positions. If it is a string, it is the path to the file containing the markers positions
        Q_init : NaturalCoordinates
            The initial guess for the inverse kinematics. If None, the initial guess is the zero vector
        solve_frame_per_frame : bool
            If True, the inverse kinematics is solved frame per frame. If False, the inverse kinematics is solved for the whole sequence at once

        Returns
        -------
        InverseKinematics
            The inverse kinematics object
        """
        return InverseKinematics(
            self, experimental_markers=experimental_markers, Q_init=Q_init, solve_frame_per_frame=solve_frame_per_frame
        )

    def external_force_set(self) -> ExternalForceSet:
        return ExternalForceSet.empty_from_nb_segment(self.nb_segments)

    def inverse_dynamics(
        self,
        Q: NaturalCoordinates,
        Qddot: NaturalAccelerations,
        external_forces: ExternalForceSet = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if external_forces is None:
            external_forces = self.external_force_set()
        else:
            if external_forces.nb_segments != self.nb_segments:
                raise ValueError(
                    f"The number of segments in the model and the external forces must be the same:"
                    f" segment number = {self.nb_segments}"
                    f" external force size = {external_forces.nb_segments}"
                )

        if Q is None:
            raise ValueError(f"The generalized coordinates must be provided")
        if Q.nb_qi() != self.nb_segments:
            raise ValueError(
                f"The number of generalized coordinates in the model and the generalized coordinates must be the same:"
                f" model number = {self.nb_segments}"
                f" generalized coordinates size = {Q.nb_qi()}"
            )
        if Qddot is None:
            raise ValueError(f"The generalized accelerations must be provided")
        if Qddot.nb_qddoti() != self.nb_segments:
            raise ValueError(
                f"The number of generalized accelerations in the model and the generalized accelerations must be the same:"
                f" model number = {self.nb_segments}"
                f" generalized accelerations size = {Qddot.nb_qddoti()}"
            )

        # last check to verify that the model doesn't contain any closed loop
        visited_segment = self._depth_first_search(0)
        if not all(visited_segment):
            raise ValueError(
                f"The model contains free segments. The inverse dynamics can't be computed."
                f" The free segments are: {np.where(np.logical_not(visited_segment))[0]}."
                f" Please consider adding joints to integer them into the kinematic tree."
            )

        # NOTE: This won't work with two independent tree in the same model
        visited_segments = [False for _ in range(self.nb_segments)]
        torques = np.zeros((3, self.nb_segments))
        forces = np.zeros((3, self.nb_segments))
        lambdas = np.zeros((6, self.nb_segments))
        _, forces, torques, lambdas = self._inverse_dynamics_recursive_step(
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

    def _inverse_dynamics_recursive_step(
        self,
        Q: NaturalCoordinates,
        Qddot: NaturalAccelerations,
        external_forces: ExternalForceSet,
        segment_index: int = 0,
        visited_segments: list[bool, ...] = None,
        torques: np.ndarray = None,
        forces: np.ndarray = None,
        lambdas: np.ndarray = None,
    ) -> tuple[list[bool, ...], np.ndarray, np.ndarray, np.ndarray]:
        """
        This function returns the segments in a depth first search order.

        Parameters
        ----------
        Q: NaturalCoordinates
            The generalized coordinates of the model
        Qddot: NaturalAccelerations
            The generalized accelerations of the model
        external_forces: ExternalForceSet
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
        tuple[list[bool, ...], np.ndarray, np.ndarray, np.ndarray]
            visited_segments: list[bool]
                The segments already visited
            torques: np.ndarray
                The intersegmental torques applied to the segments
            forces: np.ndarray
                The intersegmental forces applied to the segments
            lambdas: np.ndarray
                The lagrange multipliers applied to the segments
        """
        visited_segments[segment_index] = True

        Qi = Q.vector(segment_index)
        Qddoti = Qddot.vector(segment_index)
        external_forces_i = external_forces.to_segment_natural_external_forces(segment_index=segment_index, Q=Q)

        subtree_intersegmental_generalized_forces = np.zeros((12, 1))
        for child_index in self.children(segment_index):
            if not visited_segments[child_index]:
                visited_segments, torques, forces, lambdas = self._inverse_dynamics_recursive_step(
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
        segment_i = self.segment_from_index(segment_index)

        force_i, torque_i, lambda_i = segment_i.inverse_dynamics(
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

    def express_joint_torques_in_euler_basis(self, Q: NaturalCoordinates, torques: np.ndarray) -> np.ndarray:
        """
        This function expresses the joint torques in the euler basis.

        Parameters
        ----------
        Q: NaturalCoordinates
            The generalized coordinates of the model
        torques: np.ndarray
            The joint torques in global coordinates system

        Returns
        -------
        np.ndarray
            The joint torques expressed in the euler basis
        """
        if torques.shape != (3, self.nb_segments):
            raise ValueError(f"The shape of the joint torques must be (3, {self.nb_segments}) but is {torques.shape}")

        euler_torques = np.zeros((3, self.nb_segments))
        for i, (joint_name, joint) in enumerate(self.joints.items()):
            if joint.projection_basis is None:
                raise RuntimeError(
                    "The projection basis of the joint must be defined to express the torques in an Euler basis."
                    f"Joint {joint_name} has no projection basis defined."
                    f"Please define a projection basis for this joint, "
                    f"using argument `projection_basis` of the joint constructor"
                    f" and enum `EulerSequence` for the type of entry."
                )

            parent_segment = joint.parent
            child_segment = joint.child

            Q_parent = (
                None if joint.parent is None else Q.vector(self.segments[joint.parent.name].index)
            )  # if the joint is a joint with the ground, the parent is None
            Q_child = Q.vector(child_segment.index)

            # compute rotation matrix from Qi
            R_parent = (
                np.eye(3)
                if joint.parent is None
                else parent_segment.segment_coordinates_system(Q_parent, joint.parent_basis).rot
            )
            R_child = child_segment.segment_coordinates_system(Q_child, joint.child_basis).rot

            e1, e2, e3 = euler_axes_from_rotation_matrices(
                R_parent, R_child, sequence=joint.projection_basis, axes_source_frame="mixed"
            )

            # compute the euler torques
            euler_torques[:, i] = vector_projection_in_non_orthogonal_basis(torques[:, i], e1, e2, e3).squeeze()

        return euler_torques

    def natural_coordinates_to_joint_angles(self, Q: NaturalCoordinates) -> np.ndarray:
        """
        This function converts the natural coordinates to joint angles with Euler Sequences defined for each joint

        # todo: This should be named to_minimal_coordinates instead of joint_angles,
            because we can have translations too.

        Parameters
        ----------
        Q: NaturalCoordinates
            The natural coordinates of the model

        Returns
        -------
        np.ndarray
            The joint angles [3 x nb_joints]
        """
        euler_angles = np.zeros((3, self.nb_joints))

        for i, (joint_name, joint) in enumerate(self.joints.items()):
            if joint.projection_basis is None:
                raise RuntimeError(
                    "The projection basis of the joint must be defined to express the torques in an Euler basis."
                    f"Joint {joint_name} has no projection basis defined."
                    f"Please define a projection basis for this joint, "
                    f"using argument `projection_basis` of the joint constructor"
                    f" and enum `EulerSequence` for the type of entry."
                )

            parent_segment = joint.parent
            child_segment = joint.child

            Q_parent = (
                None if joint.parent is None else Q.vector(self.segments[joint.parent.name].index)
            )  # if the joint is a joint with the ground, the parent is None
            Q_child = Q.vector(child_segment.index)

            # compute rotation matrix from Qi
            R_parent = (
                np.eye(3)
                if joint.parent is None
                else parent_segment.segment_coordinates_system(Q_parent, joint.parent_basis).rot
            )
            R_child = child_segment.segment_coordinates_system(Q_child, joint.child_basis).rot

            euler_angles[:, i] = euler_angles_from_rotation_matrix(
                R_parent,
                R_child,
                joint_sequence=joint.projection_basis,
            )

        return euler_angles
