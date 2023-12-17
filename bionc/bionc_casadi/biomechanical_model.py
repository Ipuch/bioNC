import numpy as np
from casadi import MX, transpose, horzcat, vertcat, solve
from typing import Any

from .biomechanical_model_joints import BiomechanicalModelJoints
from .biomechanical_model_segments import BiomechanicalModelSegments
from .cartesian_vector import vector_projection_in_non_orthogonal_basis
from .external_force import ExternalForceSet, ExternalForce
from .natural_accelerations import NaturalAccelerations
from .natural_coordinates import NaturalCoordinates
from .natural_velocities import NaturalVelocities
from .rotations import euler_axes_from_rotation_matrices, euler_angles_from_rotation_matrix
from ..protocols.biomechanical_model import GenericBiomechanicalModel


class BiomechanicalModel(GenericBiomechanicalModel):
    """

    Attributes
    ----------
    _numpy_model : NumpyBiomechanicalModel
        The numpy model from which the casadi model is built

    Methods
    -------
    set_numpy_model(numpy_model: BiomechanicalModel)
        Set the numpy model from which the casadi model is built
    numpy_model
        Return the numpy model from which the casadi model is built
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
        super().__init__(segments=segments, joints=joints)
        self._numpy_model = None

    def set_numpy_model(self, numpy_model: GenericBiomechanicalModel):
        self._numpy_model = numpy_model

    @property
    def numpy_model(self):
        return self._numpy_model

    def save(self, filename: str):
        raise NotImplementedError("Saving a biomechanical model is not implemented yet with casadi models.")
        # todo: only possible with numpy models so far
        # do a method that returns a numpy model and save it
        # with open(filename, "wb") as file:
        #     pickle.dump(self, file)

    @staticmethod
    def load(filename: str):
        raise NotImplementedError("Loading a biomechanical model is not implemented yet with casadi models.")
        # todo: only possible with numpy models so far
        # load the numpy model and convert it to casadi
        # with open(filename, "rb") as file:
        #     model = pickle.load(file)
        #
        # return model

    def _update_mass_matrix(self):
        """
        This function computes the generalized mass matrix of the system, denoted G

        Returns
        -------
        MX
            generalized mass matrix of the segment [12 * nbSegment x 12 * * nbSegment]
        """
        G = MX.zeros((12 * self.nb_segments, 12 * self.nb_segments))
        for i, segment in enumerate(self.segments_no_ground.values()):
            Gi = segment.mass_matrix
            if Gi is None:
                # mass matrix is None if one the segment doesn't have any inertial properties
                self._mass_matrix = None
                return
            idx = slice(12 * i, 12 * (i + 1))
            G[idx, idx] = segment.mass_matrix

        self._mass_matrix = G

    def kinetic_energy(self, Qdot: NaturalVelocities) -> MX:
        """
        This function returns the kinetic energy of the system as a function of the natural coordinates Q and Qdot

        Parameters
        ----------
        Qdot : NaturalVelocities
            The natural velocities of the segment [12, 1]

        Returns
        -------
        MX
            The kinetic energy of the system
        """

        return 0.5 * transpose(Qdot.to_array()) @ self._mass_matrix @ Qdot.to_array()

    def potential_energy(self, Q: NaturalCoordinates) -> MX:
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

    def lagrangian(self, Q: NaturalCoordinates, Qdot: NaturalVelocities) -> MX:
        """
        This function returns the lagrangian of the system as a function of the natural coordinates Q and Qdot

        Parameters
        ----------
        Q : NaturalCoordinates
            The natural coordinates of the segment [12, 1]
        Qdot : NaturalVelocities
            The natural velocities of the segment [12, 1]

        Returns
        -------
        MX
            The lagrangian of the system
        """

        return self.kinetic_energy(Qdot) - self.potential_energy(Q)

    def markers(self, Q: NaturalCoordinates) -> MX:
        """
        This function returns the position of the markers of the system as a function of the natural coordinates Q
        also referred as forward kinematics

        Parameters
        ----------
        Q : NaturalCoordinates
            The natural coordinates of the segment [12 x n, 1]

        Returns
        -------
        MX
            The position of the markers [3, nbMarkers, nbFrames]
            in the global coordinate system/ inertial coordinate system
        """
        markers = MX.zeros((3, self.nb_markers))
        nb_markers = 0
        for segment in self.segments_no_ground.values():
            idx = slice(nb_markers, nb_markers + segment.nb_markers)
            markers[:, idx] = segment.markers(Q.vector(segment.index))
            nb_markers += segment.nb_markers

        return markers

    def center_of_mass_position(self, Q: NaturalCoordinates) -> MX:
        """
        This function returns the position of the center of mass of each segment as a function of the natural coordinates Q

        Parameters
        ----------
        Q : NaturalCoordinates
            The natural coordinates of the segment [12 x n, 1]

        Returns
        -------
        MX
            The position of the center of mass [3, nbSegments]
            in the global coordinate system/ inertial coordinate system
        """
        com = MX.zeros((3, self.nb_segments))
        for i, segment in enumerate(self.segments_no_ground.values()):
            position = segment.center_of_mass_position(Q.vector(i))
            com[:, i] = position

        return com

    def markers_constraints(self, markers: np.ndarray | MX, Q: NaturalCoordinates, only_technical: bool = True) -> MX:
        """
        This function returns the marker constraints of all segments, denoted Phi_r
        as a function of the natural coordinates Q.

        markers : np.ndarray | MX
           The markers positions [3,nb_markers]
        Q : NaturalCoordinates
           The natural coordinates of the segment [12 x n, 1]
        only_technical : bool
           If True, only technical markers are considered, by default True,
           because we only want to use technical markers for inverse kinematics, this choice can be revised.

        Returns
        -------
        MX
           Rigid body constraints of the segment [nb_markers x 3, 1]
        """
        if not isinstance(markers, MX):
            markers = MX(markers)

        nb_markers = self.nb_markers_technical if only_technical else self.nb_markers
        if markers.shape[1] != nb_markers:
            raise ValueError(
                f"markers should have {nb_markers} columns. "
                f"And should include the following markers: "
                f"{self.marker_names_technical if only_technical else self.marker_names}"
            )

        phi_m = MX.zeros((nb_markers * 3, 1))
        marker_count = 0

        for i_segment, segment in enumerate(self.segments_no_ground.values()):
            nb_segment_markers = segment.nb_markers_technical if only_technical else self.segments[name].nb_markers
            if nb_segment_markers == 0:
                continue
            constraint_idx = slice(marker_count * 3, (marker_count + nb_segment_markers) * 3)
            marker_idx = slice(marker_count, marker_count + nb_segment_markers)

            markers_temp = markers[:, marker_idx]
            phi_m[constraint_idx] = segment.marker_constraints(
                markers_temp, Q.vector(i_segment), only_technical=only_technical
            )[
                :
            ]  # [:] to flatten the array

            marker_count += nb_segment_markers

        return phi_m

    def markers_constraints_jacobian(self, only_technical: bool = True) -> MX:
        """
        This function returns the Jacobian matrix the markers constraints, denoted K_m.

        Parameters
        ----------
        only_technical : bool
            If True, only technical markers are considered, by default True,
            because we only want to use technical markers for inverse kinematics, this choice can be revised.

        Returns
        -------
        MX
            Joint constraints of the marker [nb_markers x 3, nb_Q]
        """
        nb_markers = self.nb_markers_technical if only_technical else self.nb_markers

        km = MX.zeros((3 * nb_markers, 12 * self.nb_segments))
        marker_count = 0
        for i_segment, segment in enumerate(self.segments_no_ground.values()):
            nb_segment_markers = segment.nb_markers_technical if only_technical else segment.nb_markers
            if nb_segment_markers == 0:
                continue
            constraint_idx = slice(marker_count * 3, (marker_count + nb_segment_markers) * 3)
            segment_idx = slice(12 * i_segment, 12 * (i_segment + 1))
            km[constraint_idx, segment_idx] = segment.markers_jacobian()
            marker_count += nb_segment_markers

        return km

    def holonomic_constraints(self, Q: NaturalCoordinates) -> MX:
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

        phi = MX.zeros((self.nb_holonomic_constraints, 1))
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

    def holonomic_constraints_jacobian(self, Q: NaturalCoordinates) -> MX:
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
        K = MX.zeros((self.nb_holonomic_constraints, 12 * self.nb_segments))
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

    def holonomic_constraints_jacobian_derivative(self, Qdot: NaturalVelocities) -> MX:
        """
        This function returns the Jacobian matrix the holonomic constraints, denoted Kdot.
        They are organized as follow, for each segment, the rows of the matrix are:
        [Phi_k_0, Phi_r_0, Phi_k_1, Phi_r_1, ..., Phi_k_n, Phi_r_n]

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
        Kdot = MX.zeros((self.nb_holonomic_constraints, 12 * self.nb_segments))
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

    def gravity_forces(self) -> MX:
        """
        This function returns the weights caused by the gravity forces on each segment

        Returns
        -------
            The gravity_force of each segment [12 * nb_segments, 1]
        """
        weight_vector = MX.zeros((self.nb_segments * 12, 1))
        for i, segment in enumerate(self.segments_no_ground.values()):
            idx = slice(12 * i, 12 * (i + 1))
            weight_vector[idx] = segment.gravity_force()

        return weight_vector

    def forward_dynamics(
        self,
        Q: NaturalCoordinates,
        Qdot: NaturalCoordinates,
        external_forces: ExternalForceSet = None,
        # external_forces: ExternalForces
    ):
        """
        This function computes the forward dynamics of the system, i.e. the acceleration of the segments

        Parameters
        ----------
        Q : NaturalCoordinates
            The natural coordinates of the segment [12 * nb_segments, 1]
        Qdot : NaturalCoordinates
            The natural coordinates time derivative of the segment [12 * nb_segments, 1]
        external_forces : ExternalForceSet
            The list of external forces applied on the system

        Returns
        -------
            Qddot : NaturalAccelerations
                The natural accelerations [12 * nb_segments, 1]
            lagrange_multipliers : MX
                The lagrange multipliers [nb_holonomic_constraints, 1]
        """
        G = self.mass_matrix
        K = self.holonomic_constraints_jacobian(Q)
        Kdot = self.holonomic_constraints_jacobian_derivative(Qdot)

        external_forces = self.external_force_set() if external_forces is None else external_forces
        fext = external_forces.to_natural_external_forces(Q)
        # if stabilization is not None:
        #     biais -= stabilization["alpha"] * self.rigid_body_constraint(Qi) + stabilization[
        #         "beta"
        #     ] * self.rigid_body_constraint_derivative(Qi, Qdoti)

        # KKT system
        # [G, K.T] [Qddot]  = [forces]
        # [K, 0  ] [lambda] = [biais]
        upper_KKT_matrix = horzcat(G, K.T)
        lower_KKT_matrix = horzcat(K, np.zeros((K.shape[0], K.shape[0])))
        KKT_matrix = vertcat(upper_KKT_matrix, lower_KKT_matrix)

        forces = self.gravity_forces() + fext
        biais = -Kdot @ Qdot
        B = vertcat(forces, biais)

        # solve the linear system Ax = B with casadi symbolic qr
        x = solve(KKT_matrix, B, "symbolicqr")
        Qddot = x[0 : self.nb_Qddot]
        lagrange_multipliers = x[self.nb_Qddot :]
        return NaturalAccelerations(Qddot), lagrange_multipliers

    def external_force_set(self) -> ExternalForceSet:
        return ExternalForceSet.empty_from_nb_segment(self.nb_segments)

    def inverse_dynamics(
        self,
        Q: NaturalCoordinates,
        Qddot: NaturalAccelerations,
        external_forces: ExternalForceSet = None,
    ) -> tuple[MX, MX, MX]:
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
        visited_segment = self._depth_first_search(0, visited_segments=None)
        if not all(visited_segment):
            raise ValueError(
                f"The model contains free segments. The inverse dynamics can't be computed."
                f" The free segments are: {np.where(np.logical_not(visited_segment))[0]}."
                f" Please consider adding joints to integer them into the kinematic tree."
            )

        # NOTE: This won't work with two independent tree in the same model
        visited_segments = [False for _ in range(self.nb_segments)]
        torques = MX.zeros((3, self.nb_segments))
        forces = MX.zeros((3, self.nb_segments))
        lambdas = MX.zeros((6, self.nb_segments))
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
        torques: MX = None,
        forces: MX = None,
        lambdas: MX = None,
    ):
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
        torques: MX
            The intersegmental torques applied to the segments
        forces: MX
            The intersegmental forces applied to the segments
        lambdas: MX
            The lagrange multipliers applied to the segments

        Returns
        -------
        tuple[list[bool, ...], MX, MX, MX]
            visited_segments: list[bool]
                The segments already visited
            torques: MX
                The intersegmental torques applied to the segments
            forces: MX
                The intersegmental forces applied to the segments
            lambdas: MX
                The lagrange multipliers applied to the segments
        """
        visited_segments[segment_index] = True

        Qi = Q.vector(segment_index)
        Qddoti = Qddot.vector(segment_index)
        external_forces_i = external_forces.to_segment_natural_external_forces(segment_index=segment_index, Q=Q)

        subtree_intersegmental_generalized_forces = MX.zeros((12, 1))
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
            )
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

    def express_joint_torques_in_euler_basis(self, Q: NaturalCoordinates, torques: MX) -> MX:
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

        euler_torques = MX.zeros((3, self.nb_segments))
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
            euler_torques[:, i] = vector_projection_in_non_orthogonal_basis(torques[:, i], e1, e2, e3)

        return euler_torques

    def natural_coordinates_to_joint_angles(self, Q: NaturalCoordinates) -> np.ndarray:
        """
        This function converts the natural coordinates to joint angles with Euler Sequences defined for each joint

        Parameters
        ----------
        Q: NaturalCoordinates
            The natural coordinates of the model

        Returns
        -------
        np.ndarray
            The joint angles [3 x nb_joints]
        """
        euler_angles = MX.zeros((3, self.nb_joints))

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
