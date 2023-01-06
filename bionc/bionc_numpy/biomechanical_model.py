import numpy as np
from numpy import transpose

from .natural_coordinates import NaturalCoordinates
from .natural_velocities import NaturalVelocities
from .natural_accelerations import NaturalAccelerations
from ..protocols.biomechanical_model import GenericBiomechanicalModel


class BiomechanicalModel(GenericBiomechanicalModel):
    def __init__(self):
        super().__init__()

    def to_mx(self):
        """
        This function returns the equivalent of the current BiomechanicalModel with casadi MX variables

        Returns
        -------
        BiomechanicalModel
            The equivalent of the current BiomechanicalModel with casadi MX variables
        """
        from ..bionc_casadi.biomechanical_model import BiomechanicalModel as CasadiBiomechanicalModel

        biomechanical_model = CasadiBiomechanicalModel()
        biomechanical_model.segments = {key: segment.to_mx() for key, segment in self.segments.items()}
        biomechanical_model.joints = {key: joint.to_mx() for key, joint in self.joints.items()}
        biomechanical_model._update_mass_matrix()

        return biomechanical_model

    def rigid_body_constraints(self, Q: NaturalCoordinates) -> np.ndarray:
        """
        This function returns the rigid body constraints of all segments, denoted Phi_r
        as a function of the natural coordinates Q.

        Returns
        -------
        np.ndarray
            Rigid body constraints of the segment [6 * nb_segments, 1]
        """

        Phi_r = np.zeros(6 * self.nb_segments)
        for i, segment_name in enumerate(self.segments):
            idx = slice(6 * i, 6 * (i + 1))
            Phi_r[idx] = self.segments[segment_name].rigid_body_constraint(Q.vector(i))

        return Phi_r

    def rigid_body_constraints_derivative(self, Q: NaturalCoordinates, Qdot: NaturalCoordinates) -> np.ndarray:
        """
        This function returns the derivative of the rigid body constraints of all segments, denoted Phi_r_dot
        as a function of the natural coordinates Q and Qdot.

        Returns
        -------
        np.ndarray
            Derivative of the rigid body constraints of the segment [6 * nb_segments, 1]
        """

        Phi_r_dot = np.zeros(6 * self.nb_segments)
        for i, segment_name in enumerate(self.segments):
            idx = slice(6 * i, 6 * (i + 1))
            Phi_r_dot[idx] = self.segments[segment_name].rigid_body_constraint_derivative(Q.vector(i), Qdot.vector(i))

        return Phi_r_dot

    def rigid_body_constraints_jacobian(self, Q: NaturalCoordinates) -> np.ndarray:
        """
        This function returns the rigid body constraints of all segments, denoted K_r
        as a function of the natural coordinates Q.

        Returns
        -------
        np.ndarray
            Rigid body constraints of the segment [6 * nb_segments, nbQ]
        """

        K_r = np.zeros((6 * self.nb_segments, Q.shape[0]))
        for i, segment_name in enumerate(self.segments):
            idx_row = slice(6 * i, 6 * (i + 1))
            idx_col = slice(12 * i, 12 * (i + 1))
            K_r[idx_row, idx_col] = self.segments[segment_name].rigid_body_constraint_jacobian(Q.vector(i))

        return K_r

    def rigid_body_constraint_jacobian_derivative(self, Qdot: NaturalVelocities) -> np.ndarray:
        """
        This function returns the derivative of the Jacobian matrix of the rigid body constraints denoted Kr_dot

        Parameters
        ----------
        Qdot : NaturalVelocities
            The natural velocities of the segment [12, 1]

        Returns
        -------
        np.ndarray
            The derivative of the Jacobian matrix of the rigid body constraints [6, 12]
        """

        Kr_dot = np.zeros((6 * self.nb_segments, Qdot.shape[0]))
        for i, segment_name in enumerate(self.segments):
            idx_row = slice(6 * i, 6 * (i + 1))
            idx_col = slice(12 * i, 12 * (i + 1))
            Kr_dot[idx_row, idx_col] = self.segments[segment_name].rigid_body_constraint_jacobian_derivative(
                Qdot.vector(i)
            )

        return Kr_dot

    def joint_constraints(self, Q: NaturalCoordinates) -> np.ndarray:
        """
        This function returns the joint constraints of all joints, denoted Phi_k
        as a function of the natural coordinates Q.

        Returns
        -------
        np.ndarray
            Joint constraints of the segment [nb_joint_constraints, 1]
        """

        Phi_k = np.zeros(self.nb_joint_constraints)
        nb_constraints = 0
        for joint_name, joint in self.joints.items():
            idx = slice(nb_constraints, nb_constraints + joint.nb_constraints)

            Q_parent = (
                None if joint.parent is None else Q.vector(self.segments[joint.parent.name].index)
            )  # if the joint is a joint with the ground, the parent is None
            Q_child = Q.vector(self.segments[joint.child.name].index)
            Phi_k[idx] = joint.constraint(Q_parent, Q_child)

            nb_constraints += self.joints[joint_name].nb_constraints

        return Phi_k

    def joint_constraints_jacobian(self, Q: NaturalCoordinates) -> np.ndarray:
        """
        This function returns the Jacobian matrix the joint constraints, denoted K_k
        as a function of the natural coordinates Q.

        Returns
        -------
        np.ndarray
            Joint constraints of the segment [nb_joint_constraints, nbQ]
        """

        K_k = np.zeros((self.nb_joint_constraints, Q.shape[0]))
        nb_constraints = 0
        for joint_name, joint in self.joints.items():
            idx_row = slice(nb_constraints, nb_constraints + joint.nb_constraints)

            if joint.parent is not None:  # If the joint is not a ground joint
                idx_col_parent = slice(
                    12 * self.segments[joint.parent.name].index, 12 * (self.segments[joint.parent.name].index + 1)
                )
                Q_child = Q.vector(self.segments[joint.child.name].index)
                K_k[idx_row, idx_col_parent] = joint.parent_constraint_jacobian(Q_child)

            idx_col_child = slice(
                12 * self.segments[joint.child.name].index, 12 * (self.segments[joint.child.name].index + 1)
            )
            Q_parent = (
                None if joint.parent is None else Q.vector(self.segments[joint.parent.name].index)
            )  # if the joint is a joint with the ground, the parent is None
            K_k[idx_row, idx_col_child] = joint.child_constraint_jacobian(Q_parent)

            nb_constraints += self.joints[joint_name].nb_constraints

        return K_k

    def joint_constraints_jacobian_derivative(self, Qdot: NaturalVelocities) -> np.ndarray:
        """
        This function returns the derivative of the Jacobian matrix of the joint constraints denoted K_k_dot

        Parameters
        ----------
        Qdot : NaturalVelocities
            The natural velocities of the segment [12 * nb_segments, 1]

        Returns
        -------
        MX
            The derivative of the Jacobian matrix of the joint constraints [nb_joint_constraints, 12 * nb_segments]
        """

        K_k_dot = np.zeros((self.nb_joint_constraints, Qdot.shape[0]))
        nb_constraints = 0
        for joint_name, joint in self.joints.items():
            idx_row = slice(nb_constraints, nb_constraints + joint.nb_constraints)

            if joint.parent is not None:  # If the joint is not a ground joint
                idx_col_parent = slice(
                    12 * self.segments[joint.parent.name].index, 12 * (self.segments[joint.parent.name].index + 1)
                )
                Qdot_child = Qdot.vector(self.segments[joint.child.name].index)
                K_k_dot[idx_row, idx_col_parent] = joint.parent_constraint_jacobian_derivative(Qdot_child)

            idx_col_child = slice(
                12 * self.segments[joint.child.name].index, 12 * (self.segments[joint.child.name].index + 1)
            )
            Qdot_parent = (
                None if joint.parent is None else Qdot.vector(self.segments[joint.parent.name].index)
            )  # if the joint is a joint with the ground, the parent is None
            K_k_dot[idx_row, idx_col_child] = joint.child_constraint_jacobian_derivative(Qdot_parent)

            nb_constraints += self.joints[joint_name].nb_constraints

        return K_k_dot

    def _update_mass_matrix(self):
        """
        This function computes the generalized mass matrix of the system, denoted G

        Returns
        -------
        np.ndarray
            generalized mass matrix of the segment [12 * nbSegment x 12 * * nbSegment]
        """
        G = np.zeros((12 * self.nb_segments, 12 * self.nb_segments))
        for i, segment_name in enumerate(self.segments):
            Gi = self.segments[segment_name].mass_matrix
            if Gi is None:
                # mass matrix is None if one the segment doesn't have any inertial properties
                self._mass_matrix = None
                return
            idx = slice(12 * i, 12 * (i + 1))
            G[idx, idx] = self.segments[segment_name].mass_matrix

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
        for i, segment_name in enumerate(self.segments):
            E += self.segments[segment_name].potential_energy(Q.vector(i))

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

    def markers(self, Q: NaturalCoordinates) -> np.ndarray:
        """
        This function returns the position of the markers of the system as a function of the natural coordinates Q
        also referred as forward kinematics

        Parameters
        ----------
        Q : NaturalCoordinates
            The natural coordinates of the segment [12 x n, 1]

        Returns
        -------
        np.ndarray
            The position of the markers [3, nbMarkers, nbFrames]
            in the global coordinate system/ inertial coordinate system
        """
        markers = np.zeros((3, self.nb_markers, Q.shape[1]))
        nb_markers = 0
        for _, segment in self.segments.items():
            idx = slice(nb_markers, nb_markers + segment.nb_markers)
            markers[:, idx, :] = segment.markers(Q.vector(segment.index))
            nb_markers += segment.nb_markers

        return markers

    def center_of_mass_position(self, Q: NaturalCoordinates) -> np.ndarray:
        """
        This function returns the position of the center of mass of each segment as a function of the natural coordinates Q

        Parameters
        ----------
        Q : NaturalCoordinates
            The natural coordinates of the segment [12 x n, 1]

        Returns
        -------
        np.ndarray
            The position of the center of mass [3, nbSegments]
            in the global coordinate system/ inertial coordinate system
        """
        com = np.zeros((3, self.nb_segments, Q.shape[1]))
        for i, segment in enumerate(self.segments.values()):
            position = segment.center_of_mass_position(Q.vector(i))
            com[:, i, :] = position if len(position.shape) == 2 else position[:, np.newaxis]

        return com

    def markers_constraints(
        self, markers: np.ndarray, Q: NaturalCoordinates, only_technical: bool = True
    ) -> np.ndarray:
        """
        This function returns the marker constraints of all segments, denoted Phi_r
        as a function of the natural coordinates Q.

        markers : np.ndarray
            The markers positions [3,nb_markers]
        Q : NaturalCoordinates
            The natural coordinates of the segment [12 x n, 1]
        only_technical : bool
            If True, only technical markers are considered, by default True,
            because we only want to use technical markers for inverse kinematics, this choice can be revised.

        Returns
        -------
        np.ndarray
            Rigid body constraints of the segment [nb_markers x 3, 1]
        """
        nb_markers = self.nb_markers_technical if only_technical else self.nb_markers
        if markers.shape[1] != nb_markers:
            raise ValueError(
                f"markers should have {nb_markers} columns. "
                f"And should include the following markers: "
                f"{self.marker_names_technical if only_technical else self.marker_names}"
            )

        phi_m = np.zeros((nb_markers * 3))
        marker_count = 0

        for i_segment, name in enumerate(self.segments):
            nb_segment_markers = (
                self.segments[name].nb_markers_technical if only_technical else self.segments[name].nb_markers
            )
            if nb_segment_markers == 0:
                continue
            constraint_idx = slice(marker_count * 3, (marker_count + nb_segment_markers) * 3)
            marker_idx = slice(marker_count, marker_count + nb_segment_markers)

            markers_temp = markers[:, marker_idx]
            phi_m[constraint_idx] = (
                self.segments[name].marker_constraints(markers_temp, Q.vector(i_segment)).flatten("F")
            )

            marker_count += nb_segment_markers

        return phi_m

    def markers_constraints_jacobian(self, only_technical: bool = True) -> np.ndarray:
        """
        This function returns the Jacobian matrix the markers constraints, denoted K_m.

        Parameters
        ----------
        only_technical : bool
            If True, only technical markers are considered, by default True,
            because we only want to use technical markers for inverse kinematics, this choice can be revised.

        Returns
        -------
        np.ndarray
            Joint constraints of the marker segment [nb_markers x 3, nb_Q]
        """
        nb_markers = self.nb_markers_technical if only_technical else self.nb_markers

        km = np.zeros((3 * nb_markers, 12 * self.nb_segments))
        marker_count = 0
        for i_segment, name in enumerate(self.segments):
            nb_segment_markers = (
                self.segments[name].nb_markers_technical if only_technical else self.segments[name].nb_markers
            )
            if nb_segment_markers == 0:
                continue
            constraint_idx = slice(marker_count * 3, (marker_count + nb_segment_markers) * 3)
            segment_idx = slice(12 * i_segment, 12 * (i_segment + 1))
            km[constraint_idx, segment_idx] = self.segments[name].markers_jacobian()
            marker_count += nb_segment_markers

        return km

    def Q_from_markers(self, markers: np.ndarray) -> NaturalCoordinates:
        """
        This function returns the natural coordinates of the system as a function of the markers positions

        Parameters
        ----------
        markers : np.ndarray
            The markers positions [3, nb_markers]

        Returns
        -------
        NaturalCoordinates
            The natural coordinates of the segment [12 x n, 1]
        """
        if markers.shape[1] != self.nb_markers_technical:
            raise ValueError(
                f"markers should have {self.nb_markers_technical} columns, "
                f"and should include the following markers: {self.marker_names_technical}"
            )

        # convert markers to Data
        from ..model_creation.generic_data import GenericData

        marker_data = GenericData(markers, self.marker_names_technical)

        Q = []

        for name in self.segments:
            Q.append(self.segments[name]._Qi_from_markers(marker_data, self))

        return NaturalCoordinates.from_qi(tuple(Q))

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
        for i, segment in enumerate(self.segments):
            # add the joint constraints first
            joints = self.joints_from_child_index(i)
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
        for i in range(self.nb_segments):

            # add the joint constraints first
            joints = self.joints_from_child_index(i)
            if len(joints) != 0:
                for j in joints:
                    idx_row = slice(nb_constraints, nb_constraints + j.nb_constraints)

                    if j.parent is not None:  # If the joint is not a ground joint
                        idx_col_parent = slice(
                            12 * self.segments[j.parent.name].index, 12 * (self.segments[j.parent.name].index + 1)
                        )
                        Q_child = Q.vector(self.segments[j.child.name].index)
                        K[idx_row, idx_col_parent] = j.parent_constraint_jacobian(Q_child)

                    idx_col_child = slice(
                        12 * self.segments[j.child.name].index, 12 * (self.segments[j.child.name].index + 1)
                    )
                    Q_parent = (
                        None if j.parent is None else Q.vector(self.segments[j.parent.name].index)
                    )  # if the joint is a joint with the ground, the parent is None
                    K[idx_row, idx_col_child] = j.child_constraint_jacobian(Q_parent)

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
            joints = self.joints_from_child_index(i)
            if len(joints) != 0:
                for j in joints:
                    idx_row = slice(nb_constraints, nb_constraints + j.nb_constraints)

                    if j.parent is not None:  # If the joint is not a ground joint
                        idx_col_parent = slice(
                            12 * self.segments[j.parent.name].index, 12 * (self.segments[j.parent.name].index + 1)
                        )
                        Qdot_child = Qdot.vector(self.segments[j.child.name].index)
                        Kdot[idx_row, idx_col_parent] = j.parent_constraint_jacobian_derivative(Qdot_child)

                    idx_col_child = slice(
                        12 * self.segments[j.child.name].index, 12 * (self.segments[j.child.name].index + 1)
                    )
                    Qdot_parent = (
                        None if j.parent is None else Qdot.vector(self.segments[j.parent.name].index)
                    )  # if the joint is a joint with the ground, the parent is None
                    Kdot[idx_row, idx_col_child] = j.child_constraint_jacobian_derivative(Qdot_parent)

                    nb_constraints += j.nb_constraints

            # add the rigid body constraint
            idx_row = slice(nb_constraints, nb_constraints + 6)
            idx_rigid_body_constraint = slice(6 * i, 6 * (i + 1))
            idx_segment = slice(12 * i, 12 * (i + 1))

            Kdot[idx_row, idx_segment] = rigid_body_constraints_jacobian_dot[idx_rigid_body_constraint, idx_segment]

            nb_constraints += 6

        return Kdot

    def weight(self) -> np.ndarray:
        """
        This function returns the weights caused by the gravity forces on each segment

        Returns
        -------
            The weight of each segment [12 * nb_segments, 1]
        """
        weight_vector = np.zeros((self.nb_segments * 12, 1))

        for i, segment in enumerate(self.segments.values()):
            idx = slice(12 * i, 12 * (i + 1))
            weight_vector[idx, 0] = segment.weight()

        return weight_vector

    def forward_dynamics(
        self,
        Q: NaturalCoordinates,
        Qdot: NaturalCoordinates,
        # external_forces: ExternalForces
    ) -> np.ndarray:
        """
        This function computes the forward dynamics of the system, i.e. the acceleration of the segments

        Parameters
        ----------
        Q : NaturalCoordinates
            The natural coordinates of the segment [12 * nb_segments, 1]
        Qdot : NaturalCoordinates
            The natural coordinates time derivative of the segment [12 * nb_segments, 1]

        Returns
        -------
            Qddot : NaturalAccelerations
                The natural accelerations [12 * nb_segments, 1]
        """
        G = self.mass_matrix
        K = self.holonomic_constraints_jacobian(Q)
        Kdot = self.holonomic_constraints_jacobian_derivative(Qdot)

        # if stabilization is not None:
        #     biais -= stabilization["alpha"] * self.rigid_body_constraint(Qi) + stabilization[
        #         "beta"
        #     ] * self.rigid_body_constraint_derivative(Qi, Qdoti)

        # KKT system
        # [G, K.T] [Qddot]  = [forces]
        # [K, 0  ] [lambda] = [biais]
        upper_KKT_matrix = np.concatenate((G, K.T), axis=1)
        lower_KKT_matrix = np.concatenate((K, np.zeros((K.shape[0], K.shape[0]))), axis=1)
        KKT_matrix = np.concatenate((upper_KKT_matrix, lower_KKT_matrix), axis=0)

        forces = self.weight()
        biais = -Kdot @ Qdot
        B = np.concatenate([forces, biais], axis=0)

        # solve the linear system Ax = B with numpy
        x = np.linalg.solve(KKT_matrix, B)
        Qddoti = x[0 : self.nb_Qddot]
        lambda_i = x[self.nb_Qddot :]
        return NaturalAccelerations(Qddoti), lambda_i
