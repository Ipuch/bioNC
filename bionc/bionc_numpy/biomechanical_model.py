import numpy as np
from numpy import transpose

from bionc.protocols.natural_coordinates import NaturalCoordinates
from bionc.bionc_numpy.natural_velocities import NaturalVelocities
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

        Phi_r = np.zeros(6 * self.nb_segments())
        for i, segment_name in enumerate(self.segments):
            idx = slice(6 * i, 6 * (i + 1))
            Phi_r[idx] = self.segments[segment_name].rigid_body_constraint(Q.vector(i))

        return Phi_r

    def rigid_body_constraints_jacobian(self, Q: NaturalCoordinates) -> np.ndarray:
        """
        This function returns the rigid body constraints of all segments, denoted K_r
        as a function of the natural coordinates Q.

        Returns
        -------
        np.ndarray
            Rigid body constraints of the segment [6 * nb_segments, nbQ]
        """

        K_r = np.zeros((6 * self.nb_segments(), Q.shape[0]))
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

        Kr_dot = np.zeros((6 * self.nb_segments(), Qdot.shape[0]))
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

        Phi_k = np.zeros(self.nb_joint_constraints())
        nb_constraints = 0
        for joint_name, joint in self.joints.items():
            idx = slice(nb_constraints, nb_constraints + joint.nb_constraints)

            Q_parent = Q.vector(self.segments[joint.parent.name].index)
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

        K_k = np.zeros((self.nb_joint_constraints(), Q.shape[0]))
        nb_constraints = 0
        for joint_name, joint in self.joints.items():
            idx_row = slice(nb_constraints, nb_constraints + joint.nb_constraints)
            idx_col_parent = slice(
                12 * self.segments[joint.parent.name].index, 12 * (self.segments[joint.parent.name].index + 1)
            )
            idx_col_child = slice(
                12 * self.segments[joint.child.name].index, 12 * (self.segments[joint.child.name].index + 1)
            )

            Q_parent = Q.vector(self.segments[joint.parent.name].index)
            Q_child = Q.vector(self.segments[joint.child.name].index)
            K_k[idx_row, idx_col_parent], K_k[idx_row, idx_col_child] = joint.constraint_jacobian(Q_parent, Q_child)

            nb_constraints += self.joints[joint_name].nb_constraints

        return K_k

    def _update_mass_matrix(self):
        """
        This function computes the generalized mass matrix of the system, denoted G

        Returns
        -------
        np.ndarray
            generalized mass matrix of the segment [12 * nbSegment x 12 * * nbSegment]
        """
        G = np.zeros((12 * self.nb_segments(), 12 * self.nb_segments()))
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

    def markers_constraints(self, markers: np.ndarray, Q: NaturalCoordinates) -> np.ndarray:
        """
        This function returns the marker constraints of all segments, denoted Phi_r
        as a function of the natural coordinates Q.

        markers : np.ndarray
            The markers positions [3,nb_markers]

        Q : NaturalCoordinates
            The natural coordinates of the segment [12 x n, 1]

        Returns
        -------
        np.ndarray
            Rigid body constraints of the segment [nb_segments, 1]
        """
        if markers.shape[1] != self.nb_markers():
            raise ValueError(f"markers should have {self.nb_markers()} columns")

        phi_m = np.zeros(self.nb_markers())
        marker_count = 0
        for i_segment, segment_name in enumerate(self.segments):
            marker_idx = slice(marker_count, marker_count+self.segments[segment_name].nb_markers)
            markers_temp = markers[:, marker_idx]
            phi_m[marker_idx] = self.segments[segment_name].markers_constraints(markers_temp, Q.vector(i_segment))
            marker_count += self.segments[segment_name].nb_markers

        return phi_m

    def marker_constraint_jacobian(self):
        """
        This function returns the Jacobian matrix the markers constraints, denoted k_m.

        Returns
        -------
        np.ndarray
            Joint constraints of the segment [3xnb_marker, 12xnb_segment]/[3xnb_marker, nbQ]
        """

        km = np.zeros((3*self.nb_markers(), 12 * self.nb_segments()))
        marker_count = 0
        for i_segment, segment_name in enumerate(self.segments):
            marker_idx = slice(marker_count, marker_count+self.segments[segment_name].nb_markers)
            segment_idx = slice(12 * i_segment, 12 * (i_segment + 1))
            km[marker_idx,segment_idx] = self.segments[segment_name].marker_constraint_jacobian()
            marker_count += self.segments[segment_name].nb_markers

        return km
