import numpy as np
from casadi import MX

from .natural_coordinates import SegmentNaturalCoordinates, NaturalCoordinates
from .natural_velocities import SegmentNaturalVelocities, NaturalVelocities


class BiomechanicalModel:
    def __init__(self):
        from .natural_segment import NaturalSegment  # Imported here to prevent from circular imports
        from .joint import Joint  # Imported here to prevent from circular imports

        self.segments: dict[str:NaturalSegment, ...] = {}
        self.joints: dict[str:Joint, ...] = {}
        # From Pythom 3.7 the insertion order in a dict is preserved. This is important because when writing a new
        # the order of the segment matters
        self._mass_matrix = self._update_mass_matrix()

    def __getitem__(self, name: str):
        return self.segments[name]

    def __setitem__(self, name: str, segment: "NaturalSegment"):
        if segment.name == name:  # Make sure the name of the segment fits the internal one
            self.segments[name] = segment
            self._update_mass_matrix()  # Update the generalized mass matrix
        else:
            raise ValueError("The name of the segment does not match the name of the segment")

    def __str__(self):
        out_string = "version 4\n\n"
        for name in self.segments:
            out_string += str(self.segments[name])
            out_string += "\n\n\n"  # Give some space between segments
        return out_string

    def nb_segments(self):
        return len(self.segments)

    def nb_markers(self):
        nb_markers = 0
        for key in self.segments:
            nb_markers += self.segments[key].nb_markers()
        return nb_markers

    def nb_joints(self):
        return len(self.joints)

    def nb_Q(self):
        return 12 * self.nb_segments()

    def nb_Qdot(self):
        return 12 * self.nb_segments()

    def nb_Qddot(self):
        return 12 * self.nb_segments()

    def rigid_body_constraints(self, Q: NaturalCoordinates) -> MX:
        """
        This function returns the rigid body constraints of all segments, denoted Phi_r
        as a function of the natural coordinates Q.

        Returns
        -------
        MX
            Rigid body constraints of the segment [6 * nb_segments, 1]
        """

        Phi_r = MX.zeros(6 * self.nb_segments())
        for i, segment_name in enumerate(self.segments):
            idx = slice(6 * i, 6 * (i + 1))
            Phi_r[idx] = self.segments[segment_name].rigid_body_constraint(Q.vector(i))

        return Phi_r

    def rigid_body_constraints_jacobian(self, Q: NaturalCoordinates) -> MX:
        """
        This function returns the rigid body constraints of all segments, denoted K_r
        as a function of the natural coordinates Q.

        Returns
        -------
        MX
            Rigid body constraints of the segment [6 * nb_segments, nbQ]
        """

        K_r = MX.zeros((6 * self.nb_segments(), Q.shape[0]))
        for i, segment_name in enumerate(self.segments):
            idx_row = slice(6 * i, 6 * (i + 1))
            idx_col = slice(12 * i, 12 * (i + 1))
            K_r[idx_row, idx_col] = self.segments[segment_name].rigid_body_constraint_jacobian(Q.vector(i))

        return K_r

    def rigid_body_constraint_jacobian_derivative(self, Qdot: NaturalVelocities) -> MX:
        """
        This function returns the derivative of the Jacobian matrix of the rigid body constraints denoted Kr_dot

        Parameters
        ----------
        Qdot : NaturalVelocities
            The natural velocities of the segment [12, 1]

        Returns
        -------
        MX
            The derivative of the Jacobian matrix of the rigid body constraints [6, 12]
        """

        Kr_dot = MX.zeros((6 * self.nb_segments(), Qdot.shape[0]))
        for i, segment_name in enumerate(self.segments):
            idx_row = slice(6 * i, 6 * (i + 1))
            idx_col = slice(12 * i, 12 * (i + 1))
            Kr_dot[idx_row, idx_col] = self.segments[segment_name].rigid_body_constraint_jacobian_derivative(
                Qdot.vector(i)
            )

        return Kr_dot

    def _update_mass_matrix(self):
        """
        This function computes the generalized mass matrix of the system, denoted G

        Returns
        -------
        MX
            generalized mass matrix of the segment [12 * nbSegment x 12 * * nbSegment]
        """
        G = MX.zeros((12 * self.nb_segments(), 12 * self.nb_segments()))
        for i, segment_name in enumerate(self.segments):
            Gi = self.segments[segment_name].mass_matrix
            if Gi is None:
                # mass matrix is None if one the segment doesn't have any inertial properties
                self._mass_matrix = None
                return
            idx = slice(12 * i, 12 * (i + 1))
            G[idx, idx] = self.segments[segment_name].mass_matrix

        self._mass_matrix = G

    @property
    def mass_matrix(self):
        """
        This function returns the generalized mass matrix of the system, denoted G

        Returns
        -------
        MX
            generalized mass matrix of the segment [12 * nbSegment x 12 * * nbSegment]

        """
        return self._mass_matrix


# def kinematicConstraints(self, Q):
#     # Method to calculate the kinematic constraints

# def forwardDynamics(self, Q, Qdot):
#
#     return Qddot, lambdas

# def inverseDynamics(self):
