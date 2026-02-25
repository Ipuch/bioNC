from casadi import MX
from typing import Any

from .natural_coordinates import NaturalCoordinates
from .natural_velocities import NaturalVelocities
from ..protocols.biomechanical_model import GenericBiomechanicalModelSegments


class BiomechanicalModelSegments(GenericBiomechanicalModelSegments):
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
        segments: dict[str:Any, ...] = None,
    ):
        super().__init__(segments=segments)

    def rigid_body_constraints(self, Q: NaturalCoordinates) -> MX:
        """
        This function returns the rigid body constraints of all segments, denoted Phi_r
        as a function of the natural coordinates Q.

        Returns
        -------
        MX
            Rigid body constraints of the segment [6 * nb_segments, 1]
        """

        Phi_r = MX.zeros(6 * self.nb_segments)
        for i, segment in enumerate(self.segments_no_ground.values()):
            idx = slice(6 * i, 6 * (i + 1))
            Phi_r[idx] = segment.rigid_body_constraint(Q.vector(i))

        return Phi_r

    def rigid_body_constraints_derivative(self, Q: NaturalCoordinates, Qdot: NaturalCoordinates) -> MX:
        """
        This function returns the derivative of the rigid body constraints of all segments, denoted Phi_r_dot
        as a function of the natural coordinates Q and Qdot.

        Returns
        -------
        MX
            Derivative of the rigid body constraints of the segment [6 * nb_segments, 1]
        """

        Phi_r_dot = MX.zeros(6 * self.nb_segments)
        for i, segment in enumerate(self.segments_no_ground.values()):
            idx = slice(6 * i, 6 * (i + 1))
            Phi_r_dot[idx] = segment.rigid_body_constraint_derivative(Q.vector(i), Qdot.vector(i))

        return Phi_r_dot

    def rigid_body_constraints_jacobian(self, Q: NaturalCoordinates) -> MX:
        """
        This function returns the rigid body constraints of all segments, denoted K_r
        as a function of the natural coordinates Q.

        Returns
        -------
        MX
            Rigid body constraints of the segment [6 * nb_segments, nbQ]
        """

        K_r = MX.zeros((6 * self.nb_segments, Q.shape[0]))
        for i, segment in enumerate(self.segments_no_ground.values()):
            idx_row = slice(6 * i, 6 * (i + 1))
            idx_col = slice(12 * i, 12 * (i + 1))
            K_r[idx_row, idx_col] = segment.rigid_body_constraint_jacobian(Q.vector(i))

        return K_r


