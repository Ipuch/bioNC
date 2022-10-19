import numpy as np

from ..utils.natural_velocities import NaturalVelocities
from ..utils.natural_coordinates import NaturalCoordinates
from ..utils.natural_accelerations import NaturalAccelerations


class BiomechanicalModel:
    def __init__(self):
        from .natural_segment import NaturalSegment  # Imported here to prevent from circular imports
        from .joint import Joint  # Imported here to prevent from circular imports

        self.segments: dict[str:NaturalSegment, ...] = {}
        self.joints: dict[str:Joint, ...] = {}
        # From Pythom 3.7 the insertion order in a dict is preserved. This is important because when writing a new
        # .bioMod file, the order of the segment matters
        self._generalized_mass_matrix = self._generalized_mass_matrix()

    def __getitem__(self, name: str):
        return self.segments[name]

    def __setitem__(self, name: str, segment: "NaturalSegment"):
        if segment.name == name:  # Make sure the name of the segment fits the internal one
            self.segments[name] = segment
        else:
            raise ValueError("The name of the segment does not match the name of the segment")

    def __str__(self):
        out_string = "version 4\n\n"
        for name in self.segments:
            out_string += str(self.segments[name])
            out_string += "\n\n\n"  # Give some space between segments
        return out_string

    def nbSegments(self):
        return len(self.segments)

    def nbMarkers(self):
        nbMarkers = 0
        for segment in self.segments:
            nbMarkers += len(self.segments[segment].markers)
        return nbMarkers

    def nbJoints(self):
        nbJoints = 0
        for segment in self.segments:
            if self.segments[segment].parent_name:
                nbJoints += 1
        return nbJoints

    def rigidBodyConstraints(self, Q: NaturalCoordinates) -> np.ndarray:
        """
        This function returns the rigid body constraints of all segments, denoted Phi_r
        as a function of the natural coordinates Q.

        Returns
        -------
        np.ndarray
            Rigid body constraints of the segment [6 * nbSegments, 1]
        """

        if not isinstance(Q, NaturalCoordinates):
            Q = NaturalCoordinates(Q)

        Phi_r = np.zeros(6 * self.nbSegments())
        for i, segment in enumerate(self.segments):
            idx = slice(6 * i, 6 * (i + 1))
            Phi_r[idx] = self.segments[segment].rigidBodyConstraints(Q.vector(i))

        return Phi_r

    def rigidBodyConstraintsJacobian(self, Q: NaturalCoordinates) -> np.ndarray:
        """
        This function returns the rigid body constraints of all segments, denoted K_r
        as a function of the natural coordinates Q.

        Returns
        -------
        np.ndarray
            Rigid body constraints of the segment [6 * nbSegments, nbQ]
        """
        if not isinstance(Q, NaturalCoordinates):
            Q = NaturalCoordinates(Q)

        K_r = np.zeros((6 * self.nbSegments(), Q.shape[0]))
        for i, segment in enumerate(self.segments):
            idx = slice(6 * i, 6 * (i + 1))
            K_r[idx, idx] = self.segments[segment].rigidBodyConstraintsJacobian(Q.vector(i))

        return K_r

    def rigidBodyConstraintJacobianDerivative(self, Qdot: NaturalVelocities) -> np.ndarray:
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

        if not isinstance(Qdot, NaturalVelocities):
            Qdot = NaturalVelocities(Qdot)

        Kr_dot = np.zeros((6 * self.nbSegments(), Qdot.shape[0]))
        for i, segment in enumerate(self.segments):
            idx = slice(6 * i, 6 * (i + 1))
            Kr_dot[idx, idx] = self.segments[segment].rigidBodyConstraintJacobianDerivative(Qdot.vector(i))

        return np.zeros((6, 12))

    def _generalized_mass_matrix(self):
        """
        This function computes the generalized mass matrix of the system, denoted G

        Returns
        -------
        np.ndarray
            generalized mass matrix of the segment [12 * nbSegment x 12 * * nbSegment]
        """
        G = np.zeros((12 * self.nbSegments(), 12 * self.nbSegments()))
        for i, segment in enumerate(self.segments):
            idx = slice(12 * i, 12 * (i + 1))
            G[idx, idx] = self.segments[segment].generalized_mass_matrix

        return G

    @property
    def generalized_mass_matrix(self):
        """
        This function returns the generalized mass matrix of the system, denoted G

        Returns
        -------
        np.ndarray
            generalized mass matrix of the segment [12 * nbSegment x 12 * * nbSegment]

        """
        return self._generalized_mass_matrix


# def kinematicConstraints(self, Q):
#     # Method to calculate the kinematic constraints

# def forwardDynamics(self, Q, Qdot):
#
#     return Qddot, lambdas

# def inverseDynamics(self):
