from casadi import MX

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
