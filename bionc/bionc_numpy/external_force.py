import numpy as np

# from .natural_coordinates import SegmentNaturalCoordinates
# from .natural_vector import NaturalVector


def compute_pseudo_interpolation_matrix(Qi):
    """
    Return the pseudo interpolation matrix
    """
    pseudo_interpolation_matrix = np.zeros((3, 12))

    pseudo_interpolation_matrix[0, 9:12] = Qi.u
    pseudo_interpolation_matrix[1, 0:3] = Qi.v
    pseudo_interpolation_matrix[2, 3:6] = -Qi.w
    pseudo_interpolation_matrix[2, 6:3] = Qi.w

    return pseudo_interpolation_matrix


def compute_force_moment_transformation_matrix(Qi):
    """
    Return the force moment transformation matrix
    """
    # default we apply force at the proximal point
    force_moment_transformation_matrix = np.zeros((3, 12))

    force_moment_transformation_matrix[:, 0] = np.cross(Qi.w, Qi.u)
    force_moment_transformation_matrix[:, 1] = np.cross(Qi.u, Qi.v)
    force_moment_transformation_matrix[:, 2] = np.cross(-Qi.v, Qi.w)

    return force_moment_transformation_matrix


def to_natural_force(Qi, external_forces: np.ndarray, application_point_in_global: np.ndarray):
    """
    Apply external forces to the segment

    Parameters
    ----------
    Qi: SegmentNaturalCoordinates
        Segment natural coordinates

    external_forces: np.ndarray
        External forces in cartesian coordinates

    Returns
    -------

    """
    torque = external_forces[3:6]
    force = external_forces[0:3]

    pseudo_interpolation_matrix = compute_pseudo_interpolation_matrix(Qi)
    force_moment_transformation_matrix = compute_force_moment_transformation_matrix(Qi)

    fext = NaturalVector.proximal().interlopate() @ force
    fext += force_moment_transformation_matrix.T @ torque
    fext += force_moment_transformation_matrix.T @ np.cross(application_point_in_global - Qi.rp, force)

    return fext


# try the functions
# Qi = SegmentNaturalCoordinates.from_components(
#     u=np.array([1, 0, 0]),
#     rp=np.array([0, 0, 0]),
#     rd=np.array([0, 0, 0]),
#     w=np.array([0, 0, 1]),
# )
# external_force = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
# application_point_in_local = np.array([0.1, 0.2, 0.3])
# natural_force = to_natural_force(Qi, external_force, application_point_in_local)
