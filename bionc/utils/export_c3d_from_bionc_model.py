import ezc3d
import numpy as np
from bionc.bionc_numpy import NaturalCoordinates
from math import ceil, floor, log10

def get_points_ezc3d(acq):
    """
    Extract the points from a c3d file in a more readable format for user allowing to find the index of the points in the numpy array using text using a dictionnary.
    It allows to keep the data in the numpy array format for processing, but allow a more human way of using the data.(Using str instead of indexes)
    Parameters
    ----------
    acq : ezc3d.c3d
        The c3d file to extract the points from
    Returns
    -------
    points_data : numpy.ndarray
        The points data in a numpy array of shape (3, nb_points, nb_frames)
    points_name : list
        The name of the points
    points_ind : dict
        The dictionnary allowing to find the index of the points in the numpy array
    """

    points_name = acq["parameters"]["POINT"]["LABELS"]["value"]

    points_data = acq["data"]["points"][0:3, :, :]
    points_ind = dict()
    for index_point, name_point in enumerate(points_name):
        points_ind[name_point] = index_point

    return points_data, points_name, points_ind


def add_point_from_dictionary(acq, point_to_add):
    """

    Parameters
    ----------
    acq : ezc3d.c3d
        The c3d file to add the points to
    point_to_add : dict
        The dictionnary containing the points to add to the c3d file. The key are the name of the points and the value are the position of the points in a numpy array of shape (3, nb_frames)
    Returns
    -------
    acq : ezc3d.c3d
        The c3d file with the points added
    """

    points, points_name, points_ind = get_points_ezc3d(acq)
    # copy points informations
    new_list = points_name.copy()
    new_array = acq["data"]["points"]
    nb_frame = acq["data"]["points"].shape[2]

    for ind_point, (name_point, value_point) in enumerate(point_to_add.items()):
        new_point = np.zeros((4, 1, nb_frame))
        new_list.append(name_point)
        new_point[0:3, 0, :] = value_point[:, :]
        new_point[3, 0, :] = 1
        new_array = np.append(new_array, new_point, axis=1)

    # Add the new points to the c3d file
    acq["parameters"]["POINT"]["LABELS"]["value"] = new_list
    acq["parameters"]["POINT"]["DESCRIPTIONS"]["value"] = new_list.copy()

    # Some parameters need to be modified for the c3d to be working
    temp_residuals = np.zeros((1, new_array.shape[1], new_array.shape[2]))
    temp_residuals[0, : acq["data"]["meta_points"]["residuals"].shape[1], :] = acq["data"]["meta_points"]["residuals"]
    old_camera_mask = acq["data"]["meta_points"]["camera_masks"]
    temp_camera_mask = np.zeros((old_camera_mask.shape[0], new_array.shape[1], old_camera_mask.shape[2]))
    temp_camera_mask[:, :, :] = False
    temp_camera_mask[:, : acq["data"]["meta_points"]["residuals"].shape[1], :] = old_camera_mask
    acq["data"]["meta_points"]["residuals"] = temp_residuals
    acq["data"]["meta_points"]["camera_masks"] = temp_camera_mask.astype(dtype=bool)
    # Add the new analogs to the c3d file used for the type 2 platform
    acq["data"]["points"] = new_array

    return acq


def add_technical_markers_to_c3d(acq, model, Q):
    """
    This function add the technical markers of the model to the c3d file. This point are the markers that are rigidly associated to the
    segments of the model.
    Parameters
    ----------
    acq : ezc3d.c3d
        The c3d file to add the points to
    model : BiomechanicalModel
        The biomechanical model from which the data can be exported
    Q : numpy.ndarray | NaturalCoordinates
        The natural coordinates of the model, should be given as a Natural coordinates but if given as a numpy array it will be converted

    Returns
    -------
    acq : ezc3d.c3d
        The c3d file with natural coordinate points added
    """

    if Q is not isinstance(Q, NaturalCoordinates):
        Q = NaturalCoordinates(Q)

    model_markers = model.markers(Q)

    dict_to_add = dict()
    # We add the technical markers
    for ind_marker, name_marker in enumerate(model.marker_names):
        dict_to_add[f"{name_marker}_optim"] = model_markers[:, ind_marker, :]

    add_point_from_dictionary(acq, dict_to_add)

    return acq


def add_natural_coordinate_to_c3d(acq, model, Q):
    """
    This function add the natural coordinate of the model to the c3d file. It add the segment rp,rd,u,w to the c3d file.
    Parameters
    ----------
    acq : ezc3d.c3d
        The c3d file to add the points to
    model : BiomechanicalModel
        The biomechanical model from which the data can be exported
    Q : numpy.ndarray | NaturalCoordinates
        The natural coordinates of the model, should be given as a Natural coordinates but if given as a numpy array it will be converted

    Returns
    -------
    acq : ezc3d.c3d
        The c3d file with natural coordinate points added
    """

    if Q is not isinstance(Q, NaturalCoordinates):
        Q = NaturalCoordinates(Q)
    # Calulation of a reasonable factor for the u and w
    list_factor = []
    for s in range(Q.nb_qi()):
        name_segment = model.segment_names[s]
        Qi = Q.vector(s)
        rp_temp = Qi.rp
        rd_temp = Qi.rd
        u_temp = Qi.u
        v_mean = np.mean(np.linalg.norm(rd_temp - rp_temp, axis=0))
        u_mean = np.mean(np.linalg.norm(u_temp, axis=0))
        list_factor.append(ceil(log10(u_mean / v_mean)))

    most_occurence = max(set(list_factor), key=list_factor.count)
    factor = 10**most_occurence

    dict_to_add = dict()
    # We add the segment rp,rd,u,w to the c3d file
    for s in range(Q.nb_qi()):
        name_segment = model.segment_names[s]
        Qi = Q.vector(s)
        rp_temp = Qi.rp
        rd_temp = Qi.rd
        u_temp = Qi.u
        w_temp = Qi.w

        v_mean = np.mean(np.linalg.norm(rd_temp - rp_temp, axis=0))
        u_mean = np.mean(np.linalg.norm(u_temp, axis=0))
        dict_to_add[f"u_{name_segment}"] = rp_temp + u_temp / factor
        dict_to_add[f"rp_{name_segment}"] = rp_temp
        dict_to_add[f"rd_{name_segment}"] = rd_temp
        dict_to_add[f"w_{name_segment}"] = rd_temp + w_temp / factor

    add_point_from_dictionary(acq, dict_to_add)

    return acq
