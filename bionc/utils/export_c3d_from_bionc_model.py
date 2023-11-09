import ezc3d
import numpy as np
from bionc.bionc_numpy import NaturalCoordinates


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


def export_c3d_biomechanical(model, Q, filename_export, filename_initial=None, fq_file=None):
    """

    Parameters
    ----------
    model : BiomechanicalModel
        The biomechanical model from which the data can be exported
    Q : numpy.ndarray | NaturalCoordinates
        The natural coordinates of the model, should be given as a Natural coordinates but if given as a numpy array it will be converted
    filename_export : str | path
        The path to the c3d file to export
    initial_file : str | path
        initial c3d file to add the data to, if let to the None value a new c3d file will be created
    fq_file : int
        frequency of the data if no initial file is given

    Returns
    -------

    """

    if Q is not isinstance(Q, NaturalCoordinates):
        Q = NaturalCoordinates(Q)

    model_markers = model.markers(Q)

    dict_to_add = dict()
    # We add the technical markers
    for ind_marker, name_marker in enumerate(model.marker_names):
        dict_to_add[f"{name_marker}_optim"] = model_markers[:, ind_marker, :]

    # We add the segment rp,rd,u,w to the c3d file
    for s in range(Q.nb_qi()):
        name_segment = model.segment_names[s]
        Qi = Q.vector(s)
        rp_temp = Qi.rp
        rd_temp = Qi.rd
        u_temp = Qi.u
        w_temp = Qi.w
        dict_to_add[f"u_{name_segment}"] = rp_temp + u_temp / 10
        dict_to_add[f"rp_{name_segment}"] = rp_temp
        dict_to_add[f"rd_{name_segment}"] = rd_temp
        dict_to_add[f"w_{name_segment}"] = rd_temp + w_temp / 10

    if filename_initial is None:
        original_c3d = ezc3d.ezc3dRead()
        if fq_file is not None:
            original_c3d.parameters.POINT.RATE.DATA = fq_file
        else:
            original_c3d.parameters.POINT.RATE.DATA = 100
    else:
        original_c3d = ezc3d.c3d(filename_initial)

    add_point_from_dictionary(original_c3d, dict_to_add)
    original_c3d.write(filename_export)
