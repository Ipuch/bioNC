import ezc3d
import numpy as np


def get_points_ezc3d(c3d_file):
    """
    Extract the points from a c3d file in a more readable format for user allowing to find the index of the points in the numpy array using text using a dictionnary.
    It allows to keep the data in the numpy array format for processing, but allow a more human way of using the data.(Using str instead of indexes)
    Parameters
    ----------
    c3d_file : ezc3d.c3d
        The c3d file to extract the points from
    Returns
    -------
    marker_data : numpy.ndarray
        The points data in a numpy array of shape (3, nb_points, nb_frames)
    marker_names : list
        The name of the points
    marker_idx : dict
        The dictionnary allowing to find the index of the points in the numpy array
    """

    marker_names = c3d_file["parameters"]["POINT"]["LABELS"]["value"]

    marker_data = c3d_file["data"]["points"][0:3, :, :]
    marker_idx = dict(zip(marker_names, range(len(marker_names))))

    return marker_data, marker_names, marker_idx


def add_point_from_dictionary(c3d_file, point_to_add: dict):
    """

    Parameters
    ----------
    c3d_file : ezc3d.c3d
        The c3d file to add the points to
    point_to_add : dict
        The dictionary containing the points to add to the c3d file. The key are the name of the points and the value are the position of the points in a numpy array of shape (3, nb_frames)
    Returns
    -------
    c3d_file : ezc3d.c3d
        The c3d file with the points added
    """

    points, marker_names, points_ind = get_points_ezc3d(c3d_file)
    # copy points informations
    new_list = marker_names.copy()
    new_array = c3d_file["data"]["points"]
    nb_frame = c3d_file["data"]["points"].shape[2]

    nb_markers = c3d_file["data"]["meta_points"]["residuals"].shape[1]

    for ind_point, (name_point, value_point) in enumerate(point_to_add.items()):
        new_point = np.zeros((4, 1, nb_frame))
        new_list.append(name_point)
        new_point[0:3, 0, :] = value_point[:, :]
        new_point[3, 0, :] = 1
        new_array = np.append(new_array, new_point, axis=1)

    # Add the new points to the c3d file
    c3d_file["parameters"]["POINT"]["LABELS"]["value"] = new_list
    c3d_file["parameters"]["POINT"]["DESCRIPTIONS"]["value"] = new_list.copy()

    # Some parameters need to be modified for the c3d to be working
    temp_residuals = np.zeros((1, new_array.shape[1], new_array.shape[2]))
    temp_residuals[0, : nb_markers, :] = c3d_file["data"]["meta_points"][
        "residuals"
    ]
    old_camera_mask = c3d_file["data"]["meta_points"]["camera_masks"]
    temp_camera_mask = np.zeros((old_camera_mask.shape[0], new_array.shape[1], old_camera_mask.shape[2]))
    temp_camera_mask[:, :, :] = False
    temp_camera_mask[:, : nb_markers, :] = old_camera_mask
    c3d_file["data"]["meta_points"]["residuals"] = temp_residuals
    c3d_file["data"]["meta_points"]["camera_masks"] = temp_camera_mask.astype(dtype=bool)
    c3d_file["data"]["points"] = new_array

    return c3d_file
