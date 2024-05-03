"""
This module contains helper functions to compute the total confidence for all frames
"""

import numpy as np

from .heatmap_helpers import (
    compute_total_confidence,
    compute_confidence_for_one_marker,
    rearrange_gaussian_parameters,
    compute_confidence_for_one_marker_one_camera,
)
from ..bionc_numpy.natural_coordinates import NaturalCoordinates


def subset_of_technical_markers(model, marker_positions):
    """Returns the subset of technical markers from the marker positions."""
    # todo: we only want technical markers, implement a model.markers(Q_f, only_technical=True)
    marker_names_technical = model.marker_names_technical
    marker_names = model.marker_names
    technical_index = [marker_names.index(m) for m in marker_names_technical]

    return marker_positions[:, technical_index]


def total_confidence(model, Q, camera_parameters, gaussian_parameters):
    """
    Compute the total confidence for all frames.

    Parameters
    ----------
    model: bionc_numpy.BiomechanicalModelMarkers
        The biomechanical model
    Q:  np.ndarray
        The natural coordinates of size [12 x n_segments, nb_frames]
    camera_parameters: np.ndarray
        The camera parameters of size [12, nb_cameras]
    gaussian_parameters: np.ndarray
        The gaussian parameters of size [5*nb_markers, nb_frame, nb_cameras]

    Returns
    -------
    np.ndarray
        The total confidence for all frames of size [nb_frames]
    """
    nb_frames = Q.shape[1]
    frame_total_confidence = np.zeros(nb_frames)

    for frame in range(nb_frames):
        Qf = NaturalCoordinates(Q[:, frame])
        all_marker_position = model.markers(Qf)
        marker_positions = subset_of_technical_markers(model, all_marker_position)

        frame_total_confidence[frame] = (
            compute_total_confidence(marker_positions, camera_parameters, gaussian_parameters[:, frame, :])
            .toarray()
            .squeeze()
        )

    return frame_total_confidence


def total_confidence_for_all_markers(model, Q, camera_parameters, gaussian_parameters):
    """
    Compute the total confidence for all frames.

    Parameters
    ----------
    model: bionc_numpy.BiomechanicalModelMarkers
        The biomechanical model
    Q:  np.ndarray
        The natural coordinates of size [12 x n_segments, nb_frames]
    camera_parameters: np.ndarray
        The camera parameters of size [12, nb_cameras]
    gaussian_parameters: np.ndarray
        The gaussian parameters of size [5*nb_markers, nb_frame, nb_cameras]

    Returns
    -------
    np.ndarray
        The total confidence of each marker for all frames of size [nb_frames], i.e. 3d heatmap confidence
    """
    nb_frames = Q.shape[1]
    nb_cameras = camera_parameters.shape[1]
    nb_markers = model.nb_markers_technical

    heatmap_confidences_3d = np.zeros((nb_markers, nb_frames))

    for frame in range(nb_frames):
        Qf = NaturalCoordinates(Q[:, frame])
        all_marker_position = model.markers(Qf)
        marker_positions = subset_of_technical_markers(model, all_marker_position)
        reshaped_gaussian_parameters = rearrange_gaussian_parameters(
            gaussian_parameters[:, frame, :], nb_cameras, nb_markers
        )

        for m in range(nb_markers):
            m_offset = 5 * m
            marker_gaussian_parameters = reshaped_gaussian_parameters[m_offset : m_offset + 5, :]
            the_marker_position = marker_positions[:, m]

            heatmap_confidences_3d[m, frame] = compute_confidence_for_one_marker(
                the_marker_position, camera_parameters, marker_gaussian_parameters
            )

    return heatmap_confidences_3d


def total_confidence_for_all_markers_on_each_camera(model, Q, camera_parameters, gaussian_parameters):
    """
    Compute the total confidence for all frames.

    Parameters
    ----------
    model: bionc_numpy.BiomechanicalModelMarkers
        The biomechanical model
    Q:  np.ndarray
        The natural coordinates of size [12 x n_segments, nb_frames]
    camera_parameters: np.ndarray
        The camera parameters of size [12, nb_cameras]
    gaussian_parameters: np.ndarray
        The gaussian parameters of size [5*nb_markers, nb_frame, nb_cameras]

    Returns
    -------
    np.ndarray
        The total confidence of each marker for all frames of size [nb_frames], i.e. 3d heatmap confidence
    """
    nb_frames = Q.shape[1]
    nb_cameras = camera_parameters.shape[1]
    nb_markers = model.nb_markers_technical

    heatmap_confidences_2d = np.zeros((nb_markers, nb_cameras, nb_frames))

    for frame in range(nb_frames):
        Qf = NaturalCoordinates(Q[:, frame])
        all_marker_position = model.markers(Qf)
        marker_positions = subset_of_technical_markers(model, all_marker_position)
        reshaped_gaussian_parameters = rearrange_gaussian_parameters(
            gaussian_parameters[:, frame, :], nb_cameras, nb_markers
        )

        for m in range(nb_markers):
            m_offset = 5 * m
            marker_gaussian_parameters = reshaped_gaussian_parameters[m_offset : m_offset + 5, :]
            the_marker_position = marker_positions[:, m]

            for c in range(nb_cameras):
                heatmap_confidences_2d[m, c, frame] = compute_confidence_for_one_marker_one_camera(
                    the_marker_position, camera_parameters[:, c], marker_gaussian_parameters[:, c]
                )

    return heatmap_confidences_2d


class HeatmapTimeseriesHelpers:
    """
    This class contains helper functions to compute the total confidence for all frames
    """

    total_confidence = total_confidence
    total_confidence_for_all_markers = total_confidence_for_all_markers
    total_confidence_for_all_markers_on_each_camera = total_confidence_for_all_markers_on_each_camera
