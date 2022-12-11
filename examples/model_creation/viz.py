"""
Example script for animating markers
"""

import numpy as np
from pyomeca import Markers
from bioviz.biorbd_vtk import VtkModel, VtkWindow


def cheap_markers_animation(xp_markers: np.ndarray, model_markers: np.ndarray):
    """
    This function is a cheap animation of markers

    Parameters
    ----------
    xp_markers: np.ndarray
        Markers position in the experimental space
    model_markers: np.ndarray
        Markers position in the model space
    """

    pyo_xp_markers = Markers(xp_markers)
    pyo_model_markers = Markers(model_markers)

    # Create a windows with a nice gray background
    vtkWindow = VtkWindow(background_color=(0.5, 0.5, 0.5))

    # Add marker holders to the window
    vtkModelReal = VtkModel(
        vtkWindow,
        markers_color=(1, 0, 0),
        markers_size=0.02,
        markers_opacity=1,
    )
    vtkModelModel = VtkModel(
        vtkWindow,
        markers_color=(0, 1, 0),
        markers_size=0.02,
        markers_opacity=1,
    )

    # Animate all this
    i = 0
    while vtkWindow.is_active:

        vtkModelReal.update_markers(pyo_xp_markers[:, :, i])
        vtkModelModel.update_markers(pyo_model_markers[:, :, i])

        # Update window
        vtkWindow.update_frame()
        i = (i + 1) % model_markers.shape[2]


if __name__ == "__main__":
    model_markers = np.linspace(
        start=np.array(
            [
                [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
                [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
                [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
            ]
        ),
        stop=np.array(
            [
                [0.02, 0.13, 0.24, 0.35, 0.46, 0.57, 0.68, 0.79, 0.9, 1.01, 1.12],
                [0.02, 0.13, 0.24, 0.35, 0.46, 0.57, 0.68, 0.79, 0.9, 1.01, 1.12],
                [0.02, 0.13, 0.24, 0.35, 0.46, 0.57, 0.68, 0.79, 0.9, 1.01, 1.12],
            ]
        ),
        num=100,
        axis=2,
    )

    xp_markers = (
        np.linspace(
            start=np.array(
                [
                    [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
                    [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
                    [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
                ]
            ),
            stop=np.array(
                [
                    [0.02, 0.13, 0.24, 0.35, 0.46, 0.57, 0.68, 0.79, 0.9, 1.01, 1.12],
                    [0.02, 0.13, 0.24, 0.35, 0.46, 0.57, 0.68, 0.79, 0.9, 1.01, 1.12],
                    [0.02, 0.13, 0.24, 0.35, 0.46, 0.57, 0.68, 0.79, 0.9, 1.01, 1.12],
                ]
            ),
            num=100,
            axis=2,
        )
        + np.random.random(1) * 0.01
    )

    cheap_markers_animation(xp_markers, model_markers)
