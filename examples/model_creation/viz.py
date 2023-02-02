"""
Example script for animating markers
"""
from enum import Enum

from bionc import BiomechanicalModel, NaturalCoordinates, SegmentNaturalCoordinates
import numpy as np
from pyomeca import Markers
from bioviz.biorbd_vtk import VtkModel, VtkWindow


class NaturalVectorColors(Enum):
    """Colors for the vectors."""

    # light red
    U = (255, 20, 0)
    # light green
    V = (0, 255, 20)
    # light blue
    W = (20, 0, 255)


class VectorColors(Enum):
    """Colors for the vectors."""

    # red
    X = (255, 0, 0)
    # green
    Y = (0, 255, 0)
    # blue
    Z = (0, 0, 255)


class VtkFrameModel:
    def __init__(
        self,
        vtk_window: VtkWindow,
        normalized: bool = False,
    ):
        self.vtk_window = vtk_window
        self.three_vectors = [self.vtk_vector_model(color) for color in NaturalVectorColors]
        self.normalized = normalized

    def vtk_vector_model(self, color: NaturalVectorColors):
        vtkModelReal = VtkModel(
            self.vtk_window,
            force_color=color.value,
            force_opacity=1.0,
        )
        return vtkModelReal

    def update_frame(self, Q: SegmentNaturalCoordinates):
        # in bioviz vectors are displayed through "forces"
        u_vector = np.concatenate((Q.rp, Q.rp + Q.u))
        v_vector = np.concatenate((Q.rp, Q.rd))
        w_vector = np.concatenate((Q.rp, Q.rp + Q.w))

        self.three_vectors[0].update_force(
            segment_jcs=[np.identity(4)],
            all_forces=u_vector[np.newaxis, :, np.newaxis],
            max_forces=[1],
            normalization_ratio=0.3,
        )
        self.three_vectors[1].update_force(
            segment_jcs=[np.identity(4)],
            all_forces=v_vector[np.newaxis, :, np.newaxis],
            max_forces=[1],
            normalization_ratio=1,
        )
        self.three_vectors[2].update_force(
            segment_jcs=[np.identity(4)],
            all_forces=w_vector[np.newaxis, :, np.newaxis],
            max_forces=[1],
            normalization_ratio=0.3,
        )


class VtkGroundFrame:
    def __init__(
        self,
        vtk_window: VtkWindow,
    ):
        self.vtk_window = vtk_window
        self.three_vectors = [self.vtk_vector_model(color) for color in VectorColors]

    def vtk_vector_model(self, color: VectorColors):
        vtkModelReal = VtkModel(
            self.vtk_window,
            force_color=color.value,
            force_opacity=1.0,
        )
        return vtkModelReal

    def update_frame(self):
        # in bioviz vectors are displayed through "forces"
        x_vector = np.array([0, 0, 0, 1, 0, 0])
        y_vector = np.array([0, 0, 0, 0, 1, 0])
        z_vector = np.array([0, 0, 0, 0, 0, 1])

        self.three_vectors[0].update_force(
            segment_jcs=[np.identity(4)],
            all_forces=x_vector[np.newaxis, :, np.newaxis],
            max_forces=[1],
            normalization_ratio=1,
        )
        self.three_vectors[1].update_force(
            segment_jcs=[np.identity(4)],
            all_forces=y_vector[np.newaxis, :, np.newaxis],
            max_forces=[1],
            normalization_ratio=1,
        )
        self.three_vectors[2].update_force(
            segment_jcs=[np.identity(4)],
            all_forces=z_vector[np.newaxis, :, np.newaxis],
            max_forces=[1],
            normalization_ratio=1,
        )


def cheap_animation(model: BiomechanicalModel, Q: NaturalCoordinates, markers_xp):
    """
    This function is a cheap animation of markers

    Parameters
    ----------
    model: BiomechanicalModel
        The model
    Q : NaturalCoordinates
        The natural coordinates of the segment
    """

    # Create a windows with a nice gray background
    vtkWindow = VtkWindow(background_color=(0.5, 0.5, 0.5))

    ground_frame = VtkGroundFrame(vtkWindow)
    frames = []
    for s in range(model.nb_segments):
        frames.append(VtkFrameModel(vtkWindow))

    # center_of_mass = []
    # center_of_mass_locations = np.zeros((3, model.nb_segments, Q.shape[1]))
    # for i_s in range(model.nb_segments):
    #     center_of_mass.append(
    #         VtkModel(
    #             vtkWindow,
    #             markers_color=(1, 0, 0),
    #             markers_size=0.02,
    #             markers_opacity=1,
    #         )
    #     )

    # center_of_mass_locations[:, :, :] = model.center_of_mass_position(Q)
    # center_of_mass_locations = Markers(center_of_mass_locations)

    pyo_xp_markers = Markers(markers_xp)
    model_markers = model.markers(Q)
    pyo_model_markers = Markers(model_markers)

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
        # Update the markers
        vtkModelReal.update_markers(pyo_xp_markers[:, :, i])
        vtkModelModel.update_markers(pyo_model_markers[:, :, i])

        # Update the frames
        ground_frame.update_frame()
        for s in range(Q.nb_qi()):
            frames[s].update_frame(Q.vector(s)[:, i : i + 1])

        # for s, com in enumerate(center_of_mass):
        #     com.update_markers(center_of_mass_locations[:, s : s + 1, i])

        # Update window
        vtkWindow.update_frame()
        i = (i + 1) % Q.shape[1]


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
