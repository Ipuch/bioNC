"""
Example script for animating markers
"""
from enum import Enum
from ..protocols.biomechanical_model import GenericBiomechanicalModel as BiomechanicalModel
from ..protocols.natural_coordinates import SegmentNaturalCoordinates, NaturalCoordinates
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


class Viz:
    def __init__(
        self,
        model: BiomechanicalModel,
        show_ground_frame: bool = True,
        show_frames: bool = True,
        show_model_markers: bool = True,
        show_xp_markers: bool = True,
        show_center_of_mass: bool = True,
    ):
        self.model = model
        self.show_ground_frame = show_ground_frame
        self.show_frames = show_frames
        self.show_model_markers = show_model_markers
        self.show_xp_markers = show_xp_markers
        self.show_center_of_mass = show_center_of_mass

        # Create a windows with a nice gray background
        self.vtkWindow = VtkWindow(background_color=(0.5, 0.5, 0.5))

        if self.show_ground_frame:
            self.ground_frame = VtkGroundFrame(self.vtkWindow)
        if self.show_frames:
            self.frames = []
            for s in range(model.nb_segments):
                self.frames.append(VtkFrameModel(self.vtkWindow))
        if self.show_center_of_mass:
            self.center_of_mass = []
            for i_s in range(model.nb_segments):
                self.center_of_mass.append(
                    VtkModel(
                        self.vtkWindow,
                        markers_color=(1, 0, 0),
                        markers_size=0.02,
                        markers_opacity=1,
                    )
                )
        if self.show_model_markers:
            self.vtkModelModel = VtkModel(
                self.vtkWindow,
                markers_color=(0, 1, 0),
                markers_size=0.02,
                markers_opacity=1,
            )
        if self.show_xp_markers:
            self.vtkModelReal = VtkModel(
                self.vtkWindow,
                markers_color=(1, 0, 0),
                markers_size=0.02,
                markers_opacity=1,
            )

    def animate(self, Q: NaturalCoordinates | np.ndarray, markers_xp):
        """
        This function is a cheap animation of markers

        Parameters
        ----------
        Q : NaturalCoordinates | np.ndarray
            The natural coordinates of the segment of shape (n_dofs, n_frames)
        markers_xp : np.ndarray
            The experimental markers measured in global frame of shape (3, n_markers, n_frames)
        """

        if Q.shape[1] != markers_xp.shape[2]:
            raise ValueError(
                f"Q and markers_xp must have the same number of frames. Q.shape[1]={Q.shape[1]} and markers_xp.shape[2]={markers_xp.shape[2]}"
            )
        from ..bionc_numpy import NaturalCoordinates

        Q = NaturalCoordinates(Q)

        if self.show_xp_markers:
            pyo_xp_markers = Markers(markers_xp)
        if self.show_model_markers:
            model_markers = self.model.markers(Q)
            pyo_model_markers = Markers(model_markers)

        if self.show_center_of_mass:
            center_of_mass_locations = np.zeros((3, self.model.nb_segments, Q.shape[1]))
            center_of_mass_locations[:, :, :] = self.model.center_of_mass_position(Q)
            center_of_mass_locations = Markers(center_of_mass_locations)

        # Animate all this
        i = 0
        while self.vtkWindow.is_active:
            # Update the markers
            if self.show_model_markers:
                self.vtkModelModel.update_markers(pyo_model_markers[:, :, i])
            if self.show_xp_markers:
                self.vtkModelReal.update_markers(pyo_xp_markers[:, :, i])

            # Update the frames
            if self.show_ground_frame:
                self.ground_frame.update_frame()

            if self.show_frames:
                for s in range(Q.nb_qi()):
                    self.frames[s].update_frame(Q.vector(s)[:, i : i + 1])
            if self.show_center_of_mass:
                for s, com in enumerate(self.center_of_mass):
                    com.update_markers(center_of_mass_locations[:, s : s + 1, i])

            # Update window
            self.vtkWindow.update_frame()
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
