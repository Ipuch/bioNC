"""
Example script for animating markers
"""
from enum import Enum
import time

from bioviz import VtkModel, VtkWindow, Mesh
import numpy as np
from pyomeca import Markers

from ..protocols.biomechanical_model import GenericBiomechanicalModel as BiomechanicalModel
from ..protocols.natural_coordinates import SegmentNaturalCoordinates, NaturalCoordinates
from ..bionc_numpy.joint import Joint
from .cylinder import displace_from_start_and_end, generate_cylinder_triangles, generate_cylinder_vertices


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
        show_joints: bool = True,
        size_model_marker: bool = 0.02,
        size_xp_marker: bool = 0.02,
        background_color: tuple[float, float, float] = (0.5, 0.5, 0.5),
        show_natural_mesh: bool = False,
        window_size: tuple[int, int] = (800, 600),
        camera_position: tuple[float, float, float] = None,  # may not work
        camera_focus_point: tuple[float, float, float] = None,  # may not work
        camera_zoom: float = None,  # may not work
        camera_roll: float = None,  # may not work
    ):
        """
        This class is used to visualize the biomechanical model.

        Parameters
        ----------
        model : BiomechanicalModel
            The biomechanical model to visualize.
        show_ground_frame: bool
            If True, the ground frame is displayed.
        show_frames: bool
            If True, the frames are displayed.
        show_model_markers: bool
            If True, the markers of the model are displayed.
        show_xp_markers: bool
            If True, the markers of the experimental data are displayed.
        show_center_of_mass: bool
            If True, the centers of mass are displayed.
        show_joints: bool
            If True, the joints are displayed.
        size_model_marker: float
            The size of the model markers.
        size_xp_marker: float
            The size of the experimental data markers.
        background_color: tuple[float, float, float]
            The background color of the window.
        show_natural_mesh: bool
            If True, the natural meshes of each segment are displayed.
        window_size: tuple[int, int]
            The size of the window.
        camera_position: tuple[float, float, float]
            The position of the camera. may not work
        camera_focus_point: tuple[float, float, float]
            The focus point of the camera. may not work
        camera_zoom: float
            The zoom of the camera. may not work
        camera_roll: float
            The roll of the camera. may not work
        """
        self.model = model
        self.show_ground_frame = show_ground_frame
        self.show_frames = show_frames
        self.show_model_markers = show_model_markers
        self.show_xp_markers = show_xp_markers
        self.show_center_of_mass = show_center_of_mass
        self.show_joints = show_joints
        self.show_natural_mesh = show_natural_mesh

        # Create a windows with a nice gray background
        self.vtkWindow = VtkWindow(background_color=background_color)

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
                        markers_color=(0, 0, 0),
                        markers_size=0.02,
                        markers_opacity=1,
                    )
                )
        if self.show_model_markers:
            self.vtkModelModel = VtkModel(
                self.vtkWindow,
                markers_color=(0, 1, 0),
                markers_size=size_model_marker,
                markers_opacity=1,
            )
        if self.show_xp_markers:
            self.vtkModelReal = VtkModel(
                self.vtkWindow,
                markers_color=(1, 0, 0),
                markers_size=size_xp_marker,
                markers_opacity=1,
            )
        if self.show_joints:
            self.vtkJoints = []
            for i, joint in enumerate(model.joints.values()):
                if isinstance(joint, Joint.ConstantLength):
                    self.vtkJoints = VtkModel(
                        self.vtkWindow,
                        ligament_color=(0.9, 0.9, 0.05),
                        ligament_opacity=1,
                    )

        if self.show_natural_mesh:
            self.vtkMesh = VtkModel(
                self.vtkWindow, patch_color=[(0, 0.5, 0.8) for i in range(self.model.nb_segments)], mesh_opacity=0.5
            )

        self.window_size = window_size
        self.camera_position = camera_position
        self.camera_focus_point = camera_focus_point
        self.camera_zoom = camera_zoom
        self.camera_roll = camera_roll

        self.update_window_and_view()

    def update_window_and_view(self):
        self.vtkWindow.setFixedSize(self.window_size[0], self.window_size[1])
        self.vtkWindow.set_camera_position(self.camera_position[0], self.camera_position[1], self.camera_position[2])
        self.vtkWindow.set_camera_focus_point(
            self.camera_focus_point[0], self.camera_focus_point[1], self.camera_focus_point[2]
        )
        self.vtkWindow.set_camera_roll(self.camera_roll)
        self.vtkWindow.set_camera_zoom(self.camera_zoom)

    def animate(self, Q: NaturalCoordinates | np.ndarray, markers_xp=None, frame_rate=None):
        """
        This function is a cheap animation of markers

        Parameters
        ----------
        Q : NaturalCoordinates | np.ndarray
            The natural coordinates of the segment of shape (n_dofs, n_frames)
        markers_xp : np.ndarray
            The experimental markers measured in global frame of shape (3, n_markers, n_frames)
        frame_rate : float
            The frame rate of the animation, may not be respected if there are too much stuff to display.
            very approximate, but it's just for a quick animation
        """
        from ..bionc_numpy import NaturalCoordinates

        Q = NaturalCoordinates(Q)

        if markers_xp is not None:
            if Q.shape[1] != markers_xp.shape[2]:
                raise ValueError(
                    f"Q and markers_xp must have the same number of frames. Q.shape[1]={Q.shape[1]} "
                    f"and markers_xp.shape[2]={markers_xp.shape[2]}"
                )
        else:
            self.show_xp_markers = False

        if self.show_xp_markers:
            pyo_xp_markers = Markers(markers_xp)
        if self.show_model_markers:
            model_markers = self.model.markers(Q)
            pyo_model_markers = Markers(model_markers)

        if self.show_center_of_mass:
            center_of_mass_locations = np.zeros((3, self.model.nb_segments, Q.shape[1]))
            center_of_mass_locations[:, :, :] = self.model.center_of_mass_position(Q)
            center_of_mass_locations = Markers(center_of_mass_locations)

        if self.show_joints:
            all_ligament = []
            for i, joint in enumerate(self.model.joints.values()):
                if isinstance(joint, Joint.ConstantLength):
                    origin = joint.parent_point.position_in_global(Q.vector(joint.parent.index)[:, 0:1])
                    insert = joint.child_point.position_in_global(Q.vector(joint.child.index)[:, 0:1])
                    ligament = np.concatenate((origin, insert), axis=1)
                    ligament = np.concatenate((ligament, np.ones((1, ligament.shape[1]))), axis=0)[:, :, np.newaxis]
                    all_ligament.append(Mesh(vertex=ligament))

        dt = 1 / frame_rate if frame_rate else 1e-10

        # Animate all this
        i = 0
        while self.vtkWindow.is_active:
            tic = time.time()
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

            if self.show_joints:
                all_ligament = []
                for j, joint in enumerate(self.model.joints.values()):
                    if isinstance(joint, Joint.ConstantLength):
                        origin = joint.parent_point.position_in_global(Q.vector(joint.parent.index)[:, i : i + 1])
                        insert = joint.child_point.position_in_global(Q.vector(joint.child.index)[:, i : i + 1])
                        ligament = np.concatenate((origin, insert), axis=1)
                        ligament = np.concatenate((ligament, np.ones((1, ligament.shape[1]))), axis=0)[:, :, np.newaxis]
                        all_ligament.append(Mesh(vertex=ligament))
                if len(all_ligament) > 0:
                    self.vtkJoints.update_ligament(all_ligament)

            if self.show_natural_mesh:
                self.update_natural_mesh(Q[:, i : i + 1])

            # Update window
            self.vtkWindow.update_frame()

            i = (i + 1) % Q.shape[1]

            while not (time.time() - tic > dt or (frame_rate is None)):
                pass

        # print camera parameters
        print("Camera position: ", self.get_camera_position())
        print("Camera focus point: ", self.get_camera_focus_point())
        print("Camera zoom: ", self.get_camera_zoom())
        print("Camera roll: ", self.get_camera_roll())

    def get_camera_position(self) -> tuple:
        return self.vtkWindow.get_camera_position()

    def set_camera_position(self, x: float, y: float, z: float):
        self.vtkWindow.set_camera_position(x, y, z)
        self.vtkWindow.refresh_window()

    def get_camera_roll(self) -> float:
        return self.vtkWindow.get_camera_roll()

    def set_camera_roll(self, roll: float):
        self.vtkWindow.set_camera_roll(roll)
        self.vtkWindow.refresh_window()

    def get_camera_zoom(self) -> float:
        return self.vtkWindow.get_camera_zoom()

    def set_camera_zoom(self, zoom: float):
        self.vtkWindow.set_camera_zoom(zoom)
        self.vtkWindow.refresh_window()

    def get_camera_focus_point(self) -> tuple:
        return self.vtkWindow.get_camera_focus_point()

    def set_camera_focus_point(self, x: float, y: float, z: float):
        self.vtkWindow.set_camera_focus_point(x, y, z)
        self.vtkWindow.refresh_window()

    def update_natural_mesh(self, Q: NaturalCoordinates):
        """update the mesh of the model"""
        meshes = []
        for s in range(Q.nb_qi()):
            Qi = Q.vector(s)
            height = np.linalg.norm(Qi.v) * (1 - 0.05)  # 5% shorter to see some space between segments
            vertices = generate_cylinder_vertices(height, num_segments=20)
            vertices = displace_from_start_and_end(vertices=vertices, start=Qi.rp, end=Qi.rd)
            poly = generate_cylinder_triangles(vertices)

            # vertices is list of tuple (x, y, z), convert to np.ndarray
            vertices_array = np.array([list(v) for v in vertices]).T
            poly_array = np.array(poly).T

            meshes.append(Mesh(vertex=vertices_array[:, :, None], triangles=poly_array))

        self.vtkMesh.new_mesh_set(all_meshes=meshes)


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
