"""
Example script for animating markers
"""

import numpy as np
import rerun as rr
from enum import Enum
from pyomeca import Markers

from .cylinder import displace_from_start_and_end, generate_cylinder_triangles, generate_cylinder_vertices
from .rr_utils import display_frame
from ..bionc_numpy.joints import Joint
from ..protocols.biomechanical_model import GenericBiomechanicalModel as BiomechanicalModel
from ..protocols.natural_coordinates import SegmentNaturalCoordinates, NaturalCoordinates


class NaturalVectorColors(Enum):
    """Colors for the vectors."""

    # yellow
    U = (255, 255, 0)
    # cyan
    V = (0, 255, 255)
    #  magenta
    W = (255, 0, 255)


class VectorColors(Enum):
    """Colors for the vectors."""

    # red
    X = (255, 0, 0)
    # green
    Y = (0, 255, 0)
    # blue
    Z = (0, 0, 255)


class LocalFrame:
    def __init__(
            self,
            normalized: bool = False,
    ):
        self.three_vectors = [self.vtk_vector_model(color) for color in NaturalVectorColors]
        self.normalized = normalized

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


def update_natural_mesh(Q: NaturalCoordinates):
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

        meshes.append(dict(vertex=vertices_array[:, :, None], triangles=poly_array))

    return meshes


class RerunViz:
    def __init__(
            self,
            model: BiomechanicalModel,
            show_ground_frame: bool = True,
            show_frames: bool = True,
            show_model_markers: bool = True,
            show_xp_markers: bool = True,
            show_center_of_mass: bool = True,
            show_joints: bool = True,
            size_model_marker: bool = 0.01,
            size_xp_marker: bool = 0.01,
            show_natural_mesh: bool = False,
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
        show_natural_mesh: bool
            If True, the natural meshes of each segment are displayed.
        """
        self.model = model
        self.show_ground_frame = show_ground_frame
        self.show_frames = show_frames
        self.show_model_markers = show_model_markers
        self.show_xp_markers = show_xp_markers
        self.show_center_of_mass = show_center_of_mass
        self.show_joints = show_joints
        self.show_natural_mesh = show_natural_mesh
        self.animation_name = "bionc_model"

        if self.show_model_markers:
            self.model_markers_options = dict(
                radii=self.model.nb_markers * [size_model_marker],
                color=np.tile((0, 255, 0), (self.model.nb_markers, 1)),
                opacity=1,
            )

        if self.show_xp_markers:
            self.xp_markers_options = dict(
                radii=self.model.nb_markers_technical * [size_xp_marker],
                color=np.tile((255, 0, 0), (self.model.nb_markers_technical, 1)),
                opacity=1,
            )
        if self.show_joints:
            for i, joint in enumerate(model.joints.values()):
                if isinstance(joint, Joint.ConstantLength):
                    self.joint_options = dict(
                        ligament_color=(0.9, 0.9, 0.05),
                        ligament_opacity=1,
                    )

        if self.show_natural_mesh:
            self.mesh_options = dict(patch_color=[(0, 0.5, 0.8) for _ in range(self.model.nb_segments)],
                                     mesh_opacity=0.5)

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
        nb_frames = Q.shape[1]

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
                    # all_ligament.append(Mesh(vertex=ligament))

        dt = 1 / frame_rate if frame_rate else 1e-10
        time = 0

        rr.init("bionc_animation", spawn=True)

        if self.show_ground_frame:
            display_frame(self.animation_name, 1, timeless=True)

        # Animate all this
        for i in range(nb_frames):
            rr.set_time_seconds("stable_time", time)
            # Update the markers
            if self.show_model_markers:
                point3d = rr.Points3D(
                    radii=self.model_markers_options["radii"],
                    colors=self.model_markers_options["color"],
                    positions=pyo_model_markers[:3, :, i].to_numpy().T,
                    labels=self.model.marker_names,
                )
                rr.log(self.animation_name + "/model_markers", point3d)
            if self.show_xp_markers:
                point3d = rr.Points3D(
                    radii=self.xp_markers_options["radii"],
                    colors=self.xp_markers_options["color"],
                    positions=pyo_xp_markers[:3, :, i].to_numpy().T,
                    labels=self.model.marker_names_technical,
                )
                rr.log(self.animation_name + "/xp_markers", point3d)

            if self.show_frames:
                for s in range(Q.nb_qi()):
                    qi = Q.vector(s)[:, i: i + 1]
                    display_frame(
                        animation_id=self.animation_name + f"/local_frame_{self.model.segment_names[s]}",
                        scale=[0.3, 1, 0.3],
                        timeless=False,
                        homogenous_transform=np.concatenate(
                            (qi.u[:, np.newaxis], qi.v[:, np.newaxis], qi.w[:, np.newaxis], qi.rp[:, np.newaxis]),
                            axis=1),
                        colors=[NaturalVectorColors.U.value, NaturalVectorColors.V.value, NaturalVectorColors.W.value],
                        axes=["U", "V", "W"],
                    )
                    rr.log(self.animation_name + f"/strip_{self.model.segment_names[s]}",
                           rr.LineStrips3D(
                               [(qi.rp, qi.rd)],
                               colors=[[(0, 0, 0)]],
                               radii=[0.005],
                           ))

            if self.show_center_of_mass:
                print()
                # for s, com in enumerate(self.center_of_mass):
                #     com.update_markers(center_of_mass_locations[:, s: s + 1, i])

            if self.show_joints:
                print()
                # all_ligament = []
                # for j, joint in enumerate(self.model.joints.values()):
                #     if isinstance(joint, Joint.ConstantLength):
                #         origin = joint.parent_point.position_in_global(Q.vector(joint.parent.index)[:, i: i + 1])
                #         insert = joint.child_point.position_in_global(Q.vector(joint.child.index)[:, i: i + 1])
                #         ligament = np.concatenate((origin, insert), axis=1)
                #         ligament = np.concatenate((ligament, np.ones((1, ligament.shape[1]))), axis=0)[:, :, np.newaxis]
                #         all_ligament.append(Mesh(vertex=ligament))
                # if len(all_ligament) > 0:
                #     self.vtkJoints.update_ligament(all_ligament)

            if self.show_natural_mesh:
                meshes = update_natural_mesh(Q[:, i: i + 1])
                for j, mesh in enumerate(meshes):
                    mesh3d = rr.Mesh3D(
                        vertex_positions=mesh['vertex'][:, :, 0].T,
                        indices=mesh['triangles'],
                        vertex_colors=np.tile(self.mesh_options["patch_color"][j], (mesh['vertex'].shape[1], 1)),
                    )
                    rr.log(self.animation_name + f"/natural_mesh_{j}", mesh3d)

            time += dt
