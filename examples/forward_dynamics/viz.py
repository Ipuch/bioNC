import numpy as np
import plotly.graph_objects as go
from enum import Enum

from bioviz import VtkWindow, VtkModel
from pyomeca import Markers
from bionc import BiomechanicalModel, SegmentNaturalCoordinates, NaturalCoordinates


def plot_series(time_steps, defects, legend=None):
    fig = go.Figure()
    # display defects in time
    for i in range(defects.shape[0]):
        fig.add_trace(go.Scatter(x=time_steps, y=defects[i, :], name=f"{legend}[{i}]", mode="lines+markers"))
    fig.show()


def animate_natural_segment(time_steps, all_states, center_of_mass, t_final):
    def frame_args(duration):
        return {
            "frame": {"duration": duration},
            "mode": "immediate",
            "fromcurrent": True,
            "transition": {"duration": duration, "easing": "linear"},
        }

    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=all_states[0, :] + all_states[3, :],
                y=all_states[1, :] + all_states[4, :],
                z=all_states[2, :] + all_states[5, :],
                name="u",
            ),
            go.Scatter3d(x=all_states[3, :], y=all_states[4, :], z=all_states[5, :], name="rp"),
            go.Scatter3d(x=all_states[6, :], y=all_states[7, :], z=all_states[8, :], name="rd"),
            go.Scatter3d(
                x=2 * all_states[3, :] - all_states[6, :],
                y=2 * all_states[3, :] - all_states[7, :],
                z=2 * all_states[3, :] - all_states[8, :],
                name="v",
            ),
            go.Scatter3d(
                x=all_states[3, :] + all_states[9, :],
                y=all_states[4, :] + all_states[10, :],
                z=all_states[5, :] + all_states[11, :],
                name="w",
            ),
            go.Scatter3d(
                x=center_of_mass[0, :],
                y=center_of_mass[1, :],
                z=center_of_mass[2, :],
                name="center of mass",
            ),
        ],
        layout=go.Layout(
            scene=dict(
                xaxis=dict(range=[all_states[0::3, :].min().min(), all_states[0::3, :].max().max()], autorange=False),
                yaxis=dict(range=[all_states[1::3].min().min(), all_states[1::3].max().max()], autorange=False),
                zaxis=dict(range=[all_states.min().min(), all_states.max().max()], autorange=False),
            ),
        ),
        frames=[
            go.Frame(
                data=[
                    go.Scatter3d(
                        x=all_states[0, i : i + 1] + all_states[3, i : i + 1],
                        y=all_states[1, i : i + 1] + all_states[4, i : i + 1],
                        z=all_states[2, i : i + 1] + all_states[5, i : i + 1],
                        name="u",
                    ),
                    go.Scatter3d(x=all_states[3, i : i + 1], y=all_states[4, i : i + 1], z=all_states[5, i : i + 1]),
                    go.Scatter3d(x=all_states[6, i : i + 1], y=all_states[7, i : i + 1], z=all_states[8, i : i + 1]),
                    go.Scatter3d(
                        x=2 * all_states[3, i : i + 1] - all_states[6, i : i + 1],
                        y=2 * all_states[4, i : i + 1] - all_states[7, i : i + 1],
                        z=2 * all_states[5, i : i + 1] - all_states[8, i : i + 1],
                        name="v",
                    ),
                    go.Scatter3d(
                        x=all_states[3, i : i + 1] + all_states[9, i : i + 1],
                        y=all_states[4, i : i + 1] + all_states[10, i : i + 1],
                        z=all_states[5, i : i + 1] + all_states[11, i : i + 1],
                        name="w",
                    ),
                    go.Scatter3d(
                        x=center_of_mass[0, i : i + 1],
                        y=center_of_mass[1, i : i + 1],
                        z=center_of_mass[2, i : i + 1],
                        name="center of mass",
                    ),
                ]
            )
            for i in range(len(time_steps))
        ],
    )

    sliders = [
        {
            "pad": {"b": 1, "t": t_final},
            "len": 0.9,
            "x": 1,
            "y": 0,
            "steps": [
                {
                    "args": [[f.name], frame_args(0)],
                    "label": str(k),
                    "method": "animate",
                }
                for k, f in enumerate(fig.frames)
            ],
        }
    ]

    # Layout
    fig.update_layout(
        sliders=sliders,
        title="Animation of the segment",
        updatemenus=[
            {
                "buttons": [
                    {
                        "args": [None, frame_args(time_steps[1] - time_steps[0])],
                        "label": "&#9654;",  # play symbol
                        "method": "animate",
                    },
                    {
                        "args": [[None], frame_args(0)],
                        "label": "&#9724;",  # pause symbol
                        "method": "animate",
                    },
                ],
                "direction": "left",
                "pad": {"r": 1, "t": t_final},
                "type": "buttons",
                "x": 0.1,
                "y": 0,
            }
        ],
    )
    # axis equal for each frame
    fig.update_layout(scene_aspectmode="manual", scene_aspectratio=dict(x=1, y=1, z=1))

    fig.show()


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
    ):
        self.vtk_window = vtk_window
        self.three_vectors = [self.vtk_vector_model(color) for color in NaturalVectorColors]

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
            normalization_ratio=1,
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
            normalization_ratio=1,
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
