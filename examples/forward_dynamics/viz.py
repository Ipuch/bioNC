import plotly.graph_objects as go


def plot_series(time_steps, defects, legend=None):
    fig = go.Figure()
    # display defects in time
    fig.add_trace(go.Scatter(x=time_steps, y=defects[0, :], name=f"{legend}[0]", mode="lines+markers"))
    fig.add_trace(go.Scatter(x=time_steps, y=defects[1, :], name=f"{legend}[1]", mode="lines+markers"))
    fig.add_trace(go.Scatter(x=time_steps, y=defects[2, :], name=f"{legend}[2]", mode="lines+markers"))
    fig.add_trace(go.Scatter(x=time_steps, y=defects[3, :], name=f"{legend}[3]", mode="lines+markers"))
    fig.add_trace(go.Scatter(x=time_steps, y=defects[4, :], name=f"{legend}[4]", mode="lines+markers"))
    fig.add_trace(go.Scatter(x=time_steps, y=defects[5, :], name=f"{legend}[5]", mode="lines+markers"))
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
