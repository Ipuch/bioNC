import numpy as np

from bionc import (
    NaturalSegment,
    NaturalCoordinates,
    SegmentNaturalCoordinates,
    NaturalCoordinates,
    SegmentNaturalVelocities,
)
from ode_solvers import RK4, RK8, IRK

# Let's create a segment
my_segment = NaturalSegment(
    name="box",
    alpha=np.pi / 2,
    beta=np.pi / 2,
    gamma=np.pi / 2,
    length=1,
    mass=1,
    center_of_mass=np.array([0, 0.01, 0]),
    inertia=np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
)

Qi = SegmentNaturalCoordinates.from_components(
    u=np.array([1, 2, 3]), rp=np.array([2, 2, 3]), rd=np.array([1, 5, 3]), w=np.array([4, 2, 3])
)
print(my_segment.rigidBodyConstraint(Qi))
print(my_segment.rigidBodyConstraintJacobian(Qi))
print(my_segment.rigidBodyConstraintJacobianDerivative(Qi.vector))

# Let's create a motion now
# u as x axis, w as z axis - this doesn't work
Qi = SegmentNaturalCoordinates.from_components(
    u=np.array([1, 0, 0]), rp=np.array([0, 0, 0]), rd=np.array([0, 1, 0]), w=np.array([0, 0, 1])
)
# u as y axis - this works fine
Qi = SegmentNaturalCoordinates.from_components(
    u=np.array([0, 1, 0]), rp=np.array([0, 0, 0]), rd=np.array([0, 0, 1]), w=np.array([1, 0, 0])
)
# u as z axis - this doesn't work
Qi = SegmentNaturalCoordinates.from_components(
    u=np.array([0, 0, 1]), rp=np.array([0, 0, 0]), rd=np.array([1, 0, 0]), w=np.array([0, 1, 0])
)

Qidot = SegmentNaturalVelocities.from_components(
    udot=np.array([0, 0, 0]), rpdot=np.array([0, 0, 0]), rddot=np.array([0, 0, 0]), wdot=np.array([0, 0, 0])
)

my_segment.differential_algebraic_equation(Qi, Qidot)

t_final = 1  # [s]
time_steps = np.linspace(0, t_final, 51)
states_0 = np.concatenate((Qi.vector, Qidot.vector), axis=0)
all_states = np.zeros((24, len(time_steps)))
all_lambdas = np.zeros((6, len(time_steps)))


def dynamics(t, states):
    qddot, lambdas = my_segment.differential_algebraic_equation(states[0:12], states[12:24])
    return np.concatenate((states[12:24], qddot), axis=0), lambdas


all_states = RK4(time_steps, lambda t, states: dynamics(t, states)[0], states_0)

defects = np.zeros((6, len(time_steps)))
for i in range(len(time_steps)):
    defects[:, i] = my_segment.rigidBodyConstraint(all_states[0:12, i])
    all_lambdas[:, i] = dynamics(time_steps[i], all_states[:, i])[1]

import plotly.graph_objects as go

fig = go.Figure()
# display defects in time
fig.add_trace(go.Scatter(x=time_steps, y=defects[0, :], name="defects[0]", mode="lines+markers"))
fig.add_trace(go.Scatter(x=time_steps, y=defects[1, :], name="defects[1]", mode="lines+markers"))
fig.add_trace(go.Scatter(x=time_steps, y=defects[2, :], name="defects[2]", mode="lines+markers"))
fig.add_trace(go.Scatter(x=time_steps, y=defects[3, :], name="defects[3]", mode="lines+markers"))
fig.add_trace(go.Scatter(x=time_steps, y=defects[4, :], name="defects[4]", mode="lines+markers"))
fig.add_trace(go.Scatter(x=time_steps, y=defects[5, :], name="defects[5]", mode="lines+markers"))
fig.show()

# display forces in time
fig = go.Figure()
fig.add_trace(go.Scatter(x=time_steps, y=all_lambdas[0, :], name="lambda[0]", mode="lines+markers"))
fig.add_trace(go.Scatter(x=time_steps, y=all_lambdas[1, :], name="lambda[1]", mode="lines+markers"))
fig.add_trace(go.Scatter(x=time_steps, y=all_lambdas[2, :], name="lambda[2]", mode="lines+markers"))
fig.add_trace(go.Scatter(x=time_steps, y=all_lambdas[3, :], name="lambda[3]", mode="lines+markers"))
fig.add_trace(go.Scatter(x=time_steps, y=all_lambdas[4, :], name="lambda[4]", mode="lines+markers"))
fig.add_trace(go.Scatter(x=time_steps, y=all_lambdas[5, :], name="lambda[5]", mode="lines+markers"))
fig.show()


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
            x=all_states[3, :] + all_states[9, :],
            y=all_states[4, :] + all_states[10, :],
            z=all_states[5, :] + all_states[11, :],
            name="w",
        ),
    ],
    layout=go.Layout(
        scene=dict(
            xaxis=dict(range=[-2, 2], autorange=False),
            yaxis=dict(range=[-2, 2], autorange=False),
            zaxis=dict(range=[-2, 2], autorange=False),
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
                    x=all_states[3, i : i + 1] + all_states[9, i : i + 1],
                    y=all_states[4, i : i + 1] + all_states[10, i : i + 1],
                    z=all_states[5, i : i + 1] + all_states[11, i : i + 1],
                    name="w",
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

fig.show()

# verifier la jacobian des contraintes de corps rigide.
# normaliser les contraintes ?
# tester les dae solvers ?
# stabilization de la contrainte ?
