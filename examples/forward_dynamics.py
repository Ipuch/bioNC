import numpy as np

from bioNC import NaturalSegment, NaturalCoordinates, SegmentNaturalCoordinatesCreator, NaturalCoordinatesCreator
from ode_solvers import RK4

# Let's create a segment
my_segment = NaturalSegment.XYZ(
    "box",
    alpha=np.pi / 2,
    beta=np.pi / 2,
    gamma=np.pi / 2,
    length=1,
    mass=1,
    center_of_mass=np.array([0, 0, 0]),
    inertia=np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
)

print(my_segment.alpha)
print(my_segment.beta)
print(my_segment.gamma)
print(my_segment.length)
print(my_segment.mass)
print(my_segment.center_of_mass)
print(my_segment.inertia)
# other prop
print(my_segment.interpolation_matrix_center_of_mass)
print(my_segment.generalized_mass_matrix)

# Let's create natural coordinates
Qi = SegmentNaturalCoordinatesCreator(
    u=np.array([1, 0, 0]), rp=np.array([0, 0, 0]), rd=np.array([0, 1, 0]), w=np.array([0, 0, 1])
)

print(my_segment.rigidBodyConstraint(Qi))
print(my_segment.rigidBodyConstraintJacobian(Qi))
print(my_segment.rigidBodyConstraintJacobianDerivative(Qi))

Qi = SegmentNaturalCoordinatesCreator(
    u=np.array([1, 2, 3]), rp=np.array([2, 2, 3]), rd=np.array([1, 5, 3]), w=np.array([4, 2, 3])
)
print(my_segment.rigidBodyConstraint(Qi))
print(my_segment.rigidBodyConstraintJacobian(Qi))
print(my_segment.rigidBodyConstraintJacobianDerivative(Qi))

# Let's create a motion now
Qi = SegmentNaturalCoordinatesCreator(
    u=np.array([1, 0, 0]), rp=np.array([0, 0, 0]), rd=np.array([0, 1, 0]), w=np.array([0, 0, 1])
)
Qidot = SegmentNaturalCoordinatesCreator(
    u=np.array([0, 0, 0]), rp=np.array([0, 0, 0]), rd=np.array([0, 0, 0]), w=np.array([0, 0, 0])
)

my_segment.differential_algebraic_equation(Qi, Qidot)

t_final = 1  # [s]
time_steps = np.linspace(0, t_final, 101)
states_0 = np.concatenate((Qi.vector, Qidot.vector), axis=0)
all_states = np.zeros((24, len(time_steps)))


def dynamics(t, states):
    qddot = my_segment.differential_algebraic_equation(states[0:12], states[12:24])[0]
    return np.concatenate((states[12:24], qddot), axis=0)


all_states = RK4(time_steps, dynamics, states_0)


defects = np.zeros((6, len(time_steps)))
for i in range(len(time_steps)):
    defects[:, i] = my_segment.rigidBodyConstraint(all_states[0:12, i])

import plotly.graph_objects as go

fig = go.Figure()
# display defects in time
fig.add_trace(go.Scatter(x=time_steps, y=defects[0, :], name="defects[0]"))
fig.add_trace(go.Scatter(x=time_steps, y=defects[1, :], name="defects[1]"))
fig.add_trace(go.Scatter(x=time_steps, y=defects[2, :], name="defects[2]"))
fig.add_trace(go.Scatter(x=time_steps, y=defects[3, :], name="defects[3]"))
fig.add_trace(go.Scatter(x=time_steps, y=defects[4, :], name="defects[4]"))
fig.add_trace(go.Scatter(x=time_steps, y=defects[5, :], name="defects[5]"))
fig.show()


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
    layout=go.Layout(updatemenus=[dict(type="buttons", buttons=[dict(label="Play", method="animate", args=[None])])]),
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
fig.show()
