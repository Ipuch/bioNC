import numpy as np

from bionc import (
    NaturalSegment,
    NaturalCoordinates,
    SegmentNaturalCoordinates,
    NaturalCoordinates,
    SegmentNaturalVelocities,
)
from ode_solvers import RK4, RK8, IRK
from viz import plot_series, animate_natural_segment

# Let's create a segment
my_segment = NaturalSegment(
    name="box",
    alpha=np.pi / 2,
    beta=np.pi / 2,
    gamma=np.pi / 2,
    length=1,
    mass=1,
    center_of_mass=np.array([0, 0, 0]),
    inertia=np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
)

# Let's create a motion now
# u as x-axis, w as z axis - this doesn't work
# Qi = SegmentNaturalCoordinates.from_components(
#     u=[1, 0, 0], rp=[0, 0, 0], rd=[0, -1, 0], w=[0, 0, 1]
# )
# u as y-axis - this works fine
# Qi = SegmentNaturalCoordinates.from_components(
#     u=[0, 1, 0], rp=[0, 0, 0], rd=[0, 0, -1], w=[1, 0, 0]
#     )
# u as z-axis - this doesn't work
Qi = SegmentNaturalCoordinates.from_components(u=[0, 0, 1], rp=[0, 0, 0], rd=[-1, 0, 0], w=[0, 1, 0])

# Velocities are zero
Qidot = SegmentNaturalVelocities.from_components(
    udot=np.array([0, 0, 0]), rpdot=np.array([0, 0, 0]), rddot=np.array([0, 0, 0]), wdot=np.array([0, 0, 0])
)

t_final = 2  # [s]
steps_per_second = 50
time_steps = np.linspace(0, t_final, steps_per_second * t_final + 1)

# initial conditions
states_0 = np.concatenate((Qi.vector, Qidot.vector), axis=0)

# Initialize the arrays to store the results
all_states = np.zeros((24, len(time_steps)))
all_lambdas = np.zeros((6, len(time_steps)))

# Baumgarte stabilization
stabilization = None
# stabilization = dict(alpha=0.5, beta=5)

# Create the forward dynamics function Callable (f(t, y) -> ydot)
def dynamics(t, states):
    qddot, lambdas = my_segment.differential_algebraic_equation(
        states[0:12],
        states[12:24],
        stabilization=stabilization,
    )
    return np.concatenate((states[12:24], qddot), axis=0), lambdas


# Solve the Initial Value Problem (IVP) for each time step
all_states = RK4(time_steps, lambda t, states: dynamics(t, states)[0], states_0)

# compute the quantities of interest after the integration
defects = np.zeros((6, len(time_steps)))
defects_dot = np.zeros((6, len(time_steps)))
center_of_mass = np.zeros((3, len(time_steps)))
for i in range(len(time_steps)):
    defects[:, i] = my_segment.rigid_body_constraint(all_states[0:12, i])
    defects_dot[:, i] = my_segment.rigid_body_constraint_derivative(all_states[0:12, i], all_states[12:24, i])
    all_lambdas[:, i] = dynamics(time_steps[i], all_states[:, i])[1]
    center_of_mass[:, i] = my_segment.interpolation_matrix_center_of_mass @ all_states[0:12, i]


# Plot the results
plot_series(time_steps, defects, legend="rigid_constraint")  # Phi_r
plot_series(time_steps, defects_dot, legend="rigid_constraint_derivative")  # Phi_r_dot
plot_series(time_steps, all_lambdas, legend="lagrange_multipliers")  # lambda

# animate the motion
animate_natural_segment(time_steps, all_states, center_of_mass, t_final)
