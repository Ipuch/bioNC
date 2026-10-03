"""
Pendulum test (Wartenberg) of the Feikes knee: seated subject, thigh horizontal and welded to the ground, the shank is
released at rest at 100 deg of flexion and swings under gravity on the physiological branch of the mechanism.
The Feikes mechanism has no extension stop: released closer to extension, the shank would pass the maximal flexion of
the mechanism (~160 deg) and continue on the branch where the tibia spins about its long axis.
"""

import numpy as np

from bionc import NaturalCoordinates, SegmentNaturalCoordinates, SegmentNaturalVelocities, NaturalVelocities
from bionc.vizualization.pyorerun_interface import BioncModelNoMesh
from knee_feikes import create_knee_model
from pyorerun import PhaseRerun
from utils import forward_integration, post_computations, knee_angles, knee_pose_at_flexion
from distal_femur_view import log_distal_femur_view

# right-handed frames, global X lateral, Y anterior, Z up: thigh horizontal pointing anterior, u up, w lateral
Q_thigh = np.array([0, 0, 1, 0, 0.0, 0, 0, 0.4, 0, 1, 0, 0])
Q_shank_extended = np.array([0, 0, 1, 0, 0.4, 0, 0, 0.8, 0, 1, 0, 0])

model = create_knee_model(hip="weld", thigh_pose=Q_thigh)
Q_shank = knee_pose_at_flexion(model, Q_thigh, Q_shank_extended, flexion=100)

Q = NaturalCoordinates.from_qi((SegmentNaturalCoordinates(Q_thigh), SegmentNaturalCoordinates(Q_shank)))
print("knee angles at release (flexion, abduction, axial rotation) [deg]:", knee_angles(model, Q).round(1))
print(model.rigid_body_constraints(NaturalCoordinates(Q)))
print(model.joint_constraints(NaturalCoordinates(Q)))

vizmodel = BioncModelNoMesh(model)
rerun = PhaseRerun(t_span=np.linspace(0, 1, 1))
rerun.add_animated_model(vizmodel, Q, tracked_markers=None)
rerun.rerun()

# simulation in forward dynamics
tuple_of_Qdot = [
    SegmentNaturalVelocities.from_components(udot=[0, 0, 0], rpdot=[0, 0, 0], rddot=[0, 0, 0], wdot=[0, 0, 0])
    for i in range(0, model.nb_segments)
]
Qdot = NaturalVelocities.from_qdoti(tuple(tuple_of_Qdot))

# actual simulation
t_final = 3.0  # seconds
time_steps, all_states, dynamics = forward_integration(
    model=model,
    Q_init=Q,
    Qdot_init=Qdot,
    t_final=t_final,
    steps_per_second=1000,
)

defects, defects_dot, joint_defects, all_lambdas = post_computations(
    model=model,
    time_steps=time_steps,
    all_states=all_states,
    dynamics=dynamics,
)

# animation
prr = PhaseRerun(t_span=time_steps)
vizmodel = BioncModelNoMesh(model)
prr.add_animated_model(vizmodel, NaturalCoordinates(all_states[: (12 * model.nb_segments), :]), None)
prr.rerun()

# distal femur view: condyle spheres, tibial plane slices and ligaments, in the femur frame
log_distal_femur_view(model, all_states[: (12 * model.nb_segments), ::10], time_steps[::10], femur="THIGH")

# plot results
import matplotlib.pyplot as plt

angles = np.array(
    [knee_angles(model, NaturalCoordinates(all_states[: (12 * model.nb_segments), i])) for i in range(len(time_steps))]
)
plt.figure()
for i, label in enumerate(("flexion", "abduction", "axial rotation")):
    plt.plot(time_steps, angles[:, i], label=label)
plt.xlabel("time [s]")
plt.ylabel("[deg]")
plt.title("Knee angles, tibia relative to femur (ZXY)")
plt.legend()

plt.figure()
for i in range(0, model.nb_rigid_body_constraints):
    plt.plot(time_steps, defects[i, :], marker="o", label=f"defects {i}")
plt.title("Rigid body constraints")
plt.legend()

plt.figure()
for i in range(0, model.nb_joint_constraints):
    plt.plot(time_steps, joint_defects[i, :], marker="o", label=f"joint_defects {i}")
plt.title("Joint constraints")
plt.legend()

plt.figure()
for i in range(0, model.nb_joint_constraints):
    plt.plot(time_steps, all_lambdas[i, :], marker="o", label=f"lambda {i}")
plt.title("lagrange multipliers")
plt.legend()
plt.show()
