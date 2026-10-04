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
from utils import (
    forward_integration,
    post_computations,
    knee_angles,
    knee_pose_at_flexion,
    knee_translation,
    knee_joint_loads,
)
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

nb_Q = 12 * model.nb_segments
all_Q = [NaturalCoordinates(all_states[:nb_Q, i]) for i in range(len(time_steps))]
angles = np.array([knee_angles(model, Q_i) for Q_i in all_Q])

plt.figure()
for i, label in enumerate(("flexion", "abduction", "axial rotation")):
    plt.plot(time_steps, angles[:, i], label=label)
plt.xlabel("time [s]")
plt.ylabel("[deg]")
plt.title("Knee angles, tibia relative to femur (ZXY)")
plt.legend()

# knee rhythm: coupled rotations and translations of the tibia as a function of flexion, along the whole kinematic
# path of the mechanism (femur fixed), and the part covered by the pendulum
path_flexion = np.arange(0, 146, 1.0)
path_Q, Q_shank_i = [], Q_shank_extended
for flexion in path_flexion:
    Q_shank_i = knee_pose_at_flexion(model, Q_thigh, Q_shank_i, flexion=flexion)
    path_Q.append(
        NaturalCoordinates.from_qi((SegmentNaturalCoordinates(Q_thigh), SegmentNaturalCoordinates(Q_shank_i)))
    )
path_angles = np.array([knee_angles(model, Q_i) for Q_i in path_Q])
path_translations = np.array([knee_translation(model, Q_i) for Q_i in path_Q]) * 1000
translations = np.array([knee_translation(model, Q_i) for Q_i in all_Q]) * 1000

fig, (ax_rotation, ax_translation) = plt.subplots(1, 2, figsize=(11, 4))
for i, label in ((1, "abduction"), (2, "axial rotation")):
    (line,) = ax_rotation.plot(path_angles[:, 0], path_angles[:, i], label=label)
    ax_rotation.plot(angles[:, 0], angles[:, i], ".", markersize=2, color=line.get_color())
ax_rotation.set_xlabel("flexion [deg]")
ax_rotation.set_ylabel("[deg]")
ax_rotation.set_title("Coupled rotations")
ax_rotation.legend()
for i, label in enumerate(("antero-posterior (X)", "proximo-distal (Y)", "medio-lateral (Z)")):
    (line,) = ax_translation.plot(path_angles[:, 0], path_translations[:, i], label=label)
    ax_translation.plot(angles[:, 0], translations[:, i], ".", markersize=2, color=line.get_color())
ax_translation.set_xlabel("flexion [deg]")
ax_translation.set_ylabel("[mm]")
ax_translation.set_title("Tibia origin in the femur frame")
ax_translation.legend()
fig.suptitle("Knee rhythm: kinematic path (lines), pendulum (dots)")
fig.tight_layout()

# joint loads from the Lagrange multipliers (see knee_joint_loads): contact forces, positive in compression,
# and ligament tensions, positive in tension. Negative values are not physically admissible: the bilateral
# constraints of the model then pull (contacts) or push (ligaments).
loads = [knee_joint_loads(model, Q_i, all_lambdas[:, i]) for i, Q_i in enumerate(all_Q)]
fig, (ax_contact, ax_ligament) = plt.subplots(2, 1, sharex=True)
for ax, kind, title in (
    (ax_contact, "contact", "Contact forces [N], > 0 in compression"),
    (ax_ligament, "ligament", "Ligament tensions [N], > 0 in tension"),
):
    for name in [name for name, load in loads[0].items() if load["kind"] == kind]:
        ax.plot(time_steps, [load[name]["load"] for load in loads], label=name)
    ax.axhline(0, color="k", linewidth=0.8)
    ax.axhspan(min(ax.get_ylim()[0], 0), 0, color="red", alpha=0.08)
    ax.set_title(title)
    ax.legend()
ax_ligament.set_xlabel("time [s]")
fig.tight_layout()

plt.figure()
for i in range(0, model.nb_rigid_body_constraints):
    plt.plot(time_steps, defects[i, :], label=f"defects {i}")
plt.title("Rigid body constraints")
plt.legend()

plt.figure()
for i in range(0, model.nb_joint_constraints):
    plt.plot(time_steps, joint_defects[i, :], label=f"joint_defects {i}")
plt.title("Joint constraints")
plt.legend()
plt.show()
