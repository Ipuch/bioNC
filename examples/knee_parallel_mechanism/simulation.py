import numpy as np
from bionc import NaturalCoordinates, SegmentNaturalCoordinates, Viz, SegmentNaturalVelocities, NaturalVelocities
from bionc.bionc_casadi.utils import to_numeric_MX

from knee import create_knee_model
from utils import forward_integration, post_computations

model = create_knee_model()

Q0 = SegmentNaturalCoordinates(
    np.array([0.9339, 0.3453, 0.0927, -0.0632, 0.7578, 0.1445, 0.0564, 0.4331, 0.1487, -0.0962, -0.0069, 0.9953])
)
Q1 = SegmentNaturalCoordinates(
    np.array([0.9376, 0.3445, 0.0466, 0.0571, 0.4334, 0.1486, 0.1936, 0.0612, 0.1536, -0.1557, -0.2918, 0.9437])
)

Q = NaturalCoordinates.from_qi((Q0, Q1))
print(model.rigid_body_constraints(NaturalCoordinates(Q)))
print(model.joint_constraints(NaturalCoordinates(Q)))

# viz = Viz(model, size_model_marker=0.004, show_frames=True, show_ground_frame=False, size_xp_marker=0.005)
# viz.animate(Q)


marker_names = model.marker_names
# index of medial_centre_femur and lateral_centre_femur
idx_mcf = marker_names.index("medial_centre_femur")
idx_lcf = marker_names.index("lateral_centre_femur")
markers_location = model.markers(Q)
medial_centre_femur = markers_location[:, idx_mcf]
lateral_centre_femur = markers_location[:, idx_lcf]
thigh_hip_joint_location = np.array([-0.0632, 0.7578, 0.1445])[:, np.newaxis]
thigh_condyle_center_location = np.array([0.0564, 0.4331, 0.1487])[:, np.newaxis]
shank_condyle_center_location = np.array([0.0571, 0.4334, 0.1486])[:, np.newaxis]
shank_ankle_center_location = np.array([0.1936, 0.0612, 0.1536])[:, np.newaxis]
xp_markers = np.concatenate(
    [
        thigh_hip_joint_location,
        thigh_condyle_center_location,
        medial_centre_femur,
        lateral_centre_femur,
        shank_condyle_center_location,
        shank_ankle_center_location,
    ],
    axis=1,
)[:, :, np.newaxis]

ik = model.inverse_kinematics(experimental_markers=xp_markers, Q_init=Q + 2)
Q_opt = ik.solve(method="ipopt")
# Q_opt = ik.solve(method="sqpmethod")
print(Q - Q_opt)
print(model.rigid_body_constraints(NaturalCoordinates(Q_opt)))
print(model.joint_constraints(NaturalCoordinates(Q_opt)))


viz = Viz(model, size_model_marker=0.004, show_frames=True, show_ground_frame=True, size_xp_marker=0.005)
viz.animate(Q_opt, markers_xp=xp_markers)

# simulation

tuple_of_Qdot = [
    SegmentNaturalVelocities.from_components(udot=[0, 0, 0], rpdot=[0, 0, 0], rddot=[0, 0, 0], wdot=[0, 0, 0])
    for i in range(0, model.nb_segments)
]
Qdot = NaturalVelocities.from_qdoti(tuple(tuple_of_Qdot))

# actual simulation
t_final = 0.4  # seconds
time_steps, all_states, dynamics = forward_integration(
    model=model,
    Q_init=NaturalCoordinates(Q_opt),
    Qdot_init=Qdot,
    t_final=t_final,
    steps_per_second=100,
)

defects, defects_dot, joint_defects, all_lambdas = post_computations(
    model=model,
    time_steps=time_steps,
    all_states=all_states,
    dynamics=dynamics,
)

# plot results
import matplotlib.pyplot as plt
plt.figure()
for i in range(0, model.nb_rigid_body_constraints):
    plt.plot(time_steps, defects[i, :], marker="o")
plt.show()

plt.figure()
for i in range(0, model.nb_joint_constraints):
    plt.plot(time_steps, joint_defects[i, :], marker="o")
plt.show()

# plt.plot(time_steps, defects_dot)
# plt.plot(time_steps, joint_defects)
# plt.legend(["defects", "defects_dot", "joint_defects"])
plt.show()

# viz = Viz(model)
# viz.animate(NaturalCoordinates(all_states[: (12 * model.nb_segments), :]), None)
