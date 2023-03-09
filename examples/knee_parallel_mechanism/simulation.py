import numpy as np
from bionc import NaturalCoordinates, SegmentNaturalCoordinates, Viz
from bionc.bionc_casadi.utils import to_numeric_MX

from knee import create_knee_model


Q0 = SegmentNaturalCoordinates(
    np.array([0.9339, 0.3453, 0.0927, -0.0632, 0.7578, 0.1445, 0.0564, 0.4331, 0.1487, -0.0962, -0.0069, 0.9953])
)
Q1 = SegmentNaturalCoordinates(
    np.array([0.9376, 0.3445, 0.0466, 0.0571, 0.4334, 0.1486, 0.1936, 0.0612, 0.1536, -0.1557, -0.2918, 0.9437])
)

Q = NaturalCoordinates.from_qi((Q0, Q1))

model = create_knee_model()

print(model.rigid_body_constraints(NaturalCoordinates(Q)))
print(model.joint_constraints(NaturalCoordinates(Q)))


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

ik = model.inverse_kinematics(experimental_markers=xp_markers, Q_init=Q)
# Q_opt = ik.solve(method="sqpmethod"
Q_opt = ik.solve(method="ipopt")
print(Q - Q_opt)
print(model.rigid_body_constraints(NaturalCoordinates(Q_opt)))
print(model.joint_constraints(NaturalCoordinates(Q_opt)))

mx_model = model.to_mx()
j_constraints = mx_model.joint_constraints(NaturalCoordinates(Q_opt))
print(to_numeric_MX(j_constraints))


viz = Viz(model, size_model_marker=0.004, show_frames=True, show_ground_frame=False, size_xp_marker=0.005)
viz.animate(Q_opt, markers_xp=xp_markers)
