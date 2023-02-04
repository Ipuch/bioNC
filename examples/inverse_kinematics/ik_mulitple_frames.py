import numpy as np
from casadi import vertcat, nlpsol, horzcat
import ezc3d
from pathlib import Path
from pyomeca import Markers

from bionc.bionc_casadi import (
    NaturalCoordinates,
    SegmentNaturalCoordinates,
)
from bionc.bionc_numpy import NaturalCoordinates as NaturalCoordinatesNumpy
from bionc import Viz

# models can only be loaded if a numpy model
# but not working yet with import, model_numpy.Q_from_markers ...
# model = BiomechanicalModelNumpy.load(Path.cwd().parent.absolute().__str__() + "/examples/models/lower_limb.nc")
# model_numpy = model
from tests.utils import TestUtils

bionc = TestUtils.bionc_folder()
module = TestUtils.load_module(bionc + "/examples/model_creation/main.py")

optimizer = "ipopt"
# optimizer = "sqpmethod"

# Generate c3d file
filename = module.generate_c3d_file()
# Generate model
model = module.model_creation_from_measured_data(filename)
model_numpy = model
model_mx = model.to_mx()

markers = Markers.from_c3d(filename).to_numpy()
# repeat markers along axis = 2 and dfferent noise for each element of axis 0 and 1
# markers = np.repeat(markers, , axis=2)
# markers = markers + np.random.normal(0, 0.01, markers.shape)
markers = markers[:, :, 0:1]
markers = np.repeat(markers, 1, axis=2)
Qxp = model_numpy.Q_from_markers(markers[:, :, :])
# Declare the Q symbolics
Q_sym = []
for f in range(markers.shape[2]):
    Q_f_sym = []
    for ii in range(model.nb_segments):
        Q_f_sym.append(SegmentNaturalCoordinates.sym(f"_{ii}_{f}"))
    Q_sym.append(vertcat(*Q_f_sym))
Q = horzcat(*Q_sym)
vert_Q = vertcat(*Q_sym)
# Objectives
error_m = 0
phir = []
phik = []
for f in range(markers.shape[2]):
    # Objectives
    Qf = NaturalCoordinates(Q[:, f])
    phim = model_mx.markers_constraints(markers[:3, :, f], Qf, only_technical=True)
    error_m += 1 / 2 * phim.T @ phim
    # Constraints
    phir.append(model_mx.rigid_body_constraints(Qf))
    phik.append(model_mx.joint_constraints(Qf))

constraints = vertcat(*phir, *phik)
print(constraints)
nlp = dict(
    x=vert_Q,
    f=error_m,
    g=constraints,
)

if optimizer == "sqpmethod":
    options = {
        "beta": 0.8,
        "c1": 0.0001,
        "hessian_approximation": "exact",
        "lbfgs_memory": 10,
        "max_iter": 50,
        "max_iter_ls": 3,
        "merit_memory": 4,
        "print_header": True,
        "print_time": True,
        "qpsol": "qpoases",
        "tol_du": 0.1,
        "tol_pr": 0.1,
        "qpsol_options": {"error_on_fail": False},
    }
else:
    options = {}

S = nlpsol(
    "InverseKinematics",
    optimizer,
    nlp,
    options,
)
name_technical = model_numpy.marker_names_technical
name_all = model_numpy.marker_names
# find the index of technical markers in all markers
idx_technical = [name_all.index(name) for name in name_technical]

# Display the initial guess
# bionc_viz = Viz(model_numpy, show_center_of_mass=False)
# bionc_viz.animate(NaturalCoordinatesNumpy(Qxp), markers_xp=markers[:3, :, :])

# Solve the problem
# reshape Q in one column
Q1 = Qxp.reshape((12 * model.nb_segments * markers.shape[2], 1))
r = S(
    x0=Q1,  # Initial guess
    lbg=np.zeros(model.nb_holonomic_constraints * markers.shape[2]),  # lower bound 0
    ubg=np.zeros(model.nb_holonomic_constraints * markers.shape[2]),  # upper bound 0,
)

Qopt = r["x"]
# reshape Qopt in 12 x nb_segments x nb_frames
Qopt = Qopt.reshape((12 * model.nb_segments, markers.shape[2]))

markers_model = model_numpy.markers(NaturalCoordinatesNumpy(Qopt))[:, idx_technical, :]
# print("Error on markers: ", np.linalg.norm(markers_model - markers[:3, :, 0:1], axis=0))
# print("Error on rigid body constraints: ", model_numpy.rigid_body_constraints(NaturalCoordinatesNumpy(Qopt.toarray())))
# print("Error on joint constraints: ", model_numpy.joint_constraints(NaturalCoordinatesNumpy(Qopt.toarray())))

bionc_viz = Viz(
    model_numpy,
    show_center_of_mass=False,  # no center of mass in this example
    show_xp_markers=True,
    show_model_markers=True,
)
bionc_viz.animate(NaturalCoordinatesNumpy(Qopt.toarray()), markers_xp=markers[:3, :, :])
