import numpy as np
from casadi import vertcat, nlpsol
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
Qxp = model_numpy.Q_from_markers(markers[:, :, 0:2])
Q1 = Qxp[:, 0:1]
Q2 = Qxp[:, 1:2]

# Declare the Q symbolics
Q_sym = []
for ii in range(model.nb_segments):
    Q_sym.append(SegmentNaturalCoordinates.sym(f"_{ii}"))
Q = NaturalCoordinates(vertcat(*Q_sym))

# Objectives
phim = model_mx.markers_constraints(markers[:3, :, 0], Q, only_technical=True)
error_m = 1 / 2 * phim.T @ phim
# Constraints
phir = model_mx.rigid_body_constraints(Q)
phik = model_mx.joint_constraints(Q)

nlp = dict(
    x=Q,
    f=error_m,
    g=vertcat(phir, phik),
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
    }
else:
    options = {}
S = nlpsol(
    "S",
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
# bionc_viz.animate(NaturalCoordinatesNumpy(Q1), markers_xp=markers[:3, :, 1:2])

# Solve the problem
r = S(
    x0=Q1,  # Initial guess
    lbg=np.zeros(model.nb_holonomic_constraints),  # lower bound 0
    ubg=np.zeros(model.nb_holonomic_constraints),  # upper bound 0
)

Qopt = r["x"]
markers_model = model_numpy.markers(NaturalCoordinatesNumpy(Qopt))[:, idx_technical, :]
print("Error on markers: ", np.linalg.norm(markers_model - markers[:3, :, 0:1], axis=0))
print("Error on rigid body constraints: ", model_numpy.rigid_body_constraints(NaturalCoordinatesNumpy(Qopt.toarray())))
print("Error on joint constraints: ", model_numpy.joint_constraints(NaturalCoordinatesNumpy(Qopt.toarray())))

bionc_viz = Viz(model_numpy, show_center_of_mass=False)
bionc_viz.animate(NaturalCoordinatesNumpy(Qopt.toarray()), markers_xp=markers[:3, :, 1:2])
