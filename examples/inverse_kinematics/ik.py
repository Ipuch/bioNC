"""
This script shows how to solve inverse kinematics for one frame with bionc.
If one want to use the more advanced class please see the example: inverse_kinematics.py
"""

import numpy as np
from casadi import vertcat, nlpsol
from pyomeca import Markers

from bionc.bionc_casadi import (
    NaturalCoordinates,
    SegmentNaturalCoordinates,
)
from bionc.bionc_numpy import NaturalCoordinates as NaturalCoordinatesNumpy
from bionc import Viz
from tests.utils import TestUtils

# import the lower limb model
bionc = TestUtils.bionc_folder()
module = TestUtils.load_module(bionc + "/examples/model_creation/main.py")

# Choose the optimizer
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

# Solve the problem
r = S(
    x0=Q1,  # Initial guess
    lbg=np.zeros(model.nb_holonomic_constraints),  # lower bound 0
    ubg=np.zeros(model.nb_holonomic_constraints),  # upper bound 0
)

Qopt = r["x"].toarray()

bionc_viz = Viz(model_numpy, show_center_of_mass=False)
bionc_viz.animate(NaturalCoordinatesNumpy(Qopt), markers_xp=markers[:3, :, 1:2])
