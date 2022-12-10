from casadi import MX, Function, vertcat, nlpsol
import numpy as np
from pathlib import Path
import pytest

from bionc import JointType, NaturalAxis
from bionc.bionc_numpy import BiomechanicalModel as BiomechanicalModelNumpy
from bionc.bionc_casadi import (
    NaturalCoordinates,
)



# q = MX.sym("q", 12, 1)
# b = MX.sym("b", 1, 1)
#
# def power(a, b):
#     return a ** b
#
# result = power(q, b)
#
#
# print("result = ", result)
#
# f = Function("f", [q, b], [result])
#
# val = f(np.ones((12, 1)) * 2, 3)
# np.array(val)
#
# # MX ou SX
# # MX representation matricielle
# # SX reprentation scalaire, surdocumenté, operation simplifiées, mais information multipliée...
# # MX to SX, on peut utiliser Function(...).expand(), tu peux convertir en SX. youhou.
#
#
# # voir comment fonctionne une optimization
# x = MX.sym('x')
# y = MX.sym('y')
# z = MX.sym('z')
# # nous Q = MX.sym('Q', 12 * n, 1)
#
# nlp = dict(
#     x = vertcat(x, y, z), # nous sera Q
#     f = x**2 + y**2 + z**2, # nous sera 1/2 * Phi_m.T @ Phi_m
#     g = x + y + z, # nous sera Phi_r, Phi_j
#     )
#
# S = nlpsol('S', 'ipopt', nlp)
# print(S)
#
# r = S(x0=[2.5, 3.0, 0.75], # ça serait Q_init
#       lbg=0,  # lower bound on g, nous ça serait 0
#       ubg=0,  # upper bound on g, nous ça serait 0
#       )
#
# r['x'] # ça serait Q_

model = BiomechanicalModelNumpy.load(Path.cwd().parent.absolute().__str__() + "/models/lower_limb.nc")  # models can only be loaded if a numpy model
model = model.to_mx()  # convert to casadi model

# Declare the Q init
Q = NaturalCoordinates.sym(model.nb_segments())

markers = np.zeros((3, model.nb_markers()))
# Extract the phir
phir = model.rigid_body_constraints(Q)
#
phim = model.markers_constraints(markers, Q)
#
#phik = model.joints_constraints(Q)

error_r = 1/2 * phim.T @ phim
#error_k = 1/2 * phik.T @ phik
error_m = 1/2 * phir.T @ phir

nlp = dict(
    x = Q,
    f= error_m,
    g = vertcat(phir))

S = nlpsol('S', 'ipopt', nlp)
print(S)

r = S(x0=np.zeros((Q.shape[0], 1)), # ça serait Q_init
      lbg=0,  # lower bound on g, nous ça serait 0
      ubg=0,  # upper bound on g, nous ça serait 0
      )

r['x'] # ça serait Q_
