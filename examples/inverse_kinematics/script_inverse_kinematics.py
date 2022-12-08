from casadi import MX, Function, vertcat, nlpsol
import numpy as np
import pytest
from bionc import JointType, NaturalAxis

from bionc.bionc_casadi import (
    BiomechanicalModel,
    NaturalSegment,
    NaturalCoordinates,
    SegmentNaturalCoordinates,
    Joint,
    NaturalMarker
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




box = NaturalSegment(
    name="box",
    alpha=np.pi / 2,
    beta=np.pi / 2,
    gamma=np.pi / 2,
    length=1,
    mass=1,
    center_of_mass=np.array([0, 0, 0]),  # scs
    inertia=np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),  # scs
)

bbox = NaturalSegment(
    name="bbox",
    alpha=np.pi / 5,
    beta=np.pi / 3,
    gamma=np.pi / 2.1,
    length=1.5,
    mass=1.1,
    center_of_mass=np.array([0.1, 0.11, 0.111]),  # scs
    inertia=np.array([[1.1, 0, 0], [0, 1.2, 0], [0, 0, 1.3]]),  # scs
)

bbox.add_natural_marker(marker=NaturalMarker(name="marker", position=np.array([0.1, 0.11, 0.111]), parent_name="bbox"))

model = BiomechanicalModel()
model["box"] = box
model["bbox"] = bbox



# Declare the Q init
Q = NaturalCoordinates.sym(model.nb_segments())

markers = np.zeros((3 * model.nb_markers(),1))
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

r = S(x0=np.zeros((12*2,1)), # ça serait Q_init
      lbg=0,  # lower bound on g, nous ça serait 0
      ubg=0,  # upper bound on g, nous ça serait 0
      )

r['x'] # ça serait Q_
