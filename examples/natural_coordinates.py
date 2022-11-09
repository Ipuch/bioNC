import numpy as np
from casadi import MX

from bionc import array
from bionc import casadi_type
from bionc import bionc_casadi

# CASADI EXAMPLE
print("CASADI EXAMPLE")
u = array(np.array([1, 2, 3]), instance_type=casadi_type)
rp = array([2, 2, 3], instance_type=casadi_type)
rd = MX.sym("rd", 3)
w = array([4, 2, 3], instance_type=casadi_type)
Q = bionc_casadi.SegmentNaturalCoordinates.from_components(u=u, rp=rp, rd=rd, w=w)

print(Q)
print(Q.to_array())
print(Q.u)
print(Q.rp)
print(Q.rp.shape)
print(Q.to_components())
print("done")

# NUMPY EXAMPLE
print("NUMPY EXAMPLE")
from bionc import numpy_type
from bionc import bionc_numpy

u = array(np.array([1, 2, 3]), instance_type=numpy_type)
rp = array([2, 2, 3], instance_type=numpy_type)
rd = np.array([1, 2, 3])
w = array([4, 2, 3], instance_type=numpy_type)
Q = bionc_numpy.SegmentNaturalCoordinates.from_components(u=u, rp=rp, rd=rd, w=w)

print(Q)
print(Q.to_array())
print(Q.u)
print(Q.rp)
print(Q.rp.shape)
print(Q.to_components())
print("done")
