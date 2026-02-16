"""
This example shows how to use the InverseKinematics class to solve an inverse kinematics problem.
"""

import time

import numpy as np
from pyomeca import Markers

from bionc import InverseKinematics, NaturalCoordinates
from tests.utils import TestUtils


def main():
    # build the model from the lower limb example
    bionc = TestUtils.bionc_folder()
    module = TestUtils.load_module(bionc + "/examples/model_creation/right_side_lower_limb.py")

    # Generate c3d file
    filename = module.generate_c3d_file()
    # Generate model
    model = module.model_creation_from_measured_data(filename)

    # getting noisy markers for 200 frames
    markers = Markers.from_c3d(filename).to_numpy()[:3, :, :]  # 2 frames
    markers = np.repeat(markers, 100, axis=2)  # 2 x 100 frames
    np.random.seed(42)
    markers = markers + np.random.normal(0, 0.01, markers.shape)  # add noise

    # you can import the class from bionc
    ik_solver = InverseKinematics(model, markers)

    tic0 = time.time()
    Qopt_sqp = ik_solver.solve(method="sqpmethod")  # tend to be faster (with limited-memory hessian approximation)
    toc0 = time.time()

    tic1 = time.time()
    Qopt_ipopt = ik_solver.solve(method="ipopt")  # tend to find lower cost functions but may flip axis.
    toc1 = time.time()

    print(f"Time to solve 200 frames with sqpmethod: {toc0 - tic0}")
    print(f"time to solve 200 frames with ipopt: {toc1 - tic1}")

    return ik_solver, Qopt_sqp, Qopt_ipopt, model, markers


if __name__ == "__main__":
    ik_solver, Qopt, _, model, markers = main()

    stats = ik_solver.sol()

    print(f"Max marker distance: {stats['max_marker_distance']}")
    print(f"Max rigidbody violation: {stats['max_rigidbody_violation']}")
    print(f"Max joint violation: {stats['max_joint_violation']}")

    print("RKNI residuals along x, y, z for each frame")
    idx = model.marker_names.index("RKNI")
    for f in range(ik_solver.nb_markers):
        marker_residuals = stats["marker_residuals_xyz"][:, idx, :].squeeze()
        print(f"X,\tY,\tZ\t:\t{marker_residuals[0,f]}\t{marker_residuals[1,f]}\t{marker_residuals[2,f]}")

    # convert the natural coordinates to joint angles (still experimental)
    print(model.natural_coordinates_to_joint_angles(NaturalCoordinates(Qopt[:, 0])))

    from bionc.vizualization.pyorerun_interface import BioncModelNoMesh
    from pyorerun import PhaseRerun, PyoMarkers

    model_interface = BioncModelNoMesh(model)
    prr = PhaseRerun(t_span=np.linspace(0, 1, 200))

    pyomarkers = PyoMarkers(data=markers, marker_names=model.marker_names_technical)
    prr.add_animated_model(model_interface, Qopt, tracked_markers=pyomarkers)
    prr.rerun()
