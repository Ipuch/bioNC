"""
This example shows how to use the InverseKinematics class to solve an inverse kinematics problem.
"""
from bionc import InverseKinematics
import numpy as np
from pyomeca import Markers
from tests.utils import TestUtils
import time


def main():
    # build the model from the lower limb example
    bionc = TestUtils.bionc_folder()
    module = TestUtils.load_module(bionc + "/examples/model_creation/main.py")

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
    ik_solver = InverseKinematics(model, markers, solve_frame_per_frame=True)
    # or you can use the model method
    ik_solver = model.inverse_kinematics(markers, solve_frame_per_frame=True)

    tic0 = time.time()
    Qopt_sqp = ik_solver.solve(method="sqpmethod")  # tend to be faster (with limited-memory hessian approximation)
    toc0 = time.time()

    tic1 = time.time()
    Qopt_ipopt = ik_solver.solve(method="ipopt")  # tend to find lower cost functions but may flip axis.
    toc1 = time.time()

    print(f"Time to solve 200 frames with sqpmethod: {toc0 - tic0}")
    print(f"time to solve 200 frames with ipopt: {toc1 - tic1}")

    return ik_solver, Qopt_sqp, Qopt_ipopt


if __name__ == "__main__":
    ik_solver, _, _ = main()
    # ik_solver.animate()
