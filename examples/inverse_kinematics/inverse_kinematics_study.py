"""
This example shows how to use the InverseKinematics class to solve an inverse kinematics problem.
"""
from bionc import InverseKinematics, Viz
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
    ik_solver = InverseKinematics(model, markers, solve_frame_per_frame=True, active_direct_frame_constraints=True, use_sx=True)

    tic1 = time.time()
    Qopt_ipopt = ik_solver.solve(method="sqpmethod")  # tend to find lower cost functions but may flip axis.
    toc1 = time.time()

    print(f"time to solve 200 frames with ipopt: {toc1 - tic1}")

    return ik_solver, Qopt_ipopt, model, markers


if __name__ == "__main__":
    ik_solver, Qopt, model, markers = main()

    # display the results of the optimization
    import matplotlib.pyplot as plt
    plt.figure()
    from bionc import NaturalCoordinates
    det = np.zeros((model.nb_segments, Qopt.shape[1]))
    for i in range(0, Qopt.shape[1]):
        Qi = NaturalCoordinates(Qopt)[:, i:i+1]
        for s in range(0, model.nb_segments):
            u, v, w = Qi.vector(s).to_uvw()
            matrix = np.concatenate((u[:,np.newaxis], v[:,np.newaxis], w[:,np.newaxis]), axis=1)
            det[s, i] = np.linalg.det(matrix)
            if det[s, i] < 0:
                print(f"frame {i} segment {s} has a negative determinant")

    plt.plot(det.T, label=model.segment_names, marker="o", ms=1.5)
    # set ylim -5 5
    plt.ylim(-5, 5)
    plt.legend()
    plt.show()

    viz = Viz(
        model,
        show_center_of_mass=False,  # no center of mass in this example
        show_xp_markers=True,
        show_model_markers=True,
    )
    viz.animate(Qopt[:, 197:198], markers_xp=markers[:,:,197:198])
