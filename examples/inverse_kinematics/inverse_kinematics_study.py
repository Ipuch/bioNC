"""
This example shows how to use the InverseKinematics class to solve an inverse kinematics problem.
"""

import numpy as np
import time
from pyomeca import Markers

from bionc import InverseKinematics
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
    markers = markers + np.random.normal(0, 0.05, markers.shape)  # add noise

    # you can import the class from bionc
    ik_solver = InverseKinematics(model, markers, active_direct_frame_constraints=False, use_sx=True)

    tic1 = time.time()
    Qopt_ipopt = ik_solver.solve(method="sqpmethod")  # tend to find lower cost functions but may flip axis.
    # Qopt_ipopt = ik_solver.solve(method="ipopt")  # tend to find higher cost functions but does not flip axis.
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
        Qi = NaturalCoordinates(Qopt)[:, i : i + 1]
        for s in range(0, model.nb_segments):
            u, v, w = Qi.vector(s).to_uvw()
            matrix = np.concatenate((u[:, np.newaxis], v[:, np.newaxis], w[:, np.newaxis]), axis=1)
            det[s, i] = np.linalg.det(matrix)
            if det[s, i] < 0:
                print(f"frame {i} segment {s} has a negative determinant")

    plt.plot(det.T, label=model.segment_names, marker="o", ms=1.5)
    plt.ylabel("Determinant of the non-orthogonal coordinate system")
    plt.xlabel("Frame")
    plt.ylim(-1, 1)
    plt.legend()
    plt.show()

    from bionc.vizualization.pyorerun_interface import BioncModelNoMesh
    from pyorerun import PhaseRerun

    model_interface = BioncModelNoMesh(model)

    slice_chunk = slice(197, 198)
    prr = PhaseRerun(t_span=np.linspace(0, 1, slice_chunk.stop - slice_chunk.start))
    pyomarkers = Markers(markers[:, :, slice_chunk], model.marker_names_technical)
    prr.add_animated_model(model_interface, Qopt[:, slice_chunk], tracked_markers=pyomarkers)
    prr.rerun()

    slice_chunk = slice(0, 198)
    prr = PhaseRerun(t_span=np.linspace(0, 1, slice_chunk.stop - slice_chunk.start))
    pyomarkers = Markers(markers[:, :, slice_chunk], model.marker_names_technical)
    prr.add_animated_model(model_interface, Qopt[:, slice_chunk], tracked_markers=pyomarkers)
    prr.rerun()
