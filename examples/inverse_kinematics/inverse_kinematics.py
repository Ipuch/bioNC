"""
This example shows how to use the InverseKinematics class to solve an inverse kinematics problem.
"""

import time

import numpy as np
from pyomeca import Markers

from bionc import InverseKinematics, NaturalCoordinates
from tests.utils import TestUtils

DEFAULT_METHODS = ("dik",)


def load_two_side_lower_limb_model():
    """Build the two-side lower-limb model and return it with its generated c3d file."""
    bionc = TestUtils.bionc_folder()
    right_lower_limb = TestUtils.load_module(
        bionc + "/examples/model_creation/right_side_lower_limb.py"
    )
    two_side_lower_limbs = TestUtils.load_module(
        bionc + "/examples/model_creation/two_side_lower_limbs.py"
    )

    filename = right_lower_limb.generate_c3d_file(two_side=True)
    model = two_side_lower_limbs.model_creation_from_measured_data(filename)
    return model, filename


def generate_noisy_markers(
    model, filename, n_repeats: int = 100, noise_std: float = 0.01, seed: int = 42
):
    """
    Load the generated c3d markers, reorder them like the model, duplicate the two c3d frames, and add noise.
    """
    if n_repeats < 1:
        raise ValueError("n_repeats must be greater than or equal to 1.")

    markers = Markers.from_c3d(
        filename, usecols=model.marker_names_technical
    ).to_numpy()[:3, :, :]
    markers = np.repeat(markers, n_repeats, axis=2)

    rng = np.random.default_rng(seed)
    return markers + rng.normal(0, noise_std, markers.shape)


def solve_inverse_kinematics(
    model,
    markers,
    method: str,
    options: dict | None = None,
    active_direct_frame_constraints: bool = False,
):
    ik_solver = InverseKinematics(
        model,
        markers,
        active_direct_frame_constraints=active_direct_frame_constraints,
    )

    tic = time.time()
    Qopt = ik_solver.solve(method=method, options=options)
    toc = time.time()

    return ik_solver, Qopt, toc - tic


def main(
    methods: tuple[str, ...] | list[str] | str = DEFAULT_METHODS,
    n_repeats: int = 100,
    noise_std: float = 0.01,
    seed: int = 42,
    solver_options: dict[str, dict] | None = None,
    active_direct_frame_constraints: bool = False,
    print_timing: bool = True,
):
    model, filename = load_two_side_lower_limb_model()
    markers = generate_noisy_markers(
        model, filename, n_repeats=n_repeats, noise_std=noise_std, seed=seed
    )

    if isinstance(methods, str):
        methods = (methods,)

    results = {}
    for method in methods:
        options = solver_options.get(method) if solver_options is not None else None
        ik_solver, Qopt, elapsed = solve_inverse_kinematics(
            model,
            markers,
            method=method,
            options=options,
            active_direct_frame_constraints=active_direct_frame_constraints,
        )
        results[method] = {"solver": ik_solver, "Qopt": Qopt, "elapsed": elapsed}

        if print_timing:
            print(f"Time to solve {Qopt.shape[-1]} frames with {method}: {elapsed}")

    return results, model, markers


if __name__ == "__main__":
    results, model, markers = main()
    method = next(iter(results))
    ik_solver = results[method]["solver"]
    Qopt = results[method]["Qopt"]

    stats = ik_solver.sol()

    print(f"Max marker residual: {np.max(stats['marker_residuals_norm'])}")
    print(f"Max rigidbody residual: {np.max(np.abs(stats['rigidity_residuals']))}")
    print(f"Max joint residual: {np.max(np.abs(stats['joint_residuals']))}")

    print("Joint Angles Extraction for the first frame.")
    print(model.natural_coordinates_to_joint_angles(NaturalCoordinates(Qopt[:, 0])))

    from bionc.vizualization.pyorerun_interface import BioncModelNoMesh
    from pyorerun import PhaseRerun, PyoMarkers

    model_interface = BioncModelNoMesh(model)
    prr = PhaseRerun(t_span=np.linspace(0, 1, Qopt.shape[-1]))

    pyomarkers = PyoMarkers(data=markers, marker_names=model.marker_names_technical)
    prr.add_animated_model(model_interface, Qopt, tracked_markers=pyomarkers)
    prr.rerun()
