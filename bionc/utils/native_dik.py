"""
Native (C++/Eigen + nanobind) backend for the differential inverse kinematics solver.

The pure-Python ``method="dik"`` solver spends most of its time crossing the
Python<->CasADi/NumPy boundary every Newton iteration (extracting the holonomic
constraint jacobian). This backend keeps the whole frame/Newton loop in compiled
code:

* the model's holonomic constraints + jacobian are emitted as C by CasADi codegen
  and compiled to a shared library (cached per model);
* a generic nanobind module (``dik_native``, built once) ``dlopen``'s that library
  and runs the SQP loop in Eigen.

Public entry point: :func:`solve_native`.
"""

from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

import numpy as np
from casadi import MX, Function, CodeGenerator, densify

from ..bionc_casadi import NaturalCoordinates

_MODULE_DIR = Path(__file__).parent / "dik_native"
_BUILD_DIR = _MODULE_DIR / "_build"
_CACHE_DIR = _MODULE_DIR / "_cache"

_native_module = None


def _ensure_module_built():
    """Import the generic nanobind module, building it once via CMake if needed."""
    global _native_module
    if _native_module is not None:
        return _native_module

    if str(_BUILD_DIR) not in sys.path:
        sys.path.insert(0, str(_BUILD_DIR))
    try:
        import dik_native  # type: ignore
    except ImportError:
        _build_module()
        import importlib

        importlib.invalidate_caches()
        import dik_native  # type: ignore

    _native_module = dik_native
    return _native_module


def _build_module():
    import nanobind

    _BUILD_DIR.mkdir(parents=True, exist_ok=True)
    configure = [
        "cmake",
        "-S",
        str(_MODULE_DIR),
        "-B",
        str(_BUILD_DIR),
        f"-DPython_EXECUTABLE={sys.executable}",
        f"-Dnanobind_DIR={nanobind.cmake_dir()}",
        f"-DCMAKE_PREFIX_PATH={sys.prefix}",
        "-DCMAKE_BUILD_TYPE=Release",
    ]
    subprocess.run(configure, check=True)
    subprocess.run(["cmake", "--build", str(_BUILD_DIR), "--config", "Release", "-j"], check=True)


def _build_eval_function(model) -> Function:
    """CasADi Function: Q -> [holonomic_constraints, holonomic_constraints_jacobian] (both dense)."""
    Q_sym = MX.sym("Q", 12 * model.nb_segments)
    Q = NaturalCoordinates(Q_sym)
    model_mx = model.to_mx()
    phih = densify(model_mx.holonomic_constraints(Q))
    Kh = densify(model_mx.holonomic_constraints_jacobian(Q))
    return Function("dik_eval", [Q_sym], [phih, Kh])


def _generate_evaluator(model) -> tuple[str, str]:
    """Codegen + compile the model evaluator to a shared library (cached by content hash)."""
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    f = _build_eval_function(model)

    cg = CodeGenerator("dik_eval.c", {"with_header": False})
    cg.add(f)
    code = cg.dump()

    digest = hashlib.sha1(code.encode()).hexdigest()[:16]
    so_path = _CACHE_DIR / f"dik_eval_{digest}.so"
    if not so_path.exists():
        c_path = _CACHE_DIR / f"dik_eval_{digest}.c"
        c_path.write_text(code)
        subprocess.run(
            ["gcc", "-fPIC", "-O3", "-march=native", "-shared", str(c_path), "-o", str(so_path)],
            check=True,
        )
    return str(so_path), "dik_eval"


def solve_native(model, experimental_markers: np.ndarray, Q_init: np.ndarray, options: dict) -> tuple[np.ndarray, list]:
    """
    Solve marker-based differential IK with the native backend.

    Parameters
    ----------
    model : BiomechanicalModel (bionc.numpy)
    experimental_markers : np.ndarray (3, nb_markers_technical, nb_frames)
    Q_init : np.ndarray (12*nb_segments, nb_frames)
    options : dict
        Uses keys: max_iter, regularization, constraint_eps, step_eps, objective_eps, eps.

    Returns
    -------
    (Qopt, success) : (np.ndarray (nq, nf), list[bool])
    """
    native = _ensure_module_built()
    lib_path, fname = _generate_evaluator(model)

    nq = 12 * model.nb_segments
    nf = experimental_markers.shape[2]
    nb_markers = experimental_markers.shape[1]

    Jm = np.ascontiguousarray(model.markers_constraints_jacobian(only_technical=True), dtype=float)
    regularization = options["regularization"]
    H = np.ascontiguousarray(Jm.T @ Jm + regularization * np.eye(nq), dtype=float)

    markers_flat = np.empty((3 * nb_markers, nf), dtype=float)
    for f in range(nf):
        markers_flat[:, f] = experimental_markers[:, :, f].flatten("F")

    Qinit = np.ascontiguousarray(np.asarray(Q_init, dtype=float).reshape(nq, nf))

    constraint_eps = options.get("constraint_eps", options["eps"])
    step_eps = options.get("step_eps", options["eps"])
    objective_eps = options.get("objective_eps", options["eps"])

    Qopt, success = native.solve_native(
        lib_path,
        fname,
        Jm,
        H,
        markers_flat,
        Qinit,
        int(model.nb_holonomic_constraints),
        int(options["max_iter"]),
        float(constraint_eps),
        float(step_eps),
        float(objective_eps),
        bool(options.get("warm_start_frames", False)),
    )
    return np.ascontiguousarray(Qopt), list(success)
