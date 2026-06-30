# Native (C++) differential inverse kinematics — install guide

`method="dik_native"` runs the differential inverse kinematics loop in compiled
C++ (Eigen) instead of Python. It produces the **same results** as `method="dik"`,
just faster. To use it, your machine needs a small C++ toolchain.

You only need this if you call `ik.solve(method="dik_native")`. Everything else in
bionc works without it.

---

## TL;DR — the easy way (recommended)

If you created the conda environment from `environment.yml`, **you already have
everything** (compiler, CMake, Eigen, nanobind). Nothing else to do:

```bash
conda env create -f environment.yml   # or: conda env update -f environment.yml
conda activate bionc
```

Then just use it in Python:

```python
ik = InverseKinematics(model, markers)
Qopt = ik.solve(method="dik_native")
```

The **first** call compiles the module automatically (~10–20 s). Every call after
that uses the cached build and starts instantly.

---

## What is actually needed

Four things, all installed by the conda env above:

| Tool          | What it is                                   | conda package   |
|---------------|----------------------------------------------|-----------------|
| C++ compiler  | turns C++ into a runnable library            | `cxx-compiler`  |
| CMake         | drives the build                             | `cmake`         |
| Eigen         | C++ linear-algebra library (headers only)    | `eigen`         |
| nanobind      | glue between C++ and Python                   | `nanobind`      |

## Manual install (only if you are NOT using the conda env)

- **Linux**: `sudo apt install build-essential cmake` then `pip install nanobind`
  and install Eigen (`sudo apt install libeigen3-dev`).
- **macOS**: `xcode-select --install` (compiler), `brew install cmake eigen`,
  then `pip install nanobind`.
- **Windows**: install "Visual Studio Build Tools" (C++ workload) and CMake, then
  `pip install nanobind`. Using the conda env is strongly recommended here.

---

## How it works (for the curious)

1. bionc asks CasADi to generate C code for the model's holonomic constraints and
   their Jacobian, and compiles it to a small shared library (cached per model in
   `_cache/`, keyed by a content hash).
2. A generic C++ module (`dik_native`, built once into `_build/`) loads that
   library and runs the whole frame/Newton loop in Eigen.

Both `_build/` and `_cache/` are build artifacts and are git-ignored.

---

## Troubleshooting

- **`ImportError: No module named 'nanobind'`** → `pip install nanobind`
  (or recreate the conda env).
- **CMake error: "Could NOT find Eigen3"** → install Eigen (see above). With the
  conda env it is found automatically.
- **A compiler error mentioning `gcc`/`cl.exe` not found** → no C++ compiler is
  installed; install one (see "Manual install") or use the conda env.
- **Force a clean rebuild** → delete the `_build/` and `_cache/` folders next to
  this README; they are regenerated on the next call.
