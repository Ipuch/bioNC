
This repository contains the source code for the bioNC library, a python package for biomechanics and natural coordinates formalism.
Inverse and Forward approach are implemented.

## Status
![Stars](https://img.shields.io/github/stars/Ipuch/bioNC?style=social)
![Forks](https://img.shields.io/github/forks/Ipuch/bioNC?style=social)

![Release Version](https://img.shields.io/github/v/release/Ipuch/bioNC) <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/license-MIT-success" alt="License"/></a>  
[![Build status](https://github.com/Ipuch/bioNC/actions/workflows/run_tests.yml/badge.svg)](https://github.com/Ipuch/bioNC/actions/) <a href="https://codeclimate.com/github/Ipuch/bioNC/maintainability"><img src="https://api.codeclimate.com/v1/badges/1f46b245f72858ae8bd5/maintainability" /></a> [![codecov](https://codecov.io/gh/ipuch/bionc/branch/main/graph/badge.svg)](https://codecov.io/gh/ipuch/bionc) [![Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

![Last Commit](https://img.shields.io/github/last-commit/Ipuch/bioNC)
![Contributors](https://img.shields.io/github/contributors/Ipuch/bioNC)
![Merged PRs](https://img.shields.io/github/issues-pr-closed/Ipuch/bioNC)
![Open Issues](https://img.shields.io/github/issues/Ipuch/bioNC)
![Closed Issues](https://img.shields.io/github/issues-closed/Ipuch/bioNC)


# Table of Contents

-[Installation](#installation)

-[A first practical example](#a-first-practical-example)

-[Mathematical backends](#mathematical-backends)

-[Natural coordinates reminders](#natural-coordinates-reminders)

# Installation
- from conda:

![Conda-version](https://anaconda.org/conda-forge/bionc/badges/version.svg)
![Last-update](https://anaconda.org/conda-forge/bionc/badges/latest_release_relative_date.svg)
```
    conda install -c conda-forge bionc
```
- from source:
```
    python setup.py install
```

# A first practical example
The easiest way to learn bionc is to dive into it.
So let's build our first model.
Please note that this tutorial is designed to recreate example which builds a lower limb model (Pelvis, Thigh, Shank, Foot). You can have look to [https://github.com/Ipuch/bioNC/blob/main/examples/model_creation/main.py](https://github.com/Ipuch/bioNC/blob/main/examples/model_creation/main.py)

<img src="./docs/inverse_kinematics_viz.png" alt="Inverse kinematics" width="200"/>


# Mathematical backends
This toolbox support two mathematical backends: `numpy` and `casadi`.

# Natural coordinates reminders

The $i$-th body in a biomechanism with $N$ bodies has generalized coordinates:  

```math 
Q_i = (u_{i}, r_{p_{i}}, r_{d_{i}}, w_{i}) \in \mathbf{R}^3 \times \mathbf{R}^3 \times \mathbf{R}^3 \times \mathbf{R}^3,
``` 

represented in natural coordinates. 

- $u$: is the proximal vector in the global coordinate system.
- $r_p$: the position of the proximal point in the global coordinate system.
- $r_d$:  the position of the distal point in the global coordinate system.
- $w$: is the distal vector in the global coordinate system.

The biomechanism generalized coordinates:   

```math 
Q = (Q_1, \dots, Q_N).
```
is the concatenation of all body coordinates.

To rigidify the body segments and to articulate them, two types of holonomic constraints are handled in this formalism: rigid-body constraints and joint constraints (also termed as kinematic constraints), denoted $\Phi^r(Q)$ and $\Phi^j(Q)$, respectively, and gathered in a common constraint function $\Phi$:

```math 
\begin{align}
\Phi(Q) = \left(
    \Phi^r(Q) \quad
    \Phi^j(Q)
\right)^\top
\in \mathbf{R}^{6 \times N} \times \mathbf{R}^M.
\end{align}
```

# How to cite

Pierre Puchaud, Alexandre Naaim, & aaiaueil. (2024). Ipuch/bioNC: CarpentersTogether (0.9.0). Zenodo. https://doi.org/10.5281/zenodo.10822669

```
@software{pierre_puchaud_2024_10822669,
  author       = {Pierre Puchaud and
                  Alexandre Naaim and
                  aaiaueil},
  title        = {Ipuch/bioNC: CarpentersTogether},
  month        = mar,
  year         = 2024,
  publisher    = {Zenodo},
  version      = {0.9.0},
  doi          = {10.5281/zenodo.10822669},
  url          = {https://doi.org/10.5281/zenodo.10822669}
}
```

