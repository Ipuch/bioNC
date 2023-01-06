
This repository contains the source code for the bioNC library, a python package for biomechanics and natural coordinates formalism.
Inverse and Forward approach are implemented.

## Status

| Type | Status |
|---|---|
| License | <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/license-MIT-success" alt="License"/></a> |
| Continuous integration | [![Build status](https://github.com/Ipuch/bioNC/actions/workflows/run_tests.yml/badge.svg)](https://github.com/Ipuch/bioNC/actions/) |
| Code coverage | [![codecov](https://codecov.io/gh/ipuch/bionc/branch/main/graph/badge.svg)](https://codecov.io/gh/ipuch/bionc) |

# Table of Contents

[Installation - from source](#installation---from-source)

[A first practical example](#a-first-practical-example)
- [The import](#the-import)
- [Building a model from scratch](#building-a-model-from-scratch)
- [Building a model from a c3d file](#building-a-model-from-a-c3d-file)
[Mathematical backends](#mathematical-backends)
[Natural coordinates reminders](#natural-coordinates-reminders)

[A more in depth look at the `bionc` API](#a-more-in-depth-look-at-the-bionc-api)
 todo

# Installation - from source
One can install the package from source using the following command:
```
    python setup.py install
```

# A first practical example
The easiest way to learn bionc is to dive into it.
So let's build our first model.
Please note that this tutorial is designed to recreate example which builds a lower limb model (Pelvis, Thigh, Shank, Foot).

## The import
The first step is to import the classes we will need.
```python
from bionc import (
    AxisTemplate,
    BiomechanicalModelTemplate,
    MarkerTemplate,
    SegmentTemplate,
    NaturalSegmentTemplate,
    C3dData,
    BiomechanicalModel,
    JointType,
)
```

## Building a model from scratch

todo

## Building a model from a c3d file

todo

# Mathematical backends
This toolbox support two mathematical backends: `numpy` and `casadi`.
todo

# Natural coordinates reminders

The ``i``-th body in a biomechanism with ``N`` bodies has generalized coordinates:  

```math 
Q_i = (u_{i}, r_p_{i}, r_d_{i}, w_{i}) \in \mathbf{R}^3 \times \mathbf{R}^3 \times \mathbf{R}^3 \times \mathbf{R}^3,
``` 

represented in natural coordinates. 

- ``u``: is the proximal vector in the global coordinate system.
- ``r_p``: the position of the proximal point in the global coordinate system.
- ``r_d``:  the position of the distal point in the global coordinate system.
- ``w``: is the distal vector in the global coordinate system.

The biomechanism generalized coordinates:   

```math 
Q = (Q_1, \dots, Q_N}).
```

is the concatentation of all body coordinates.



