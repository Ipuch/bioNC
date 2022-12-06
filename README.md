
This repository contains the source code for the bioNC library, a python package for biomechanics and natural coordinates formalism.
Inverse and Forward approach are implemented.

## Status

| Type | Status |
|---|---|
| License | <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/license-MIT-success" alt="License"/></a> |
| Continuous integration | [![Build status](https://ci.appveyor.com/api/projects/status/axakr33c79eo0xal/branch/main?svg=true)](https://ci.appveyor.com/project/Ipuch/bionc/branch/main) |
| Code coverage | [![codecov](https://codecov.io/gh/ipuch/bionc/branch/main/graph/badge.svg)](https://codecov.io/gh/ipuch/bionc) |

# Table of Contents

[Installation - from source](#installation---from-source)

[A first practical example](#a-first-practical-example)
- [The import](#the-import)
- [Building a model from scratch](#building-a-model-from-scratch)
- [Building a model from a c3d file](#building-a-model-from-a-c3d-file)
[Mathematical backends](#mathematical-backends)
[Natural coordinates reminders](#natural-coordinates-reminders)

[A more in depth look at the `bioptim` API](#a-more-in-depth-look-at-the-bioptim-api)
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