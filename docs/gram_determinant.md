# The Gram determinant `delta`, and where it comes from

This note explains the scalar that every transformation matrix `B` and every analytical inverse in
`bionc` is built on:

```
delta = 1 - cos²α - cos²β - cos²γ + 2·cos α·cos β·cos γ
```

implemented once in [`bionc/utils/gram.py`](../bionc/utils/gram.py) as `gram_determinant`, and used
by [`bionc/bionc_numpy/transformation_matrix.py`](../bionc/bionc_numpy/transformation_matrix.py)
(via `gram_determinant_sqrt`) and by `AbstractNaturalSegment._angle_sanity_check` in
[`bionc/protocols/natural_segment.py`](../bionc/protocols/natural_segment.py).

It is not a `bionc` invention and needs no reference to be trusted — it is a 3×3 determinant you can
expand by hand in a minute. But it has standard names in two other fields, and those names are what
you cite.

---

## 1. Where it comes from: it is `det(BᵀB)`

Write `B = A · diag(1, L, 1)`, where `A` has the three **unit** natural vectors as columns:
`u`, `v̂ = (rp − rd)/L`, and `w`. Then `AᵀA` is just the table of pairwise dot products. With
`bionc`'s angle convention — α between `v` and `w`, β between `u` and `w`, γ between `u` and `v`:

```
G = AᵀA = [[1,     cos γ, cos β],
           [cos γ, 1,     cos α],
           [cos β, cos α, 1    ]]
```

Expanding that determinant:

```
det G = 1·(1 − cos²α) − cos γ·(cos γ − cos α cos β) + cos β·(cos γ cos α − cos β)
      = 1 − cos²α − cos²β − cos²γ + 2 cos α cos β cos γ
      = delta
```

Two consequences, and they are the whole reason the quantity matters here:

- **`det(B) = L·√delta`, for every `TransformationMatrixType`.** Because `det G = det(A)²` and
  `det B = L·det A`. All the matrix types have the same columns, only expressed in different
  orthonormal frames, so they all share this determinant.
- **`delta > 0` is exactly the condition for the three angles to be realisable.** A Gram determinant
  is `≥ 0` always, and `= 0` precisely when the vectors are linearly dependent. `delta ≤ 0` means no
  three vectors in 3D can hold those pairwise angles at all.

It also gives the guard for free: `delta > 0` implies `sin α`, `sin β`, `sin γ` are all non-zero
(if `cos γ = ±1` then `delta = −(cos α ∓ cos β)² ≤ 0`, and likewise for the others). That is what
makes the divisions by `sin β` and `sin γ` in the analytical inverses safe — one criterion covers
both singularities, which is why `_angle_sanity_check` tests only this.

## 2. What to cite

**As linear algebra.** `G = AᵀA` is the *Gram matrix* (Gramian) of the three vectors, after
Jørgen Pedersen Gram. Its determinant is the squared volume of the parallelepiped they span, and
vanishes iff they are linearly dependent. Any standard reference covers it — Horn & Johnson,
*Matrix Analysis*, or [Wikipedia: Gram matrix](https://en.wikipedia.org/wiki/Gram_determinant).

**As crystallography.** This is *literally* the triclinic unit-cell volume formula,

```
V = a·b·c·√(1 − cos²α − cos²β − cos²γ + 2 cos α cos β cos γ)
```

A natural segment and a triclinic unit cell are the same object: three edge lengths and three
pairwise angles. See any crystallography text, or
[Triclinic unit cell](https://msestudent.com/triclinic-unit-cell/).

**For the realisability condition specifically**, the best citation is:

> J. Foadi and G. Evans, *On the allowed values for the triclinic unit-cell angles*,
> **Acta Crystallographica A67, 93–95 (2011)**.
> [journals.iucr.org/a/issues/2011/01/00/au5114](https://journals.iucr.org/a/issues/2011/01/00/au5114/au5114.pdf)

It makes exactly the point `_angle_sanity_check` encodes — that the three angles are *not*
independent and that manuals and tables do not say so. Their headline example is that a cell with
angles 60°, 60°, 130° cannot exist despite looking perfectly sensible. That is essentially the case
`tests/test_natural_segment.py::test_angle_sanity_check` rejects (36°, 36°, 120°).

## 3. The cleaner equivalent: it is a spherical triangle

`delta` factorizes. With the semi-perimeter `s = (α + β + γ)/2`:

```
delta = 4·sin(s)·sin(s − α)·sin(s − β)·sin(s − γ)
```

(verified against the expanded form over 200 000 random triples: agreement to `3.6e-15`).

This is the spherical analogue of Heron's formula, and it says something much more intuitive than
the polynomial does. Three unit vectors from the origin cut a **spherical triangle** on the unit
sphere whose *side lengths* are α, β, γ. So `delta > 0` is nothing but the **spherical triangle
inequality**:

```
α < β + γ,    β < α + γ,    γ < α + β,    α + β + γ < 2π
```

Read that way, the rejected test case is obvious at a glance: α = β = 36°, γ = 120°, and
`120° > 36° + 36°`. No triangle, no segment.

**This form is the one to reach for when explaining a rejection to a user** — it identifies *which*
inequality failed, where `delta = −1.21` says nothing. It is not, however, what the library
evaluates; see below.

## 4. Which form the code actually evaluates

Three algebraically identical candidates, counted in atomic operations:

| form | expression | ops |
|---|---|---|
| symmetric | `1 - ca² - cb² - cg² + 2·ca·cb·cg` | 3 cos, 6 mul, 4 add/sub |
| **Horner in `ca`** | **`ca·(2·cb·cg − ca) + 1 − cb·cb − cg·cg`** | **3 cos, 5 mul, 4 add/sub** |
| spherical | `4·sin(s)·sin(s−α)·sin(s−β)·sin(s−γ)` | 4 sin, 4 mul, 6 add/sub |

Three transcendental calls is the floor — you need all three cosines regardless — so the spherical
form's fourth `sin` is a real cost that its fewer multiplies do not recover. Measured against the
symmetric form (best of 7 runs, agreement to `1e-15` throughout):

| form | pure scalar (`math`) | numpy scalar | numpy vectorized, 100k triples |
|---|---|---|---|
| symmetric | 1.00× | 1.00× | 1.00× |
| **Horner** | **0.69×** | **0.93×** | **0.97×** |
| spherical | 0.67× | 1.06× | 1.22× |

The Horner form is the only candidate faster than the symmetric one in *all three* regimes, so that
is what `gram_determinant` evaluates. The spherical form is fastest on pure Python scalars but the
slowest vectorized, by 22%.

Two caveats worth keeping in mind before treating this as important:

- The margins are small and the call is not in a hot loop — `gram_determinant` runs a handful of
  times per segment construction, not per frame. The symmetric form would be perfectly defensible on
  readability grounds; it is kept in the docstring for exactly that reason.
- `bionc_casadi` keeps its own copy in
  [`bionc/bionc_casadi/transformation_matrix.py`](../bionc/bionc_casadi/transformation_matrix.py),
  because casadi needs `casadi.cos` rather than `np.cos`. It is written in the symmetric form. For
  `MX` arguments these timings do not transfer at all — casadi builds an expression graph and its
  own simplifier decides the arithmetic — so there is no reason to mirror the Horner form there.

`tests/test_transformation_matrix.py` pins all three forms against each other, so the optimisation
cannot silently drift from the definition.
