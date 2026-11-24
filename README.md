# KEOSOP

KEOSOP constructs molecular rovibrational kinetic-energy operators (KEOs) directly in numerical sum-of-products (SOP) form from curvilinear coordinate transformations.
It combines reverse-mode automatic differentiation with SOP arithmetic, without first deriving a complete symbolic KEO or fitting it on a multidimensional grid.

The package computes the vibrational, rovibrational, and rotational $G$-matrix blocks, together with an optional pseudopotential.

## Installation

KEOSOP requires Python 3.12.11 or newer.
For installation, use the command below:
```bash
python -m pip install --upgrade git+https://github.com/robochimps/keosop.git
```

## Quick start: water in user-defined coordinates

The following example uses two O-H bond lengths, `r1` and `r2`, and the H-O-H valence angle `alpha`, the O atom is placed at the origin, and the Cartesian $z$-axis is aligned along the H-O-H bisector:

```python
import numpy as np

from keosop.gmat import Gmat
from keosop.revad import angle, distance

# Atom order: O, H1, H2
masses = np.array([15.99491463, 1.00782505, 1.00782505])


def internal_to_cartesian(q):
    r1, r2, alpha = q
    return np.array(
        [
            [0.0, 0.0, 0.0],
            [r1 * np.sin(alpha / 2), 0.0, r1 * np.cos(alpha / 2)],
            [-r2 * np.sin(alpha / 2), 0.0, r2 * np.cos(alpha / 2)],
        ],
        dtype=object,
    )


def cartesian_to_internal(xyz):
    O, H1, H2 = xyz
    return np.array(
        [distance(O, H1), distance(O, H2), angle(H1, O, H2)],
        dtype=object,
    )


# Coordinate order: r1, r2, alpha
grids = [
    np.linspace(0.8, 1.4, 12),
    np.linspace(0.8, 1.4, 12),
    np.deg2rad(np.linspace(80.0, 140.0, 16)),
]

g = Gmat(grids, masses, internal_to_cartesian, cartesian_to_internal)
gvib, grovib, grot = g.run(return_rot=True)
pseudo = g.build_pseudopotential_fdf(stencil=5, fdf_step=1e-3)

print("Gvib shape:", gvib.shape)
print("Grovib shape:", grovib.shape)
print("Grot shape:", grot.shape)
print("SOP terms in Gvib[0, 0]:", gvib[0, 0].nterms)
print("SOP terms in U:", pseudo.nterms)
```

The shape output for this three-coordinate example is:

```text
Gvib shape: (3, 3)
Grovib shape: (3, 3)
Grot shape: (3, 3)
```

The pseudopotential is constructed separately.

## Inputs, outputs, and SOP factors

For `N` atoms and `M` internal coordinates, the main inputs are:

| Input | Format | Meaning |
| --- | --- | --- |
| `grids` | Sequence of `M` one-dimensional arrays | One grid per coordinate, in the same order as both transformations. |
| `masses` | Array of shape `(N,)` | Positive masses, in the same atom order as the Cartesian arrays. |
| `internal_to_cartesian(q)` | Returns an object array of shape `(N, 3)` | Maps internal coordinates to Cartesian components ordered as `x, y, z`. During SOP construction, `q[k]` is an SOP object. |
| `cartesian_to_internal(xyz)` | Returns `M` traced expressions | Defines the inverse coordinate map. During tracing, `xyz` contains `keosop.revad.Expr` objects, not ordinary floating-point values. |
| `frame_conditions(xyz)` | Optional callable returning three expressions | Specifies the body-fixed frame explicitly. If omitted, compatible conditions are inferred from the Cartesian embedding. |

Use scalar arithmetic, supported NumPy functions, and object arrays in the transformation callbacks. 
Geometry helpers such as `distance`, `angle`, `dihedral`, `dot`, and `norm` are available from `keosop.revad`.
Avoid casting traced objects to floats or introducing coordinate-dependent `if` statements.

The two transformations must describe the same coordinates, atom ordering, and angular branches on the chosen domain.

The outputs of `g.run(return_rot=True)` are object arrays:

| Output | Shape | Meaning |
| --- | --- | --- |
| `gvib` | `(M, M)` | Vibrational block. |
| `grovib` | `(3, M)` | Rovibrational block, where rows index body-fixed frame axes. |
| `grot` | `(3, 3)` | Rotational block. |
| `pseudo` | One SOP object | Pseudopotential returned by `build_pseudopotential_fdf()`. |

With default arguments, `builder.run()` returns only `gvib`.

An SOP element is represented as

$$
G_{\lambda\mu}(q_1,\ldots,q_M)
= \sum_{t=1}^{N_t}\prod_{k=1}^{M}g^{(k)}_{t,\lambda\mu}(q_k).
$$

The one-dimensional factors are stored as arrays on user-defined `grids` rather than analytical formulas.
For an SOP object `sop`, `sop.terms[t][k]` contains the values of factor for coordinate `k` in term `t` on `grids[k]`, and `sop.nterms` is the number of stored products.
This term count is not necessarily the minimal tensor rank.

For example, evaluate a regular SOP element at a tuple of grid indices:

```python
sop = gvib[0, 0]
indices = tuple(len(grid) // 2 for grid in grids)
value = sum(
    np.prod([factor[indices[k]] for k, factor in enumerate(term)])
    for term in sop.terms
)
print("Gvib[0, 0] at the selected grid point:", value)
```

## Saving and loading operators

The following continues the water example:

```python
from keosop.hdf5 import load_gmat_hdf5, save_gmat_hdf5

save_gmat_hdf5(
    "keo_h2o.h5",
    grids,
    gvib,
    grovib,
    grot,
    pseudo,
    coo_names=("r1", "r2", "alpha"),
    coo_units=("angstrom", "angstrom", "rad"),
    metadata={"molecule": "H2O", "coordinates": "valence"},
)

(
    loaded_grids,
    loaded_gvib,
    loaded_grovib,
    loaded_grot,
    loaded_pseudo,
    metadata,
) = load_gmat_hdf5("keo_h2o.h5", return_metadata=True)

print("Loaded coordinates:", metadata["coo_names"])
```

HDF5 files contain the coordinate grids, numerical SOP factors, optional coordinate metadata, and protected-factor metadata when present.
The default load result is `(grids, gvib, grovib, grot, pseudo)`, while `return_metadata=True` appends a metadata dictionary.
Omitted operator blocks are returned as `None`.

## Z-matrix coordinates

A conventional Z-matrix provides both coordinate transformations.
Here is a complete formaldehyde example:

```python
import numpy as np

from keosop.gmat import Gmat
from keosop.zmat import cartesian_to_internal, internal_to_cartesian

# Atom order: C, O, H1, H2
zmat_masses = np.array([12.0, 15.99491463, 1.00782505, 1.00782505])
zmat_connections = [
    (0, 0, 0, 0),  # C
    (1, 0, 0, 0),  # C-O
    (1, 2, 0, 0),  # H1-C-O
    (1, 2, 3, 0),  # H2-C-O-H1
]

# Order: rCO, rCH1, rCH2, alpha_H1CO, alpha_H2CO, tau
zmat_grids = [
    np.linspace(1.1, 1.3, 8),
    np.linspace(0.95, 1.15, 8),
    np.linspace(0.95, 1.15, 8),
    np.deg2rad(np.linspace(110.0, 130.0, 8)),
    np.deg2rad(np.linspace(110.0, 130.0, 8)),
    np.deg2rad(np.linspace(150.0, 175.0, 8)),
]

g = Gmat(
    zmat_grids,
    zmat_masses,
    lambda q: internal_to_cartesian(zmat_connections, q),
    lambda xyz: cartesian_to_internal(zmat_connections, xyz),
)
gvib_zmat, grovib_zmat, grot_zmat = g.run(return_rot=True)
pseudo_zmat = g.build_pseudopotential_fdf()
print(gvib_zmat.shape, grovib_zmat.shape, grot_zmat.shape)
```
For four atoms, the $G$-matrix block shapes are `(6, 6)`, `(3, 6)`, and `(3, 3)`.

The current exact SOP implementation requires each new atom in a Z-matrix to be placed using reference bonds and planes generated in earlier construction steps.
Other reference patterns, including some second-angle definitions, may introduce multivariate square roots that the current SOP algebra cannot handle, which will raise an error.

## Body-fixed frame

The rotational construction follows [Sørensen's formulation](https://doi.org/10.1007/BFb0048009), where three independent conditions on the Cartesian coordinates specify the body-fixed frame.

By default, KEOSOP infers compatible conditions from the Cartesian embedding supplied by the transformation.
It does not optimize the frame for symmetry, basis convergence, or SOP compactness. A `frame_conditions(xyz)` callback can supply the three conditions explicitly.

To inspect the inferred conditions and auxiliary data:

```python
gvib, grovib, grot, aux = g.run(return_rot=True, return_svec=True)
print("Inferred frame conditions:", aux["frame_conditions"])
```

The auxiliary data also include the Cartesian SOP coordinates, $s$-vectors, and rotational frame-condition Jacobian `J`.

## Singularity tracing

Recognized coordinate singularities can be retained as protected analytical prefactors using the separate experimental `keosop.sing` implementation.
For example, the following code traces sine factors associated with the two valence angles in the formaldehyde Z-matrix example:

```python
from keosop.sing.gmat import Gmat
from keosop.sing.zmat import (
    cartesian_to_internal as singular_cartesian_to_internal,
    internal_to_cartesian as singular_internal_to_cartesian,
)

g = Gmat(
    zmat_grids,
    zmat_masses,
    lambda q: singular_internal_to_cartesian(zmat_connections, q),
    lambda xyz: singular_cartesian_to_internal(zmat_connections, xyz),
    trace_singularities=True,
    singularity_coo=(3, 4),
    singularity_kinds=("sin",),
)

gvib, _, _ = g.run(return_rot=True)
print(gvib[5, 5].singularity_summary()) # G_{tau,tau} element (dihedral angle)
```

Protected factors are stored separately from the numerical arrays in `sop.terms`.
For these objects, use `sop.evaluate(indices)` to include the analytical prefactors, and `sop.singularity_summary()` to inspect them.
Summing only the numerical factors does not reconstruct the complete singular expression.
See the [HCN](examples/hcn_jacobi.ipynb) and [HCCH](examples/hcch_zmat.ipynb) notebooks for examples.

## Optional N-mode approximation

A truncated $N$-mode expansion can restrict the number of simultaneously coupled coordinates in each $G$-matrix element:

```python
g = Gmat(
    grids,
    masses,
    builder.internal_to_cartesian,
    builder.cartesian_to_internal,
    nmode_order=2,
    nmode_ref_indices=[len(grid) // 2 for grid in grids],
)
gvib_2mode = g.run()
```

The reference indices `nmode_ref_indices` select the reference expansion point for each coordinate grid. 

## Supported operations and limitations

The standard SOP backend supports sums, products, and nonnegative integer powers.
It also expands sine and cosine functions of additively separable arguments, such as `cos(q0 + q1)`, using addition formulas.
Reciprocals are supported when the denominator reduces to a single separable product or depends on one coordinate, and not equal to zero.

General multivariate nonlinear functions are not automatically factorized.
Examples of functions that are currently outside the supported rules include (even when an exact factorization exists):
- `1 / (1 + q0 * q1)`
- `np.exp(q0 * q1)`
- `np.exp(q0 + q1)`

An unsupported operation raises an error rather than silently introducing a fit.
The singularity-tracing backend has a more restricted set of function rules.
For example, it does not currently include the standard multivariate sine and cosine addition formulas.

Other practical limitations:
- The rotational construction requires inversion of a geometry-dependent $3\times 3$ frame matrix.
A singular determinant or an unsupported reciprocal can prevent construction.
- Runtime and memory depend on intermediate term growth, coordinate definitions, grid sizes, and simplification settings.
- The formulation uses $M=3N-6$ internal coordinates.
Linear configurations can occur as singular limits, as illustrated by the HCN and HCCH examples.
- Automatic numerical low-rank fitting of unsupported intermediates remains a possible extension and is not currently implemented.

## More examples and numerical validation

The [`examples`](examples) directory contains complete scripts and notebooks.

## Citation

If you use KEOSOP in your research, please cite the associated manuscript:

> A. Yachmenev and G. Rauhut, “Automated sum-of-products construction of rovibrational kinetic energy operators in curvilinear coordinates without Taylor expansion, multidimensional fitting, or symbolic algebra” (2026).

```bibtex
@article{Yachmenev_JCPXXX_2026,
  author  = {A. Yachmenev, G. Rauhut},
  title   = {Automated sum-of-products construction of rovibrational kinetic energy operators in curvilinear coordinates without Taylor expansion, multidimensional fitting, or symbolic algebra},
  year    = {2026},
  journal = {J. Chem. Phys.},
  volume = {},
  pages = {},
  url = {},
  DOI = {},
  archiveprefix = {arXiv},
  eprint = {},
  primaryclass = {physics},
  url = {},
}
```

## License

KEOSOP is distributed under the [MIT License](LICENSE).

## Contact

For questions, bug reports, and feature requests, open a [GitHub issue](https://github.com/robochimps/keosop/issues) or contact [andrey.yachmenev@robochimps.com](mailto:andrey.yachmenev@robochimps.com).