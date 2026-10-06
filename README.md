# KEOSOP

Python package for constructing exact molecular rovibrational kinetic-energy operators (KEO) in sum-of-products (SOP) form.

The method requires the user to provide internal-to-Cartesian and Cartesian-to-internal coordinate transformations, together with a one-dimensional grid for each internal coordinate $q_k$ ($k=1,\ldots,M$), where $M=3N-6$ for a nonlinear molecule with $N$ atoms.

The elements of the KEO $G$-matrix and pseudopotential are constructed directly in SOP form,

$$
G_{\lambda\mu}(q_1,\ldots,q_M) = \sum_{t=1}^{N_t} g_{t,\lambda\mu}^{(1)}(q_1) \cdots g_{t,\lambda\mu}^{(M)}(q_M),
$$

where each one-dimensional factor $g_{t,\lambda\mu}^{(k)}(q_k)$ is represented numerically by its values on the corresponding one-dimensional grid rather than by an explicit analytical expression.
These factors are generated automatically by propagating SOP objects through the computational graphs defining the coordinate transformations.
The full multi-dimensional direct-product grid is never constructed.

For the rotational and rovibrational contributions to the $G$-matrix, KEOSOP follows the approach of Sørensen [G. O. Sørensen, "A new approach to the Hamiltonian of nonrigid molecules" (1979). In: Large Amplitude Motion in Molecules II. Topics in Current Chemistry, vol 82, https://doi.org/10.1007/BFb0048009].
This formulation requires three conditions, expressed in terms of the Cartesian coordinates of atoms, that uniquely define the orientation of the body-fixed molecular frame.
For example, such conditions may say that a particular atom lies in the $xz$ plane or that a particular bond is aligned with the $z$ axis, or more generally, define an Eckart frame.

KEOSOP includes an experimental procedure for automatically inferring body-frame conditions from the user-supplied internal-to-Cartesian transformation function.
The procedure searches for linear and quadratic relations among the Cartesian coordinates that are identically satisfied over the molecular geometries represented by the coordinate grids.
Automatic selection of body-fixed frames that respect molecular symmetry is yet to be implemented (see, e.g., T. Mellor, *Mol. Phys.* **120** (2023) e2118638, https://doi.org/10.1080/00268976.2022.2118638).

The construction of the rotational contribution requires the inversion of a geometry-dependent $3\times3$ matrix.
In general, the resulting inverse may not possess an exact finite SOP representation.
Furthermore, the internal-to-Cartesian coordinate transformation may introduce nonseparable coordinate dependencies.
This represents a potential limitation of the fully automatic construction of the KEO in SOP form.

Installation
-----
```bash
pip install --upgrade git+https://github.com/robochimps/keosop.git
```

Quick examples
-----
Sum-of-products representation of the kinetic energy operator ($G$-matrix and pseudopotential) for a triatomic molecule, expressed in terms of valence coordinates: two bond length $r_1$, $r_2$, and the bond angle $\alpha$:
```py
import numpy as np

from keosop.gmat import Gmat
from keosop.hdf5 import load_gmat_hdf5, save_gmat_hdf5
from keosop.revad import dot, norm

masses = np.array([15.99491463, 1.00782505, 1.00782505])  # masses of O, H, H atoms


# User-supplied function for computing Cartesian coordinates from internal valence coordinates
def internal_to_cartesian(q):
    r1, r2, a = q
    return np.array(
        [
            [0.0, 0.0, 0.0],
            [r1 * np.sin(a / 2), 0.0, r1 * np.cos(a / 2)],
            [-r2 * np.sin(a / 2), 0.0, r2 * np.cos(a / 2)],
        ],
        dtype=object,
    )


# User-supplied function for computing internal valence coordinates from Cartesian coordinates
def cartesian_to_internal(xyz):
    C, H1, H2 = xyz

    CH1 = H1 - C
    CH2 = H2 - C

    r1 = norm(CH1)
    r2 = norm(CH2)
    a = np.arccos(dot(CH1, CH2) / (r1 * r2))

    return np.array([r1, r2, a], dtype=object)


# User-supplied one-dimensional grids for each internal coordinate
# on which sum-of-products representation will be built

rad = np.pi / 180
grids = [
    np.linspace(0.8, 3.0, 1000),  # r1
    np.linspace(0.8, 3.0, 1000),  # r2
    np.linspace(30 * rad, 170 * rad, 1000),  # alpha
]

ncoo = len(grids)

# Generate G-matrix and pseudopotential in sum-of-products form

gmat = Gmat(grids, masses, internal_to_cartesian, cartesian_to_internal)

gvib_sop, grovib_sop, grot_sop = gmat.run(return_rot=True)
pseudo_sop = gmat.build_pseudopotential_fdf()

print("Number of terms in Gvib_{i,j}\n i  j no.terms")
for i in range(ncoo):
    for j in range(i, ncoo):
        print(f"{i:2d} {j:2d} {gvib_sop[i, j].nterms:6d}")

print("\nNumber of terms in Grovib_{i,j}\n i  j no.terms")
for i in range(3):
    for j in range(ncoo):
        print(f"{i:2d} {j:2d} {grovib_sop[i, j].nterms:6d}")

print("\nNumber of terms in Grot_{i,j}\n i  j no.terms")
for i in range(3):
    for j in range(i, 3):
        print(f"{i:2d} {j:2d} {grot_sop[i, j].nterms:6d}")

print(f"\nNumber of terms in pseudopotential: {pseudo_sop.nterms:6d}")

# Store data in file

filename = "keo_h2o.h5"

save_gmat_hdf5(
    filename,
    grids,
    gvib_sop,
    grovib_sop,
    grot_sop,
    pseudo_sop,
    coo_names=("r1", "r2", "alpha"),
    coo_units=("angs", "angs", "rad"),
)

# Load data from file

grids, gvib, grovib, grot, pseudo = load_gmat_hdf5(filename)
```
-----
Sum-of-products representation of the kinetic energy operator ($G$-matrix and pseudopotential) for formaldehyde ($\mathrm{H_2CO}$) molecule, expressed in terms of the standard Z-matrix coordinates:
```py
import numpy as np

from keosop.gmat import Gmat
from keosop.hdf5 import load_gmat_hdf5, save_gmat_hdf5
from keosop.zmat import cartesian_to_internal, internal_to_cartesian

# Define Z-matrix coordinates for H2CO

masses = np.array([12.0, 15.99491463, 1.00782505, 1.00782505])  # masses of C, O, H, H

zmat_connections = [
    (0, 0, 0, 0),  # C
    (1, 0, 0, 0),  # O  - C
    (1, 2, 0, 0),  # H1 - C - O
    (1, 2, 3, 0),  # H2 - C - O - H1
]


# Define one-dimensional grids for the internal coordinate representation
# and construct the G-matrix and pseudopotential in sum-of-products form

rad = np.pi / 180

# Define one-dimensional grids for each coordinate
grids = [
    np.linspace(0.8, 3.0, 1000),  # r_{CO}
    np.linspace(0.8, 3.0, 1000),  # r_{CH1}
    np.linspace(0.8, 3.0, 1000),  # r_{CH2}
    np.linspace(30 * rad, 170 * rad, 1000),  # alpha_{H1CO}
    np.linspace(30 * rad, 170 * rad, 1000),  # alpha_{H2CO}
    np.linspace(30 * rad, 330 * rad, 1000),  # tau
]

ncoo = len(grids)

# Compute for G-matrix and pseudopotential
gmat = Gmat(
    grids,
    masses,
    lambda q: internal_to_cartesian(zmat_connections, q),
    lambda xyz: cartesian_to_internal(zmat_connections, xyz),
)

gvib_sop, grovib_sop, grot_sop = gmat.run(return_rot=True)
pseudo_sop = gmat.build_pseudopotential_fdf()

print("Number of terms for vibrational elements of G-matrix\n i  j no.terms")
for i in range(ncoo):
    for j in range(i, ncoo):
        print(f"{i:2d} {j:2d} {gvib_sop[i, j].nterms:6d}")

print("\nNumber of terms for rovibrational elements of G-matrix\n i  j no.terms")
for i in range(3):
    for j in range(ncoo):
        print(f"{i:2d} {j:2d} {grovib_sop[i, j].nterms:6d}")

print("\nNumber of terms for rotational elements of G-matrix\n i  j no.terms")
for i in range(3):
    for j in range(i, 3):
        print(f"{i:2d} {j:2d} {grot_sop[i, j].nterms:6d}")

print(f"Number of terms for pseudopotential: {pseudo_sop.nterms:6d}")

# Store into file

filename = "keo_h2co.h5"

save_gmat_hdf5(filename, grids, gvib_sop, grovib_sop, grot_sop, pseudo_sop)

# Load from file

grids, gvib, grovib, grot, pseudo = load_gmat_hdf5(filename)
```

Examples
-----
More examples and user contributions for various molecules can be found in the [examples](examples) folder.

Citation
-----
If you use this code in your research, please cite:
> A. Yachmenev, G. Rauhut, "Automated sum-of-products construction of rovibrational kinetic energy operators in curvilinear coordinates without Taylor expansion, multidimensional fitting, or symbolic algebra", *J. Chem. Phys.* **XXX**, XXXXXX (2026) [DOI:](https://doi.org/), [arXiv:](https://doi.org/)

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

Contact
-----
For questions or feedback, feel free to open an issue or reach out to the authors directly via andrey.yachmenev@robochimps.com