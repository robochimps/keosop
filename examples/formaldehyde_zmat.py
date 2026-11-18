"""KEO for formaldehyde acid in Z-matrix coordinates"""

import numpy as np

from keosop.gmat import Gmat
from keosop.zmat import internal_to_cartesian, cartesian_to_internal
from keosop.hdf5 import save_gmat_hdf5, load_gmat_hdf5
from test_numeric import compare_sop_numeric

# Atom order: C, O, H1, H2
masses = np.array([12.0, 15.99491463, 1.00782505, 1.00782505])

zmat_connections = [
    (0, 0, 0, 0),  # C
    (1, 0, 0, 0),  # O  - C
    (1, 2, 0, 0),  # H1 - C - O
    (1, 2, 3, 0),  # H2 - C - O - H1
]

# ---- Define 1D grids for each coordinate

q0 = np.r_[[1.0, 1.0, 1.0], np.deg2rad([120.0, 120.0, 180.0])]

nbonds = len(masses) - 1

npoints = 1000

grids = [np.linspace(0.7, 2.5, npoints) for _ in range(nbonds)]  # angstrom
grids += [
    np.linspace(q - np.deg2rad(40.0), q + np.deg2rad(40.0), npoints)
    for q in q0[nbonds:]
]

# ---- Build sum-of-products representations

g = Gmat(
    grids,
    masses,
    lambda q: internal_to_cartesian(zmat_connections, q),
    lambda xyz: cartesian_to_internal(zmat_connections, xyz),
)

gvib, grovib, grot = g.run(return_rot=True)
pseudo = g.build_pseudopotential_fdf()

# ---- Print number of sum-of-products terms

counts = {
    name: max(sop.nterms for sop in block.flat)
    for name, block in [("Gvib", gvib), ("Grovib", grovib), ("Grot", grot)]
}
counts["pseudo"] = pseudo.nterms

print(f"\n{'Quantity':<10} {'Max SOP terms':>16}")
print("-" * 27)
for name, count in counts.items():
    print(f"{name:<10} {count:>16,d}")

# ---- Numerical test

comparison_report = compare_sop_numeric(
    grids,
    masses,
    gvib,
    grovib,
    grot,
    pseudo,
    internal_to_cartesian=g.internal_to_cartesian,
    max_points=1_000_000,
    batch_size=1024,
    seed=123,
)

# ---- Store data into HDF5 file

filename = "keo_h2co_zmat.h5"
save_gmat_hdf5(
    filename,
    grids,
    gvib,
    grovib,
    grot,
    pseudo,
    coo_names=("r_CO", "r_CH1", "r_CH2", "alpha_H1CO", "alpha_H2CO", "tau_H1COH2"),
    coo_units=("angs", "angs", "angs", "rad", "rad", "rad"),
)

# ---- Read data from HDF5 file

grids, gvib, grovib, grot, pseudo = load_gmat_hdf5(filename)
