"""KEO for formic acid in Z-matrix coordinates"""

import numpy as np

from keosop.gmat import Gmat
from keosop.zmat import internal_to_cartesian, cartesian_to_internal
from test_numeric import compare_sop_numeric

# Atom order: C, O1, O2, H1, H2
masses = np.array([12.000000, 15.994910, 15.994910, 1.007825, 1.007825])

zmat_connections = [
    (0, 0, 0, 0),  # C
    (1, 0, 0, 0),  # O1 - C
    (1, 2, 0, 0),  # O2 - C - O1
    (1, 2, 3, 0),  # H1 - C - O1 - O2
    (3, 1, 2, 0),  # H2 - O2 - C - O1
]

# ---- Define 1D grids for each coordinate

q0 = np.r_[[1.0, 1.0, 1.0, 1.0], np.deg2rad([120.0, 120.0, 120.0, 360.0, 0.0])]

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
    batch_size=256,
    seed=123,
)
