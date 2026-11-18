"""KEO for ethene in Z-matrix coordinates"""

import numpy as np
from test_numeric import compare_sop_numeric

from keosop.gmat import Gmat
from keosop.zmat import cartesian_to_internal, internal_to_cartesian

# Atom order: C1, C2, H1, H2, H3, H4, H5, H6
masses = np.array([12.0, 12.0] + [1.00782505] * 6)

zmat_connections = [
    (0, 0, 0, 0),  # C1
    (1, 0, 0, 0),  # C2 – C1
    (1, 2, 0, 0),  # H1 – C1 – C2
    (1, 2, 3, 0),  # H2 – C1 – C2 – H1
    (1, 2, 3, 0),  # H3 – C1 – C2 – H1
    (2, 1, 3, 0),  # H4 – C2 – C1 – H1
    (2, 1, 3, 0),  # H5 – C2 – C1 – H1
    (2, 1, 3, 0),  # H6 – C2 – C1 – H1
]

# ---- Define 1D grids for each coordinate

q0 = np.r_[
    [1.0] * 7, np.deg2rad([120] * 6), np.deg2rad([120.0, 240.0, 60.0, 180.0, 300.0])
]

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
