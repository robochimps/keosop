"""KEO for water in user-defined valence coordinates"""

import numpy as np

from keosop.gmat import Gmat
from keosop.revad import dot, norm
from test_numeric import compare_sop_numeric

# Atom order: O, H1, H2
masses = np.array([15.99491463, 1.00782505, 1.00782505])

# ---- Define internal-to-Cartesian and reverse transformations


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


def cartesian_to_internal(xyz):
    C, H1, H2 = xyz

    CH1 = H1 - C
    CH2 = H2 - C

    r1 = norm(CH1)
    r2 = norm(CH2)
    a = np.arccos(dot(CH1, CH2) / (r1 * r2))

    return np.array([r1, r2, a], dtype=object)


# ---- Define 1D grids for each coordinate

q0 = np.r_[[1.0, 1.0], np.deg2rad([120.0])]

nbonds = len(masses) - 1

npoints = 1000

grids = [np.linspace(0.7, 2.5, npoints) for _ in range(nbonds)]  # angstrom
grids += [
    np.linspace(q - np.deg2rad(40.0), q + np.deg2rad(40.0), npoints)
    for q in q0[nbonds:]
]

# ---- Build sum-of-products representations

g = Gmat(grids, masses, internal_to_cartesian, cartesian_to_internal)

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
