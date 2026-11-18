"""KEO for formaldehyde (H2CO) in user-defined valence coordinates"""

import numpy as np

from keosop.gmat import Gmat
from keosop.revad import dihedral, dot, norm
from test_numeric import compare_sop_numeric

# Atom order: C, O, H1, H2
masses = np.array([12.0, 15.99491463, 1.00782505, 1.00782505])

# ---- Define internal-to-Cartesian and reverse transformations


def internal_to_cartesian(q):
    rCO, rCH1, rCH2, aOCH1, aOCH2, dtau = q
    tau = np.pi + dtau
    xyz = [
        [0.0, 0.0, 0.0],
        [0.0, 0.0, rCO],
        [
            rCH1 * np.sin(aOCH1) * np.cos(tau * 0.5),
            -rCH1 * np.sin(aOCH1) * np.sin(tau * 0.5),
            rCH1 * np.cos(aOCH1),
        ],
        [
            rCH2 * np.sin(aOCH2) * np.cos(tau * 0.5),
            rCH2 * np.sin(aOCH2) * np.sin(tau * 0.5),
            rCH2 * np.cos(aOCH2),
        ],
    ]
    return np.array(xyz, dtype=object)


def cartesian_to_internal(xyz):
    C, O, H1, H2 = xyz

    CO = O - C
    CH1 = H1 - C
    CH2 = H2 - C

    rCO = norm(CO)
    rCH1 = norm(CH1)
    rCH2 = norm(CH2)

    aOCH1 = np.arccos(dot(CO, CH1) / (rCO * rCH1))
    aOCH2 = np.arccos(dot(CO, CH2) / (rCO * rCH2))

    tau = dihedral(H1, C, O, H2)
    dtau = tau + np.pi

    return np.array([rCO, rCH1, rCH2, aOCH1, aOCH2, dtau], dtype=object)


# ---- Define 1D grids for each coordinate

q0 = np.r_[[1.0, 1.0, 1.0], np.deg2rad([120.0, 120.0, 0.0])]

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

# optionally, compress the sum-of-product expression using looser thresholds
pseudo = pseudo.compress(
    nterms_thresh=1,
    tol=1e-6,
    zero_tol=1e-6,
    identity_tol=1e-6,
)

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
