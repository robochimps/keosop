"""KEO for fluoromethane (CH3F) in user-defined valence coordinates"""

import numpy as np

from keosop.gmat import Gmat
from test_numeric import compare_sop_numeric
from keosop.revad import signed_plane_angle, distance, angle

# Atom order: C, F, H1, H2, H3
masses = np.array([12.0, 18.9984, 1.007825, 1.007825, 1.007825])

# ---- Define internal-to-Cartesian and reverse transformations


def internal_to_cartesian(coords):
    r0, r1, r2, r3, beta1, beta2, beta3, stau1, stau2 = coords

    dbeta1 = np.pi - beta1
    dbeta2 = np.pi - beta2
    dbeta3 = np.pi - beta3

    tau1 = np.sqrt(6) / 3 * stau1 + 2 * np.pi / 3
    tau2 = -1 / np.sqrt(6) * stau1 + 1 / np.sqrt(2) * stau2 + 2 * np.pi / 3
    tau3 = -1 / np.sqrt(6) * stau1 - 1 / np.sqrt(2) * stau2 + 2 * np.pi / 3

    xyz = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, r0],
            [r1 * np.sin(dbeta1), 0.0, -r1 * np.cos(dbeta1)],
            [
                r2 * np.sin(dbeta2) * np.cos(tau3),
                r2 * np.sin(dbeta2) * np.sin(tau3),
                -r2 * np.cos(dbeta2),
            ],
            [
                r3 * np.sin(dbeta3) * np.cos(tau2),
                -r3 * np.sin(dbeta3) * np.sin(tau2),
                -r3 * np.cos(dbeta3),
            ],
        ],
        dtype=object,
    )

    return xyz


def cartesian_to_internal(xyz):
    C, F, H1, H2, H3 = xyz

    r0 = distance(C, F)
    r1 = distance(C, H1)
    r2 = distance(C, H2)
    r3 = distance(C, H3)

    beta1 = angle(F, C, H1)
    beta2 = angle(F, C, H2)
    beta3 = angle(F, C, H3)

    phi2 = signed_plane_angle((H1, C, F), (H2, C, F), (C, F))
    phi3 = signed_plane_angle((H1, C, F), (H3, C, F), (C, F))

    tau3 = phi2
    tau2 = -phi3
    tau1 = 2.0 * np.pi - tau2 - tau3

    stau1 = np.sqrt(6.0) / 2.0 * (tau1 - 2.0 * np.pi / 3.0)
    stau2 = (tau2 - tau3) / np.sqrt(2.0)

    return np.array([r0, r1, r2, r3, beta1, beta2, beta3, stau1, stau2], dtype=object)


# ---- Define 1D grids for each coordinate

q0 = np.r_[[1.0, 1.0, 1.0, 1.0], np.deg2rad([120.0, 120.0, 120.0, 0.0, 0.0])]

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
