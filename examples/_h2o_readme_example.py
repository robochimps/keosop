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
