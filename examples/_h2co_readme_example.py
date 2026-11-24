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
