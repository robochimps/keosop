import numpy as np

from keosop.sing.gmat import Gmat
from keosop.sing.zmat import cartesian_to_internal as singular_cartesian_to_internal
from keosop.sing.zmat import internal_to_cartesian as singular_internal_to_cartesian

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
    lambda q: singular_internal_to_cartesian(zmat_connections, q),
    lambda xyz: singular_cartesian_to_internal(zmat_connections, xyz),
    trace_singularities=True,
    singularity_coo=(3, 4),
    singularity_kinds=("sin",),
)
gvib, _, _ = g.run(return_rot=True)
print(gvib[5, 5].singularity_summary())  # G_{tau,tau} element
