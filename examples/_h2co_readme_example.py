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
