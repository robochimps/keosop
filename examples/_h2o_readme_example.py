import numpy as np
from keosop.gmat import Gmat
from keosop.revad import dot, norm

masses = np.array([15.99491463, 1.00782505, 1.00782505])  # masses of O, H, H atoms

ncoo = 3


# User-supplied function for computing Cartesian coordinates from internal valence coordinates
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


# User-supplied function for computing internal valence coordinates from Cartesian coordinates
def cartesian_to_internal(xyz):
    C, H1, H2 = xyz

    CH1 = H1 - C
    CH2 = H2 - C

    r1 = norm(CH1)
    r2 = norm(CH2)
    a = np.arccos(dot(CH1, CH2) / (r1 * r2))

    return np.array([r1, r2, a], dtype=object)


# User-supplied one-dimensional grids for each internal coordinate
# on which sum-of-products representation will be built
rad = np.pi / 180
grids = [
    np.linspace(0.8, 2.0, 10000),  # r1
    np.linspace(0.8, 2.0, 10000),  # r2
    np.linspace(30 * rad, 170 * rad, 10000),  # alpha
]

# Generate Gvib, Grot, and Grovib in sum-of-products form
gmat = Gmat(grids, masses, internal_to_cartesian, cartesian_to_internal)
gvib_sop, grovib_sop, grot_sop = gmat.run(return_rot=True)

print("Number of terms in Gvib_{i,j}\n i  j no.terms")
for i in range(ncoo):
    for j in range(i, ncoo):
        print(f"{i:2d} {j:2d} {gvib_sop[i, j].nterms:6d}")

print("\nNumber of terms in Grovib_{i,j}\n i  j no.terms")
for i in range(3):
    for j in range(ncoo):
        print(f"{i:2d} {j:2d} {grovib_sop[i, j].nterms:6d}")

print("\nNumber of terms in Grot_{i,j}\n i  j no.terms")
for i in range(3):
    for j in range(i, 3):
        print(f"{i:2d} {j:2d} {grot_sop[i, j].nterms:6d}")
