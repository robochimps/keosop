import numpy as np

from keosop.revad import angle, dihedral, distance


def _vec(x, y, z):
    return np.array([x, y, z], dtype=object)


def _vadd(a, b):
    return _vec(*(a[i] + b[i] for i in range(3)))


def _vsub(a, b):
    return _vec(*(a[i] - b[i] for i in range(3)))


def _vscale(s, a):
    return _vec(*(s * a[i] for i in range(3)))


def _cross(a, b):
    return _vec(
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _neg(a):
    return _vscale(-1.0, a)


def _put_plane(planes, a, b, c, n):
    """n || (b-a) x (c-b), cyclic permutations keep the orientation,
    odd permutations reverse it
    """
    planes[(a, b, c)] = n
    planes[(b, c, a)] = n
    planes[(c, a, b)] = n
    nn = _neg(n)
    planes[(a, c, b)] = nn
    planes[(c, b, a)] = nn
    planes[(b, a, c)] = nn


def internal_to_cartesian(zmat_connections, internal, pm=1, exact_sop=True):
    if pm not in (-1, 1):
        raise ValueError("pm must be +1 or -1")

    natoms = len(zmat_connections)
    nbonds = max(natoms - 1, 0)
    nangles = max(natoms - 2, 0)

    q = np.asarray(internal, dtype=object)

    expected = nbonds + nangles + max(natoms - 3, 0)
    if len(q) != expected:
        raise ValueError(
            f"Expected {expected} internal coordinates for {natoms} atoms, got {len(q)}"
        )

    xyz = []

    # directed unit vectors: units[(a,b)] points a -> b
    units = {}

    # oriented unit normals for reference planes
    planes = {}

    if natoms >= 1:
        xyz.append(_vec(0.0, 0.0, 0.0))

    if natoms >= 2:
        p1 = zmat_connections[1][0] - 1
        if p1 < 0 or p1 >= 1:
            raise ValueError("The second atom must reference the first atom")

        r = q[0]
        u = _vec(0.0, 0.0, 1.0)
        xyz.append(_vadd(xyz[p1], _vscale(r, u)))

        units[(p1, 1)] = u
        units[(1, p1)] = _neg(u)

    if natoms >= 3:
        p1, p2, _, _ = zmat_connections[2]
        p1 -= 1
        p2 -= 1

        if (p1, p2) not in units:
            raise ValueError(
                "Third atom requires an already known reference bond (p1,p2)"
            )

        r = q[1]
        alpha = q[nbonds]
        n2 = units[(p1, p2)]

        n1 = _vec(float(pm), 0.0, 0.0)
        u = _vadd(_vscale(np.cos(alpha), n2), _vscale(np.sin(alpha), n1))
        xyz.append(_vadd(xyz[p1], _vscale(r, u)))

        units[(p1, 2)] = u
        units[(2, p1)] = _neg(u)

        n3 = _cross(n2, n1)
        _put_plane(planes, p1, p2, 2, n3)

    for iatom in range(3, natoms):
        p1, p2, p3, J = zmat_connections[iatom]
        p1 -= 1
        p2 -= 1
        p3 -= 1

        r = q[iatom - 1]
        alpha = q[nbonds + iatom - 2]
        beta = q[nbonds + nangles + iatom - 3]

        if (p1, p2) not in units:
            raise NotImplementedError(
                f"Exact SOP Z-matrix frame unavailable for atom {iatom+1}: "
                f"reference bond ({p1+1},{p2+1}) was not generated earlier. "
                "A generic normalization would introduce a multi-term sqrt."
            )

        if (p1, p2, p3) not in planes:
            raise NotImplementedError(
                f"Exact SOP Z-matrix frame unavailable for atom {iatom+1}: "
                f"reference plane ({p1+1},{p2+1},{p3+1}) was not generated earlier. "
                "Choose conventional Z-matrix references or add a specialized frame rule."
            )

        n2 = units[(p1, p2)]
        n3 = planes[(p1, p2, p3)]
        n1 = _cross(n3, n2)

        if J == 0:
            cb = np.cos(beta)
            sb = np.sin(beta)

        elif J in (-1, 1):
            v13 = _vsub(xyz[p3], xyz[p1])
            r13 = np.sqrt(_dot(v13, v13))
            ca3 = _dot(n2, v13) / r13
            sa3 = np.sqrt(1.0 - ca3 * ca3)
            cb = (np.cos(beta) - np.cos(alpha) * ca3) / (np.sin(alpha) * sa3)
            sb = J * np.sqrt(1.0 - cb * cb)

        else:
            raise ValueError(
                f"Illegal J={J} for atom {iatom+1}; only J=0, +1, and -1 are supported"
            )

        u = _vadd(
            _vscale(np.cos(alpha), n2),
            _vadd(_vscale(np.sin(alpha) * cb, n1), _vscale(np.sin(alpha) * sb, n3)),
        )
        xyz.append(_vadd(xyz[p1], _vscale(r, u)))

        units[(p1, iatom)] = u
        units[(iatom, p1)] = _neg(u)

        new_normal = _vadd(_vscale(cb, n3), _vscale(-sb, n1))
        _put_plane(planes, p1, p2, iatom, new_normal)

    return np.asarray(xyz, dtype=object)


def cartesian_to_internal(zmat_connections, xyz):
    xyz = np.asarray(xyz, dtype=object)

    natoms = len(zmat_connections)
    bonds = []
    angles = []
    betas = []

    for iatom in range(1, natoms):
        p1 = zmat_connections[iatom][0] - 1
        bonds.append(distance(xyz[iatom], xyz[p1]))

    for iatom in range(2, natoms):
        p1, p2, _, _ = zmat_connections[iatom]
        angles.append(angle(xyz[iatom], xyz[p1 - 1], xyz[p2 - 1]))

    for iatom in range(3, natoms):
        p1, p2, p3, J = zmat_connections[iatom]
        p1 -= 1
        p2 -= 1
        p3 -= 1

        if J == 0:
            betas.append(-dihedral(xyz[iatom], xyz[p1], xyz[p2], xyz[p3]))
        elif J in (-1, 1):
            betas.append(angle(xyz[iatom], xyz[p1], xyz[p3]))
        else:
            raise ValueError(
                f"Illegal J={J} for atom {iatom+1}; only J=0, +1, and -1 are supported"
            )

    return np.asarray(bonds + angles + betas, dtype=object)
