import jax
from jax import numpy as jnp

jax.config.update("jax_enable_x64", True)


def internal_to_cartesian_jax(zmat_connections, internal, pm=1):

    if pm not in (-1, 1):
        raise ValueError("pm must be +1 or -1")

    natoms = len(zmat_connections)

    q = jnp.asarray(internal)

    nbonds = max(natoms - 1, 0)
    nangles = max(natoms - 2, 0)

    expected = nbonds + nangles + max(natoms - 3, 0)
    if q.ndim != 1 or q.shape[0] != expected:
        raise ValueError(
            f"Expected {expected} internal coordinates for {natoms} atoms, got shape {q.shape}"
        )

    def vec(x, y, z):
        return jnp.stack(
            (
                jnp.asarray(x, dtype=q.dtype),
                jnp.asarray(y, dtype=q.dtype),
                jnp.asarray(z, dtype=q.dtype),
            )
        )

    def put_plane(planes, a, b, c, n):
        planes[(a, b, c)] = n
        planes[(b, c, a)] = n
        planes[(c, a, b)] = n
        nn = -n
        planes[(a, c, b)] = nn
        planes[(c, b, a)] = nn
        planes[(b, a, c)] = nn

    xyz = []
    units = {}
    planes = {}

    if natoms >= 1:
        xyz.append(vec(0.0, 0.0, 0.0))

    if natoms >= 2:
        p1 = zmat_connections[1][0] - 1

        if p1 < 0 or p1 >= 1:
            raise ValueError("The second atom must reference the first atom")

        u = vec(0.0, 0.0, 1.0)
        xyz.append(xyz[p1] + q[0] * u)

        units[(p1, 1)] = u
        units[(1, p1)] = -u

    if natoms >= 3:
        p1, p2, _, _ = zmat_connections[2]
        p1 -= 1
        p2 -= 1

        if (p1, p2) not in units:
            raise ValueError(
                "Third atom requires an already known reference bond (p1, p2)"
            )

        r = q[1]
        alpha = q[nbonds]
        n2 = units[(p1, p2)]
        n1 = vec(float(pm), 0.0, 0.0)
        u = jnp.cos(alpha) * n2 + jnp.sin(alpha) * n1
        xyz.append(xyz[p1] + r * u)

        units[(p1, 2)] = u
        units[(2, p1)] = -u

        n3 = jnp.cross(n2, n1)
        put_plane(planes, p1, p2, 2, n3)

    for iatom in range(3, natoms):
        p1, p2, p3, J = zmat_connections[iatom]
        p1 -= 1
        p2 -= 1
        p3 -= 1

        r = q[iatom - 1]
        alpha = q[nbonds + iatom - 2]
        beta = q[nbonds + nangles + iatom - 3]

        if (p1, p2) not in units:
            raise ValueError(
                f"Z-matrix frame unavailable for atom {iatom+1}: reference bond "
                f"({p1+1}, {p2+1}) was not generated earlier"
            )

        if (p1, p2, p3) not in planes:
            raise ValueError(
                f"Z-matrix frame unavailable for atom {iatom+1}: reference plane "
                f"({p1+1}, {p2+1}, {p3+1}) was not generated earlier"
            )

        n2 = units[(p1, p2)]
        n3 = planes[(p1, p2, p3)]
        n1 = jnp.cross(n3, n2)

        if J == 0:
            cb = jnp.cos(beta)
            sb = jnp.sin(beta)
        elif J in (-1, 1):
            v13 = xyz[p3] - xyz[p1]
            r13 = jnp.linalg.norm(v13)
            ca3 = jnp.dot(n2, v13) / r13
            # this is sin(angle)
            sa3 = jnp.sqrt(jnp.maximum(0.0, 1.0 - ca3 * ca3))
            cb = (jnp.cos(beta) - jnp.cos(alpha) * ca3) / (jnp.sin(alpha) * sa3)
            cb = jnp.clip(cb, -1.0, 1.0)
            sb = J * jnp.sqrt(jnp.maximum(0.0, 1.0 - cb * cb))
        else:
            raise ValueError(
                f"Illegal J={J} for atom {iatom+1}; only J=0, +1, and -1 are supported"
            )

        u = jnp.cos(alpha) * n2 + jnp.sin(alpha) * cb * n1 + jnp.sin(alpha) * sb * n3
        xyz.append(xyz[p1] + r * u)

        units[(p1, iatom)] = u
        units[(iatom, p1)] = -u

        new_normal = cb * n3 - sb * n1
        put_plane(planes, p1, p2, iatom, new_normal)

    return jnp.stack(xyz, axis=0)
