import numpy as np
from sumprod import init_internal
from revad import trace, norm, dot, reverse_derivatives


def internal_to_cartesian(q):
    r1, r2, theta = q
    xyz = np.array(
        [
            [0.0, 0.0, 0.0],
            [r1, 0.0, 0.0],
            [r2 * np.cos(theta), r2 * np.sin(theta), 0.0],
            # [r2 * theta, r2 * theta, 0.0],
        ],
        dtype=object,
    )
    return xyz


def cartesian_to_internal(x):
    u = x[0] - x[1]
    v = x[2] - x[1]

    r1 = norm(u)
    r2 = norm(v)
    theta = np.arccos(dot(u, v) / (r1 * r2))

    return np.array([r1, r2, theta], dtype=object)


if __name__ == "__main__":
    masses = [16.0, 1.0, 1.0]

    grids = [
        np.linspace(0.8, 1.2, 10),
        np.linspace(0.8, 1.2, 10),
        np.linspace(60 * np.pi / 180, 160 * np.pi / 180, 10),
    ]

    q = init_internal(grids)
    xyz = internal_to_cartesian(q)
    print(xyz[2, 0].terms)

    expr = trace(3, 3, cartesian_to_internal)
    print(expr[0].oper, expr[0].args, expr[0].data, expr[0].id)

    deriv = reverse_derivatives(expr[0])
    print(deriv)
