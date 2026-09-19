import numpy as np
from sumprod import init_internal


def internal_to_cartesian(q):
    r1, r2, theta = q
    xyz = np.array(
        [
            [0.0, 0.0, 0.0],
            [r1, 0.0, 0.0],
            # [r2 * np.cos(theta), r2 * np.sin(theta), 0.0],
            [r2 * theta, r2 * theta, 0.0],
        ],
        dtype=object,
    )
    return xyz


if __name__ == "__main__":
    masses = [16.0, 1.0, 1.0]

    grids = [
        np.linspace(0.8, 1.2, 10),
        np.linspace(0.8, 1.2, 10),
        np.linspace(60 * np.pi / 180, 160 * np.pi / 180, 10),
    ]

    q = init_internal(grids)
    xyz = internal_to_cartesian(q)
    print(xyz)
