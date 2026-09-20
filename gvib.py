import numpy as np

from revad import as_expr, init_cartesian, reverse_derivatives
from sumprod import SumProd, init_internal


def eval_expr(root, xyz_sop, grids, qnode_to_q, cache=None):
    """Recursively evaluates derivative graph into sum-of-products"""
    args = root.args
    oper = root.oper

    eval = lambda x: eval_expr(x, xyz_sop, grids, qnode_to_q, cache)

    if oper == "const":
        out = SumProd.constant(root.data, grids)
    elif oper == "x_coo":
        out = env[root.data]
    elif oper == "add":
        out = eval(args[0]) + eval(args[1])
    else:
        raise NotImplemented(oper)

    cache[root.id] = out
    return out


class Gvib:
    def __init__(self, grids, masses, internal_to_cartesian, cartesian_to_internal):
        self.grids = [np.asarray(elem, dtype=float) for elem in grids]
        self.masses = np.asarray(masses, dtype=float)
        self.internal_to_cartesian = internal_to_cartesian
        self.cartesian_to_internal = cartesian_to_internal
        self.ncoo = len(grids)
        self.natoms = len(masses)

    def sumprod_cartesian(self):
        q = init_internal(self.grids)
        x = np.asarray(self.internal_to_cartesian(q), dtype=object)

        if x.shape != (self.natoms, 3):
            raise ValueError(
                f"'internal_to_cartesian' must return shape ({self.natoms}, 3), got {x.shape}"
            )

        xyz = np.empty(x.shape, dtype=object)
        for i in range(self.natoms):
            for ialpha in range(3):
                if not isinstance(x[i, ialpha], SumProd):
                    val = SumProd.constant(x[i, ialpha], self.grids)
                    x[i, ialpha] = val
                xyz[i, ialpha] = x[i, ialpha]
        return xyz

    def trace_internal(self):
        xyz = np.array(
            [
                [init_cartesian(f"{alpha}{i}") for alpha in ("x", "y", "z")]
                for i in range(self.natoms)
            ]
        )

        q_expr = self.cartesian_to_internal(xyz)

        if len(q_expr) != self.ncoo:
            raise ValueError(
                f"'cartesian_to_internal' output size = {len(q_expr)} does not match 'ncoo' = {self.ncoo}"
            )

        return xyz, q_expr

    def sumprod_derivatives(self, xyz):
        _, q_expr = self.trace_internal()

        # sum-of-products for Cartesian coordinates
        xyz_sop = {
            f"{alpha}{i}": xyz[i, alpha]
            for i in range(self.natoms)
            for alpha in ("x", "y", "z")
        }

        qnode_to_q = {
            q_expr[icoo].id: SumProd.init_coordinate(icoo, self.grids)
            for icoo in range(self.ncoo)
        }

        svec = np.empty((self.ncoo, self.natoms, 3), dtype=object)

        for icoo, root in enumerate(q_expr):

            deriv = reverse_derivatives(root)

            for i in range(self.natoms):
                for ialpha, alpha in enumerate(("x", "y", "z")):

                    svec[icoo, i, ialpha] = eval_expr(
                        deriv.get(f"{alpha}{i}", as_expr(0)),
                        xyz_sop,
                        self.grids,
                        qnode_to_q,
                    )
