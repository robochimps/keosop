import numpy as np

from revad import Expr, as_expr, init_cartesian, reverse_derivatives
from sumprod import SumProd, init_internal, sop_ufunc


def eval_expr(root, x_bindings, grids, q_bindings, cache=None):
    """Recursively evaluates derivative graph into sum-of-products"""
    if cache is None:
        cache = {}

    if root.id in cache:
        return cache[root.id]

    if root.id in q_bindings:
        cache[root.id] = q_bindings[root.id]
        return cache[root.id]

    args = root.args
    oper = root.oper
    eval = lambda node: eval_expr(node, x_bindings, grids, q_bindings, cache)

    if oper == "const":
        out = SumProd.constant(root.data, grids)
    elif oper == "x_coo":
        out = x_bindings[root.data]
    elif oper == "add":
        out = eval(args[0]) + eval(args[1])
    elif oper == "neg":
        out = -eval(args[0])
    elif oper == "mul":
        out = eval(args[0]) * eval(args[1])
    elif oper == "div":
        out = eval(args[0]) / eval(args[1])
    elif oper == "pow":
        p = root.data
        out = eval(args[0]) ** p
    elif oper == "sin":
        out = sop_ufunc(eval(args[0]), grids, np.sin, "sin")
    elif oper == "cos":
        out = sop_ufunc(eval(args[0]), grids, np.cos, "cos")
    elif oper == "tan":
        out = sop_ufunc(eval(args[0]), grids, np.tan, "tan")
    elif oper == "sqrt":
        out = sop_ufunc(eval(args[0]), grids, np.sqrt, "sqrt")
    elif oper == "exp":
        out = sop_ufunc(eval(args[0]), grids, np.exp, "exp")
    elif oper == "log":
        out = sop_ufunc(eval(args[0]), grids, np.log, "log")
    elif oper == "acos":
        out = sop_ufunc(eval(args[0]), grids, np.arccos, "acos")
    elif oper == "asin":
        out = sop_ufunc(eval(args[0]), grids, np.arcsin, "asin")
    elif oper == "atan":
        out = sop_ufunc(eval(args[0]), grids, np.arctan, "atan")
    else:
        raise NotImplementedError(f"{oper}")

    cache[root.id] = out
    return out


class Gvib:
    def __init__(
        self,
        grids,
        masses,
        internal_to_cartesian,
        cartesian_to_internal,
        compress_tol: float = 1e-10,
        compress_zero_tol: float = 1e-12,
        compress_nterms_thresh: int = 10,
    ):
        self.grids = [np.asarray(elem, dtype=float) for elem in grids]
        self.masses = np.asarray(masses, dtype=float)
        self.internal_to_cartesian = internal_to_cartesian
        self.cartesian_to_internal = cartesian_to_internal
        self.ncoo = len(grids)
        self.natoms = len(masses)
        self.compress_tol = compress_tol
        self.compress_zero_tol = compress_zero_tol
        self.compress_nterms_thresh = compress_nterms_thresh

    def sumprod_x(self):
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

    def trace_q(self):
        xyz = np.array(
            [
                [init_cartesian(f"{alpha}{i}") for alpha in ("x", "y", "z")]
                for i in range(self.natoms)
            ]
        )

        q_expr = np.asarray(self.cartesian_to_internal(xyz), dtype=object).ravel()

        if len(q_expr) != self.ncoo:
            raise ValueError(
                f"'cartesian_to_internal' output size = {len(q_expr)} does not match 'len(grids)' = {self.ncoo}"
            )

        for icoo, q in enumerate(q_expr):
            if not isinstance(q, Expr):
                raise TypeError(f"cartesian_to_internal output {icoo} is not an Expr")

        return q_expr

    def sumprod_dq_dx(self, xyz_sop):
        # trace x->q
        q_expr = self.trace_q()

        # sum-of-products for Cartesian coordinates
        xyz_bindings = {
            f"{alpha}{i}": xyz_sop[i, ialpha]
            for i in range(self.natoms)
            for ialpha, alpha in enumerate(("x", "y", "z"))
        }

        q_bindings = {
            q_expr[icoo].id: SumProd.init_coordinate(icoo, self.grids)
            for icoo in range(self.ncoo)
        }

        svec = np.empty((self.ncoo, self.natoms, 3), dtype=object)

        for icoo, root in enumerate(q_expr):

            # graph for derivative dq_icoo / dx
            deriv = reverse_derivatives(root)

            for i in range(self.natoms):
                for ialpha, alpha in enumerate(("x", "y", "z")):

                    # evaluate graph: substitute Cartesian coordinates
                    # and internal coordinates by sum-of-product expressions
                    # defined in `xyz_bindings` and `q_bindings`, respectively
                    svec[icoo, i, ialpha] = eval_expr(
                        deriv.get(f"{alpha}{i}", as_expr(0)),
                        xyz_bindings,
                        self.grids,
                        q_bindings,
                    )

        return svec

    def build_gvib(self, svec):
        gvib = np.empty((self.ncoo, self.ncoo), dtype=object)

        for lam in range(self.ncoo):
            for mu in range(lam, self.ncoo):

                g = SumProd.constant(0, self.grids)

                for i, mass in enumerate(self.masses):
                    for ialpha in range(3):

                        g = g + svec[lam, i, ialpha] * svec[mu, i, ialpha] / mass

                        if g.nterms > self.compress_nterms_thresh:
                            g = g.compress(
                                nterms_thresh=1,
                                tol=self.compress_tol,
                                zero_tol=self.compress_zero_tol,
                            )

                g = g.compress(
                    nterms_thresh=1,
                    tol=self.compress_tol,
                    zero_tol=self.compress_zero_tol,
                )

                gvib[lam, mu] = gvib[mu, lam] = g

        return gvib

    def run(self):
        xyz = self.sumprod_x()
        svec = self.sumprod_dq_dx(xyz)
        gvib = self.build_gvib(svec)
        return gvib
