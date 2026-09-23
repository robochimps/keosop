import numpy as np

from revad import (
    Expr,
    _geom_cache,
    as_expr,
    expr_signature,
    init_cartesian,
    reverse_derivatives,
)
from sumprod import SumProd, init_internal, sop_ufunc


def eval_expr(root, x_bindings, grids, q_bindings, cache=None):
    """Recursively evaluates derivative graph into sum-of-products"""
    if cache is None:
        cache = {}

    if root.id in cache:
        return cache[root.id]

    binding = q_bindings.get(root.id)
    if binding is None:
        binding = q_bindings.get(("sig", expr_signature(root)))
    if binding is not None:
        cache[root.id] = binding
        return binding

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
    elif oper == "arccos":
        out = sop_ufunc(eval(args[0]), grids, np.arccos, "acos")
    elif oper == "arcsin":
        out = sop_ufunc(eval(args[0]), grids, np.arcsin, "asin")
    elif oper == "arctan":
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
        nmode_order: int | None = None,
        nmode_ref_indices: list | tuple | None = None,
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

        # optional exact truncated N-mode projection of each G-matrix element
        self.nmode_order = nmode_order

        if self.nmode_order is not None:

            if nmode_ref_indices is None:
                raise ValueError(
                    f"'nmode_ref_indices' must be specified for N-mode truncation"
                )

            if len(nmode_ref_indices) != self.ncoo:
                raise ValueError(
                    f"'nmode_ref_indices' must have length 'len(grids)' = {self.ncoo}, got {len(nmode_ref_indices)}"
                )

            if self.nmode_order < 0:
                raise ValueError(
                    f"'nmode_order' = {nmode_order} must be non-negative or None"
                )

            self.nmode_order = min(self.nmode_order, self.ncoo)
            self.nmode_ref_indices = tuple(int(i) for i in nmode_ref_indices)

            for k, (i, g) in enumerate(zip(self.nmode_ref_indices, self.grids)):
                if i < 0 or i >= len(g):
                    raise IndexError(
                        f"'nmode_ref_indices[{k}]={i}' is outside the grid range [0, {len(g)})"
                    )
        else:
            self.nmode_ref_indices = None

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
        # clear geometry primitive cache
        _geom_cache.clear()

        xyz = np.array(
            [
                [init_cartesian(f"{alpha}{i}") for alpha in ("x", "y", "z")]
                for i in range(self.natoms)
            ]
        )

        q_expr = np.asarray(self.cartesian_to_internal(xyz), dtype=object).ravel()

        if len(q_expr) != self.ncoo:
            raise ValueError(
                f"'cartesian_to_internal' output size = {len(q_expr)} must have length 'len(grids)' = {self.ncoo}, "
                f"got {len(q_expr)}"
            )

        for icoo, q in enumerate(q_expr):
            if not isinstance(q, Expr):
                raise TypeError(f"'cartesian_to_internal' output {icoo} is not an Expr")

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

        q_bindings = {}
        for icoo, qnode in enumerate(q_expr):
            q_sop = SumProd.init_coordinate(icoo, self.grids)
            q_bindings[qnode.id] = q_sop
            q_bindings[("sig", expr_signature(qnode))] = q_sop

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

                        if self.nmode_order is None:
                            contrib = svec[lam, i, ialpha] * svec[mu, i, ialpha] / mass
                        else:
                            # exact truncated N-mode projection of the product
                            contrib = (
                                svec[lam, i, ialpha].nmode_mul(
                                    svec[mu, i, ialpha],
                                    self.nmode_order,
                                    self.nmode_ref_indices,
                                    zero_tol=self.compress_zero_tol,
                                )
                                / mass
                            )

                        g = g + contrib

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

                # don't know if this is necessary, as we already do svec[].nmode_mul()
                # if self.nmode_order is not None:
                #     g = g.nmode_project(
                #         self.nmode_order,
                #         self.nmode_ref_indices,
                #         zero_tol=self.compress_zero_tol,
                #     )

                gvib[lam, mu] = gvib[mu, lam] = g

        return gvib

    def run(self):
        xyz = self.sumprod_x()
        svec = self.sumprod_dq_dx(xyz)
        gvib = self.build_gvib(svec)
        return gvib
