from itertools import combinations

import numpy as np

from keosop.sing.revad import (
    Expr,
    _geom_cache,
    as_expr,
    expr_signature,
    init_cartesian,
    reverse_derivatives,
)
from keosop.sing.sumprod import (
    SumProd,
    configure_singularity_tracing,
    init_internal,
    sop_reciprocal,
    sop_ufunc,
)


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


class Gmat:
    def __init__(
        self,
        grids,
        masses,
        internal_to_cartesian,
        cartesian_to_internal,
        frame_conditions=None,
        compress_tol: float = 1e-10,
        compress_zero_tol: float = 1e-14,
        compress_identity_tol: float = 1e-12,
        compress_nterms_thresh: int = 10,
        nmode_order: int | None = None,
        nmode_ref_indices: list | tuple | None = None,
        trace_singularities: bool = False,
        singularity_coo: list | tuple | None = None,
        singularity_kinds: tuple = ("q", "sin", "cos"),
    ):
        self.grids = [np.asarray(elem, dtype=float) for elem in grids]
        self.masses = np.asarray(masses, dtype=float)
        self.internal_to_cartesian = internal_to_cartesian
        self.cartesian_to_internal = cartesian_to_internal
        self.frame_conditions = frame_conditions
        self.inferred_frame_conditions = None
        self.ncoo = len(grids)
        self.natoms = len(masses)

        self.compress_tol = compress_tol
        self.compress_zero_tol = compress_zero_tol
        self.compress_identity_tol = compress_identity_tol
        self.compress_nterms_thresh = compress_nterms_thresh

        self.trace_singularities = bool(trace_singularities)
        self.singularity_coo = (
            None if singularity_coo is None else tuple(int(k) for k in singularity_coo)
        )
        self.singularity_kinds = tuple(singularity_kinds)
        if self.singularity_coo is not None:
            bad = [k for k in self.singularity_coo if k < 0 or k >= self.ncoo]
            if bad:
                raise IndexError(f"singularity coordinate indices out of range: {bad}")
        configure_singularity_tracing(
            self.trace_singularities,
            self.singularity_coo,
            self.singularity_kinds,
        )

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

    def build_vib_svec(self, xyz_sop):
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

    def prod_svec(self, svec1, svec2):
        n1, n2 = svec1.shape[0], svec2.shape[0]
        gmat = np.empty((n1, n2), dtype=object)

        for lam in range(n1):
            for mu in range(n2):
                g = SumProd.constant(0.0, self.grids)

                for i, mass in enumerate(self.masses):
                    for ialpha in range(3):
                        g = g + svec1[lam, i, ialpha] * svec2[mu, i, ialpha] / mass

                        if g.nterms > self.compress_nterms_thresh:
                            g = g.compress(
                                nterms_thresh=1,
                                tol=self.compress_tol,
                                zero_tol=self.compress_zero_tol,
                                identity_tol=self.compress_identity_tol,
                            )

                g = g.compress(
                    nterms_thresh=1,
                    tol=self.compress_tol,
                    zero_tol=self.compress_zero_tol,
                    identity_tol=self.compress_identity_tol,
                )

                # finalise protected coordinate powers (singular factors)
                # before N-mode projection
                if self.trace_singularities:
                    g = g.normalize_protected_factors(
                        tol=max(self.compress_identity_tol, 1e-10),
                        coordinates=self.singularity_coo,
                        kinds=self.singularity_kinds,
                    ).compress_hashed(
                        zero_tol=self.compress_zero_tol,
                        identity_tol=self.compress_identity_tol,
                    )

                    # terms with the same protected analytical prefactor may still cancel
                    # even when their 1D arrays differ only by an overall scalar or sign
                    g = g.collect_final_terms(
                        zero_tol=self.compress_zero_tol,
                        identity_tol=max(self.compress_identity_tol, 1e-12),
                    )

                if self.nmode_order is not None:
                    g = g.nmode_project(
                        self.nmode_order,
                        self.nmode_ref_indices,
                        zero_tol=self.compress_zero_tol,
                        identity_tol=self.compress_identity_tol,
                    )
                    g = g.compress(
                        nterms_thresh=1,
                        tol=self.compress_tol,
                        zero_tol=self.compress_zero_tol,
                        identity_tol=self.compress_identity_tol,
                    )

                gmat[lam, mu] = g

        return gmat

    def build_gvib(self, svec):
        gvib = self.prod_svec(svec, svec)

        for lam in range(self.ncoo):
            for mu in range(lam + 1, self.ncoo):
                gvib[mu, lam] = gvib[lam, mu]

        return gvib

    def trace_frame_conditions(self):
        """Traces three user-defined rotational frame conditions C_alpha(r)"""
        if self.frame_conditions is None or self.frame_conditions == "auto":
            raise ValueError("No frame_conditions callable provided")

        if not callable(self.frame_conditions):
            raise TypeError("'frame_conditions' must be a callable, None, or 'auto'")

        _geom_cache.clear()
        xyz = np.array(
            [
                [init_cartesian(f"{alpha}{i}") for alpha in ("x", "y", "z")]
                for i in range(self.natoms)
            ],
            dtype=object,
        )

        roots = np.asarray(self.frame_conditions(xyz), dtype=object).ravel()

        if len(roots) != 3:
            raise ValueError(
                f"'frame_conditions' must return exactly 3 scalars, got {len(roots)}"
            )

        roots = np.array([as_expr(root) for root in roots], dtype=object)

        return roots

    def sop_constant_value(self, sop, tol=None):
        """Returns scalar value if sum-of-products is constant on every grid, else None"""
        if not isinstance(sop, SumProd):
            return float(sop) if np.isscalar(sop) else None

        if tol is None:
            tol = self.compress_zero_tol

        value = 0.0

        for term in sop.terms:
            coeff = 1.0

            for factor in term:
                if not SumProd.factor_is_constant(factor, tol=tol):
                    return None

                coeff *= float(np.asarray(factor)[0])

            value += coeff

        return value

    def sop_is_zero(self, sop, tol=None):
        """Returns True when sum-of-products expression compresses to zero"""
        if tol is None:
            tol = self.compress_zero_tol

        z = sop.compress_hashed(
            zero_tol=tol,
            identity_tol=self.compress_identity_tol,
        )
        z = z.compress_similar(
            tol=self.compress_tol,
            zero_tol=tol,
            identity_tol=self.compress_identity_tol,
        )

        value = self.sop_constant_value(z, tol=tol)

        return value is not None and abs(value) <= tol

    def sop_point_value(self, sop, indices):
        """Evaluates sum-of-products at a direct-product grid point without building one"""
        value = 0.0

        for term in sop.terms:
            prod = 1.0

            for k, factor in enumerate(term):
                prod *= float(np.asarray(factor)[indices[k]])

            value += prod

        return value

    def det_3x3_mat(self, jmat):
        a, b, c = jmat[0]
        d, e, f = jmat[1]
        g, h, i = jmat[2]

        det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)
        det = det.compress(
            nterms_thresh=1,
            tol=self.compress_tol,
            zero_tol=self.compress_zero_tol,
            identity_tol=self.compress_identity_tol,
        )

        return det

    def cvec_elem(self, C_alpha, xyz_sop):
        """Computes derivatives c_{i,beta} = dC(r)/dr_{i,beta} for one inferred fixed, linear, or bilinear
        molecular-frame rotational constraint C(r)
        """
        zero = SumProd.constant(0.0, self.grids)
        one = SumProd.constant(1.0, self.grids)

        c = np.empty((self.natoms, 3), dtype=object)
        for i in range(self.natoms):
            for a in range(3):
                c[i, a] = zero

        kind = C_alpha["kind"]
        if kind == "fixed":
            i, a = C_alpha["component"]
            c[i, a] = one

        elif kind == "linear":
            (i, a), (j, b) = C_alpha["components"]
            sign = C_alpha["sign"]
            c[i, a] = c[i, a] + one
            c[j, b] = c[j, b] + sign * one

        elif kind == "bilinear":
            # C = r_a r_b + sign r_c r_d
            (ia, aa), (ib, ab), (ic, ac), (id_, ad) = C_alpha["components"]
            sign = C_alpha["sign"]
            c[ia, aa] = c[ia, aa] + xyz_sop[ib, ab]
            c[ib, ab] = c[ib, ab] + xyz_sop[ia, aa]
            c[ic, ac] = c[ic, ac] + sign * xyz_sop[id_, ad]
            c[id_, ad] = c[id_, ad] + sign * xyz_sop[ic, ac]

        elif kind == "polynomial":
            # C = sum_p coeff_p M_p(r), where M_p is 1, r_A, or r_A*r_B
            for coeff, mono in zip(C_alpha["coeffs"], C_alpha["monomials"]):
                if abs(coeff) <= C_alpha.get("coef_tol", 0.0):
                    continue
                degree = len(mono)
                if degree == 0:
                    continue
                if degree == 1:
                    i, a = mono[0]
                    c[i, a] = c[i, a] + coeff * one
                elif degree == 2:
                    (i, a), (j, b) = mono
                    c[i, a] = c[i, a] + coeff * xyz_sop[j, b]
                    c[j, b] = c[j, b] + coeff * xyz_sop[i, a]
                else:
                    raise ValueError("Polynomial inference supports degree <= 2")

        else:
            raise ValueError(f"Unknown kind of candidate constraint: {kind}")

        return c

    def quadratic_relation_candidates(self, xyz_sop, max_candidates=100, zero_tol=1e-8):
        """Discover general degree<=2 Cartesian identities from the exact SOP Gram matrix

        1. Form the monomial SOP basis M = {1, r_A, r_A r_B}, where A = {x_1, y_1, z_1, x_2, ... , z_N}
        2. Form the Gramm matrix <M_p, M_q> over the complete direct-product grid, evaluate it by factorized
        one-dimensional SOP contractions (never sample multidimensional grid)
        3. Diagonalize the column-normalized Gram matrix and use its null space to propose candidates
        for polynomial identities
        4. Validate proposed candidates with the exact SOP algebra
        """
        comps = [((i, a), xyz_sop[i, a]) for i in range(self.natoms) for a in range(3)]

        # keep nonzero Cartesian components
        active = [(ia, sop) for ia, sop in comps if not self.sop_is_zero(sop)]
        one = SumProd.constant(1.0, self.grids)

        monomials = [()]
        funcs = [one]
        labels = ["1"]

        for ia, a in active:
            monomials.append((ia,))
            funcs.append(a)
            labels.append(f"{'xyz'[ia[1]]}{ia[0]}")

        for u, (ia, a) in enumerate(active):
            for ib, b in active[u:]:
                monomials.append((ia, ib))
                funcs.append(a * b)
                labels.append(f"{'xyz'[ia[1]]}{ia[0]}*{'xyz'[ib[1]]}{ib[0]}")

        nmono = len(funcs)
        if nmono <= 1:
            return []

        def sop_inner(a, b):
            # inner product <a, b> using SOP factorization
            value = 0.0
            for ta in a.terms:
                for tb in b.terms:
                    prod = 1.0
                    for fa, fb in zip(ta, tb):
                        prod *= float(np.dot(fa, fb))
                    value += prod
            return float(value)

        # diagonal norms
        norm2 = np.empty(nmono, dtype=float)
        for p in range(nmono):
            norm2[p] = sop_inner(funcs[p], funcs[p])

        # funcs should be nonzero
        norm2[np.abs(norm2) < 1e-30] = 0.0
        if np.any(norm2 < 0.0):
            raise RuntimeError(
                "Negative Gram-matrix diagonal encountered during quadratic frame inference"
            )
        scale = np.sqrt(norm2)
        scale[scale == 0.0] = 1.0

        # normalized Gram matrix Ks[p,q] = <M_p/scale_p, M_q/scale_q>
        Ks = np.empty((nmono, nmono), dtype=float)
        for p in range(nmono):
            Ks[p, p] = norm2[p] / (scale[p] * scale[p])
            for q in range(p + 1, nmono):
                val = sop_inner(funcs[p], funcs[q]) / (scale[p] * scale[q])
                Ks[p, q] = val
                Ks[q, p] = val

        # diagonalize
        Ks = 0.5 * (Ks + Ks.T)
        evals, evecs = np.linalg.eigh(Ks)
        emax = float(np.max(evals)) if len(evals) else 1.0

        # null vector tolerance
        rank_tol = max(1e-11 * max(emax, 1.0), 1e-13)
        null_idx = np.flatnonzero(evals <= rank_tol)

        out = []
        for idx in null_idx:
            # evecs[:,idx] is a null vector in the normalized monomial basis
            # convert it back to coefficients multiplying the original funcs
            coeff = evecs[:, idx] / scale
            m = np.max(np.abs(coeff))
            if not np.isfinite(m) or m == 0.0:
                continue

            coeff = coeff / m
            coeff[np.abs(coeff) < zero_tol] = 0.0
            nz = np.flatnonzero(coeff)
            if len(nz) < 2:
                continue

            # null vectors are only proposals, check if reconstructed
            # relation vanishes in the SOP algebra
            expr = SumProd.constant(0.0, self.grids)
            for j in nz:
                expr = expr + float(coeff[j]) * funcs[j]

            if not self.sop_is_zero(expr, tol=max(self.compress_zero_tol, 1e-9)):
                continue

            terms = [f"{coeff[j]:+.6g}*{labels[j]}" for j in nz]
            out.append(
                {
                    "kind": "polynomial",
                    "coeffs": tuple(float(x) for x in coeff),
                    "monomials": tuple(monomials),
                    "coef_tol": 1e-12,
                    "label": " ".join(terms) + " = 0",
                }
            )

            if len(out) >= max_candidates:
                break

        return out

    def infer_frame_constraints(self, xyz_sop, tvec=None):
        """Searches for three molecular-frame rotational constraints C_{alpha}(r)
        using inernal-to-Cartesian transformation r=F_X(q) and SOP identities

        Returns c_{alpha,i,beta} = dC_{alpha}(r)/dr_{i,beta} in SOP form

        The search collects relations in increasing complexity order:
        1. Fixed Cartesian components
        2. Simple linear identities r_a +/- r_b = 0
        3. Simple bilinear identities r_a*r_b +/- r_c*r_d = 0
        4. Arbitrary degree<=2 polynomial identities found from SOP Gramm matrix

        The three conditions are accepted when the rotational Jacobian J=dC/dr*tvec is nonsingular
        """
        if tvec is None:
            tvec = self.build_tvec(xyz_sop)

        comps = [((i, a), xyz_sop[i, a]) for i in range(self.natoms) for a in range(3)]
        C_cand = []

        # fixed components C = r_ia - const
        for ia, sop in comps:
            value = self.sop_constant_value(sop)

            if value is not None:
                C_cand.append(
                    {
                        "kind": "fixed",
                        "component": ia,
                        "value": value,
                        "label": f"{'xyz'[ia[1]]}{ia[0]} = {value:g}",
                    }
                )

        # simple linear identities
        for n1 in range(len(comps)):
            ia, a = comps[n1]

            for n2 in range(n1 + 1, len(comps)):
                ib, b = comps[n2]

                for sign in (-1.0, 1.0):
                    if self.sop_is_zero(a + sign * b):
                        op = "-" if sign < 0 else "+"

                        C_cand.append(
                            {
                                "kind": "linear",
                                "components": (ia, ib),
                                "sign": sign,
                                "label": f"{'xyz'[ia[1]]}{ia[0]} {op} {'xyz'[ib[1]]}{ib[0]} = 0",
                            }
                        )

        # test triples of candidate constraints
        def test_candidates(pool):
            for chosen in combinations(pool, 3):
                cvec = np.empty((3, self.natoms, 3), dtype=object)

                for row, cand in enumerate(chosen):
                    cvec[row] = self.cvec_elem(cand, xyz_sop)

                J = self.build_jmat(cvec, tvec)
                det = self.det_3x3_mat(J)

                # probe det(J) directly at grid points and accept a candidate
                # as soon as it is nonsingular at one of them;
                # exact SOP simplification or inversion is delayed
                # until after a frame has been selected
                probes = [
                    tuple(n // 2 for n in det.npoints),
                    tuple(0 for _ in det.npoints),
                    tuple(n - 1 for n in det.npoints),
                ]
                det_scale = 0.0
                nonsingular = False
                for p in probes:
                    try:
                        v = float(det.evaluate(p, include_singular=True))
                    except (
                        FloatingPointError,
                        ZeroDivisionError,
                        ValueError,
                        OverflowError,
                    ):
                        continue

                    if np.isfinite(v):
                        det_scale = max(det_scale, abs(v))
                        if abs(v) > self.compress_zero_tol:
                            nonsingular = True
                            break

                if not nonsingular:
                    continue

                self.inferred_frame_conditions = tuple(dict(c) for c in chosen)

                return cvec

            return None

        cvec = test_candidates(C_cand)
        if cvec is not None:
            return cvec

        # bilinear identities
        products = []
        for n1 in range(len(comps)):
            ia, a = comps[n1]

            if self.sop_is_zero(a):
                continue

            for n2 in range(n1, len(comps)):
                ib, b = comps[n2]

                if self.sop_is_zero(b):
                    continue

                products.append((ia, ib, a * b))

        seen = set()
        for p in range(len(products)):
            ia, ib, ab = products[p]

            for q in range(p + 1, len(products)):
                ic, id_, cd = products[q]

                if {ia, ib} == {ic, id_}:
                    continue

                for sign in (-1.0, 1.0):
                    if self.sop_is_zero(ab + sign * cd):
                        key = (tuple(sorted((ia, ib))), tuple(sorted((ic, id_))), sign)

                        if key in seen:
                            continue

                        seen.add(key)
                        op = "-" if sign < 0 else "+"

                        C_cand.append(
                            {
                                "kind": "bilinear",
                                "components": (ia, ib, ic, id_),
                                "sign": sign,
                                "label": (
                                    f"{'xyz'[ia[1]]}{ia[0]}*{'xyz'[ib[1]]}{ib[0]} "
                                    f"{op} {'xyz'[ic[1]]}{ic[0]}*{'xyz'[id_[1]]}{id_[0]} = 0"
                                ),
                            }
                        )

        cvec = test_candidates(C_cand)
        if cvec is not None:
            return cvec

        # general quadratic identities
        C_cand += self.quadratic_relation_candidates(xyz_sop)
        cvec = test_candidates(C_cand)
        if cvec is not None:
            return cvec

        raise RuntimeError(
            "Automatic search could not find three rotationally independent Cartesian constraints. "
            f"Provide 'frame_conditions' callable explicitly for a more general molecular-frame convention."
        )

    def build_cvec(self, xyz_sop, tvec=None):
        """Computes c_{alpha,i,beta} = dC_alpha/d r_{i,beta}"""
        if self.frame_conditions is None or self.frame_conditions == "auto":
            return self.infer_frame_constraints(xyz_sop, tvec=tvec)

        self.inferred_frame_conditions = None
        roots = self.trace_frame_conditions()

        xyz_bindings = {
            f"{alpha}{i}": xyz_sop[i, ialpha]
            for i in range(self.natoms)
            for ialpha, alpha in enumerate(("x", "y", "z"))
        }

        cvec = np.empty((3, self.natoms, 3), dtype=object)

        for a, root in enumerate(roots):
            deriv = reverse_derivatives(root)
            cache = {}

            for i in range(self.natoms):
                for ialpha, alpha in enumerate(("x", "y", "z")):

                    cvec[a, i, ialpha] = eval_expr(
                        deriv.get(f"{alpha}{i}", as_expr(0)),
                        xyz_bindings,
                        self.grids,
                        {},
                        cache,
                    )

        return cvec

    def com_xyz(self, xyz_sop):
        """Return Cartesian SOP coordinates relative to the center of mass"""
        mtot = float(np.sum(self.masses))

        com = np.empty(3, dtype=object)

        for alpha in range(3):
            val = SumProd.constant(0.0, self.grids)

            for i, mass in enumerate(self.masses):
                val = val + float(mass) * xyz_sop[i, alpha]

            com[alpha] = (val / mtot).compress(
                nterms_thresh=1,
                tol=self.compress_tol,
                zero_tol=self.compress_zero_tol,
                identity_tol=self.compress_identity_tol,
            )

        xyz_com = np.empty((self.natoms, 3), dtype=object)

        for i in range(self.natoms):
            for alpha in range(3):

                xyz_com[i, alpha] = (xyz_sop[i, alpha] - com[alpha]).compress(
                    nterms_thresh=1,
                    tol=self.compress_tol,
                    zero_tol=self.compress_zero_tol,
                    identity_tol=self.compress_identity_tol,
                )

        return xyz_com, com

    def cvec_translation_free(self, cvec):
        """Project frame gradients onto the translation-free Cartesian subspace
        c_{alpha,i,beta} = c_{alpha,i,beta} - m_i/M * sum_j c_{alpha,j,beta}
        """
        mtot = float(np.sum(self.masses))

        out = np.empty_like(cvec, dtype=object)

        for lam in range(3):
            for alpha in range(3):
                csum = SumProd.constant(0.0, self.grids)

                for j in range(self.natoms):
                    csum = csum + cvec[lam, j, alpha]

                for i, mass in enumerate(self.masses):
                    out[lam, i, alpha] = (
                        cvec[lam, i, alpha] - (float(mass) / mtot) * csum
                    ).compress(
                        nterms_thresh=1,
                        tol=self.compress_tol,
                        zero_tol=self.compress_zero_tol,
                        identity_tol=self.compress_identity_tol,
                    )

        return out

    def build_tvec(self, xyz_sop):
        """Computes tvec_{i,alpha,beta} = sum_{gamma} eps_{alpha,beta,gamma} r_{i,gamma}"""
        zero = SumProd.constant(0.0, self.grids)
        tvec = np.empty((self.natoms, 3, 3), dtype=object)

        for i in range(self.natoms):
            x, y, z = xyz_sop[i]
            vals = ((zero, z, -y), (-z, zero, x), (y, -x, zero))

            for alpha in range(3):
                for beta in range(3):
                    tvec[i, alpha, beta] = vals[alpha][beta]

        return tvec

    def build_jmat(self, cvec, tvec):
        """Computes J_{alpha,beta} = sum_{i,gamma} c_{alpha,i,gamma} t[{i,gamma,beta}"""
        jmat = np.empty((3, 3), dtype=object)

        for a in range(3):
            for b in range(3):
                val = SumProd.constant(0.0, self.grids)

                for i in range(self.natoms):
                    for gamma in range(3):
                        val = val + cvec[a, i, gamma] * tvec[i, gamma, b]

                jmat[a, b] = val.compress(
                    nterms_thresh=1,
                    tol=self.compress_tol,
                    zero_tol=self.compress_zero_tol,
                    identity_tol=self.compress_identity_tol,
                )

        return jmat

    def inverse_3x3_mat(self, A):
        """Exact 3x3 inverse in SOP algebra; raise error if det(A)^-1 is not exact SOP"""
        if np.shape(A) != (3, 3):
            raise ValueError("inverse_3x3_mat expects a 3x3 matrix")

        a, b, c = A[0]
        d, e, f = A[1]
        g, h, i = A[2]

        cof = np.empty((3, 3), dtype=object)
        cof[0, 0] = e * i - f * h
        cof[0, 1] = -(d * i - f * g)
        cof[0, 2] = d * h - e * g
        cof[1, 0] = -(b * i - c * h)
        cof[1, 1] = a * i - c * g
        cof[1, 2] = -(a * h - b * g)
        cof[2, 0] = b * f - c * e
        cof[2, 1] = -(a * f - c * d)
        cof[2, 2] = a * e - b * d

        det = (a * cof[0, 0] + b * cof[0, 1] + c * cof[0, 2]).compress(
            nterms_thresh=1,
            tol=self.compress_tol,
            zero_tol=self.compress_zero_tol,
            identity_tol=self.compress_identity_tol,
        )

        # det(J) is a special motherfucker, try to simplify it here
        # rather than carying it through every SOP multiplication

        # first cancel any exact q/sin/cos powers against
        # protected factors, then perform exact compression
        if self.trace_singularities:
            det = det.normalize_protected_factors(
                tol=max(self.compress_identity_tol, 1e-12),
                coordinates=self.singularity_coo,
                kinds=self.singularity_kinds,
            )

        det = det.compress_hashed(
            zero_tol=self.compress_zero_tol,
            identity_tol=self.compress_identity_tol,
        )

        # may be opposite proportional terms can cancel
        # before the one-coordinate collection below
        if det.nterms > 1:
            det = det.cancel_proportional_terms(zero_tol=self.compress_zero_tol)

        if det.nterms > 1:
            det = det.collect_one_coordinate(
                zero_tol=self.compress_zero_tol, identity_tol=self.compress_identity_tol
            )

            det = det.compress_hashed(
                zero_tol=self.compress_zero_tol, identity_tol=self.compress_identity_tol
            )

        # before giving up on an exact SOP reciprocal, test if det(J) can be refactored
        # to a single separable product
        if det.nterms > 1:
            det = det.refactor_rank1(
                tol=max(self.compress_tol, self.compress_identity_tol, 1e-11)
            )

            det = det.compress_hashed(
                zero_tol=self.compress_zero_tol, identity_tol=self.compress_identity_tol
            )

        try:
            invdet = sop_reciprocal(det)
        except (NotImplementedError, ZeroDivisionError) as exc:
            raise RuntimeError(
                "Rotational frame Jacobian cannot be inverted exactly in the current SOP algebra. "
                "det(J) is either singular on the grid or its reciprocal is not representable as an "
                f"exact supported SOP (det(J).nterms = {det.nterms})."
            ) from exc

        ainv = np.empty((3, 3), dtype=object)

        for row in range(3):
            for col in range(3):
                ainv[row, col] = (cof[col, row] * invdet).compress(
                    nterms_thresh=1,
                    tol=self.compress_tol,
                    zero_tol=self.compress_zero_tol,
                    identity_tol=self.compress_identity_tol,
                )

        return ainv, det

    def build_rot_svec(self, Jinv, cvec):
        """srot_{alpha,i,gamma} = sum_{beta} Jinv_{alpha,beta} c{beta,i,gamma}"""
        srot = np.empty((3, self.natoms, 3), dtype=object)

        for a in range(3):
            for i in range(self.natoms):
                for gamma in range(3):

                    val = SumProd.constant(0.0, self.grids)

                    for b in range(3):
                        val = val + Jinv[a, b] * cvec[b, i, gamma]

                    srot[a, i, gamma] = val.compress(
                        nterms_thresh=1,
                        tol=self.compress_tol,
                        zero_tol=self.compress_zero_tol,
                        identity_tol=self.compress_identity_tol,
                    )

        return srot

    def build_full_g(self, svib, srot):
        """Return vibrational, rotation-vibration, and rotational G blocks"""
        gvv = self.build_gvib(svib)  # (ncoo, ncoo)
        grv = self.prod_svec(srot, svib)  # (3, ncoo)
        grr = self.prod_svec(srot, srot)  # (3, 3)

        for a in range(3):
            for b in range(a + 1, 3):
                grr[b, a] = grr[a, b]

        return gvv, grv, grr

    def run(self, return_rot=False, return_svec=False):
        """Build SOP G matrices"""
        xyz = self.sumprod_x()
        svib = self.build_vib_svec(xyz)

        if not return_rot:
            gvib = self.build_gvib(svib)

            if return_svec:
                return gvib, {"svib": svib, "xyz": xyz}

            return gvib

        # Keep the user's original Cartesian SOP representation
        # for the vibrational branch above

        # -------------------------------------------------------------
        # What is implemented below is described in detail in
        #   Sørensen, G.O. (1979). A new approach to the Hamiltonian of nonrigid molecules.
        #   In: Large Amplitude Motion in Molecules II. Topics in Current Chemistry, vol 82.
        #   https://doi.org/10.1007/BFb0048009
        #   The relevant Section 2.2.3.2 "Axis conventions and rotational s-vectors"
        # -------------------------------------------------------------

        # Rotational-frame branch need to be expressed in translation-reduced
        # (COM-centered) coordinates, so we construct rho_i = r_i - R_COM only
        # for frame inference/differentiation and rotational tangents
        xyz_com, com = self.com_xyz(xyz)
        tvec = self.build_tvec(xyz_com)

        # build_cvec(xyz_com) evaluates dC/drho at the COM-centered geometry,
        # and to obtain derivatives with respect to the original Cartesian
        # coordinates r, apply the COM chain rule in cvec_translation_free(cvec)
        cvec_raw = self.build_cvec(xyz_com, tvec=tvec)
        cvec = self.cvec_translation_free(cvec_raw)

        # cvec is now translation-reduced and tvec is a pure rotation about the COM,
        # so J will be built in the translation-reduced frame
        J = self.build_jmat(cvec, tvec)
        Jinv, detJ = self.inverse_3x3_mat(J)
        srot = self.build_rot_svec(Jinv, cvec)
        gvv, grv, grr = self.build_full_g(svib, srot)

        if return_svec:
            return (
                gvv,
                grv,
                grr,
                {
                    "xyz": xyz,
                    "xyz_com": xyz_com,
                    "com": com,
                    "svib": svib,
                    "srot": srot,
                    "cvec": cvec,
                    "cvec_raw": cvec_raw,
                    "tvec": tvec,
                    "J": J,
                    "Jinv": Jinv,
                    "detJ": detJ,
                    "frame_conditions": self.inferred_frame_conditions,
                },
            )

        return gvv, grv, grr
