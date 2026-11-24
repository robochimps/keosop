import operator
import weakref
from collections import OrderedDict
from itertools import combinations, combinations_with_replacement
from time import perf_counter

import numpy as np

from keosop.revad import (
    Expr,
    _geom_cache,
    as_expr,
    dot,
    expr_signature,
    init_cartesian,
    norm,
    reverse_derivatives,
)
from keosop.sumprod import (
    COMPRESS_NTERMS_THRESH,
    SumProd,
    init_internal,
    sop_reciprocal,
    sop_ufunc,
)


def geom_expanded_signatures(node):
    """Return structural signatures corresponding to an atomic geometry node.

    Used as aliases when user-defined Cartesian-to-internal coordinates
    are written using norm, dot, or arccos functions rather than the atomic
    distance() or angle() primitives.
    """
    args = node.args

    if node.oper == "distance":
        A = np.array(args[0:3], dtype=object)
        B = np.array(args[3:6], dtype=object)
        # distance() primitive uses defined endpoints, but user may
        # write either norm(A-B) or norm(B-A), so try both
        return {expr_signature(norm(A - B)), expr_signature(norm(B - A))}

    if node.oper == "angle":
        A = np.array(args[0:3], dtype=object)
        B = np.array(args[3:6], dtype=object)
        C = np.array(args[6:9], dtype=object)

        def expanded(P, Q, R):
            u = P - Q
            v = R - Q
            return np.arccos(dot(u, v) / (norm(u) * norm(v)))

        # angle() primitive uses defined ordering of the two outer atoms,
        # so try both representations to match user's convention
        return {expr_signature(expanded(A, B, C)), expr_signature(expanded(C, B, A))}

    return set()


def add_user_geometry_aliases(q_expr, q_bindings):
    """Bind atomic distance or angle nodes to equivalent user-defined nodes.

    This is on purpose made local to vibrational derivative evaluation.
    It does not affect reverse AD, the geometry cache, or the rotational-frame path.
    """
    sig_to_sop = {expr_signature(qnode): q_bindings[qnode.id] for qnode in q_expr}

    out = dict(q_bindings)

    for node in tuple(_geom_cache.values()):
        if node.oper not in ("distance", "angle"):
            continue

        # if already an explicit internal-coordinate primitive
        if node.id in out or ("sig", expr_signature(node)) in out:
            continue

        for sig in geom_expanded_signatures(node):
            q_sop = sig_to_sop.get(sig)

            if q_sop is not None:
                out[node.id] = q_sop
                out[("sig", expr_signature(node))] = q_sop
                break

    return out


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


class _PseudopotentialFactor:
    __slots__ = (
        "id",
        "values",
        "finite",
        "has_nan",
        "constant",
        "zeros",
        "__weakref__",
    )

    def __init__(self, identifier, values):
        self.id = identifier
        self.values = values
        self.finite = bool(np.all(np.isfinite(values)))
        self.has_nan = bool(np.any(np.isnan(values)))
        self.constant = bool(np.all(values == values[0]))
        self.zeros = {0.0: bool(np.all(values == 0.0))}


class _PseudopotentialContext:
    def __init__(self, grids, fd_info, max_comparisons):
        self.grids, self.fd_info = grids, fd_info
        self.ncoo = len(grids)
        self.max_comparisons = max_comparisons
        self.factors = weakref.WeakValueDictionary()
        self.products = weakref.WeakValueDictionary()
        self.scales = weakref.WeakValueDictionary()
        self.derivatives = weakref.WeakValueDictionary()
        self.simplifications = weakref.WeakValueDictionary()
        self.comparisons = OrderedDict()
        self.comparison_tables = OrderedDict()
        self.comparison_table_bytes = 0
        self.next_id = 0
        self.ones = tuple(self.intern(np.ones(len(g))) for g in grids)
        self.stats = dict(
            factors_created=self.next_id,
            factor_reuses=0,
            factor_comparison_requests=0,
            factor_comparison_hits=0,
            factor_array_comparisons=0,
            factor_comparison_evictions=0,
            peak_comparison_cache_entries=0,
            peak_live_factors=len(self.factors),
            product_cache_hits=0,
            derivative_cache_hits=0,
            comparison_table_hits=0,
            comparison_table_builds=0,
            comparison_table_evictions=0,
            peak_comparison_table_bytes=0,
            peak_comparison_table_entries=0,
            hashed_compressions=0,
            similar_compressions=0,
        )

    def intern(self, value):
        array = np.asarray(value, dtype=float)

        # Use the same signed-zero normalization as SumProd.factor_key
        key = SumProd.factor_key(array)
        existing = self.factors.get(key)

        if existing is not None:
            if hasattr(self, "stats"):
                self.stats["factor_reuses"] += 1
            return existing

        owned = array.copy()
        owned.flags.writeable = False
        factor = _PseudopotentialFactor(self.next_id, owned)

        self.next_id += 1
        self.factors[key] = factor

        if hasattr(self, "stats"):
            self.stats["factors_created"] += 1
            self.stats["peak_live_factors"] = max(
                self.stats["peak_live_factors"], len(self.factors)
            )
        return factor

    def constant(self, value):
        factors = list(self.ones)
        factors[0] = self.scale(factors[0], float(value))
        return _PseudopotentialSOP(self, (tuple(factors),))

    def import_sop(self, sop):
        if not isinstance(sop, SumProd) or any(getattr(sop, "protected", ())):
            raise TypeError(
                "build_pseudopotential_fdf supports regular SumProd factors only"
            )
        return _PseudopotentialSOP(
            self, tuple(tuple(self.intern(f) for f in term) for term in sop.terms)
        )

    def zero_factor(self, factor, tol):
        if tol not in factor.zeros:
            factor.zeros[tol] = bool(SumProd.is_zero_factor(factor.values, tol))
        return factor.zeros[tol]

    def zero_sop(self, sop):
        return all(
            any(self.zero_factor(f, 0.0) for f in term) and all(f.finite for f in term)
            for term in sop.terms
        )

    def multiply(self, left, right):
        key = left.id, right.id
        result = self.products.get(key)
        if result is not None:
            self.stats["product_cache_hits"] += 1
            return result

        result = self.intern(left.values * right.values)
        self.products[key] = result
        return result

    def factor_table(self, factors, tol):
        key = tuple(f.id for f in factors), float(tol)

        result = self.comparison_tables.get(key)
        if result is not None:
            self.comparison_tables.move_to_end(key)
            self.stats["comparison_table_hits"] += 1
            return result

        self.stats["comparison_table_builds"] += 1

        result = np.empty((len(factors), len(factors)), dtype=bool)
        for a, left in enumerate(factors):
            result[a, a] = self.close(left, left, tol)
            for b in range(a + 1, len(factors)):
                result[a, b] = result[b, a] = self.close(left, factors[b], tol)

        result.flags.writeable = False

        if self.max_comparisons and result.nbytes <= 4 * 1024**2:
            self.comparison_tables[key] = result
            self.comparison_table_bytes += result.nbytes

            while (
                len(self.comparison_tables) > 128
                or self.comparison_table_bytes > 4 * 1024**2
            ):
                _, old = self.comparison_tables.popitem(last=False)
                self.comparison_table_bytes -= old.nbytes
                self.stats["comparison_table_evictions"] += 1

            self.stats["peak_comparison_table_bytes"] = max(
                self.stats["peak_comparison_table_bytes"], self.comparison_table_bytes
            )
            self.stats["peak_comparison_table_entries"] = max(
                self.stats["peak_comparison_table_entries"], len(self.comparison_tables)
            )
        return result

    def scale(self, factor, value):
        key = factor.id, np.float64(value).tobytes()

        result = self.scales.get(key)
        if result is not None:
            return result

        result = self.intern(factor.values * value)

        self.scales[key] = result
        return result

    def close(self, left, right, tol):
        self.stats["factor_comparison_requests"] += 1
        if left.id == right.id:
            self.stats["factor_comparison_hits"] += 1
            return left.finite if tol > 0.0 else not left.has_nan

        key = min(left.id, right.id), max(left.id, right.id), float(tol)

        result = self.comparisons.get(key)
        if result is not None:
            self.comparisons.move_to_end(key)
            self.stats["factor_comparison_hits"] += 1
            return result

        if left.values.shape != right.values.shape:
            result = False
        elif left.constant and right.constant and left.finite and right.finite:
            delta = abs(left.values[0] - right.values[0])
            result = bool(delta <= tol) if tol > 0.0 else bool(delta == 0.0)
        else:
            self.stats["factor_array_comparisons"] += 1
            result = bool(SumProd.factors_close(left.values, right.values, tol))

        if self.max_comparisons:
            self.comparisons[key] = result
            if len(self.comparisons) > self.max_comparisons:
                self.comparisons.popitem(last=False)
                self.stats["factor_comparison_evictions"] += 1
            self.stats["peak_comparison_cache_entries"] = max(
                self.stats["peak_comparison_cache_entries"], len(self.comparisons)
            )

        return result

    def simplify(self, factor, tol):
        key = factor.id, float(tol)

        result = self.simplifications.get(key)
        if result is not None:
            return result

        values, changed = SumProd.simplify_1d_factor(factor.values, tol)
        result = self.intern(values) if changed else factor

        self.simplifications[key] = result
        return result

    def fdf(self, sop, n=None, m=None):
        orders = [0] * self.ncoo

        if n is not None:
            orders[n] = 1

        if m is not None:
            orders[m] += 1

        terms = []
        for term in sop.terms:
            factors = []

            for k, (factor, order) in enumerate(zip(term, orders)):
                key = factor.id, k, order
                result = self.derivatives.get(key)

                if result is None:
                    ci = self.fd_info[k]
                    if order == 0:
                        values = factor.values[ci["orig_inds"]]
                    else:
                        values = np.array(
                            [
                                np.dot(w1 if order == 1 else w2, factor.values[inds])
                                for inds, w1, w2 in ci["points"]
                            ]
                        )

                    result = self.intern(values)
                    self.derivatives[key] = result

                else:
                    self.stats["derivative_cache_hits"] += 1

                factors.append(result)

            terms.append(tuple(factors))

        return _PseudopotentialSOP(self, tuple(terms))


class _PseudopotentialSOP:

    def __init__(self, context, terms):
        self.context, self.terms = context, tuple(terms)

    @property
    def nterms(self):
        return len(self.terms)

    def export(self):
        return SumProd([[f.values for f in term] for term in self.terms])

    def __add__(self, other):
        if np.isscalar(other):
            other = self.context.constant(other)

        if (
            not isinstance(other, _PseudopotentialSOP)
            or other.context is not self.context
        ):
            return NotImplemented

        return _PseudopotentialSOP(self.context, self.terms + other.terms).compress()

    __radd__ = __add__

    def __neg__(self):
        return self * -1.0

    def __sub__(self, other):
        return self + (-other)

    def __mul__(self, other):
        context = self.context

        if np.isscalar(other):
            return _PseudopotentialSOP(
                context,
                tuple(
                    (context.scale(term[0], float(other)),) + term[1:]
                    for term in self.terms
                ),
            )

        if not isinstance(other, _PseudopotentialSOP) or other.context is not context:
            return NotImplemented

        return _PseudopotentialSOP(
            context,
            tuple(
                tuple(context.multiply(a, b) for a, b in zip(left, right))
                for left in self.terms
                for right in other.terms
            ),
        ).compress()

    __rmul__ = __mul__

    def _nonzero_terms(self, zero_tol):
        c = self.context
        return [
            term
            for term in self.terms
            if not any(c.zero_factor(f, zero_tol) for f in term)
        ]

    def compress_hashed(self, zero_tol=0.0, max_sweeps=None, identity_tol=0.0):
        c = self.context
        c.stats["hashed_compressions"] += 1

        terms = self._nonzero_terms(zero_tol)
        sweep, changed = 0, True
        while terms and changed and (max_sweeps is None or sweep < max_sweeps):
            sweep += 1
            changed = False

            for k in range(c.ncoo):

                groups = {}
                for term in terms:
                    key = tuple(f.id for j, f in enumerate(term) if j != k)
                    groups.setdefault(key, []).append(term)

                new_terms = []
                for group in groups.values():
                    if len(group) == 1:
                        new_terms.append(group[0])
                        continue
                    merged = list(group[0])
                    merged[k] = c.simplify(
                        c.intern(np.add.reduce([term[k].values for term in group])),
                        identity_tol,
                    )

                    changed = True

                    if not c.zero_factor(merged[k], zero_tol):
                        new_terms.append(tuple(merged))

                terms = new_terms

                if not terms:
                    break

        return _PseudopotentialSOP(c, tuple(terms)) if terms else c.constant(0.0)

    def _column_matches(self, terms, k, tol):
        c = self.context
        unique = {}
        for term in terms:
            factor = term[k]
            unique.setdefault(factor.id, factor)
        factors = [unique[identifier] for identifier in sorted(unique)]
        lookup = {factor.id: index for index, factor in enumerate(factors)}
        indices = np.fromiter((lookup[t[k].id] for t in terms), dtype=int)
        close = c.factor_table(factors, tol)
        return close[indices[:, None], indices[None, :]]

    def _mismatches(self, terms, tol):
        counts = np.zeros((len(terms), len(terms)), dtype=np.int32)
        for k in range(self.context.ncoo):
            counts += ~self._column_matches(terms, k, tol)
        return counts

    def compress_similar(
        self, tol=1e-10, zero_tol=0.0, max_sweeps=None, identity_tol=0.0
    ):
        c = self.context
        c.stats["similar_compressions"] += 1

        terms = self._nonzero_terms(zero_tol)
        sweep, changed = 0, True
        mismatches = None

        while terms and changed and (max_sweeps is None or sweep < max_sweeps):
            sweep += 1
            changed = False

            for k in range(c.ncoo):
                if len(terms) <= 512:
                    if mismatches is None:
                        mismatches = self._mismatches(terms, tol)
                    column = self._column_matches(terms, k, tol)
                    match = mismatches == ~column
                else:
                    match = None
                used = np.zeros(len(terms), dtype=bool)

                new_terms = []
                representatives = []
                merged_in_phase = False

                for i, term in enumerate(terms):
                    if used[i]:
                        continue

                    used[i] = True
                    merged = list(term)
                    values = None

                    if match is not None:
                        candidates = np.flatnonzero(match[i] & ~used)
                    else:
                        candidates = (
                            j
                            for j in range(i + 1, len(terms))
                            if not used[j]
                            and all(
                                c.close(term[a], terms[j][a], tol)
                                for a in range(c.ncoo)
                                if a != k
                            )
                        )

                    for j in candidates:
                        if values is None:
                            values = term[k].values.copy()
                        values += terms[j][k].values
                        used[j] = True
                        changed = True
                        merged_in_phase = True

                    if values is not None:
                        merged[k] = c.simplify(c.intern(values), identity_tol)

                    if not c.zero_factor(merged[k], zero_tol):
                        new_terms.append(tuple(merged))
                        representatives.append(i)

                if mismatches is not None and new_terms and merged_in_phase:
                    selected = np.ix_(representatives, representatives)
                    mismatches = (
                        mismatches[selected]
                        + column[selected].astype(np.int32)
                        - self._column_matches(new_terms, k, tol).astype(np.int32)
                    )

                terms = new_terms

                if not terms:
                    break

        return _PseudopotentialSOP(c, tuple(terms)) if terms else c.constant(0.0)

    def compress(
        self,
        tol=0.0,
        zero_tol=0.0,
        max_sweeps=None,
        nterms_thresh=COMPRESS_NTERMS_THRESH,
        identity_tol=0.0,
    ):
        if self.nterms < nterms_thresh:
            return self

        result = self.compress_hashed(zero_tol, max_sweeps, identity_tol)

        if tol > 0.0:
            result = result.compress_similar(tol, zero_tol, max_sweeps, identity_tol)
        return result


class Gmat:
    """Build molecular kinetic-energy G-matrices in sum-of-products (SOP) form.

    Parameters
    ----------
    grids : sequence of one-dimensional array_like
        One grid per internal coordinate, in the order used by both coordinate
        transformations.
        Grids must be nonempty and strictly increasing when computing the
        finite-difference pseudopotential.
    masses : array_like, shape (N,)
        Positive atomic masses of N atoms, in the same atom order as the Cartesian arrays.
    internal_to_cartesian : callable
        Function `xyz = internal_to_cartesian(q)` accepting one sequence of M internal coordinates
        and returning an array_like of shape (N, 3), with Cartesian components ordered as x, y, z.
        During construction, each q[k] is a `keosop.sumprod.SumProd` object, rather than a numeric scalar.
        Returned components must be SumProd objects or numeric constants.
    cartesian_to_internal : callable
        Function `q = cartesian_to_internal(xyz)` accepting an object array of shape (N, 3) containing
        `keosop.revad.Expr` Cartesian variables.
        Return M `Expr` objects in the same coordinate order as `grids`.
    frame_conditions : callable, None, or "auto", optional
        Three independent constraints defining the rotational frame.
        A callable `frame_conditions(xyz)` receives an (N, 3) `Expr` array and returns three
        scalar expressions, evaluated with center-of-mass-centered geometry.
        None or "auto" infers constraints from the Cartesian SOP geometry.
        Used for rotational blocks and the pseudopotential.
    compress_tol : float, default=1e-10
        Absolute similarity tolerance for G-matrix compression.
    compress_zero_tol : float, default=1e-14
        Absolute threshold for discarding a product containing a near-zero factor.
    compress_identity_tol : float, default=1e-12
        Scaled tolerance for simplifying nearly constant factors during merging.
    compress_nterms_thresh : int, default=10
        Term-count trigger for intermediate G-matrix and pseudopotential compression.
        Final nonzero G-matrix elements and pseudopotential are compressed regardless of this trigger.
    nmode_order : int or None, optional
        Maximum number of simultaneously coupled coordinates retained in each G-matrix element truncated
        N-mode expansion. None keeps the full SOP.
    nmode_ref_indices : sequence of int or None, optional
        One valid grid index per coordinate defining the N-mode reference point.
        Required when `nmode_order` is not None.

    Notes
    -----
    Both transformations must describe the same coordinates and atom order:
    `cartesian_to_internal(internal_to_cartesian(q))` should recover q on the
    chosen coordinate domain, with consistent angular branches.

    For Cartesian-to-internal transformation, use `keosop.revad` helpers such as
    `distance`, `angle`, `dihedral`, `dot`, and `norm`.

    All intermediate expressions must be supported by the exact SOP algebra.
    General multivariate nonlinear functions and reciprocals may raise NotImplementedError
    when no implemented finite SOP representation exists.
    The pseudopotential evaluates displaced geometries outside the supplied grid
    endpoints, so the transformations must remain valid there as well.

    Call `run` to build the G-matrix blocks and `build_pseudopotential_fdf` separately to build the pseudopotential.

    See Also
    --------
    Gmat.run,
    Gmat.build_pseudopotential_fdf,
    keosop.zmat.internal_to_cartesian,
    keosop.zmat.cartesian_to_internal
    """

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

    @staticmethod
    def _structural_zero(value):
        """Recognize explicit zero products, without tolerance or cancellation"""
        if not isinstance(value, SumProd):
            return bool(value == 0.0)
        for term in value.terms:
            if not any(not np.any(factor) for factor in term):
                return False
            if not all(np.all(np.isfinite(factor)) for factor in term):
                return False
        return True

    @classmethod
    def _component_support(cls, svec):
        """Return active flattened Cartesian components for every s-vector row"""
        zero_by_id = {}
        support = []
        for row in svec:
            active = set()

            for component, value in enumerate(row.flat):
                key = id(value)

                if key not in zero_by_id:
                    zero_by_id[key] = cls._structural_zero(value)

                if not zero_by_id[key]:
                    active.add(component)

            support.append(frozenset(active))

        return tuple(support)

    @staticmethod
    def _rows_by_component(support):
        rows = {}
        for index, components in enumerate(support):
            for component in components:
                rows.setdefault(component, []).append(index)
        return rows

    def _compress_g(self, value):
        return value.compress(
            nterms_thresh=1,
            tol=self.compress_tol,
            zero_tol=self.compress_zero_tol,
            identity_tol=self.compress_identity_tol,
        )

    def _contract_svec(self, svec1, svec2, support1, support2, block_name="G"):
        """Accumulate candidate pairs using only shared Cartesian components"""
        block_start = perf_counter()
        symmetric = svec1 is svec2

        rows1 = self._rows_by_component(support1)
        rows2 = rows1 if symmetric else self._rows_by_component(support2)

        pairs = set()
        for component in rows1.keys() & rows2.keys():
            if symmetric:
                pairs.update(combinations_with_replacement(rows1[component], 2))
            else:
                pairs.update((i, j) for i in rows1[component] for j in rows2[component])

        zero = SumProd.constant(0.0, self.grids)
        gmat = np.empty((len(svec1), len(svec2)), dtype=object)

        for lam in range(len(svec1)):
            for mu in range(lam if symmetric else 0, len(svec2)):
                start = perf_counter()
                value = zero
                structural_zero = (lam, mu) not in pairs

                if not structural_zero:
                    for component in sorted(support1[lam] & support2[mu]):
                        atom, alpha = divmod(component, 3)
                        value = value + (
                            svec1[lam, atom, alpha]
                            * svec2[mu, atom, alpha]
                            / self.masses[atom]
                        )
                        if value.nterms > self.compress_nterms_thresh:
                            value = self._compress_g(value)
                    value = self._compress_g(value)
                    if self.nmode_order is not None:
                        value = value.nmode_project(
                            self.nmode_order,
                            self.nmode_ref_indices,
                            zero_tol=self.compress_zero_tol,
                            identity_tol=self.compress_identity_tol,
                        )
                        value = self._compress_g(value)
                gmat[lam, mu] = value
                elapsed = perf_counter() - start
                suffix = " (structural zero)" if structural_zero else ""

                # print(
                #     f"{block_name}[{lam}, {mu}]: {elapsed:.6f} s{suffix}",
                #     flush=True,
                # )

                if symmetric and lam != mu:
                    start = perf_counter()
                    gmat[mu, lam] = value
                    elapsed = perf_counter() - start

                    # print(
                    #     f"{block_name}[{mu}, {lam}]: {elapsed:.6f} s"
                    #     f" (symmetry copy of [{lam}, {mu}])",
                    #     flush=True,
                    # )

        print(f"{block_name} total: {perf_counter() - block_start:.3f} s", flush=True)
        return gmat

    def build_vib_svec(self, xyz_sop):
        """Evaluate present AD derivatives, sharing a cache within each row"""
        q_expr = self.trace_q()

        xyz_bindings = {
            f"{alpha}{i}": xyz_sop[i, a]
            for i in range(self.natoms)
            for a, alpha in enumerate("xyz")
        }

        components = {
            f"{alpha}{i}": (i, a)
            for i in range(self.natoms)
            for a, alpha in enumerate("xyz")
        }

        q_bindings = {}
        for index, root in enumerate(q_expr):
            value = SumProd.init_coordinate(index, self.grids)
            q_bindings[root.id] = value
            q_bindings[("sig", expr_signature(root))] = value

        zero = SumProd.constant(0.0, self.grids)
        svec = np.empty((self.ncoo, self.natoms, 3), dtype=object)
        svec.fill(zero)

        for index, root in enumerate(q_expr):
            deriv = reverse_derivatives(root)
            # Reverse AD can introduce geometry primitives needed by dihedral derivatives,
            # need to create aliases after differentiating this
            bindings = add_user_geometry_aliases(q_expr, q_bindings)

            cache = {}
            for name, derivative in deriv.items():
                atom, alpha = components[name]
                svec[index, atom, alpha] = eval_expr(
                    derivative, xyz_bindings, self.grids, bindings, cache
                )

        return svec

    def prod_svec(self, svec1, svec2):
        """Contract s-vectors"""
        return self._prod_svec(svec1, svec2, "G")

    def _prod_svec(self, svec1, svec2, block_name):
        """Contract only overlapping Cartesian components"""
        support1 = self._component_support(svec1)
        support2 = support1 if svec1 is svec2 else self._component_support(svec2)
        return self._contract_svec(svec1, svec2, support1, support2, block_name)

    def build_gvib(self, svec):
        support = self._component_support(svec)
        return self._contract_svec(svec, svec, support, support, "Gvib")

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

                if self.sop_is_zero(det):
                    continue

                # expression may be formally nonzero but have zeros at all grid points
                probes = [
                    tuple(n // 2 for n in det.npoints),
                    tuple(0 for _ in det.npoints),
                    tuple(n - 1 for n in det.npoints),
                ]

                if not any(
                    abs(self.sop_point_value(det, p)) > self.compress_zero_tol
                    for p in probes
                ):
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
        """Return Gvib, Grovib, Grot, reusing component supports across blocks"""
        vsupport = self._component_support(svib)
        rsupport = self._component_support(srot)

        return (
            self._contract_svec(svib, svib, vsupport, vsupport, "Gvib"),
            self._contract_svec(srot, svib, rsupport, vsupport, "Grovib"),
            self._contract_svec(srot, srot, rsupport, rsupport, "Grot"),
        )

    @staticmethod
    def fdf_weights(x0, x, order):
        """Finite-difference weights for d**order/dx**order at x0"""
        x = np.asarray(x, dtype=float)
        n = len(x)
        if order < 0 or order >= n:
            raise ValueError(
                "finite-difference derivative order must be < stencil size"
            )
        dx = x - float(x0)
        A = np.vstack([dx**k for k in range(n)])
        b = np.zeros(n, dtype=float)
        b[order] = (
            float(np.math.factorial(order))
            if hasattr(np, "math")
            else float(__import__("math").factorial(order))
        )
        return np.linalg.solve(A, b)

    def fdf_augmented_grids(self, stencil=5, fdf_step=1.0e-3):
        """Augment every grid point with a fixed finite-difference stencil"""
        if stencil < 3 or stencil % 2 == 0:
            raise ValueError("stencil must be an odd integer >= 3")

        ncoo = len(self.grids)
        steps = np.asarray(fdf_step, dtype=float)

        if steps.ndim == 0:
            steps = np.full(ncoo, float(steps), dtype=float)
        else:
            steps = steps.ravel()
            if len(steps) != ncoo:
                raise ValueError(
                    "fdf_step must be a scalar or have one value per coordinate"
                )

        if np.any(~np.isfinite(steps)) or np.any(steps <= 0.0):
            raise ValueError("all fdf_step values must be finite and positive")

        radius = stencil // 2
        aug_grids = []
        info = []

        for icoo, g in enumerate(self.grids):
            g = np.asarray(g, dtype=float)

            if g.ndim != 1 or len(g) < 1:
                raise ValueError(
                    "each user grid must be a non-empty one-dimensional array"
                )

            if len(g) > 1 and np.any(np.diff(g) <= 0.0):
                raise ValueError("user grids must be strictly increasing")

            pts = list(g)
            stencils = []

            for x0 in g:
                xs = x0 + steps[icoo] * np.arange(-radius, radius + 1, dtype=float)

                pts.extend(xs.tolist())
                stencils.append(xs)

            aug = np.unique(np.asarray(pts, dtype=float))
            aug.sort()

            coord_info = []
            for x0, xs in zip(g, stencils):
                inds = np.array(
                    [int(np.argmin(np.abs(aug - x))) for x in xs], dtype=int
                )

                # stencil points should match to roundoff
                if np.max(np.abs(aug[inds] - xs)) > 1e-11 * max(
                    1.0, np.max(np.abs(xs))
                ):
                    raise RuntimeError(
                        "failed to locate finite-difference stencil in augmented grid"
                    )

                w1 = self.fdf_weights(x0, aug[inds], 1)
                w2 = self.fdf_weights(x0, aug[inds], 2)
                coord_info.append((inds, w1, w2))

            orig_inds = np.array(
                [int(np.argmin(np.abs(aug - x))) for x in g], dtype=int
            )

            aug_grids.append(aug)
            info.append({"points": coord_info, "orig_inds": orig_inds})

        return aug_grids, info

    def fdf_sop(self, sop, fd_info, deriv_orders=None):
        """Differentiate sum-of-products terms using 1D finite-differences"""
        if deriv_orders is None:
            deriv_orders = [0] * self.ncoo

        if len(deriv_orders) != self.ncoo:
            raise ValueError("deriv_orders has wrong length")

        terms = []

        for term in sop.terms:
            newterm = []

            for k, factor in enumerate(term):
                factor = np.asarray(factor, dtype=float)
                order = int(deriv_orders[k])
                ci = fd_info[k]

                if order == 0:
                    vals = factor[ci["orig_inds"]]

                elif order in (1, 2):
                    vals = np.empty(len(self.grids[k]), dtype=float)

                    for j, (inds, w1, w2) in enumerate(ci["points"]):
                        vals[j] = np.dot(w1 if order == 1 else w2, factor[inds])

                else:
                    raise ValueError("only derivative orders 0, 1, and 2 are supported")

                newterm.append(vals)

            terms.append(newterm)

        return SumProd(terms)

    def fdf_add(self, acc, value):
        out = acc + value
        if out.nterms >= self.compress_nterms_thresh:
            out = out.compress(
                tol=self.compress_tol,
                zero_tol=self.compress_zero_tol,
                identity_tol=self.compress_identity_tol,
            )
        return out

    def build_pseudopotential_fdf(
        self,
        stencil=5,
        fdf_step=1.0e-3,
        *,
        compress_tol=1.0e-8,
        zero_tol=1.0e-10,
        identity_tol=1.0e-10,
        max_comparisons=200_000,
    ):
        """Build the pseudopotential U in sum-of-products (SOP) form using finite differences.

        Parameters
        ----------
        stencil : int, default=5
            Number of points in each centered finite-difference stencil.
            First and second derivatives are computed from these points.
        fdf_step : float or array_like of shape (M,), default=1e-3
            Absolute finite-difference displacement in each coordinate's units.
            A scalar applies to every coordinate, otherwise provide one step per coordinate.
            Stencils extending beyond the original grid endpoints require valid geometries there.
        compress_tol : float, default=1e-8
            Absolute similarity tolerance for pseudopotential compression.
        zero_tol : float, default=1e-10
            Absolute threshold for discarding a product containing a near-zero factor.
        identity_tol : float, default=1e-10
            Scaled tolerance for simplifying nearly constant factors during merging.

        Returns
        -------
        pseudo : keosop.sumprod.SumProd
            Pseudopotential evaluated on the original coordinate grids.

        Notes
        -----

        Exact factors, products, derivatives, and comparisons are reused in caches local to this call, comparison tables are limited to 128 entries and 4 MB.

        See Also
        --------
        Gmat,
        Gmat.run
        """
        started = perf_counter()

        compression = dict(
            tol=compress_tol, zero_tol=zero_tol, identity_tol=identity_tol
        )

        for name, value in compression.items():
            label = "compress_tol" if name == "tol" else name
            try:
                if (
                    isinstance(value, (bool, np.bool_))
                    or not np.isscalar(value)
                    or not np.isrealobj(value)
                ):
                    raise ValueError
                value = float(value)

            except (TypeError, ValueError, OverflowError):
                raise ValueError(
                    f"{label} must be a finite nonnegative scalar"
                ) from None

            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{label} must be a finite nonnegative scalar")

            compression[name] = value

        max_comparisons = operator.index(max_comparisons)

        grids, info = self.fdf_augmented_grids(stencil, fdf_step)
        aug = type(self)(
            grids,
            self.masses,
            self.internal_to_cartesian,
            self.cartesian_to_internal,
            frame_conditions=self.frame_conditions,
            compress_tol=self.compress_tol,
            compress_zero_tol=self.compress_zero_tol,
            compress_identity_tol=self.compress_identity_tol,
            compress_nterms_thresh=self.compress_nterms_thresh,
            nmode_order=None,
        )

        aux = aug.build_svec(return_rot=True)

        svib_a, srot_a = aux["svib"], aux["srot"]
        del aux

        prepared = perf_counter()
        context = _PseudopotentialContext(self.grids, info, max_comparisons)
        imported = {}

        def import_active(array):
            raw = {}
            for row in range(len(array)):
                for atom in range(self.natoms):
                    for alpha in range(3):

                        source = array[row, atom, alpha]

                        if id(source) not in imported:
                            imported[id(source)] = context.import_sop(source)

                        sop = imported[id(source)]

                        if not context.zero_sop(sop):
                            raw[row, 3 * atom + alpha] = sop
            return raw

        vraw, rraw = import_active(svib_a), import_active(srot_a)
        del imported, svib_a, srot_a, aug

        vrows = {}
        for row, component in vraw:
            vrows.setdefault(component, []).append(row)

        vib = {key: context.fdf(sop) for key, sop in vraw.items()}
        rot = {key: context.fdf(sop) for key, sop in rraw.items()}
        zero = context.constant(0.0)

        def add(acc, value):
            result = acc + value
            if result.nterms >= self.compress_nterms_thresh:
                result = result.compress(nterms_thresh=1, **compression)
            return result

        eps = (
            (0, 1, 2, 1.0),
            (1, 2, 0, 1.0),
            (2, 0, 1, 1.0),
            (0, 2, 1, -1.0),
            (2, 1, 0, -1.0),
            (1, 0, 2, -1.0),
        )

        result = zero

        for atom, mass in enumerate(self.masses):
            rows = [vrows.get(3 * atom + a, ()) for a in range(3)]
            term1 = zero
            for alpha in range(3):
                w = zero
                for aa, beta, gamma, sign in eps:
                    key = beta, 3 * atom + gamma
                    if aa == alpha and key in rot:
                        w = add(w, sign * rot[key])
                if not context.zero_sop(w):
                    term1 = add(term1, 0.5 * w * w)

            term2 = zero
            for alpha, beta, gamma, sign in eps:
                key = beta, 3 * atom + alpha
                if key in rraw:
                    for n in rows[gamma]:
                        term2 = add(
                            term2,
                            sign * vib[n, 3 * atom + gamma] * context.fdf(rraw[key], n),
                        )

            term3 = zero
            for alpha in range(3):
                component = 3 * atom + alpha
                for n in rows[alpha]:
                    for m in rows[alpha]:
                        term3 = add(
                            term3,
                            vib[n, component] * context.fdf(vraw[m, component], n, m),
                        )

            term4 = zero
            for alpha in range(3):
                component = 3 * atom + alpha
                divergence = zero
                for n in rows[alpha]:
                    divergence = add(divergence, context.fdf(vraw[n, component], n))
                if not context.zero_sop(divergence):
                    term4 = add(term4, 0.5 * divergence * divergence)

            result = add(result, (0.25 / float(mass)) * (term1 + term2 - term3 - term4))

        result = result.compress(nterms_thresh=1, **compression).export()

        finished = perf_counter()
        print(f"Pseudopotential total: {finished - started:.6f} s", flush=True)

        return result

    def build_svec(self, return_rot=False):
        """Build the same auxiliary data as `run`, without G contractions"""
        xyz = self.sumprod_x()
        svib = self.build_vib_svec(xyz)
        aux = {"xyz": xyz, "svib": svib}
        if not return_rot:
            return aux

        # Axis conventions and rotational s-vectors follow Sørensen (1979),
        # Large Amplitude Motion in Molecules II, Section 2.2.3.2:
        # https://doi.org/10.1007/BFb0048009
        # Vibrational derivatives use the original geometry.
        # Only the frame and rotational tangents use COM-centered coordinates.
        # Apply the COM chain rule to frame derivatives in cvec_translation_free below.
        xyz_com, com = self.com_xyz(xyz)

        tvec = self.build_tvec(xyz_com)
        cvec_raw = self.build_cvec(xyz_com, tvec=tvec)
        cvec = self.cvec_translation_free(cvec_raw)

        J = self.build_jmat(cvec, tvec)
        Jinv, detJ = self.inverse_3x3_mat(J)

        srot = self.build_rot_svec(Jinv, cvec)

        aux.update(
            xyz_com=xyz_com,
            com=com,
            srot=srot,
            cvec=cvec,
            cvec_raw=cvec_raw,
            tvec=tvec,
            J=J,
            Jinv=Jinv,
            detJ=detJ,
            frame_conditions=self.inferred_frame_conditions,
        )
        return aux

    def run(self, return_rot=False, return_svec=False):
        """Build dense arrays of G-matrix elements in sum-of-products (SOP) form.

        Parameters
        ----------
        return_rot : bool, default=False
            Also construct the rovibrational and rotational blocks.
            This requires three independent frame constraints, supplied or inferred from geometry.
        return_svec : bool, default=False
            Append a dictionary of intermediate geometry, s-vectors, and frame data to the returned result.

        Returns
        -------
        gvib : numpy.ndarray, shape (M, M), dtype=object
            Vibrational G-matrix block, with SumProd elements on the original grids,
            where M is the number of internal coordinates.
            Returned alone when both flags are False,
            or as `(gvib, aux)` when only `return_svec` is True.
        grovib : numpy.ndarray, shape (3, M), dtype=object
            Rovibrational G-matrix block, returned if `return_rot=True`.
            Rows index body-fixed frame axes, columns index internal coordinates.
        grot : numpy.ndarray, shape (3, 3), dtype=object
            Rotational G-matrix block, returned if `return_rot=True`.
            The three blocks are returned as `(gvib, grovib, grot)` when `return_svec=False`.
        aux : dict
            Included only if `return_svec=True`.
            With rotations the return value is `(gvib, grovib, grot, aux)`.
            The dictionary contains Cartesian SOP coordinates `xyz` of shape (N, 3)
            and vibrational derivatives `svib` of shape (M, N, 3), where N is the number of atoms.
            With rotations included, it also contains `xyz_com`, `com`, `srot`, `cvec`, `cvec_raw`,
            `tvec`, `J`, `Jinv`, `detJ`, and `frame_conditions`.

        See Also
        --------
        Gmat,
        Gmat.build_pseudopotential_fdf
        """
        start = perf_counter()

        aux = self.build_svec(return_rot=return_rot)

        if not return_rot:
            gvib = self.build_gvib(aux["svib"])

            print(
                "G-matrix total (including geometry and s-vectors): "
                f"{perf_counter() - start:.3f} s",
                flush=True,
            )

            return (gvib, aux) if return_svec else gvib

        blocks = self.build_full_g(aux["svib"], aux["srot"])

        print(
            "G-matrix total (including geometry and s-vectors): "
            f"{perf_counter() - start:.3f} s",
            flush=True,
        )

        return (*blocks, aux) if return_svec else blocks
