from itertools import combinations

import numpy as np

# Performs internal-to-Cartesian pass 'q->r' for building sum-of-product form,
# where 'q' are internal coordinates and 'r' are Cartesian coordinate

# TODO: add forward-mode AD for consistency check between user's input internal-to-Cartesian and Cartesian-to-internal functions

RECIPROCAL_TOL = 1e-9
COMPRESS_NTERMS_THRESH = 100

_TRACE_SINGULARITIES = False
_TRACE_COORDINATES = None
_TRACE_KINDS = ("q", "sin", "cos")


def configure_singularity_tracing(
    enabled=False, coordinates=None, kinds=("q", "sin", "cos")
):
    global _TRACE_SINGULARITIES, _TRACE_COORDINATES, _TRACE_KINDS

    _TRACE_SINGULARITIES = bool(enabled)
    _TRACE_COORDINATES = (
        None if coordinates is None else tuple(int(k) for k in coordinates)
    )
    _TRACE_KINDS = tuple(str(k) for k in kinds)


class SumProd:

    def __init__(self, terms, protected=None, grids=None):
        self.terms = [[np.copy(np.array(f, dtype=float)) for f in t] for t in terms]
        self.grids = (
            None
            if grids is None
            else [np.asarray(g, dtype=float).copy() for g in grids]
        )

        if protected is None:
            protected = [()] * len(self.terms)
        if len(protected) != len(self.terms):
            raise ValueError("Protected metadata must have one entry per SOP term")
        self.protected = [self.protected_signature(sig) for sig in protected]

        if len(self.terms) == 0:
            raise ValueError("SumProd must contain at least one term")

        if len(self.terms[0]) == 0:
            raise ValueError("SumProd terms must contain at least one coordinate")

        # check shapes
        ncoo = len(self.terms[0])
        npoints = tuple(len(f) for f in self.terms[0])
        for t in self.terms:
            if len(t) != ncoo or tuple(len(f) for f in t) != npoints:
                raise ValueError("Inconsistent shapes of terms in SumProd")

    @staticmethod
    def protected_signature(sig):
        acc = {}
        for k, kind, power in sig:
            key = (int(k), str(kind))
            acc[key] = acc.get(key, 0.0) + float(power)
        return tuple(
            (k, kind, p) for (k, kind), p in sorted(acc.items()) if abs(p) > 1e-14
        )

    @property
    def singularities(self):
        """Residual negative protected factors"""
        return [tuple(x for x in sig if x[2] < 0.0) for sig in self.protected]

    def singularity_summary(self):
        out = {}
        for sig in self.singularities:
            out[sig] = out.get(sig, 0) + 1
        return out

    def collect_by_singularity(self):
        groups = {}
        for term, sig in zip(self.terms, self.singularities):
            groups.setdefault(sig, []).append(term)
        return {
            sig: SumProd(ts, protected=[sig] * len(ts), grids=self.grids)
            for sig, ts in groups.items()
        }

    def evaluate(self, inds, grids=None, include_singular=True):
        grids = self.grids if grids is None else grids
        val = 0.0

        for term, sig in zip(self.terms, self.protected):
            x = 1.0

            for k, f in enumerate(term):
                x *= f[inds[k]]

            if include_singular:
                if grids is None and sig:
                    raise ValueError("Grids are required to evaluate protected factors")

                for k, kind, power in sig:
                    q = grids[k][inds[k]]

                    base = (
                        q
                        if kind == "q"
                        else (
                            np.sin(q)
                            if kind == "sin"
                            else np.cos(q) if kind == "cos" else None
                        )
                    )

                    if base is None:
                        raise ValueError(f"Unknown protected factor kind {kind}")

                    x *= base**power

            val += x

        return val

    def normalize_protected_factors(self, tol=1e-10, coordinates=None, kinds=None):
        """Performs cancellation step for protected factors.

        If a numerical factor is equal to 'const*q**p', 'const*sin(q)**p', ...,
        its coordinate power is moved into 'protected'  and only 'const'
        is left on the numerical grid. The operation is exact up to 'tol'.
        """
        if not _TRACE_SINGULARITIES or self.grids is None:
            return self.copy()

        coords = _TRACE_COORDINATES if coordinates is None else coordinates
        trace_set = None if coords is None else set(int(k) for k in coords)
        allowed = set(_TRACE_KINDS if kinds is None else kinds)

        new_terms, new_sigs = [], []
        for term, sig in zip(self.terms, self.protected):
            fs = [np.asarray(f, dtype=float).copy() for f in term]

            extra = []
            targets = {(int(k), kind) for k, kind, power in sig if power < 0.0}

            for k, f in enumerate(fs):
                if trace_set is not None and k not in trace_set:
                    continue

                target_kinds = {
                    kind for kk, kind in targets if kk == k and kind in allowed
                }

                if not target_kinds:
                    continue

                rec = recognize_protected(f, self.grids[k], tol=tol)
                if rec is None:
                    continue

                kind, power, scale = rec
                if kind not in target_kinds:
                    continue

                fs[k] = np.full_like(f, scale, dtype=float)

                extra.append((k, kind, power))

            new_terms.append(fs)
            new_sigs.append(self.protected_signature(tuple(sig) + tuple(extra)))

        return SumProd(new_terms, protected=new_sigs, grids=self.grids)

    def absorb_selected_protected(self, predicate):
        """Move selected protected factors exactly into their numerical 1D arrays"""
        if self.grids is None:
            return self.copy()

        terms = []
        sigs = []

        for term, sig in zip(self.terms, self.protected):
            fs = [np.asarray(f, dtype=float).copy() for f in term]

            keep = []

            for k, kind, power in sig:
                if predicate(int(k), str(kind), float(power)):
                    q = self.grids[int(k)]
                    base = (
                        q if kind == "q" else np.sin(q) if kind == "sin" else np.cos(q)
                    )

                    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                        fs[int(k)] *= base ** float(power)

                else:
                    keep.append((k, kind, power))

            terms.append(fs)
            sigs.append(self.protected_signature(keep))

        return SumProd(terms, protected=sigs, grids=self.grids)

    def split_protected_even_remainder(self, tol=1e-10, max_degree=6):
        """For a protected factor `base(q)^(-2m)` on coordinate k, try to write
        the numerical 1D factor on that same coordinate as a low-degree polynomial
        in `x = base(q)^2`:

            f(q) = a0 + a1*x + ... + ad*x**d

        The terms a_j*x**j with j < m remain protected with pole order
        `-2*(m-j)`, terms with j >= m are regular and are moved completely
        into the numerical factor
        """
        if self.grids is None:
            return self.copy()

        out_terms = []
        out_sigs = []

        for term, sig0 in zip(self.terms, self.protected):
            pieces = [([np.asarray(f, dtype=float).copy() for f in term], list(sig0))]

            # consider only negative even integer powers
            for k, kind, power in sig0:
                k = int(k)
                power = float(power)
                n = int(round(-power))

                if kind not in ("q", "sin", "cos") or power >= 0.0 or n < 2 or n % 2:
                    continue

                if abs(power + n) > 1e-12:
                    continue

                m = n // 2
                q = np.asarray(self.grids[k], dtype=float)
                base = q if kind == "q" else np.sin(q) if kind == "sin" else np.cos(q)
                x = base * base

                newpieces = []

                for fs, sig in pieces:
                    cur = None
                    for ii, (kk, kd, pp) in enumerate(sig):
                        if (
                            int(kk) == k
                            and kd == kind
                            and abs(float(pp) - power) < 1e-12
                        ):
                            cur = ii
                            break

                    if cur is None:
                        newpieces.append((fs, sig))
                        continue

                    y = np.asarray(fs[k], dtype=float)
                    if len(y) < 2:
                        newpieces.append((fs, sig))
                        continue

                    degmax = min(int(max_degree), len(y) - 1)
                    coeff = None
                    scale = max(1.0, float(np.max(np.abs(y))))

                    for deg in range(degmax + 1):
                        V = np.vander(x, N=deg + 1, increasing=True)
                        c, *_ = np.linalg.lstsq(V, y, rcond=None)
                        err = float(np.max(np.abs(V @ c - y)))

                        if err <= tol * scale:
                            coeff = c
                            break

                    if coeff is None:
                        newpieces.append((fs, sig))
                        continue

                    # replace this one SOP term by the polynomial pieces,
                    # the scalar polynomial coefficient is put on coordinate k
                    sig_without = sig[:cur] + sig[cur + 1 :]

                    for j, a in enumerate(coeff):
                        if abs(float(a)) <= tol * scale:
                            continue

                        fsj = [f.copy() for f in fs]

                        if j < m:
                            fsj[k] = np.full_like(y, float(a))
                            ps = -2.0 * (m - j)
                            sigj = self.protected_signature(
                                tuple(sig_without) + ((k, kind, ps),)
                            )

                        else:
                            fsj[k] = float(a) * (x ** (j - m))
                            sigj = self.protected_signature(tuple(sig_without))

                        newpieces.append((fsj, list(sigj)))

                pieces = newpieces

            for fs, sig in pieces:
                out_terms.append(fs)
                out_sigs.append(self.protected_signature(sig))

        if not out_terms:
            return self.copy()

        return SumProd(out_terms, protected=out_sigs, grids=self.grids)

    @property
    def npoints(self):
        return tuple(len(f) for f in self.terms[0])

    @property
    def ncoo(self):
        return len(self.terms[0])

    @property
    def nterms(self):
        return len(self.terms)

    def refactor_rank1(self, tol=1e-10, max_reference_trials=64):
        """Tests if sum-of-products function can be written as single-term (rank-1) function.

        Used for small, special expressions, such as the rotational-frame determinant det(J),
        not as a general compression routine.
        """
        if self.nterms == 1:
            return self.copy()

        if any(sig != self.protected[0] for sig in self.protected[1:]):
            return self.copy()

        T, M = self.nterms, self.ncoo
        terms = self.terms

        candidates = []
        for k in range(M):
            score = np.add.reduce([np.abs(terms[t][k]) for t in range(T)])
            order = np.argsort(score)[::-1]
            candidates.append([int(x) for x in order[: min(4, len(order))]])

        refs_to_try = [tuple(c[0] for c in candidates)]
        rng = np.random.default_rng(12345)
        for _ in range(max(0, int(max_reference_trials) - 1)):
            refs_to_try.append(tuple(c[int(rng.integers(len(c)))] for c in candidates))

        scale_norm = 0.0
        for t in range(T):
            nt = 1.0
            for k in range(M):
                nt *= float(np.linalg.norm(terms[t][k]))
            scale_norm += nt
        zero_cut = max(float(tol) * max(scale_norm, 1.0), np.finfo(float).eps)

        for refs in refs_to_try:
            coeff = np.ones(T, dtype=float)
            for t in range(T):
                for k in range(M):
                    coeff[t] *= terms[t][k][refs[k]]
            pivot = float(np.sum(coeff))
            if abs(pivot) <= zero_cut:
                continue

            factors = []
            for k in range(M):
                fiber = np.zeros_like(terms[0][k], dtype=float)
                for t in range(T):
                    c = 1.0
                    for j in range(M):
                        if j != k:
                            c *= terms[t][j][refs[j]]
                    fiber += c * terms[t][k]
                factors.append(fiber / pivot)
            factors[0] = factors[0] * pivot

            # verify ||F-G|| without using full product grid
            norm_f2 = 0.0
            for a in range(T):
                for b in range(T):
                    v = 1.0
                    for k in range(M):
                        v *= float(np.dot(terms[a][k], terms[b][k]))
                    norm_f2 += v

            norm_g2 = 1.0
            for k in range(M):
                norm_g2 *= float(np.dot(factors[k], factors[k]))

            cross = 0.0
            for t in range(T):
                v = 1.0
                for k in range(M):
                    v *= float(np.dot(terms[t][k], factors[k]))
                cross += v

            err2 = max(0.0, norm_f2 + norm_g2 - 2.0 * cross)
            denom = max(np.sqrt(max(norm_f2, 0.0)), np.finfo(float).tiny)

            if np.sqrt(err2) / denom <= tol:
                return SumProd(
                    [factors], protected=[self.protected[0]], grids=self.grids
                )

        return self.copy()

    @classmethod
    def init_coordinate(cls, icoo, grids, func=None):
        f = [np.ones(len(g), dtype=float) for g in grids]
        v = np.asarray(grids[icoo], dtype=float)
        f[icoo] = np.copy(np.asarray(func(v) if func else v, dtype=float))
        return cls([f], grids=grids)

    @classmethod
    def constant(cls, c, grids):
        f = [np.ones(len(g), dtype=float) for g in grids]
        f[0] *= float(c)
        return cls([f], grids=grids)

    def dummy_grids(self):
        return [np.empty(n) for n in self.npoints]

    def as_sumprod(self, x):
        if isinstance(x, SumProd):
            return x
        elif np.isscalar(x):
            return SumProd.constant(x, self.dummy_grids())
        else:
            return NotImplemented

    def copy(self):
        return SumProd(self.terms, protected=self.protected, grids=self.grids)

    @staticmethod
    def factor_is_constant(f, tol=0.0):
        """Return True if a one-dimensional factor is constant on its grid"""
        f = np.asarray(f, dtype=float)
        if tol <= 0.0:
            return np.all(f == f[0])
        return np.all(np.abs(f - f[0]) <= tol)

    @classmethod
    def term_active_set(cls, term, tol=0.0):
        """Coordinates on which sum-of-products term is not a constant on its grid"""
        return frozenset(
            k for k, f in enumerate(term) if not cls.factor_is_constant(f, tol)
        )

    def nmode_project(
        self, max_order, ref_indices, zero_tol=0.0, identity_tol=0.0, compress=True
    ):
        """Return the exact truncated N-mode projection of sum-of-products.

        For every product P(q)=prod_k f_k(q_k), write

            f_k(q_k) = f_k(q_k^0) + d_k(q_k)

        where q_k^0 is selected by `ref_indices[k]`.

        Expanding the product and retaining subsets containing at most `max_order`
        d-factors gives the truncated N-mode expansion.
        Every returned nonconstant term contains at most `max_order`
        simultaneously coupled coordinates.
        """
        if max_order < 0:
            raise ValueError(f"'max_order' = {max_order} must be non-negative")
        max_order = min(max_order, self.ncoo)

        ref_indices = tuple(int(i) for i in ref_indices)

        if len(ref_indices) != self.ncoo:
            raise ValueError(
                f"'ref_indices' must have length {self.ncoo}, got {len(ref_indices)}"
            )

        for k, (i, n) in enumerate(zip(ref_indices, self.npoints)):
            if i < 0 or i >= n:
                raise IndexError(
                    f"'ref_indices[{k}]={i}' is outside the grid range [0, {n})"
                )

        ones = [np.ones(n, dtype=float) for n in self.npoints]
        new_terms = []
        new_protected = []

        for term, sig in zip(self.terms, self.protected):
            c = np.array([term[k][ref_indices[k]] for k in range(self.ncoo)])
            delta = [term[k] - c[k] for k in range(self.ncoo)]

            # coordinates with identically zero delta are not active
            variable = [
                k
                for k in range(self.ncoo)
                if not self.is_zero_factor(delta[k], zero_tol)
            ]

            for order in range(min(max_order, len(variable)) + 1):
                for active_tuple in combinations(variable, order):
                    active = set(active_tuple)
                    coeff = float(
                        np.prod([c[k] for k in range(self.ncoo) if k not in active])
                    )

                    if abs(coeff) <= zero_tol:
                        continue

                    factors = [ones[k].copy() for k in range(self.ncoo)]
                    for k in active:
                        factors[k] = delta[k].copy()

                    factors[0] *= coeff

                    new_terms.append(factors)
                    new_protected.append(sig)

        if not new_terms:
            return SumProd.constant(0.0, self.dummy_grids())

        out = SumProd(new_terms, protected=new_protected, grids=self.grids)

        return (
            out.compress_hashed(zero_tol=zero_tol, identity_tol=identity_tol)
            if compress
            else out
        )

    def nmode_mul(
        self, x, max_order, ref_indices, zero_tol=0.0, identity_tol=0.0, compress=True
    ):
        """Multiply two sum-of-products and return their exact truncated N-mode product"""
        if np.isscalar(x):
            return self.nmode_project(max_order, ref_indices, zero_tol) * x

        if not isinstance(x, SumProd):
            return NotImplemented

        if self.npoints != x.npoints:
            raise ValueError("Cannot multiply SumProd objects with different grids")

        a = self.nmode_project(max_order, ref_indices, zero_tol, compress=False)
        b = x.nmode_project(max_order, ref_indices, zero_tol, compress=False)

        a_data = [(t, self.term_active_set(t, zero_tol)) for t in a.terms]
        b_data = [(t, self.term_active_set(t, zero_tol)) for t in b.terms]

        terms = []
        protected = []

        for ia, (ta, aa) in enumerate(a_data):
            for ib, (tb, ab) in enumerate(b_data):
                if len(aa | ab) > max_order:
                    continue

                terms.append([fa * fb for fa, fb in zip(ta, tb)])
                protected.append(
                    self.protected_signature(a.protected[ia] + b.protected[ib])
                )

        if not terms:
            return SumProd.constant(0.0, self.dummy_grids())

        out = SumProd(terms, protected=protected, grids=self.grids)

        return (
            out.compress_hashed(zero_tol=zero_tol, identity_tol=identity_tol)
            if compress
            else out
        )

    @staticmethod
    def factor_key(f):
        """Hash key for a 1D sum-of-products factor"""
        a = np.ascontiguousarray(f, dtype=float).copy()

        a[a == 0.0] = 0.0
        return (a.shape, a.dtype.str, a.tobytes())

    @staticmethod
    def is_zero_factor(f, zero_tol=0.0):
        if zero_tol == 0.0:
            return np.all(f == 0.0)
        return np.all(np.abs(f) <= zero_tol)

    @staticmethod
    def simplify_1d_factor(f, tol=1e-12):
        """Simplify 1D sum-of-products factor if it is numerically constant"""
        a0 = np.asarray(f, dtype=float)
        if a0.size == 0 or not np.all(np.isfinite(a0)):
            return a0, False

        c = float(a0.flat[0])
        scale0 = max(1.0, abs(c))
        atol0 = float(tol) * scale0

        if a0.size > 1 and abs(float(a0.flat[-1]) - c) > atol0:
            return a0, False
        if a0.size > 2 and abs(float(a0.flat[a0.size // 2]) - c) > atol0:
            return a0, False

        scale = max(1.0, float(np.max(np.abs(a0))))
        atol = float(tol) * scale
        if np.max(np.abs(a0 - c)) > atol:
            return a0, False

        # snap common constants
        if abs(c) <= atol:
            c = 0.0
        elif abs(c - 1.0) <= atol:
            c = 1.0
        elif abs(c + 1.0) <= atol:
            c = -1.0

        out = np.full_like(a0, c, dtype=float)
        return out, not np.array_equal(out, a0)

    @staticmethod
    def factors_close(f1, f2, tol=0.0):
        """Compare two 1D sum-of-products factors"""
        f1 = np.asarray(f1)
        f2 = np.asarray(f2)

        if f1.shape != f2.shape:
            return False

        if tol <= 0.0:
            return np.array_equal(f1, f2)

        # return np.max(np.abs(f1 - f2)) <= tol
        return np.all(np.abs(f1 - f2) <= tol)

    @staticmethod
    def canonical_factor_scale(f):
        """Return (canonical_factor, scale) with f == scale * canonical_factor.

        This is for proportional factors to hash identically while keeping the scalar
        coefficient exact up to floating-point arithmetic.
        """
        a = np.asarray(f, dtype=float)
        if a.size == 0 or not np.all(np.isfinite(a)):
            return a.copy(), 1.0

        i = int(np.argmax(np.abs(a)))
        scale = float(a.flat[i])

        if scale == 0.0:
            return np.zeros_like(a), 0.0

        c = a / scale
        c[c == 0.0] = 0.0
        return c, scale

    def _cancel_proportional_terms(self, terms, sigs, zero_tol=0.0):
        """Cancel or combine sum-of-products terms that are proportional in every coordinate"""
        groups = {}
        reps = {}
        coeffs = {}

        for term, sig in zip(terms, sigs):
            canon = []
            coeff = 1.0
            zero = False

            for f in term:
                cf, sc = self.canonical_factor_scale(f)

                if sc == 0.0:
                    zero = True
                    break

                canon.append(cf)
                coeff *= sc

            if zero:
                continue

            key = (sig,) + tuple(self.factor_key(f) for f in canon)

            if key not in groups:
                groups[key] = sig
                reps[key] = canon
                coeffs[key] = coeff
            else:
                coeffs[key] += coeff

        out_terms, out_sigs = [], []

        for key, sig in groups.items():
            coeff = coeffs[key]

            if abs(coeff) <= zero_tol:
                continue

            term = [f.copy() for f in reps[key]]

            # keep the overall scalar in one factor
            term[0] = term[0] * coeff
            out_terms.append(term)
            out_sigs.append(sig)

        return out_terms, out_sigs

    def compress_hashed(self, zero_tol=0.0, max_sweeps=None, identity_tol=0.0):
        """Exactly combine sum-of-products terms using hash grouping.

        This on purpose does not canonicalize proportional factors.
        It is used globally, including during automatic frame-condition search, so
        only terms with identical factors are collected.

        Proportional-factor cancellation is coded in cancel_proportional_terms()
        as a separate targeted operation for the rotational-frame determinant det(J).
        """
        pairs = [
            ([f.copy() for f in t], sig) for t, sig in zip(self.terms, self.protected)
        ]

        pairs = [
            (t, sig)
            for t, sig in pairs
            if not any(self.is_zero_factor(f, zero_tol) for f in t)
        ]

        terms = [t for t, _ in pairs]
        sigs = [sig for _, sig in pairs]
        if not terms:
            return SumProd.constant(0.0, self.dummy_grids())

        sweep = 0
        changed = True
        while changed and (max_sweeps is None or sweep < max_sweeps):
            changed = False
            sweep += 1

            for k in range(self.ncoo):

                groups = {}
                for term, sig in zip(terms, sigs):
                    key = (sig,) + tuple(
                        self.factor_key(term[j]) for j in range(self.ncoo) if j != k
                    )

                    groups.setdefault(key, []).append((term, sig))

                new_terms, new_sigs = [], []
                for group in groups.values():
                    if len(group) == 1:
                        new_terms.append(group[0][0])
                        new_sigs.append(group[0][1])
                        continue

                    merged = [f.copy() for f in group[0][0]]
                    merged[k] = np.add.reduce([x[0][k] for x in group])
                    merged[k], _ = self.simplify_1d_factor(merged[k], tol=identity_tol)
                    changed = True

                    if not self.is_zero_factor(merged[k], zero_tol):
                        new_terms.append(merged)
                        new_sigs.append(group[0][1])

                terms, sigs = new_terms, new_sigs

                if not terms:
                    return SumProd.constant(0.0, self.dummy_grids())

        return SumProd(terms, protected=sigs, grids=self.grids)

    def cancel_proportional_terms(self, zero_tol=0.0):
        """Targeted exact (up to round-off) cancellation of fully proportional terms.

        Unlike compress_hashed() this function canonicalizes every 1D factor up to a scalar.
        Don't use this in the global sum-of-products path, only for small special expressions,
        such as the rotational-frame determinant det(J).
        """
        pairs = [
            ([f.copy() for f in t], sig) for t, sig in zip(self.terms, self.protected)
        ]

        pairs = [
            (t, sig)
            for t, sig in pairs
            if not any(self.is_zero_factor(f, zero_tol) for f in t)
        ]

        if not pairs:
            return SumProd.constant(0.0, self.dummy_grids())

        terms, sigs = zip(*pairs)
        terms, sigs = self._cancel_proportional_terms(
            list(terms), list(sigs), zero_tol=zero_tol
        )

        if not terms:
            return SumProd.constant(0.0, self.dummy_grids())

        return SumProd(terms, protected=sigs, grids=self.grids)

    def collect_final_terms(self, zero_tol=0.0, identity_tol=1e-12, max_sweeps=None):
        """Performs final G-matrix compression for terms with the same protected factor.

        This is not used in the general sum-of-products path, but only for G-matrix:
        canonicalizes each 1D factor up to a scalar, combines fully proportional terms,
        and then exactly collects terms that are proportional in every coordinate except one.

        Terms carrying different analytical prefactors are never merged.
        """
        sop = self.compress_hashed(zero_tol=zero_tol, identity_tol=identity_tol)
        terms = [[np.asarray(f, dtype=float).copy() for f in t] for t in sop.terms]
        sigs = list(sop.protected)

        def canonicalize(term):
            canon = []
            coeff = 1.0

            for f in term:
                cf, sc = self.canonical_factor_scale(f)

                if sc == 0.0:
                    return None, 0.0

                canon.append(cf)
                coeff *= sc

            return canon, coeff

        def same_factor(a, b):
            return self.factors_close(a, b, tol=identity_tol)

        def cancel_fully_proportional(terms, sigs):
            groups = []  # (signature, canonical factors, accumulated coefficient)

            for term, sig in zip(terms, sigs):
                canon, coeff = canonicalize(term)

                if canon is None or abs(coeff) <= zero_tol:
                    continue

                found = False
                for g in groups:
                    if sig == g[0] and all(
                        same_factor(a, b) for a, b in zip(canon, g[1])
                    ):
                        g[2] += coeff
                        found = True
                        break

                if not found:
                    groups.append([sig, canon, coeff])

            out_t = []
            out_s = []
            for sig, canon, coeff in groups:
                if abs(coeff) <= zero_tol:
                    continue

                term = [f.copy() for f in canon]
                term[0] *= coeff

                out_t.append(term)
                out_s.append(sig)

            return out_t, out_s

        terms, sigs = cancel_fully_proportional(terms, sigs)
        if not terms:
            return SumProd.constant(0.0, self.dummy_grids())

        sweep = 0
        changed = True

        while changed and (max_sweeps is None or sweep < max_sweeps):
            changed = False
            sweep += 1

            for k in range(self.ncoo):
                used = [False] * len(terms)
                new_terms = []
                new_sigs = []

                for i in range(len(terms)):
                    if used[i]:
                        continue

                    used[i] = True
                    base, sig = terms[i], sigs[i]
                    cbase, _ = canonicalize(base)
                    group = [i]

                    for j in range(i + 1, len(terms)):
                        if used[j] or sigs[j] != sig:
                            continue

                        cj, _ = canonicalize(terms[j])

                        if all(
                            same_factor(cbase[m], cj[m])
                            for m in range(self.ncoo)
                            if m != k
                        ):
                            group.append(j)
                            used[j] = True

                    if len(group) == 1:
                        new_terms.append([f.copy() for f in base])
                        new_sigs.append(sig)
                        continue

                    # place all proportional scales of coordinates != k into k,
                    # e.g., a*b*c + a*b*d = a*b*(c+d)
                    ref = cbase
                    merged_k = np.zeros_like(base[k], dtype=float)
                    for idx in group:
                        cterm, coeff = canonicalize(terms[idx])
                        merged_k += coeff * cterm[k]

                    merged = [f.copy() for f in ref]
                    merged[k] = merged_k
                    merged[k], _ = self.simplify_1d_factor(merged[k], tol=identity_tol)

                    changed = True
                    if not self.is_zero_factor(merged[k], zero_tol):
                        new_terms.append(merged)
                        new_sigs.append(sig)

                terms, sigs = cancel_fully_proportional(new_terms, new_sigs)

                if not terms:
                    return SumProd.constant(0.0, self.dummy_grids())

        sop = SumProd(terms, protected=sigs, grids=self.grids).compress_hashed(
            zero_tol=zero_tol, identity_tol=identity_tol
        )
        return sop

    def collect_one_coordinate(self, zero_tol=0.0, max_sweeps=None, identity_tol=0.0):
        """Exactly collect terms that differ in only one coordinate.

        Separated from compress_hashed(), used for the rotational-frame determinant det(J).
        """
        sop = self.compress_hashed(zero_tol=zero_tol, identity_tol=identity_tol)

        terms = [[np.asarray(f, dtype=float).copy() for f in t] for t in sop.terms]
        sigs = list(sop.protected)

        if not terms:
            return SumProd.constant(0.0, self.dummy_grids())

        sweep = 0
        changed = True

        while changed and (max_sweeps is None or sweep < max_sweeps):
            changed = False
            sweep += 1

            for k in range(self.ncoo):
                groups = {}

                for term, sig in zip(terms, sigs):
                    work = [f.copy() for f in term]
                    scale = 1.0

                    key_factors = []

                    for j in range(self.ncoo):
                        if j == k:
                            continue

                        cf, sc = self.canonical_factor_scale(work[j])
                        work[j] = cf
                        key_factors.append(cf)

                        scale *= sc

                    work[k] = work[k] * scale
                    key = (sig,) + tuple(self.factor_key(f) for f in key_factors)

                    groups.setdefault(key, []).append((work, sig))

                new_terms, new_sigs = [], []
                for group in groups.values():
                    if len(group) == 1:
                        new_terms.append(group[0][0])
                        new_sigs.append(group[0][1])
                        continue

                    merged = [f.copy() for f in group[0][0]]
                    merged[k] = np.add.reduce([x[0][k] for x in group])
                    merged[k], _ = self.simplify_1d_factor(merged[k], tol=identity_tol)
                    changed = True

                    if not self.is_zero_factor(merged[k], zero_tol):
                        new_terms.append(merged)
                        new_sigs.append(group[0][1])

                terms, sigs = new_terms, new_sigs

                if not terms:
                    return SumProd.constant(0.0, self.dummy_grids())

            # newly collected terms may be proportional duplicates
            n_before = len(terms)
            terms, sigs = self._cancel_proportional_terms(
                terms, sigs, zero_tol=zero_tol
            )

            if len(terms) != n_before:
                changed = True

            if not terms:
                return SumProd.constant(0.0, self.dummy_grids())

        return SumProd(terms, protected=sigs, grids=self.grids)

    @classmethod
    def terms_equal_except_one(cls, term1, term2, k, tol=0.0):
        """Compare two sum-of-product terms in all coordinates except k"""
        return all(
            cls.factors_close(term1[j], term2[j], tol)
            for j in range(len(term1))
            if j != k
        )

    def compress_similar(
        self, tol=1e-10, zero_tol=0.0, max_sweeps=None, identity_tol=0.0
    ):
        # protected analytical factors are part of a term's identity, so to avoid approximate
        # merging across different signatures, use exact compression
        return self.compress_hashed(
            zero_tol=zero_tol, max_sweeps=max_sweeps, identity_tol=identity_tol
        )

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

        # fast exact compression
        sop = self.compress_hashed(
            zero_tol=zero_tol, max_sweeps=max_sweeps, identity_tol=identity_tol
        )

        # optional approximate compression
        if tol > 0.0:
            sop = sop.compress_similar(
                tol=tol,
                zero_tol=zero_tol,
                max_sweeps=max_sweeps,
                identity_tol=identity_tol,
            )

        return sop

    def __add__(self, x):
        x = self.as_sumprod(x)
        if x is NotImplemented:
            return NotImplemented
        return SumProd(
            self.terms + x.terms,
            protected=self.protected + x.protected,
            grids=self.grids or x.grids,
        ).compress()

    __radd__ = __add__

    def __neg__(self):
        sop = self.copy()
        for t in sop.terms:
            t[0] *= -1
        return sop

    def __sub__(self, x):
        x = self.as_sumprod(x)
        if x is NotImplemented:
            return NotImplemented
        return self + (-x)

    def __rsub__(self, x):
        x = self.as_sumprod(x)
        if x is NotImplemented:
            return NotImplemented
        return x + (-self)

    def __mul__(self, x):
        if np.isscalar(x):
            sop = self.copy()
            for t in sop.terms:
                t[0] *= float(x)
            return sop
        if not isinstance(x, SumProd):
            return NotImplemented

        terms = []
        protected = []
        for i, t1 in enumerate(self.terms):
            for j, t2 in enumerate(x.terms):
                terms.append([a * b for a, b in zip(t1, t2)])
                protected.append(
                    self.protected_signature(self.protected[i] + x.protected[j])
                )

        out = SumProd(terms, protected=protected, grids=self.grids or x.grids)

        # keep it cheap, propagate protected factors exactly by signature addition
        # and delay expensive numerical factor recognition until a complete G-matrix
        # element is available
        return out.compress()

    __rmul__ = __mul__

    def __pow__(self, p):
        if isinstance(p, (int, np.integer)):
            if p == 0:
                return SumProd.constant(1.0, self.dummy_grids())

            if p > 0:
                sop = SumProd.constant(1.0, self.dummy_grids())
                for _ in range(int(p)):
                    sop = sop * self
                return sop

            # negative integer power
            return sop_reciprocal(self ** (-int(p)), tol=RECIPROCAL_TOL)

        if np.isscalar(p):
            return sop_ufunc(
                self,
                (self.grids or self.dummy_grids()),
                lambda term: term**p,
                f"pow({p})",
            )

        return NotImplemented

    def __truediv__(self, x):
        if np.isscalar(x):
            if abs(x) <= RECIPROCAL_TOL:
                raise ZeroDivisionError(
                    f"Division by a value close to zero: {x}: |x| <= {RECIPROCAL_TOL}"
                )
            return self * (1.0 / float(x))

        if isinstance(x, SumProd):
            try:
                return self * sop_reciprocal(x, tol=RECIPROCAL_TOL)
            except NotImplementedError as exc:
                raise NotImplementedError(
                    "Division by a general sum-of-products requires a low-rank approximation."
                ) from exc

        return NotImplemented

    def __rtruediv__(self, x):
        if np.isscalar(x):
            return sop_reciprocal(self, tol=RECIPROCAL_TOL) * x
        return NotImplemented

    def __array_ufunc__(self, ufunc, method, *args, **kwargs):
        if method != "__call__" or kwargs.get("out") is not None:
            return NotImplemented

        if ufunc is np.add:
            a, b = args
            if a is self:
                return self.__add__(b)
            else:
                return self.__radd__(a)
        elif ufunc is np.subtract:
            a, b = args
            if a is self:
                return self.__sub__(b)
            else:
                return self.__rsub__(a)
        elif ufunc is np.multiply:
            a, b = args
            if a is self:
                return self.__mul__(b)
            else:
                return self.__rmul__(a)
        elif ufunc is np.divide:
            a, b = args
            if a is self:
                return self.__truediv__(b)
            else:
                return self.__rtruediv__(a)
        elif ufunc is np.power:
            a, b = args
            if a is self:
                return self.__pow__(b)
            return NotImplemented
        elif ufunc is np.negative:
            return self.__neg__()
        elif ufunc is np.positive:
            return self
        elif ufunc is np.sin:
            return sop_ufunc(self, (self.grids or self.dummy_grids()), np.sin, "sin")
        elif ufunc is np.cos:
            return sop_ufunc(self, (self.grids or self.dummy_grids()), np.cos, "cos")
        elif ufunc is np.tan:
            return sop_ufunc(self, (self.grids or self.dummy_grids()), np.tan, "tan")
        elif ufunc is np.sqrt:
            return sop_ufunc(self, (self.grids or self.dummy_grids()), np.sqrt, "sqrt")
        elif ufunc is np.exp:
            return sop_ufunc(self, (self.grids or self.dummy_grids()), np.exp, "exp")
        elif ufunc is np.log:
            return sop_ufunc(self, (self.grids or self.dummy_grids()), np.log, "log")
        elif ufunc is np.arccos:
            return sop_ufunc(
                self, (self.grids or self.dummy_grids()), np.arccos, "arccos"
            )
        elif ufunc is np.arcsin:
            return sop_ufunc(
                self, (self.grids or self.dummy_grids()), np.arcsin, "arcsin"
            )
        elif ufunc is np.arctan:
            return sop_ufunc(
                self, (self.grids or self.dummy_grids()), np.arctan, "arctan"
            )
        else:
            return NotImplemented


def init_internal(grids):
    ncoo = len(grids)
    sop = [SumProd.init_coordinate(icoo, grids) for icoo in range(ncoo)]
    return sop


def check_sop_unary(sop: SumProd):
    """Classify sum-of-products by its total coordinate dependence"""
    varying = set()

    for term in sop.terms:
        term_varying = [icoo for icoo, f in enumerate(term) if not np.all(f == f[0])]

        if len(term_varying) > 1:
            return ("depends_on_more_than_one_coo", None, None)
        varying.update(term_varying)

        if len(varying) > 1:
            return ("depends_on_more_than_one_coo", None, None)

    if not varying:
        value = 0.0
        for term in sop.terms:
            value += np.prod([f[0] for f in term])
        return ("const", None, np.array([value], dtype=float))

    icoo = next(iter(varying))
    values = np.zeros(sop.npoints[icoo], dtype=float)
    for term in sop.terms:
        scale = np.prod([f[0] for j, f in enumerate(term) if j != icoo])
        values += scale * term[icoo]
    return ("depends_on_one_coo", icoo, values)


def sop_ufunc(sop: SumProd, grids, func, func_name):
    kind, icoo, terms = check_sop_unary(sop)
    if kind in ("sum_of_more_than_one_term", "depends_on_more_than_one_coo"):
        raise NotImplementedError(
            f"{func_name} of a multi-term sum-of-products requires a low-rank approximation."
        )
    elif kind == "const":
        return SumProd.constant(float(func(terms[0])), grids)
    else:
        return SumProd.init_coordinate(icoo, grids, lambda _: func(terms))


def recognize_protected(values, grid, tol=1e-10):
    """Recognizes if a 1D array is actually a simple analytical function,
    such as const * q^p, sin(q)^p, or cos(q)^p.
    """
    values = np.asarray(values, dtype=float)
    q = np.asarray(grid, dtype=float)
    candidates = (("q", q), ("sin", np.sin(q)), ("cos", np.cos(q)))

    # integer powers
    powers = tuple(float(p) for p in range(1, 13)) + (0.5,)

    for kind, base in candidates:
        for power in powers:
            with np.errstate(invalid="ignore", over="ignore"):
                bp = base**power

            if not np.all(np.isfinite(bp)):
                continue

            # determine an arbitrary constant scale without dividing at zeros
            mask = np.abs(bp) > tol
            if np.any(mask):
                denom = float(np.dot(bp[mask], bp[mask]))
                scale = float(np.dot(values[mask], bp[mask]) / denom)

                if abs(scale) <= tol:
                    continue

                if np.allclose(values, scale * bp, rtol=tol, atol=tol):
                    return kind, power, scale

            elif np.allclose(values, 0.0, rtol=tol, atol=tol):
                continue

    return None


def sop_reciprocal(
    sop: SumProd, tol: float = 1e-9, trace_coordinates=None, trace_kinds=None
):
    """Exact reciprocal when representable in the supported sum-of-products algebra.

    With singularity tracing enabled, recognized 1D factors are inverted analytically
    and moved into 'protected' metadata before any numerical reciprocal is evaluated.
    """
    sop = sop.compress_hashed()

    kind, icoo, values = check_sop_unary(sop)
    if kind == "const":
        value = float(values[0])
        if abs(value) <= tol:
            raise ZeroDivisionError(f"Division by a value close to zero: {value}")

        return SumProd.constant(1.0 / value, sop.grids or sop.dummy_grids())

    if trace_kinds is None:
        trace_kinds = _TRACE_KINDS

    if trace_coordinates is None:
        trace_coordinates = _TRACE_COORDINATES

    trace_set = None if trace_coordinates is None else set(trace_coordinates)

    # simple one-coordinate expression, protect it before pointwise inversion
    if (
        _TRACE_SINGULARITIES
        and kind == "depends_on_one_coo"
        and sop.grids is not None
        and (trace_set is None or icoo in trace_set)
    ):
        rec = recognize_protected(values, sop.grids[icoo])

        if rec is not None:
            pkind, power, scale = rec

            if pkind in trace_kinds:
                fs = [np.ones(n, float) for n in sop.npoints]
                fs[icoo] *= 1.0 / scale

                return SumProd(
                    [fs], protected=[((icoo, pkind, -power),)], grids=sop.grids
                )

    if kind == "depends_on_one_coo":
        if np.any(np.abs(values) <= tol):
            raise ZeroDivisionError(
                f"Division by a value close to zero: min(abs(f)) = "
                f"{np.min(np.abs(values))} <= tolerance {tol}"
            )

        return SumProd.init_coordinate(
            icoo, sop.grids or sop.dummy_grids(), lambda _: 1.0 / values
        )

    if sop.nterms != 1:
        raise NotImplementedError(
            "Reciprocal of a multivariate multi-term sum-of-products requires a low-rank approximation."
        )

    term = sop.terms[0]
    grids = sop.grids
    fs = []

    # invert already protected factors first
    invsig = [(k, pkind, -p) for k, pkind, p in sop.protected[0]]

    for k, f in enumerate(term):
        can_trace = (
            _TRACE_SINGULARITIES
            and grids is not None
            and (trace_set is None or k in trace_set)
        )

        rec = recognize_protected(f, grids[k]) if can_trace else None

        if rec is not None and rec[0] in trace_kinds:
            pkind, power, scale = rec

            if abs(scale) <= tol:
                raise ZeroDivisionError(
                    f"Cannot invert zero scale in coordinate factor {k}"
                )

            # 1 / (const * x^p) = (1/const) * x^{-p},
            # keep the constant numerically and the coordinate power analytically
            fs.append(
                np.full_like(np.asarray(f, dtype=float), 1.0 / scale, dtype=float)
            )

            invsig.append((k, pkind, -power))
            continue

        if np.any(np.abs(f) <= tol):
            raise ZeroDivisionError(
                f"Division by a value close to zero in coordinate factor {k}: "
                f"min(abs(f)) = {np.min(np.abs(f))} <= tolerance {tol}. "
                "The zero factor was not recognized as a supported protected q/sin/cos power."
            )

        fs.append(1.0 / f)

    return SumProd([fs], protected=[tuple(invsig)], grids=grids)
