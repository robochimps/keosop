from itertools import combinations

import numpy as np

# Performs internal-to-Cartesian pass 'q->r' for building sum-of-product form,
# where 'q' are internal coordinates and 'r' are Cartesian coordinate

# TODO: add forward-mode AD for consistency check between user's input internal-to-Cartesian and Cartesian-to-internal functions

RECIPROCAL_TOL = 1e-9
COMPRESS_NTERMS_THRESH = 100


class SumProd:

    def __init__(self, terms):
        self.terms = [[np.copy(np.array(f, dtype=float)) for f in t] for t in terms]

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

    @property
    def npoints(self):
        return tuple(len(f) for f in self.terms[0])

    @property
    def ncoo(self):
        return len(self.terms[0])

    @property
    def nterms(self):
        return len(self.terms)

    @classmethod
    def init_coordinate(cls, icoo, grids, func=None):
        f = [np.ones(len(g), dtype=float) for g in grids]
        v = np.asarray(grids[icoo], dtype=float)
        f[icoo] = np.copy(np.asarray(func(v) if func else v, dtype=float))
        return cls([f])

    @classmethod
    def constant(cls, c, grids):
        f = [np.ones(len(g), dtype=float) for g in grids]
        f[0] *= float(c)
        return cls([f])

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
        return SumProd(self.terms)

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

    def nmode_project(self, max_order, ref_indices, zero_tol=0.0, compress=True):
        """Return the exact truncated N-mode projection of sum-of-products

        For every product P(q)=prod_k f_k(q_k), write

            f_k(q_k) = c_k + d_k(q_k),  c_k=f_k(q_k^0),

        where q_k^0 is selected by `ref_indices[k]`.
        Expanding the product and retaining subsets containing at most `max_order`
        d-factors gives the truncated N-mode expansion.
        Therefore, every returned nonconstant term contains at most `max_order`
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

        for term in self.terms:
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

        if not new_terms:
            return SumProd.constant(0.0, self.dummy_grids())

        out = SumProd(new_terms)

        return out.compress_hashed(zero_tol=zero_tol) if compress else out

    def nmode_mul(self, x, max_order, ref_indices, zero_tol=0.0, compress=True):
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

        for ta, aa in a_data:
            for tb, ab in b_data:
                if len(aa | ab) > max_order:
                    continue

                terms.append([fa * fb for fa, fb in zip(ta, tb)])

        if not terms:
            return SumProd.constant(0.0, self.dummy_grids())

        out = SumProd(terms)

        return out.compress_hashed(zero_tol=zero_tol) if compress else out

    @staticmethod
    def factor_key(f):
        """Hash key for a 1D sum-of-product factor"""
        a = np.ascontiguousarray(f, dtype=float).copy()

        a[a == 0.0] = 0.0
        return (a.shape, a.dtype.str, a.tobytes())

    @staticmethod
    def is_zero_factor(f, zero_tol=0.0):
        if zero_tol == 0.0:
            return np.all(f == 0.0)
        return np.all(np.abs(f) <= zero_tol)

    @staticmethod
    def factors_close(f1, f2, tol=0.0):
        """Compare two 1D sum-of-product factors"""
        f1 = np.asarray(f1)
        f2 = np.asarray(f2)

        if f1.shape != f2.shape:
            return False

        if tol <= 0.0:
            return np.array_equal(f1, f2)

        # return np.max(np.abs(f1 - f2)) <= tol
        return np.all(np.abs(f1 - f2) <= tol)

    def compress_hashed(self, zero_tol=0.0, max_sweeps=None):
        """Exactly combine sum-of-product terms using hash grouping

        For each coordinate k, terms that are identical in every factor except k
        are grouped and their k-th factors are added:

            f1(q_k) P(q_{l!=k}) + f2(q_k) P(q_{l!=k}) -> [f1(q_k) + f2(q_k)] P(q_{l!=k})
        """
        terms = [[f.copy() for f in t] for t in self.terms]

        # any product containing zero factor is a zero term
        terms = [
            t for t in terms if not any(self.is_zero_factor(f, zero_tol) for f in t)
        ]
        if not terms:
            return SumProd.constant(0.0, self.dummy_grids())

        sweep = 0
        changed = True

        while changed and (max_sweeps is None or sweep < max_sweeps):
            changed = False
            sweep += 1

            for k in range(self.ncoo):
                groups = {}
                for term in terms:

                    key = tuple(
                        self.factor_key(term[j]) for j in range(self.ncoo) if j != k
                    )

                    groups.setdefault(key, []).append(term)

                new_terms = []
                for group in groups.values():

                    if len(group) == 1:
                        new_terms.append(group[0])
                        continue

                    merged = [f.copy() for f in group[0]]
                    merged[k] = np.add.reduce([t[k] for t in group])
                    changed = True

                    if not self.is_zero_factor(merged[k], zero_tol):
                        new_terms.append(merged)

                terms = new_terms
                if not terms:
                    return SumProd.constant(0.0, self.dummy_grids())

        return SumProd(terms)

    @classmethod
    def terms_equal_except_one(cls, term1, term2, k, tol=0.0):
        """Compare two sum-of-product terms in all coordinates except k"""
        return all(
            cls.factors_close(term1[j], term2[j], tol)
            for j in range(len(term1))
            if j != k
        )

    def compress_similar(self, tol=1e-10, zero_tol=0.0, max_sweeps=None):
        """Approximately combine sum-of-product terms using O(N^2) grouping"""
        terms = [[f.copy() for f in t] for t in self.terms]

        # any product containing zero factor is a zero term
        terms = [
            t for t in terms if not any(self.is_zero_factor(f, zero_tol) for f in t)
        ]

        if not terms:
            return SumProd.constant(0.0, self.dummy_grids())

        sweep = 0
        changed = True

        while changed and (max_sweeps is None or sweep < max_sweeps):
            changed = False
            sweep += 1

            for k in range(self.ncoo):

                used = np.zeros(len(terms), dtype=bool)
                new_terms = []

                for i in range(len(terms)):
                    if used[i]:
                        continue

                    merged = [f.copy() for f in terms[i]]
                    used[i] = True

                    for j in range(i + 1, len(terms)):
                        if used[j]:
                            continue

                        if self.terms_equal_except_one(terms[i], terms[j], k, tol):
                            merged[k] += terms[j][k]
                            used[j] = True
                            changed = True

                    if not self.is_zero_factor(merged[k], zero_tol):
                        new_terms.append(merged)

                terms = new_terms

                if not terms:
                    return SumProd.constant(0.0, self.dummy_grids())

        return SumProd(terms)

    def compress(
        self,
        tol=0.0,
        zero_tol=0.0,
        max_sweeps=None,
        nterms_thresh=COMPRESS_NTERMS_THRESH,
    ):
        if self.nterms < nterms_thresh:
            return self

        # fast exact compression
        sop = self.compress_hashed(zero_tol=zero_tol, max_sweeps=max_sweeps)

        # optional approximate compression
        if tol > 0.0:
            sop = sop.compress_similar(
                tol=tol, zero_tol=zero_tol, max_sweeps=max_sweeps
            )

        return sop

    def __add__(self, x):
        x = self.as_sumprod(x)
        if x is NotImplemented:
            return NotImplemented
        return SumProd(self.terms + x.terms).compress()

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
        return SumProd(
            [[a * b for a, b in zip(t1, t2)] for t1 in self.terms for t2 in x.terms]
        ).compress()

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
                self, self.dummy_grids(), lambda term: term**p, f"pow({p})"
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
            return sop_ufunc(self, self.dummy_grids(), np.sin, "sin")
        elif ufunc is np.cos:
            return sop_ufunc(self, self.dummy_grids(), np.cos, "cos")
        elif ufunc is np.tan:
            return sop_ufunc(self, self.dummy_grids(), np.tan, "tan")
        elif ufunc is np.sqrt:
            return sop_ufunc(self, self.dummy_grids(), np.sqrt, "sqrt")
        elif ufunc is np.exp:
            return sop_ufunc(self, self.dummy_grids(), np.exp, "exp")
        elif ufunc is np.log:
            return sop_ufunc(self, self.dummy_grids(), np.log, "log")
        elif ufunc is np.arccos:
            return sop_ufunc(self, self.dummy_grids(), np.arccos, "arccos")
        elif ufunc is np.arcsin:
            return sop_ufunc(self, self.dummy_grids(), np.arcsin, "arcsin")
        elif ufunc is np.arctan:
            return sop_ufunc(self, self.dummy_grids(), np.arctan, "arctan")
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


def sop_reciprocal(sop: SumProd, tol: float = 1e-9):
    # combine exactly compatible terms, may be it will collapse
    # identities generated by the AD graph before inversion
    sop = sop.compress_hashed()

    # a multi-term sum-of-product can still be inverted exactly
    # if the whole expression depends on only one coordinate
    kind, icoo, values = check_sop_unary(sop)
    if kind == "const":
        value = float(values[0])
        if abs(value) <= tol:
            raise ZeroDivisionError(f"Division by a value close to zero: {value}")
        return SumProd.constant(1.0 / value, sop.dummy_grids())

    if kind == "depends_on_one_coo":
        if np.any(np.abs(values) <= tol):
            raise ZeroDivisionError(
                f"Division by a value close to zero: min(abs(f)) = {np.min(np.abs(values))} <= tolerance {tol}"
            )
        return SumProd.init_coordinate(icoo, sop.dummy_grids(), lambda _: 1.0 / values)

    if sop.nterms != 1:
        raise NotImplementedError(
            "Reciprocal of a multivariate multi-term sum-of-products requires a low-rank approximation."
        )

    term = sop.terms[0]
    fs = []
    for f in term:
        f = np.asarray(f)

        if np.any(np.abs(f) <= tol):
            raise ZeroDivisionError(
                f"Division by a value close to zero: min(abs(f)) = {np.min(np.abs(f))} <= tolerance {tol}"
            )

        fs.append(1.0 / f)

    return SumProd([fs])
