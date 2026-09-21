import numpy as np

# Performs internal-to-Cartesian pass 'q->r' for building sum-of-product form,
# where 'q' are internal coordinates and 'r' are Cartesian

# TODO: add forward-mode AD for consistency check between user's input internal-to-Cartesian and Cartesian-to-internal functions

RECIPROCAL_TOL = 1e-9
COMPRESS_THRESH = 100


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
    def factor_key(f, hash_tol=0.0):
        """Hash key for a 1D sum-of-product factor"""
        a = np.ascontiguousarray(f, dtype=float)

        if hash_tol <= 0.0:
            a = a.copy()
            a[a == 0.0] = 0.0
            return (a.shape, a.dtype.str, a.tobytes())

        q = np.rint(a / hash_tol).astype(np.int64)

        return (q.shape, q.dtype.str, q.tobytes())

    @staticmethod
    def factors_similar(f, g, tol):
        if tol <= 0.0:
            return np.array_equal(f, g)
        return np.max(np.abs(np.asarray(f) - np.asarray(g))) <= tol

    @staticmethod
    def is_zero_factor(f, zero_tol=0.0):
        if zero_tol == 0.0:
            return np.all(f == 0.0)
        return np.all(np.abs(f) <= zero_tol)

    def compress_hashed(self, zero_tol=0.0, hash_tol=0.0, max_sweeps=None):
        """Exactly combine sum-of-product terms using hash grouping

        For each coordinate k, terms that are identical in every factor except k are grouped
        and their k-th factors are added:

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
                        self.factor_key(term[j], hash_tol)
                        for j in range(self.ncoo)
                        if j != k
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

    def compress(self, thresh=COMPRESS_THRESH, zero_tol=0.0):
        """Run exact hashed compression when the number of terms grows large"""
        if self.nterms >= thresh:
            return self.compress_hashed(zero_tol=zero_tol)
        return self

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
        if ufunc is np.sin:
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
    if sop.nterms != 1:
        return ("sum_of_more_than_one_term", None, None)
    else:
        terms = sop.terms[0]
        varying_coo = [icoo for icoo, f in enumerate(terms) if not np.allclose(f, f[0])]
        if len(varying_coo) == 0:
            scaling = np.prod([f[0] for f in terms])
            return ("const", None, np.array([scaling]))
        elif len(varying_coo) == 1:
            icoo = varying_coo[0]
            scaling = np.prod([f[0] for jcoo, f in enumerate(terms) if jcoo != icoo])
            return ("depends_on_one_coo", icoo, scaling * terms[icoo])
        else:
            return ("depends_on_more_than_one_coo", None, None)


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
    if sop.nterms != 1:
        raise NotImplementedError(
            "Reciprocal of a multi-term sum-of-products requires a low-rank approximation."
        )

    term = sop.terms[0]
    fs = []

    for f in term:
        f = np.asarray(f)

        mask = np.abs(f) <= tol
        if np.any(mask):
            raise ZeroDivisionError(
                f"Division by a value close to zero: min(abs(f)) = {np.min(np.abs(f))} < tolerance {tol}"
            )

        fs.append(1.0 / f)

    return SumProd([fs])
