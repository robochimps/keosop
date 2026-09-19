import numpy as np


class SumProd:

    def __init__(self, terms):
        self.terms = [[np.copy(np.array(f, dtype=float)) for f in t] for t in terms]

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

    def check_type(self, x):
        if isinstance(x, SumProd):
            return x
        elif np.isscalar(x):
            return SumProd.constant(x, self.dummy_grids())
        else:
            return NotImplemented

    def copy(self):
        return SumProd(self.terms)

    def __add__(self, x):
        x = self.check_type(x)
        if x is NotImplemented:
            return NotImplemented
        else:
            return SumProd(self.terms + x.terms)

    def __neg__(self):
        sop = self.copy()
        for t in sop.terms:
            t[0] *= -1
        return sop

    def __sub__(self, x):
        x = self.check_type(x)
        if x is NotImplemented:
            return NotImplemented
        else:
            return self + (-x)

    def __rsub__(self, x):
        x = self.check_type(x)
        if x is NotImplemented:
            return NotImplemented
        else:
            return x + (-self)

    def __mul__(self, x):
        if np.isscalar(x):
            sop = self.copy()
            for t in sop.terms:
                t[0] *= float(x)
                return sop
        elif not isinstance(x, SumProd):
            return NotImplemented
        else:
            sop = SumProd(
                [[a * b for a, b in zip(t1, t2)] for t1 in self.terms for t2 in x.terms]
            )
            return sop

    __rmul__ = __mul__

    def __pow__(self, p):
        if isinstance(p, (int, np.integer)) and p >= 0:
            sop = SumProd.constant(1.0, self.dummy_grids())
            for _ in range(int(p)):
                sop = sop * self
            return sop
        else:
            sop = apply_unary_func(
                self, self.dummy_grids(), lambda term: term**p, "pow"
            )

    def __truediv__(self, x):
        if np.isscalar(x):
            return self * (1.0 / float(x))
        else:
            raise TypeError(
                "Division by sum-of-products has no closed expression."
                "A low-rank approximation is required (to be implemented)"
            )

    def __array_ufunc__(self, ufunc, method, *args, **kwargs):
        if method != "__call__" or kwargs.get("out") is not None:
            return NotImplemented
        if ufunc is np.sin:
            return apply_unary_func(self, self.dummy_grids(), np.sin, "sin")
        elif ufunc is np.cos:
            return apply_unary_func(self, self.dummy_grids(), np.cos, "cos")
        elif ufunc is np.tan:
            return apply_unary_func(self, self.dummy_grids(), np.tan, "tan")
        elif ufunc is np.sqrt:
            return apply_unary_func(self, self.dummy_grids(), np.sqrt, "sqrt")
        elif ufunc is np.exp:
            return apply_unary_func(self, self.dummy_grids(), np.exp, "exp")
        elif ufunc is np.log:
            return apply_unary_func(self, self.dummy_grids(), np.log, "log")
        elif ufunc is np.arccos:
            return apply_unary_func(self, self.dummy_grids(), np.arccos, "arccos")
        elif ufunc is np.arcsin:
            return apply_unary_func(self, self.dummy_grids(), np.arcsin, "arcsin")
        elif ufunc is np.arctan:
            return apply_unary_func(self, self.dummy_grids(), np.arctan, "arctan")
        else:
            return NotImplemented


def check_if_sumprod_unary(sop: SumProd):
    if sop.nterms != 1:
        return ("sum_of_more_than_one_term", None, None)
    else:
        terms = sop.terms[0]
        varying_coo = [icoo for icoo, f in enumerate(terms) if not np.allclose(f, f[0])]
        if len(varying_coo) == 0:
            scaling = np.prod([f[0] for f in terms])
            return ("constant", None, np.array([scaling]))
        elif len(varying_coo) == 1:
            icoo = varying_coo[0]
            scaling = np.prod([f[0] for jcoo, f in enumerate(terms) if jcoo != icoo])
            return ("depends_on_one_coo", icoo, scaling * terms[icoo])
        else:
            return ("depends_on_more_than_one_coo", None, None)


def apply_unary_func(sop: SumProd, grids, func, func_name):
    kind, icoo, terms = check_if_sumprod_unary(sop)
    if kind in ("sum_of_more_than_one_term", "depends_on_more_than_one_coo"):
        raise RuntimeError(
            f"{func_name} of a multidimensional sum-of-product has no closed expression."
            f"A low-rank nonlinear approximation is required (to be implemented)"
        )
    elif kind == "constant":
        return SumProd.constant(float(func(terms[0])), grids)
    else:
        return SumProd.init_coordinate(icoo, grids, lambda _: func(terms))


def init_internal(grids):
    ncoo = len(grids)
    sop = [SumProd.init_coordinate(icoo, grids) for icoo in range(ncoo)]
    return sop
