import numpy as np


class SumProd:

    def __init__(self, terms):
        self.terms = [[np.copy(np.array(f, dtype=float)) for f in t] for t in terms]

        # check shapes
        ncoo = len(self.terms[0])
        npoints = tuple(len(f) for f in self.terms[0])
        for t in self.terms:
            if len(t) != ncoo or tuple(len(f) for f in t) != npoints:
                raise ValueError("Inconsistent shapes of terms in SOP")

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
            return NotImplemented  # can actually implement if function depends only on one coordinate


def init_internal(grids):
    ncoo = len(grids)
    sop = [SumProd.init_coordinate(icoo, grids) for icoo in range(ncoo)]
    return sop
