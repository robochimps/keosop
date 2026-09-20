import numpy as np

# This is Cartesian-to-internal pass 'r->q' for building dq_k/dr_{ia} derivatives,
# where q_k (k=1..3N-6) are internal coordinates and r_{ia} are Cartesian (i=1..N, a=x,y,z)

expr_id = 0


def id_next():
    global expr_id
    expr_id += 1
    return expr_id


class Expr:
    def __init__(self, oper, args=(), data=None):
        self.oper = oper
        self.args = tuple(args)
        self.data = data
        self.id = id_next()

    def as_expr(self, x):
        if isinstance(x, Expr):
            return x
        elif np.isscalar(x):
            return Expr("const", data=float(x))
        else:
            return NotImplemented

    def __add__(self, x):
        return Expr("add", (self, self.as_expr(x)))

    def __radd__(self, x):
        return self.as_expr(x) + self

    def __neg__(self):
        return Expr("neg", (self,))

    def __sub__(self, x):
        return self + (-self.as_expr(x))

    def __rsub__(self, x):
        return self.as_expr(x) + (-self)

    def __mul__(self, x):
        return Expr("mul", (self, self.as_expr(x)))

    def __rmul__(self, x):
        return self.as_expr(x) * self

    def __truediv__(self, x):
        return Expr("div", (self, self.as_expr(x)))

    def __rtruediv__(self, x):
        return Expr("div", (self.as_expr(x), self))

    def __pow__(self, p):
        if not np.isscalar(p):
            return NotImplemented
        else:
            return Expr("pow", (self,), float(p))

    def __array_ufunc__(self, ufunc, method, *args, **kwargs):
        if method != "__call__" or kwargs.get("out") is not None:
            return NotImplemented
        x = tuple(self.as_expr(inp) for inp in args)
        ufunc_dict = {
            np.sin: "sin",
            np.cos: "cos",
            np.tan: "tan",
            np.sqrt: "sqrt",
            np.exp: "exp",
            np.log: "log",
            np.arccos: "arccos",
            np.arcsin: "arcsin",
            np.arctan: "arctan",
        }
        if ufunc in ufunc_dict:
            return Expr(ufunc_dict[ufunc], x)
        else:
            return NotImplemented


def init_cartesian(name):
    return Expr("coordinate", data=name)


def trace(natoms, ncoo, cartesian_to_internal):
    r = np.array(
        [[init_cartesian(f"r_{i}_{x}") for x in range(3)] for i in range(natoms)]
    )
    q_expr = cartesian_to_internal(r)
    if len(q_expr) != ncoo:
        raise ValueError(
            f"'cartesian_to_internal' output size = {len(q_expr)} does not match 'ncoo' = {ncoo}"
        )
    return q_expr


def dot(x, y):
    xx = np.asarray(x, dtype=object).ravel()
    yy = np.asarray(y, dtype=object).ravel()
    if len(xx) != len(yy):
        raise ValueError(
            f"In dot(x, y), size of 'x' = {len(xx)} does not match that of 'y' = {len(yy)}"
        )
    z = xx[0] * yy[0]
    for x, y in zip(xx[1:], yy[1:]):
        z = z + x * y
    return z


def sum(x):
    xx = np.asarray(x, dtype=object).ravel()
    if len(xx) == 0:
        return 0.0
    y = xx[0]
    for x in xx[1:]:
        y = y + x
    return y


def norm(x):
    return np.sqrt(dot(x, x))


def cross(x, y):
    x = np.asarray(x, dtype=object)
    y = np.asarray(y, dtype=object)
    return np.array(
        [
            x[1] * y[2] - x[2] * y[1],
            x[2] * y[0] - x[0] * y[2],
            x[0] * y[1] - x[1] * y[0],
        ],
        dtype=object,
    )
