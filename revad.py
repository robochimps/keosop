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

    def __add__(self, x):
        return Expr("add", (self, as_expr(x)))

    def __radd__(self, x):
        return as_expr(x) + self

    def __neg__(self):
        return Expr("neg", (self,))

    def __sub__(self, x):
        return self + (-as_expr(x))

    def __rsub__(self, x):
        return as_expr(x) + (-self)

    def __mul__(self, x):
        return Expr("mul", (self, as_expr(x)))

    def __rmul__(self, x):
        return as_expr(x) * self

    def __truediv__(self, x):
        return Expr("div", (self, as_expr(x)))

    def __rtruediv__(self, x):
        return Expr("div", (as_expr(x), self))

    def __pow__(self, p):
        if not np.isscalar(p):
            return NotImplemented
        else:
            return Expr("pow", (self,), float(p))

    def __array_ufunc__(self, ufunc, method, *args, **kwargs):
        if method != "__call__" or kwargs.get("out") is not None:
            return NotImplemented
        x = tuple(as_expr(inp) for inp in args)
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


def as_expr(x):
    if isinstance(x, Expr):
        return x
    elif np.isscalar(x):
        return Expr("const", data=float(x))
    else:
        return NotImplemented


def init_cartesian(name):
    return Expr("coordinate", data=name)


def trace(natoms, ncoo, cartesian_to_internal):
    r = np.array(
        [[init_cartesian(f"{x}{i}") for x in ("x", "y", "z")] for i in range(natoms)]
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


def topo_order(root):
    seen = set()
    order = []

    def visit(node):
        if node.id in seen:
            return
        else:
            seen.add(node.id)
            for arg in node.args:
                visit(arg)
            order.append(node)

    visit(root)
    return order


def reverse_derivatives(root):
    order = topo_order(root)
    adj = {root.id: as_expr(1.0)}

    def add(node, grad):
        adj[node.id] = adj[node.id] + grad if node.id in adj else grad

    for node in reversed(order):
        if node.id not in adj:
            continue
        args = node.args
        grad = adj[node.id]
        if node.oper in ("x_coo", "const"):
            pass
        elif node.oper == "add":
            add(args[0], grad)
            add(args[1], grad)
        elif node.oper == "neg":
            add(args[0], -grad)
        elif node.oper == "mul":
            add(args[0], grad * args[1])
            add(args[1], grad * args[0])
        elif node.oper == "div":
            add(args[0], grad / args[1])
            add(args[1], -grad * args[0] / (args[1] * args[1]))
        elif node.oper == "pow":
            p = node.data
            add(args[0], grad * p * (args[0] ** (p - 1)))
        elif node.oper == "sqrt":
            add(args[0], grad / (2 * node))
        elif node.op == "sin":
            add(args[0], grad * np.cos(args[0]))
        elif node.op == "cos":
            add(args[0], -grad * np.sin(args[0]))
        elif node.op == "tan":
            add(args[0], grad / (np.cos(args[0]) * np.cos(args[0])))
        elif node.op == "exp":
            add(args[0], grad * node)
        elif node.op == "log":
            add(args[0], grad / args[0])
        elif node.op == "acos":
            add(args[0], -grad / np.sin(node))
        elif node.op == "asin":
            add(args[0], grad / np.cos(node))
        elif node.op == "atan":
            add(args[0], grad * np.cos(node) * np.cos(node))
        else:
            return NotImplemented

    return {
        node.data: adj.get(node.id, as_expr(0.0))
        for node in order
        if node.oper == "x_coo"
    }
