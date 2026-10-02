import numpy as np

# Performs Cartesian-to-internal pass 'r->q' for building dq_k/dr_{ia} derivatives,
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

        # numpy scalar constants
        if ufunc is np.add and len(args) == 2:
            return Expr("add", (as_expr(args[0]), as_expr(args[1])))
        if ufunc is np.subtract and len(args) == 2:
            return Expr("add", (as_expr(args[0]), Expr("neg", (as_expr(args[1]),))))
        if ufunc is np.multiply and len(args) == 2:
            return Expr("mul", (as_expr(args[0]), as_expr(args[1])))
        if ufunc in (np.divide, np.true_divide) and len(args) == 2:
            return Expr("div", (as_expr(args[0]), as_expr(args[1])))
        if ufunc is np.negative and len(args) == 1:
            return Expr("neg", (as_expr(args[0]),))
        if ufunc is np.positive and len(args) == 1:
            return as_expr(args[0])
        if ufunc is np.power and len(args) == 2 and np.isscalar(args[1]):
            return Expr("pow", (as_expr(args[0]),), float(args[1]))

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
            np.arctan2: "arctan2",
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
    return Expr("x_coo", data=name)


def trace(natoms, ncoo, cartesian_to_internal):
    _geom_cache.clear()
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


# Geometry primitive cache.
# It is cleared at the start of each trace so that repeated distances and angles
# created while building one coordinate graph share the same Expr nodes.
# This sharing is important because Gvib can then bind those nodes directly
# to the corresponding one-dimensional internal grids.
_geom_cache = {}


def _geom_key(name, *pts):
    return (name,) + tuple(p.id for p in pts)


def _point_xyz(p):
    p = np.asarray(p, dtype=object).ravel()
    if len(p) != 3 or not all(isinstance(x, Expr) for x in p):
        raise TypeError("geometry primitives require 3-vectors of Expr objects")
    return p


def distance(a, b):
    """Distance |a-b| as an atomic geometry primitive"""
    a, b = _point_xyz(a), _point_xyz(b)
    # convention for the endpoint order for correct caching
    if tuple(x.id for x in a) > tuple(x.id for x in b):
        a, b = b, a
    pts = tuple(a) + tuple(b)
    key = _geom_key("distance", *pts)
    if key not in _geom_cache:
        _geom_cache[key] = Expr("distance", pts)
    return _geom_cache[key]


def angle(a, b, c):
    """Angle a-b-c as an atomic geometry primitive"""
    a, b, c = _point_xyz(a), _point_xyz(b), _point_xyz(c)
    # convention for the out atoms for correct caching
    if tuple(x.id for x in a) > tuple(x.id for x in c):
        a, c = c, a
    pts = tuple(a) + tuple(b) + tuple(c)
    key = _geom_key("angle", *pts)
    if key not in _geom_cache:
        _geom_cache[key] = Expr("angle", pts)
    return _geom_cache[key]


def dihedral(a, b, c, d):
    """Signed dihedral a-b-c-d as an atomic geometry primitive.
    The derivative convention is the standard signed torsion about b-c.
    """
    a, b, c, d = map(_point_xyz, (a, b, c, d))
    pts = tuple(a) + tuple(b) + tuple(c) + tuple(d)
    key = _geom_key("dihedral", *pts)
    if key not in _geom_cache:
        _geom_cache[key] = Expr("dihedral", pts)
    return _geom_cache[key]


def _same_point(a, b):
    """True when two traced 3-vectors represent the same Cartesian point."""
    a, b = _point_xyz(a), _point_xyz(b)
    return tuple(x.id for x in a) == tuple(x.id for x in b)


def _off_axis_point(plane, p, q):
    """Return the point of a 3-point plane that is not one of axis endpoints p, q"""
    plane = tuple(_point_xyz(x) for x in plane)
    p, q = _point_xyz(p), _point_xyz(q)
    has_p = any(_same_point(x, p) for x in plane)
    has_q = any(_same_point(x, q) for x in plane)
    if not (has_p and has_q):
        raise ValueError(
            "signed_plane_angle requires the directed axis endpoints to belong to both planes"
        )
    off = [x for x in plane if not _same_point(x, p) and not _same_point(x, q)]
    if len(off) != 1:
        raise ValueError("each plane must contain exactly one point off the axis")
    return off[0]


def signed_plane_angle(plane1, plane2, axis):
    """Signed angle between two planes about a directed common axis as an atomic geometry primitive.

    The reverse AD rule uses the exact sum-of-product-friendly dihedral gradient
    rather than expanding atan2/cross/norm.
    """
    if len(plane1) != 3 or len(plane2) != 3 or len(axis) != 2:
        raise ValueError("expected plane1=(A,B,C), plane2=(D,E,F), axis=(P,Q)")
    P, Q = map(_point_xyz, axis)
    R1 = _off_axis_point(plane1, P, Q)
    R2 = _off_axis_point(plane2, P, Q)
    pts = tuple(R1) + tuple(P) + tuple(Q) + tuple(R2)
    key = _geom_key("signed_plane_angle", *pts)
    if key not in _geom_cache:
        _geom_cache[key] = Expr("signed_plane_angle", pts)
    return _geom_cache[key]


def _vsub(a, b):
    return np.array([a[i] - b[i] for i in range(3)], dtype=object)


def _vscale(s, v):
    return np.array([s * v[i] for i in range(3)], dtype=object)


def _vadd(a, b):
    return np.array([a[i] + b[i] for i in range(3)], dtype=object)


def _add_vec(add, point, vec, grad):
    for i in range(3):
        add(point[i], grad * vec[i])


def expr_signature(node):
    """Structural signature used to recognize algebraically identical traced subexpressions"""
    if not isinstance(node, Expr):
        return ("literal", repr(node))

    if node.oper == "const":
        return ("const", float(node.data))

    if node.oper == "x_coo":
        return ("x_coo", node.data)

    sigs = [expr_signature(a) for a in node.args]
    if node.oper in ("add", "mul"):
        sigs = sorted(sigs, key=repr)

    return (node.oper, node.data, tuple(sigs))


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
            # node = sqrt(x), so reuse the output d/dx sqrt(x) = 1/(2*node)
            add(args[0], grad / (2 * node))
        elif node.oper == "sin":
            add(args[0], grad * np.cos(args[0]))
        elif node.oper == "cos":
            add(args[0], -grad * np.sin(args[0]))
        elif node.oper == "tan":
            # node = tan(x), so reuse the output: d tan(x)/dx = 1 + node^2.
            add(args[0], grad * (1.0 + node * node))
        elif node.oper == "exp":
            add(args[0], grad * node)
        elif node.oper == "log":
            add(args[0], grad / args[0])
        elif node.oper == "arccos":
            add(args[0], -grad / np.sin(node))
        elif node.oper == "arcsin":
            add(args[0], grad / np.cos(node))
        elif node.oper == "arctan":
            # node = arctan(x), so reuse the output d atan(x)/dx = cos(node)^2
            c = np.cos(node)
            add(args[0], grad * c * c)
        elif node.oper == "arctan2":
            y, x = args
            denom = x * x + y * y
            add(y, grad * x / denom)
            add(x, -grad * y / denom)
        elif node.oper == "distance":
            A = np.array(args[0:3], dtype=object)
            B = np.array(args[3:6], dtype=object)
            u = _vsub(A, B)
            # reuse the output |A-B|; if this distance is an internal coordinate,
            # q_bindings turns 1/node into a 1D factor
            gA = _vscale(1.0 / node, u)
            _add_vec(add, A, gA, grad)
            _add_vec(add, B, _vscale(-1.0, gA), grad)
        elif node.oper == "angle":
            A = np.array(args[0:3], dtype=object)
            B = np.array(args[3:6], dtype=object)
            C = np.array(args[6:9], dtype=object)
            u = _vsub(A, B)
            v = _vsub(C, B)
            ru = distance(A, B)
            rv = distance(C, B)
            # reuse node, ru and rv; if these are internal coordinates,
            # all denominators will be univariate
            ct = np.cos(node)
            st = np.sin(node)
            gu = np.array(
                [(ct * u[i] / ru - v[i] / rv) / (ru * st) for i in range(3)],
                dtype=object,
            )
            gv = np.array(
                [(ct * v[i] / rv - u[i] / ru) / (rv * st) for i in range(3)],
                dtype=object,
            )
            _add_vec(add, A, gu, grad)
            _add_vec(add, C, gv, grad)
            _add_vec(add, B, _vscale(-1.0, _vadd(gu, gv)), grad)
        elif node.oper in ("dihedral", "signed_plane_angle"):
            A = np.array(args[0:3], dtype=object)
            B = np.array(args[3:6], dtype=object)
            C = np.array(args[6:9], dtype=object)
            D = np.array(args[9:12], dtype=object)
            b1 = _vsub(B, A)
            b2 = _vsub(C, B)
            b3 = _vsub(D, C)
            n1 = cross(b1, b2)
            n2 = cross(b2, b3)

            # keep all factors as atomic geometry primitives,
            # which are themselves internal coordinates (at least for Z-matrix coo);
            # q_bindings in Gmat will replace them by one-dimensional SOP factors
            # before any Cartesian norm or sqrt is evaluated
            r12 = distance(A, B)
            r23 = distance(C, B)
            th1 = angle(A, B, C)

            # second-plane normal n2=(C-B)x(D-C) can be obtained
            # from either triangle representation of plane B-C-D:
            # |n2|^2 = BC^2 BD^2 sin(angle(C,B,D))^2
            # or
            # |n2|^2 = BC^2 CD^2 sin(angle(B,C,D))^2

            # prefer representation whose distance and angle nodes already exist
            # in the geometry cache (i.e. are themselves internal coordinates)
            def _cached_distance(P, Q):
                P, Q = _point_xyz(P), _point_xyz(Q)
                if tuple(x.id for x in P) > tuple(x.id for x in Q):
                    P, Q = Q, P
                key = _geom_key("distance", *(tuple(P) + tuple(Q)))
                return _geom_cache.get(key)

            def _cached_angle(P, Q, R):
                P, Q, R = _point_xyz(P), _point_xyz(Q), _point_xyz(R)
                if tuple(x.id for x in P) > tuple(x.id for x in R):
                    P, R = R, P
                key = _geom_key("angle", *(tuple(P) + tuple(Q) + tuple(R)))
                return _geom_cache.get(key)

            rBD = _cached_distance(B, D)
            thB = _cached_angle(C, B, D)
            rCD = _cached_distance(C, D)
            thC = _cached_angle(B, C, D)

            if rBD is not None and thB is not None:
                n2sq = (r23 * r23) * (rBD * rBD) * np.sin(thB) * np.sin(thB)
            elif rCD is not None and thC is not None:
                n2sq = (r23 * r23) * (rCD * rCD) * np.sin(thC) * np.sin(thC)
            else:
                # make compatible with user-defined transforms:
                # create the primitives and let local structural aliasing
                # in Gmat match them to norm or arccos coordinates if possible
                rBD = distance(B, D)
                thB = angle(C, B, D)
                n2sq = (r23 * r23) * (rBD * rBD) * np.sin(thB) * np.sin(thB)

            # first-plane normal is obtained form direct A-B-C quantities
            n1sq = (r12 * r12) * (r23 * r23) * np.sin(th1) * np.sin(th1)
            # standard signed-dihedral endpoint gradients for b1=B-A, b2=C-B, b3=D-C
            gA = _vscale(-r23 / n1sq, n1)
            gD = _vscale(r23 / n2sq, n2)
            c1 = dot(b1, b2) / (r23 * r23)
            c2 = dot(b3, b2) / (r23 * r23)
            gB = _vadd(_vscale(-(1.0 + c1), gA), _vscale(c2, gD))
            gC = _vscale(-1.0, _vadd(_vadd(gA, gB), gD))
            _add_vec(add, A, gA, grad)
            _add_vec(add, B, gB, grad)
            _add_vec(add, C, gC, grad)
            _add_vec(add, D, gD, grad)
        else:
            raise NotImplementedError(f"{node.oper}")

    return {
        node.data: adj.get(node.id, as_expr(0.0))
        for node in order
        if node.oper == "x_coo"
    }
