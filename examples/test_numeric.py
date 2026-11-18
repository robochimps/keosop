import math

import jax
import numpy as np

from keosop.numerical.keo import G_to_invcm, batch_Gmat, com, pseudo


def _index_batches(sizes, npoints, batch_size, rng, full_grid):
    """Visit a small full grid, or sample a large grid one batch at a time"""
    order = rng.permutation(npoints) if full_grid else None
    for start in range(0, npoints, batch_size):
        count = min(batch_size, npoints - start)
        if full_grid:
            flat = order[start : start + count].copy()
            indices = np.empty((count, len(sizes)), dtype=int)
            for k in range(len(sizes) - 1, -1, -1):
                indices[:, k] = flat % sizes[k]
                flat //= sizes[k]
        else:
            indices = rng.integers(0, sizes, size=(count, len(sizes)))
        yield indices


def _compile_sop(sop, grids):
    """Fold constant factors and restore any protected analytical factors"""
    if np.isscalar(sop):
        return [(float(sop), [])]
    signatures = getattr(sop, "protected", None) or [()] * len(sop.terms)
    compiled = []
    for term, signature in zip(sop.terms, signatures):
        powers = {}
        for coordinate, kind, power in signature:
            powers.setdefault(coordinate, []).append((kind, power))
        coefficient, factors = 1.0, []
        for k, factor in enumerate(term):
            factor = np.asarray(factor, dtype=float)
            for kind, power in powers.get(k, ()):
                base = {"q": lambda x: x, "sin": np.sin, "cos": np.cos}[kind](grids[k])
                factor = factor * base**power
            if np.all(factor == factor[0]):
                coefficient *= factor[0]
            else:
                factors.append((k, factor))
        compiled.append((coefficient, factors))
    return compiled


def _compile_block(block, grids, cache):
    """Group repeated matrix entries so each distinct SOP is evaluated once"""
    array = np.asarray(block, dtype=object)
    groups = {}
    for index in np.ndindex(array.shape):
        sop = array[index]
        if id(sop) not in cache:
            cache[id(sop)] = (sop, _compile_sop(sop, grids))
        groups.setdefault(id(sop), []).append(index)
    return array.shape, [(entries, cache[key][1]) for key, entries in groups.items()]


def _evaluate_block(plan, indices, scale):
    shape, groups = plan
    result = np.empty((len(indices),) + shape)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        for entries, terms in groups:
            values = np.zeros(len(indices))
            for coefficient, factors in terms:
                product = np.full(len(indices), coefficient)
                for coordinate, factor in factors:
                    product *= factor[indices[:, coordinate]]
                values += product
            for entry in entries:
                result[(slice(None),) + entry] = values * scale
    return result


class _Errors:
    """Accumulate PASS/FAIL, maximum, mean, and RMS absolute errors"""

    def __init__(self, atol, rtol):
        self.atol, self.rtol = atol, rtol
        self.passed = True
        self.count = 0
        self.maximum = self.total = self.squared = 0.0

    def update(self, actual, reference):
        with np.errstate(invalid="ignore", over="ignore"):
            difference = actual - reference
            finite = np.isfinite(difference)
            errors = np.where(finite, np.abs(difference), np.inf)
            self.passed = self.passed and bool(
                np.all(finite & (errors <= self.atol + self.rtol * np.abs(reference)))
            )
            self.maximum = max(self.maximum, float(np.max(errors)))
            self.total += float(np.sum(errors))
            self.squared += float(np.sum(errors**2))
        self.count += errors.size

    def finish(self):
        return dict(
            passed=self.passed,
            n_values=self.count,
            max_abs_error=self.maximum,
            mean_abs_error=self.total / self.count,
            rms_error=math.sqrt(self.squared / self.count),
            atol=self.atol,
            rtol=self.rtol,
        )


def _pad_points(points, size):
    """Repeat the last geometry to keep JAX batch shapes fixed"""
    return np.concatenate((points, np.repeat(points[-1:], size - len(points), axis=0)))


def _numeric_transform(internal_to_cartesian, ncoo):
    """Use JAX directly, or translate a NumPy/SOP expression graph to JAX"""
    import jax
    import jax.numpy as jnp

    from keosop.revad import Expr, as_expr

    def native(q):
        return jnp.asarray(internal_to_cartesian(q), dtype=jnp.float64)

    try:
        jax.eval_shape(native, jax.ShapeDtypeStruct((ncoo,), jnp.float64))
        return native
    except (TypeError, ValueError, jax.errors.TracerArrayConversionError):
        pass

    q = np.array([Expr("x_coo", data=k) for k in range(ncoo)], dtype=object)
    xyz = np.asarray(internal_to_cartesian(q), dtype=object)
    roots = [as_expr(value) for value in xyz.flat]
    operations = {
        "add": lambda a, b: a + b,
        "neg": lambda a: -a,
        "mul": lambda a, b: a * b,
        "div": lambda a, b: a / b,
        **{
            name: getattr(jnp, name)
            for name in (
                "sin",
                "cos",
                "tan",
                "sqrt",
                "exp",
                "log",
                "arccos",
                "arcsin",
                "arctan",
                "arctan2",
            )
        },
    }

    def traced(q):
        values = {}

        def evaluate(node):
            if node.id not in values:
                if node.oper == "const":
                    value = jnp.asarray(node.data, dtype=q.dtype)
                elif node.oper == "x_coo":
                    value = q[node.data]
                elif node.oper == "pow":
                    value = evaluate(node.args[0]) ** node.data
                else:
                    value = operations[node.oper](*(evaluate(arg) for arg in node.args))
                values[node.id] = value
            return values[node.id]

        return jnp.stack([evaluate(root) for root in roots]).reshape(xyz.shape)

    return traced


def compare_sop_numeric(
    grids,
    masses,
    gvib_sop=None,
    grovib_sop=None,
    grot_sop=None,
    pseudo_sop=None,
    *,
    internal_to_cartesian,
    max_points=1_000_000,
    batch_size=256,
    pseudo_batch_size=16,
    seed=0,
    atol=1e-8,
    rtol=1e-8,
    pseudo_atol=1e-5,
    pseudo_rtol=1e-6,
):
    # Prepare the SOPs and the numerical Cartesian transformation

    grids = [np.asarray(grid, dtype=float) for grid in grids]
    masses = np.asarray(masses, dtype=float)
    ncoo = len(grids)

    inputs = dict(
        Gvib=gvib_sop, Grovib=grovib_sop, Grot=grot_sop, pseudopotential=pseudo_sop
    )
    cache = {}
    plans = {
        name: _compile_block(block, grids, cache)
        for name, block in inputs.items()
        if block is not None
    }

    errors = {
        name: (
            _Errors(pseudo_atol, pseudo_rtol)
            if name == "pseudopotential"
            else _Errors(atol, rtol)
        )
        for name in plans
    }

    transform = com(masses)(_numeric_transform(internal_to_cartesian, ncoo))

    batch_pseudo = jax.jit(jax.vmap(pseudo, in_axes=(0, None, None)), static_argnums=2)

    slices = {
        "Gvib": (slice(None), slice(0, ncoo), slice(0, ncoo)),
        "Grovib": (slice(None), slice(ncoo, ncoo + 3), slice(0, ncoo)),
        "Grot": (slice(None), slice(ncoo, ncoo + 3), slice(ncoo, ncoo + 3)),
    }

    gnames = [name for name in plans if name != "pseudopotential"]

    # Sample at most max_points tuples of 1D grid indices

    sizes = np.array([len(grid) for grid in grids])
    total_points = math.prod(int(size) for size in sizes)
    npoints = min(max_points, total_points)
    full_grid = total_points <= max_points
    batch_size = min(batch_size, npoints)
    pseudo_batch_size = min(pseudo_batch_size, batch_size)
    rng = np.random.default_rng(seed)

    print(f"\nComparing {npoints:,} product-grid points", flush=True)

    # Evaluate numerical and SOP values for each batch, then accumulate errors

    for indices in _index_batches(sizes, npoints, batch_size, rng, full_grid):
        points = np.column_stack([grid[indices[:, k]] for k, grid in enumerate(grids)])

        if gnames:
            reference = np.asarray(
                batch_Gmat(_pad_points(points, batch_size), masses, transform)
            )[: len(points)]

            for name in gnames:
                actual = _evaluate_block(plans[name], indices, G_to_invcm)
                errors[name].update(actual, reference[slices[name]])

        if "pseudopotential" in plans:
            for start in range(0, len(points), pseudo_batch_size):
                q = points[start : start + pseudo_batch_size]
                inds = indices[start : start + pseudo_batch_size]

                reference = np.asarray(
                    batch_pseudo(_pad_points(q, pseudo_batch_size), masses, transform)
                )[: len(q)]

                actual = _evaluate_block(plans["pseudopotential"], inds, G_to_invcm)
                errors["pseudopotential"].update(actual, reference)

    # Print summary

    blocks = {name: error.finish() for name, error in errors.items()}

    for name, block in blocks.items():
        status = "PASS" if block["passed"] else "FAIL"
        print(
            f"{name}: {status}; max abs={block['max_abs_error']:.6e}, "
            f"mean abs={block['mean_abs_error']:.6e}, "
            f"RMS={block['rms_error']:.6e} cm^-1",
            flush=True,
        )
