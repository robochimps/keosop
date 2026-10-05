import importlib
from pathlib import Path

import h5py
import numpy as np

FORMAT_NAME = "KEOSOP-GMAT"
FORMAT_VERSION = "1.0"
PROTECTED_KIND_TO_CODE = {"q": 0, "sin": 1, "cos": 2}
PROTECTED_CODE_TO_KIND = {v: k for k, v in PROTECTED_KIND_TO_CODE.items()}


def _as_text(x):
    return x.decode("utf-8") if isinstance(x, (bytes, np.bytes_)) else str(x)


def _grid_layout(grids):
    grids = [np.asarray(g, dtype=np.float64).ravel() for g in grids]
    sizes = np.asarray([g.size for g in grids], dtype=np.int64)
    offsets = np.concatenate(([0], np.cumsum(sizes))).astype(np.int64)
    values = np.concatenate(grids) if grids else np.empty(0, dtype=np.float64)
    return grids, sizes, offsets, values


def _write_element(group, sop, grid_sizes, compress):
    if sop is None:
        group.attrs["present"] = False
        return

    group.attrs["present"] = True

    terms = sop.terms
    nterms = len(terms)
    ncoords = len(grid_sizes)

    if nterms < 1:
        raise ValueError("Cannot export an empty SumProd")

    if any(len(t) != ncoords for t in terms):
        raise ValueError("SumProd coordinate count does not match the exported grids")

    total = int(np.sum(grid_sizes))
    factors = np.empty((nterms, total), dtype=np.float64)

    for iterm, term in enumerate(terms):

        pos = 0
        for k, (factor, n) in enumerate(zip(term, grid_sizes)):
            factor = np.asarray(factor, dtype=np.float64).ravel()

            if factor.size != int(n):
                raise ValueError(
                    f"Term {iterm}, coordinate {k}: factor length {factor.size} != grid length {n}"
                )

            factors[iterm, pos : pos + int(n)] = factor
            pos += int(n)

    kwargs = {}
    if compress is not None and factors.size:
        kwargs.update(compression=compress, shuffle=True)

    group.create_dataset("factors", data=factors, **kwargs)
    group.attrs["nterms"] = nterms

    protected = getattr(sop, "protected", None)

    if protected is None:
        protected = [()] * nterms

    if len(protected) != nterms:
        raise ValueError("SumProd protected metadata has inconsistent length")

    offsets = [0]
    coords, kinds, powers = [], [], []

    for sig in protected:
        for coord, kind, power in sig:
            kind = str(kind)
            if kind not in PROTECTED_KIND_TO_CODE:
                raise ValueError(f"Unsupported protected factor kind {kind}")

            coords.append(int(coord))
            kinds.append(PROTECTED_KIND_TO_CODE[kind])
            powers.append(float(power))

        offsets.append(len(coords))

    group.create_dataset("protected_offsets", data=np.asarray(offsets, dtype=np.int64))
    group.create_dataset("protected_coord", data=np.asarray(coords, dtype=np.int32))
    group.create_dataset("protected_kind", data=np.asarray(kinds, dtype=np.int8))
    group.create_dataset("protected_power", data=np.asarray(powers, dtype=np.float64))


def _write_matrix(parent, name, matrix, grid_sizes, compress):
    if matrix is None:
        return

    arr = np.asarray(matrix, dtype=object)
    mat_group = parent.create_group(name)

    mat_group.attrs["shape"] = np.asarray(arr.shape, dtype=np.int64)
    mat_group.attrs["storage"] = "full"

    for idx in np.ndindex(arr.shape):
        key = "i_" + "_".join(map(str, idx))
        el_group = mat_group.create_group(key)

        el_group.attrs["index"] = np.asarray(idx, dtype=np.int64)
        _write_element(el_group, arr[idx], grid_sizes, compress)


def save_gmat_hdf5(
    filename,
    grids,
    gvib_sop,
    grovib_sop=None,
    grot_sop=None,
    pseudo_sop=None,
    coo_names=None,
    coo_units=None,
    metadata=None,
    compress="gzip",
):
    """Save G-matrices in an HDF5 file"""
    grids, sizes, offsets, values = _grid_layout(grids)
    ncoo = len(grids)

    if coo_names is not None and len(coo_names) != ncoo:
        raise ValueError("coo_names must have one entry per coordinate")

    if coo_units is not None and len(coo_units) != ncoo:
        raise ValueError("coo_units must have one entry per coordinate")

    with h5py.File(Path(filename), "w") as h5:
        h5.attrs["format"] = FORMAT_NAME
        h5.attrs["format_version"] = FORMAT_VERSION
        h5.attrs["index_base"] = 0
        h5.attrs["protected_kind_q"] = PROTECTED_KIND_TO_CODE["q"]
        h5.attrs["protected_kind_sin"] = PROTECTED_KIND_TO_CODE["sin"]
        h5.attrs["protected_kind_cos"] = PROTECTED_KIND_TO_CODE["cos"]

        if metadata:
            for key, value in metadata.items():
                if value is None:
                    continue

                if np.isscalar(value) or isinstance(value, (str, bytes)):
                    h5.attrs[str(key)] = value
                else:
                    raise TypeError(f"Metadata[{key}] must be a scalar or string")

        coo_group = h5.create_group("grids")
        coo_group.create_dataset("values", data=values)
        coo_group.create_dataset("sizes", data=sizes)
        coo_group.create_dataset("offsets", data=offsets)

        if coo_names is not None:
            dt = h5py.string_dtype(encoding="utf-8")
            coo_group.create_dataset(
                "names", data=np.asarray(coo_names, dtype=object), dtype=dt
            )

        if coo_units is not None:
            dt = h5py.string_dtype(encoding="utf-8")
            coo_group.create_dataset(
                "units", data=np.asarray(coo_units, dtype=object), dtype=dt
            )

        mat_group = h5.create_group("matrices")
        _write_matrix(mat_group, "gvib", gvib_sop, sizes, compress)
        _write_matrix(mat_group, "grovib", grovib_sop, sizes, compress)
        _write_matrix(mat_group, "grot", grot_sop, sizes, compress)
        if pseudo_sop is not None:
            pseudo_group = h5.create_group("pseudo")
            _write_element(pseudo_group, pseudo_sop, sizes, compress)


def _default_sumprod_class():
    mod_candidates = ("keosop.sing.sumprod", "keosop.sumprod")

    err = []
    for mod_name in mod_candidates:
        try:
            return importlib.import_module(mod_name).SumProd
        except (ImportError, AttributeError) as exc:
            err.append(f"{mod_name}: {exc}")

    raise ImportError(
        "Could not import a KEOSOP SumProd class; pass sumprod_cls explicitly"
        + "; ".join(err)
    )


def _read_element(group, grids, sumprod_cls):
    if not bool(group.attrs.get("present", True)):
        return None

    factors = np.asarray(group["factors"], dtype=np.float64)

    nterms = factors.shape[0]
    sizes = [len(g) for g in grids]
    goff = np.concatenate(([0], np.cumsum(sizes))).astype(int)
    terms = [
        [factors[t, goff[k] : goff[k + 1]].copy() for k in range(len(grids))]
        for t in range(nterms)
    ]

    poff = np.asarray(group["protected_offsets"], dtype=np.int64)
    pcoord = np.asarray(group["protected_coord"], dtype=np.int32)
    pkind = np.asarray(group["protected_kind"], dtype=np.int8)
    ppower = np.asarray(group["protected_power"], dtype=np.float64)

    protected = []

    for t in range(nterms):
        sig = []

        for j in range(int(poff[t]), int(poff[t + 1])):
            code = int(pkind[j])

            if code not in PROTECTED_CODE_TO_KIND:
                raise ValueError(f"Unknown protected kind code {code}")

            sig.append((int(pcoord[j]), PROTECTED_CODE_TO_KIND[code], float(ppower[j])))

        protected.append(tuple(sig))

    # standard SumProd does not contain protected metadata (only singular version does)
    try:
        return sumprod_cls(terms, protected=protected, grids=grids)
    except TypeError:
        if any(protected):
            raise TypeError(
                "The selected SumProd class does not accept protected metadata, "
                "but the HDF5 file contains protected factors"
            )

        try:
            return sumprod_cls(terms, grids=grids)

        except TypeError:
            return sumprod_cls(terms)


def _read_matrix(parent, name, grids, sumprod_cls):
    if name not in parent:
        return None

    mat_group = parent[name]

    shape = tuple(int(x) for x in np.asarray(mat_group.attrs["shape"]).ravel())
    out = np.empty(shape, dtype=object)

    for idx in np.ndindex(shape):
        key = "i_" + "_".join(map(str, idx))

        if key not in mat_group:
            raise ValueError(f"Missing HDF5 matrix element {name}/{key}")

        out[idx] = _read_element(mat_group[key], grids, sumprod_cls)

    return out


def load_gmat_hdf5(filename, sumprod_cls=None, return_metadata=False):
    """Load an HDF5 file written by save_gmat_hdf5"""
    if sumprod_cls is None:
        sumprod_cls = _default_sumprod_class()

    with h5py.File(Path(filename), "r") as h5:
        fmt = _as_text(h5.attrs.get("format", ""))
        version = _as_text(h5.attrs.get("format_version", ""))

        if fmt != FORMAT_NAME:
            raise ValueError(f"Unsupported {FORMAT_NAME} file (format={fmt})")

        if version.split(".")[0] != FORMAT_VERSION.split(".")[0]:
            raise ValueError(f"Unsupported {FORMAT_NAME} version {version}")

        coo_group = h5["grids"]

        values = np.asarray(coo_group["values"], dtype=np.float64)
        offsets = np.asarray(coo_group["offsets"], dtype=np.int64)
        grids = [
            values[offsets[k] : offsets[k + 1]].copy() for k in range(len(offsets) - 1)
        ]

        mat_group = h5["matrices"]

        gvib = _read_matrix(mat_group, "gvib", grids, sumprod_cls)
        grovib = _read_matrix(mat_group, "grovib", grids, sumprod_cls)
        grot = _read_matrix(mat_group, "grot", grids, sumprod_cls)
        if "pseudo" in h5:
            pseudo = _read_element(h5["pseudo"], grids, sumprod_cls)
        else:
            pseudo = None

        metadata = {str(k): h5.attrs[k] for k in h5.attrs.keys()}

        if "names" in coo_group:
            metadata["coo_names"] = [_as_text(x) for x in coo_group["names"][...]]

        if "units" in coo_group:
            metadata["coo_units"] = [_as_text(x) for x in coo_group["units"][...]]

    result = (grids, gvib, grovib, grot, pseudo)

    return result + (metadata,) if return_metadata else result
