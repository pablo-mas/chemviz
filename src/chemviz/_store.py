"""Saving and loading maps as .chemviz files.

A .chemviz file is a zip archive of NumPy arrays (.npy, read without pickle, so opening
a file cannot run code) and a JSON header: format version, map fields, method
parameters and how to rebuild each data column. Text is stored as UTF-8 bytes with
offsets, so long strings cost no padding.
"""

import json
import warnings
import zipfile
from importlib.metadata import version

import numpy as np
import pandas as pd

from chemviz.maps import Map

FORMAT = "chemviz-map"
FORMAT_VERSION = 1


def _write(zf, name, array, compress=True):
    info = zipfile.ZipInfo(name)
    info.compress_type = zipfile.ZIP_DEFLATED if compress else zipfile.ZIP_STORED
    with zf.open(info, "w", force_zip64=True) as f:
        np.lib.format.write_array(f, np.ascontiguousarray(array), allow_pickle=False)


def _read(zf, name):
    with zf.open(name) as f:
        return np.lib.format.read_array(f, allow_pickle=False)


def _encode_text(values, isna):
    data = [b"" if na else str(v).encode("utf-8") for v, na in zip(values, isna)]
    offsets = np.zeros(len(data) + 1, dtype=np.int64)
    np.cumsum([len(b) for b in data], out=offsets[1:])
    return np.frombuffer(b"".join(data), dtype=np.uint8), offsets


def _decode_text(data, offsets, isna):
    raw = data.tobytes()
    return np.array([None if na else raw[a:b].decode("utf-8")
                     for a, b, na in zip(offsets[:-1], offsets[1:], isna)], dtype=object)


def _put_series(zf, key, s):
    """Write a Series (or Index) under ``key``; returns its JSON description."""
    dtype = s.dtype
    if isinstance(dtype, pd.CategoricalDtype):
        spec = {"kind": "categorical", "ordered": bool(dtype.ordered),
                "categories": _put_series(zf, f"{key}.categories", pd.Series(dtype.categories))}
        _write(zf, f"{key}.npy", np.asarray(s.cat.codes if isinstance(s, pd.Series) else s.codes))
        return spec
    if isinstance(dtype, np.dtype) and dtype.kind in "biufcmM":
        _write(zf, f"{key}.npy", np.asarray(s))
        return {"kind": "numpy"}
    isna = np.asarray(pd.isna(s), dtype=bool)
    if isinstance(dtype, pd.api.extensions.ExtensionDtype) and hasattr(dtype, "numpy_dtype") \
            and np.dtype(dtype.numpy_dtype).kind in "biuf" and not isinstance(dtype, pd.StringDtype):
        _write(zf, f"{key}.npy", np.asarray(s.to_numpy(dtype=dtype.numpy_dtype, na_value=0)))
        _write(zf, f"{key}.mask.npy", isna)
        return {"kind": "masked", "dtype": str(dtype)}
    values = np.asarray(s, dtype=object)
    if not (isinstance(dtype, pd.StringDtype) or dtype == object) or \
            not all(isinstance(v, str) for v, na in zip(values, isna) if not na):
        warnings.warn(f"Column {s.name!r} ({dtype}) is saved as text", stacklevel=4)
    data, offsets = _encode_text(values, isna)
    _write(zf, f"{key}.npy", data)
    _write(zf, f"{key}.offsets.npy", offsets)
    _write(zf, f"{key}.mask.npy", isna)
    return {"kind": "text", "dtype": str(dtype) if isinstance(dtype, pd.StringDtype) else "object"}


def _get_series(zf, key, spec):
    kind = spec["kind"]
    if kind == "numpy":
        return pd.Series(_read(zf, f"{key}.npy"))
    if kind == "categorical":
        categories = _get_series(zf, f"{key}.categories", spec["categories"])
        return pd.Series(pd.Categorical.from_codes(_read(zf, f"{key}.npy"), categories, ordered=spec["ordered"]))
    isna = _read(zf, f"{key}.mask.npy")
    if kind == "masked":
        return pd.Series(pd.array(_read(zf, f"{key}.npy"), dtype=spec["dtype"])).mask(isna)
    if kind == "text":
        values = _decode_text(_read(zf, f"{key}.npy"), _read(zf, f"{key}.offsets.npy"), isna)
        return pd.Series(values, dtype=spec["dtype"])
    raise ValueError(f"Unknown column kind {kind!r}")


def _json_name(name):
    return name if name is None or isinstance(name, (str, int, float, bool)) else str(name)


def save_map(m: Map, path, model: bool = True) -> None:
    """Write ``m`` to a .chemviz file (see ``Map.save``)."""
    header = {"format": FORMAT, "version": FORMAT_VERSION, "chemviz": version("chemviz"),
              "method": m.method, "smiles": m.smiles, "n": len(m.x)}
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for name in ("x", "y", "edges", "invalid"):
            _write(zf, f"{name}.npy", getattr(m, name))
        columns = []
        for i, (name, s) in enumerate(m.data.items()):
            columns.append({"name": _json_name(name), **_put_series(zf, f"data/{i}", s)})
        header["columns"] = columns
        index = m.data.index
        if isinstance(index, pd.RangeIndex):
            header["index"] = {"kind": "range", "start": index.start, "step": index.step, "name": _json_name(index.name)}
        else:
            header["index"] = {"kind": "levels", "levels": [
                {"name": _json_name(index.names[i]),
                 **_put_series(zf, f"index/{i}", pd.Series(index.get_level_values(i)))}
                for i in range(index.nlevels)]}
        params, arrays = {}, []
        for key, value in (m.model or {}).items() if model else ():
            if isinstance(value, np.ndarray):
                arrays.append(key)
                # MinHash values are random: compressing them only costs time.
                _write(zf, f"model/{key}.npy", value, compress=value.dtype != np.uint32)
            else:
                params[key] = value
        header["model"] = {"params": params, "arrays": arrays} if model and m.model is not None else None
        zf.writestr("header.json", json.dumps(header))


def load(path) -> Map:
    """Load a map saved with ``Map.save("map.chemviz")``.

    Files are read without pickle, so loading a file from someone else cannot run code.
    """
    try:
        zf = zipfile.ZipFile(path)
    except zipfile.BadZipFile:
        raise ValueError(f"{path} is not a chemviz map file") from None
    with zf:
        try:
            header = json.loads(zf.read("header.json"))
        except KeyError:
            raise ValueError(f"{path} is not a chemviz map file") from None
        if header.get("format") != FORMAT:
            raise ValueError(f"{path} is not a chemviz map file")
        if header["version"] > FORMAT_VERSION:
            raise ValueError(f"{path} was written by a newer chemviz ({header['chemviz']}); please upgrade")
        x, y, edges, invalid = (_read(zf, f"{name}.npy") for name in ("x", "y", "edges", "invalid"))
        n = header["n"]
        spec = header["index"]
        if spec["kind"] == "range":
            index = pd.RangeIndex(spec["start"], spec["start"] + n * spec["step"], spec["step"], name=spec["name"])
        else:
            levels = [_get_series(zf, f"index/{i}", lv) for i, lv in enumerate(spec["levels"])]
            names = [lv["name"] for lv in spec["levels"]]
            index = pd.Index(levels[0], name=names[0]) if len(levels) == 1 else pd.MultiIndex.from_arrays(levels, names=names)
        data = pd.DataFrame({i: _get_series(zf, f"data/{i}", c).set_axis(index)
                             for i, c in enumerate(header["columns"])}, index=index)
        data.columns = [c["name"] for c in header["columns"]]
        model = None
        if header["model"] is not None:
            model = dict(header["model"]["params"])
            for key in header["model"]["arrays"]:
                model[key] = _read(zf, f"model/{key}.npy")
    if not (len(x) == len(y) == len(data) == n):
        raise ValueError(f"{path} is corrupted: inconsistent number of molecules")
    return Map(x, y, edges, data, invalid, header["smiles"], header["method"], model)
