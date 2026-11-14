"""Self-contained interactive WebGL viewer for tree maps.

The map is written as a single HTML file: a JSON header describing the columns and
one zlib-compressed, base64-encoded binary blob holding coordinates, edges and
column values, which the page decompresses natively (``DecompressionStream``).
Structures are drawn on hover by RDKit.js, loaded from a CDN.
"""

import base64
import json
import zlib
from importlib.resources import files

import numpy as np
import pandas as pd

MAX_CATEGORIES = 1000  # object columns with more distinct values are treated as free text


def _column_payload(name, values: pd.Series, force_text: bool = False):
    """Header entry and binary buffer of one column."""
    if not force_text and (pd.api.types.is_numeric_dtype(values) and not pd.api.types.is_bool_dtype(values)):
        arr = values.to_numpy(dtype=np.float32, na_value=np.nan)
        finite = arr[np.isfinite(arr)]
        stats = {}
        if len(finite):
            lo, hi = np.percentile(finite, [1, 99])
            stats = {"min": float(finite.min()), "max": float(finite.max()), "lo": float(lo), "hi": float(hi)}
        integer = bool(len(finite)) and pd.api.types.is_integer_dtype(values)
        return {"name": name, "kind": "num", "integer": integer, **stats}, arr.tobytes()

    strings = values.astype(object).where(values.notna(), "").astype(str)
    counts = strings.value_counts(sort=True)
    if not force_text and len(counts) <= MAX_CATEGORIES and len(counts) < max(2, len(values) // 2):
        categories = counts.index.tolist()  # most frequent first, so rare ones can share a colour
        codes = pd.Categorical(strings, categories=categories).codes.astype(np.uint32)
        return {"name": name, "kind": "cat", "categories": categories,
                "counts": counts.to_numpy().tolist()}, codes.tobytes()
    text = "\n".join(s.replace("\n", " ") for s in strings)
    return {"name": name, "kind": "text"}, text.encode("utf-8")


def build_payload(x, y, edges, data: pd.DataFrame, smiles: str | None, columns=None, title: str = "chemviz"):
    """JSON header and compressed binary blob for the viewer."""
    if columns is None:
        columns = list(data.columns)
    columns = [c for c in columns if c != smiles]
    if smiles is not None:
        columns = [smiles, *columns]
    missing = [c for c in columns if c not in data.columns]
    if missing:
        raise ValueError(f"Unknown column(s) {missing}; available: {list(data.columns)}")

    buffers, entries = [], []
    offset = 0

    def add(buf: bytes):
        nonlocal offset
        start = offset
        buffers.append(buf)
        offset += len(buf)
        pad = -offset % 4  # keep typed-array views aligned
        if pad:
            buffers.append(b"\0" * pad)
            offset += pad
        return start, len(buf)

    pos_ref = add(np.column_stack([x, y]).astype(np.float32).tobytes())
    edge_ref = add(np.ascontiguousarray(edges, dtype=np.uint32).tobytes())
    for c in columns:
        entry, buf = _column_payload(str(c), data[c], force_text=(c == smiles))
        entry["offset"], entry["length"] = add(buf)
        entries.append(entry)

    header = {
        "title": title,
        "n": int(len(x)),
        "n_edges": int(len(edges)),
        "smiles": str(smiles) if smiles is not None else None,
        "positions": pos_ref,
        "edges": edge_ref,
        "columns": entries,
    }
    blob = zlib.compress(b"".join(buffers), 6)
    return header, blob


def to_html(x, y, edges, data, smiles, columns=None, title: str = "chemviz") -> str:
    header, blob = build_payload(x, y, edges, data, smiles, columns, title)
    template = files("chemviz").joinpath("viewer.html").read_text(encoding="utf-8")
    header_json = json.dumps(header, allow_nan=False).replace("</", "<\\/")
    return (
        template.replace("__CHEMVIZ_TITLE__", _escape(title))
        .replace("__CHEMVIZ_HEADER__", header_json)
        .replace("__CHEMVIZ_DATA__", base64.b64encode(blob).decode("ascii"))
    )


def _escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
