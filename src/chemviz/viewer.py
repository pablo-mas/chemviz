"""Self-contained interactive WebGL viewer for tree maps.

The map is written as a single HTML file: a JSON header describing the columns and
one zlib-compressed, base64-encoded binary blob holding coordinates, edges and
column values, which the page decompresses natively (``DecompressionStream``).
Structures are drawn on hover by RDKit.js, loaded from a CDN, or embedded in the
page for offline use (downloaded once, then cached).
"""

import base64
import hashlib
import json
import os
import sys
import urllib.request
import zlib
from importlib.resources import files
from pathlib import Path

import numpy as np
import pandas as pd

MAX_CATEGORIES = 1000  # object columns with more distinct values are treated as free text

RDKIT_JS_VERSION = "2026.9.1"  # keep in sync with RDKIT_URL in viewer.html
RDKIT_JS_URL = f"https://cdn.jsdelivr.net/npm/@rdkit/rdkit@{RDKIT_JS_VERSION}/dist/"
RDKIT_JS_FILES = {
    "RDKit_minimal.js": "ea427db9cc39be245065603de0df10c3053cb49af3b63fdf6bc2bf73d6fc17da",
    "RDKit_minimal.wasm": "594ceaed387adc70ea13bfbb2e8eace916f6c782254ed5a6f0f0a0a7a4fe4061",
}


def cache_dir() -> Path:
    """Where downloaded assets are kept: $CHEMVIZ_CACHE, else the platform user cache."""
    if "CHEMVIZ_CACHE" in os.environ:
        return Path(os.environ["CHEMVIZ_CACHE"])
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Caches"
    else:
        base = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return base / "chemviz"


def rdkit_js_files() -> dict[str, bytes]:
    """RDKit.js script and WebAssembly module, downloaded on first use and checked by SHA-256."""
    folder = cache_dir() / f"rdkit-js-{RDKIT_JS_VERSION}"
    out = {}
    for name, sha256 in RDKIT_JS_FILES.items():
        path = folder / name
        content = path.read_bytes() if path.exists() else None
        if content is None or hashlib.sha256(content).hexdigest() != sha256:
            try:
                with urllib.request.urlopen(RDKIT_JS_URL + name, timeout=60) as response:
                    content = response.read()
            except OSError as e:
                raise RuntimeError(
                    f"offline=True needs RDKit.js, which could not be downloaded from {RDKIT_JS_URL} ({e}). "
                    f"Connect once to cache it in {folder}, or copy {list(RDKIT_JS_FILES)} there."
                ) from e
            if hashlib.sha256(content).hexdigest() != sha256:
                raise RuntimeError(f"Downloaded {name} does not match its expected SHA-256; not using it")
            folder.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(path.suffix + ".tmp")
            tmp.write_bytes(content)
            tmp.replace(path)
        out[name] = content
    return out


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


def to_html(x, y, edges, data, smiles, columns=None, title: str = "chemviz", offline: bool = False) -> str:
    header, blob = build_payload(x, y, edges, data, smiles, columns, title)
    template = files("chemviz").joinpath("viewer.html").read_text(encoding="utf-8")
    header_json = json.dumps(header, allow_nan=False).replace("</", "<\\/")
    rdkit = ""
    if offline and smiles is not None:
        rdkit = "".join(
            f'<script type="application/octet-stream" id="chemviz-{name.replace(".", "-").lower()}">'
            f"{_pack(content)}</script>\n"
            for name, content in rdkit_js_files().items()
        )
    return (
        template.replace("__CHEMVIZ_TITLE__", _escape(title))
        .replace("__CHEMVIZ_HEADER__", header_json)
        .replace("__CHEMVIZ_RDKIT__", rdkit)
        .replace("__CHEMVIZ_DATA__", _pack(blob, compressed=True))
    )


def _pack(content: bytes, compressed: bool = False) -> str:
    """zlib-compressed, base64-encoded bytes (decoded in the page by DecompressionStream)."""
    return base64.b64encode(content if compressed else zlib.compress(content, 9)).decode("ascii")


def _escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
