"""Compare candidate tree layouts on FXA and ZINC250k.

Usage: uv run python prototypes/compare_layouts.py [fxa|zinc] [n_molecules]
Writes PNGs to prototypes/out/ and prints timing and quality metrics.
"""

import csv
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).parent))
from layouts import force_layout, multilevel_layout, radial_layout  # noqa: E402

from chemviz import lsh_knn, minhash_ecfp, minimum_spanning_forest  # noqa: E402

ROOT = Path(__file__).parent.parent
OUT = Path(__file__).parent / "out"


def load(name, limit=None):
    if name == "fxa":
        rows = list(csv.DictReader(open(ROOT / "data/fxa/fxa.csv")))
        smiles = [r["SMILES"] for r in rows]
        labels = np.unique([r["SERIES_ID"] for r in rows], return_inverse=True)[1]
        return smiles[:limit], labels[:limit], True
    rows = list(csv.DictReader(open(ROOT / "data/zinc250k/zinc250k.csv")))
    smiles = [r["smiles"].strip() for r in rows]
    values = np.array([float(r["logP"]) for r in rows])
    return smiles[:limit], values[:limit], False


def knn_preservation(pos, knn_idx, k=10, n_sample=5000, seed=0):
    """Mean fraction of each molecule's k chemical neighbours that are among its k nearest 2D points."""
    rng = np.random.default_rng(seed)
    q = rng.choice(len(pos), min(n_sample, len(pos)), replace=False)
    _, nn2d = cKDTree(pos).query(pos[q], k=k + 1)
    hits = [len(set(nn2d[i, 1:]) & set(knn_idx[qi, :k]) - {-1}) / k for i, qi in enumerate(q)]
    return float(np.mean(hits))


def label_agreement(pos, labels, k=5):
    """Fraction of 2D k-nearest neighbours sharing the molecule's series (categorical labels)."""
    _, nn2d = cKDTree(pos).query(pos, k=k + 1)
    return float((labels[nn2d[:, 1:]] == labels[:, None]).mean())


def plot(pos, edges, color, categorical, title, path):
    fig, ax = plt.subplots(figsize=(10, 10), dpi=120)
    segs = np.stack([pos[edges[:, 0]], pos[edges[:, 1]]], axis=1)
    lw = 0.4 if len(pos) < 20_000 else 0.05
    ax.add_collection(LineCollection(segs, colors="0.6", linewidths=lw, zorder=1))
    s = 6 if len(pos) < 20_000 else 0.1
    ax.scatter(pos[:, 0], pos[:, 1], c=color, s=s, cmap="tab20" if categorical else "viridis", zorder=2, linewidths=0)
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "fxa"
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else None
    OUT.mkdir(exist_ok=True)
    smiles, color, categorical = load(name, limit)
    n = len(smiles)

    t = time.perf_counter()
    hashes = minhash_ecfp(smiles)
    idx, dist = lsh_knn(hashes, k=10)
    edges, _ = minimum_spanning_forest(idx, dist)
    print(f"{name} n={n}: MinHash+kNN+MST {time.perf_counter() - t:.1f}s, {n - len(edges)} components")

    layouts = {
        "radial": lambda: radial_layout(n, edges),
        "force": lambda: force_layout(n, edges),
        "force600": lambda: force_layout(n, edges, n_iter=600),
        "multilevel": lambda: multilevel_layout(n, edges),
    }
    for lname, fn in layouts.items():
        fn() if n < 2000 else None  # warm numba JIT on small inputs only
        t = time.perf_counter()
        pos = fn()
        dt = time.perf_counter() - t
        metrics = f"knn_pres={knn_preservation(pos, idx):.3f}"
        if categorical:
            metrics += f" series_agree={label_agreement(pos, color):.3f}"
        print(f"  {lname:10s} {dt:7.2f}s  {metrics}")
        plot(pos, edges, color, categorical, f"{name} - {lname} ({dt:.1f}s, {metrics})", OUT / f"{name}_{n}_{lname}.png")


if __name__ == "__main__":
    main()
