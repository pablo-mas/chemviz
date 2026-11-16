"""Benchmark chemviz's mapping methods, and its UMAP and t-SNE against umap-learn and openTSNE.

Usage: uv run python benchmarks/compare_methods.py [fxa|zinc|chembl] [n_molecules|all] [--reference]

1. Each method (pca, tmap, umap, tsne) maps the molecules end to end from SMILES.
2. With --reference, umap-learn and openTSNE (in .venv-ref, see run_reference.py) embed the
   same LSH kNN graphs as chemviz's umap_layout and tsne_layout, so only the embedding
   algorithms are compared.

Metrics: kNN preservation (share of each molecule's 10 exact MinHash neighbours among its
10 nearest points in 2D, on 2000 molecules) and distance correlation (Spearman correlation
between MinHash Jaccard distance and 2D distance, on 20,000 random pairs).
"""

import json
import subprocess
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).parent.parent
OUT = Path(__file__).parent / "out"
sys.path.insert(0, str(ROOT / "prototypes"))
sys.path.insert(0, str(Path(__file__).parent))
from compare_layouts import load  # noqa: E402
from compare_tmap import exact_knn, knn_preservation  # noqa: E402

import chemviz as cv  # noqa: E402


def metrics(pos, hashes, queries, true_knn, pairs):
    a, b = pairs
    jaccard = 1 - (hashes[a] == hashes[b]).mean(axis=1)
    corr = spearmanr(jaccard, np.linalg.norm(pos[a] - pos[b], axis=1)).statistic
    return f"knn {knn_preservation(pos, queries, true_knn):.3f}  dist_corr {corr:.3f}"


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "fxa"
    limit = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2] != "all" else None
    reference = "--reference" in sys.argv
    warnings.simplefilter("ignore")
    OUT.mkdir(exist_ok=True)
    smiles, _, _ = load(name, limit)
    n = len(smiles)
    hashes = cv.minhash_ecfp(smiles)
    rng = np.random.default_rng(0)
    queries = rng.choice(n, min(2000, n), replace=False)
    true_knn = exact_knn(hashes, queries, 10)
    pairs = np.random.default_rng(1).integers(0, n, (2, 20000))
    for f in (cv.pca, cv.tmap, cv.umap, cv.tsne):  # compile outside the timings
        f(smiles[:300])

    print(f"{name} n={n}, end to end from SMILES")
    for method in ("pca", "tmap", "umap", "tsne"):
        start = time.perf_counter()
        m = getattr(cv, method)(smiles)
        elapsed = time.perf_counter() - start
        pos = np.column_stack([m.x, m.y]).astype(np.float64)
        print(f"  {method:5s} {elapsed:6.1f} s  {metrics(pos, hashes, queries, true_knn, pairs)}", flush=True)

    if not reference:
        return
    python = ROOT / ".venv-ref/bin/python"
    print("same kNN graph, embedding only")
    for method, k, layout in (("umap", 15, cv.umap_layout), ("tsne", 90, cv.tsne_layout)):
        idx, dist = cv.lsh_knn(hashes, k=k, kc=10 if method == "umap" else 3)
        knn_path = OUT / f"{name}_{n}_knn{k}.npz"
        np.savez(knn_path, idx=idx, dist=dist)
        start = time.perf_counter()
        pos = layout(idx, dist).astype(np.float64)
        elapsed = time.perf_counter() - start
        print(f"  chemviz {method:5s} {elapsed:6.1f} s  {metrics(pos, hashes, queries, true_knn, pairs)}", flush=True)
        out_path = OUT / f"{name}_{n}_{method}_reference.npz"
        res = subprocess.run([str(python), str(Path(__file__).parent / "run_reference.py"), method,
                              str(knn_path), str(out_path)], capture_output=True, text=True, check=True)
        elapsed = json.loads(res.stdout.strip().splitlines()[-1])["time_s"]
        pos = np.load(out_path)["pos"].astype(np.float64)
        label = "umap-learn" if method == "umap" else "openTSNE"
        print(f"  {label:13s} {elapsed:6.1f} s  {metrics(pos, hashes, queries, true_knn, pairs)}", flush=True)


if __name__ == "__main__":
    main()
