"""Benchmark chemviz against the original tmap on the same MinHash input.

Usage: uv run python benchmarks/compare_tmap.py [fxa|zinc|chembl] [n_molecules|all] [chemviz,tmap]
Requires the original tmap in .venv-tmap (Python 3.10, `uv pip install tmap-viz "numpy<2" scipy`).
Each tool runs in its own process so timings and peak memory are isolated.
"""

import json
import subprocess
import sys
from pathlib import Path

import numba as nb
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import shortest_path
from scipy.spatial import cKDTree
from scipy.stats import spearmanr

ROOT = Path(__file__).parent.parent
OUT = Path(__file__).parent / "out"
sys.path.insert(0, str(ROOT / "prototypes"))
from compare_layouts import label_agreement, load, plot  # noqa: E402

from chemviz import minhash_ecfp  # noqa: E402

RUNNERS = {
    "chemviz": [sys.executable, str(Path(__file__).parent / "run_chemviz.py")],
    "tmap": [str(ROOT / ".venv-tmap/bin/python"), str(Path(__file__).parent / "run_tmap_original.py")],
}


@nb.njit(cache=True, parallel=True)
def exact_knn(hashes, queries, k):
    """Exact MinHash kNN of the query rows by brute force (self excluded)."""
    n, d = hashes.shape
    out = np.empty((queries.shape[0], k), dtype=np.int64)
    for qi in nb.prange(queries.shape[0]):
        q = queries[qi]
        dist = np.empty(n, dtype=np.float32)
        for j in range(n):
            eq = 0
            for c in range(d):
                if hashes[q, c] == hashes[j, c]:
                    eq += 1
            dist[j] = 1.0 - eq / d
        dist[q] = np.inf
        out[qi] = np.argsort(dist, kind="mergesort")[:k]
    return out


def knn_preservation(pos, queries, true_knn):
    k = true_knn.shape[1]
    _, nn2d = cKDTree(pos).query(pos[queries], k=k + 1)
    return float(np.mean([len(set(nn2d[i, 1:]) & set(true_knn[i])) / k for i in range(len(queries))]))


def global_structure(pos, edges, n_src=20, n_tgt=20000, seed=1):
    """Spearman correlation between tree hop distance and 2D distance on random node pairs.

    kNN preservation only measures local structure; a layout that collapses into a
    uniform disc can score well on it while losing the readable branch structure.
    """
    n = len(pos)
    rng = np.random.default_rng(seed)
    g = coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(n, n)).tocsr()
    src = rng.choice(n, min(n_src, n), replace=False)
    tgt = rng.choice(n, min(n_tgt, n), replace=False)
    hops = shortest_path(g, directed=False, unweighted=True, indices=src)[:, tgt]
    dist2d = np.linalg.norm(pos[src][:, None] - pos[tgt][None], axis=2)
    ok = np.isfinite(hops)
    return float(spearmanr(hops[ok], dist2d[ok]).statistic)


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "fxa"
    limit = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2] != "all" else None
    tools = sys.argv[3].split(",") if len(sys.argv) > 3 else list(RUNNERS)
    OUT.mkdir(exist_ok=True)
    smiles, color, categorical = load(name, limit)
    n = len(smiles)
    hashes = minhash_ecfp(smiles)
    hashes_path = OUT / f"{name}_{n}_hashes.npy"
    np.save(hashes_path, hashes)

    queries = np.random.default_rng(0).choice(n, min(2000, n), replace=False)
    true_knn = exact_knn(hashes, queries, 10)

    print(f"{name} n={n}")
    for tool in tools:
        cmd = RUNNERS[tool]
        out_path = OUT / f"{name}_{n}_{tool}.npz"
        res = subprocess.run([*cmd, str(hashes_path), str(out_path)], capture_output=True, text=True, check=True)
        stats = json.loads(res.stdout.strip().splitlines()[-1])
        data = np.load(out_path)
        pos, edges = data["pos"], data["edges"].astype(np.int64)
        metrics = {"knn_pres": knn_preservation(pos, queries, true_knn), "global": global_structure(pos, edges)}
        if categorical:
            metrics["series_agree"] = label_agreement(pos, color)
        timing = " ".join(f"{k}={v:.1f}" for k, v in stats.items())
        quality = " ".join(f"{k}={v:.3f}" for k, v in metrics.items())
        print(f"  {tool:8s} {timing}  {quality}  edges={len(edges)}")
        plot(pos, edges, color, categorical, f"{name} - {tool} ({stats['total_s']:.1f}s, {quality})",
             OUT / f"{name}_{n}_{tool}.png")


if __name__ == "__main__":
    main()
