"""Run umap-learn or openTSNE on a precomputed kNN graph (for compare_methods.py).

Usage: run_reference.py umap|tsne knn.npz out.npz
Requires .venv-ref (`uv venv .venv-ref --python 3.12; uv pip install --python .venv-ref
umap-learn openTSNE`). Missing neighbours (index -1) get a far neighbour, so they weigh nothing.
Prints a JSON line with the time (s).
"""

import json
import sys
import time

import numpy as np

method, knn_path, out_path = sys.argv[1:4]
data = np.load(knn_path)
idx, dist = data["idx"].copy(), data["dist"].astype(np.float64)
n, k = idx.shape
missing = idx < 0
idx[missing] = ((np.arange(n)[:, None] + 1) % n).repeat(k, axis=1)[missing]
dist[missing] = 1e3

start = time.perf_counter()
if method == "umap":
    import umap

    # umap-learn expects each point as its own first neighbour.
    idx = np.hstack([np.arange(n)[:, None], idx])
    dist = np.hstack([np.zeros((n, 1)), dist]).astype(np.float32)
    pos = umap.UMAP(n_neighbors=k + 1, precomputed_knn=(idx, dist, None)).fit_transform(np.zeros((n, 1)))
else:
    from openTSNE import TSNE, affinity

    class Precomputed(affinity.Affinities):
        def __init__(self, P):
            self.P = P

    P = affinity.joint_probabilities_nn(idx, dist, [k / 3], symmetrize=True)
    pos = np.asarray(TSNE(n_jobs=-1).fit(affinities=Precomputed(P), initialization="spectral"))
elapsed = time.perf_counter() - start
np.savez(out_path, pos=pos)
print(json.dumps({"time_s": elapsed}))
