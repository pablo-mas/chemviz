"""Run chemviz kNN + MST + layout on a MinHash matrix.

Usage: run_chemviz.py hashes.npy out.npz
Prints a JSON line with timings (s) and peak memory (MB).
"""

import json
import resource
import sys
import time

import numpy as np

from chemviz import lsh_knn, minimum_spanning_forest, tree_layout

hashes = np.load(sys.argv[1])
n = len(hashes)

# Warm the numba cache on a tiny input so JIT loading is not timed.
_e, _ = minimum_spanning_forest(*lsh_knn(hashes[:200]))
tree_layout(200, _e)

t0 = time.perf_counter()
idx, dist = lsh_knn(hashes, k=10, kc=10)
t1 = time.perf_counter()
edges, _ = minimum_spanning_forest(idx, dist)
t2 = time.perf_counter()
pos = tree_layout(n, edges)
t3 = time.perf_counter()

np.savez(sys.argv[2], pos=pos, edges=edges)
print(json.dumps({
    "knn_s": t1 - t0,
    "mst_s": t2 - t1,
    "layout_s": t3 - t2,
    "total_s": t3 - t0,
    "max_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
}))
