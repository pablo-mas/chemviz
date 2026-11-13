"""Run the original tmap (tmap-viz) on a MinHash matrix. Executed with .venv-tmap/bin/python.

Usage: run_tmap_original.py hashes.npy out.npz
Prints a JSON line with timings (s) and peak memory (MB).
"""

import json
import os
import resource
import sys
import time

import numpy as np
import tmap as tm

hashes = np.load(sys.argv[1])
n, d = hashes.shape

t0 = time.perf_counter()
lf = tm.LSHForest(d, 8)
lf.batch_add([tm.VectorUint(h.tolist()) for h in hashes])
lf.index()
t1 = time.perf_counter()
cfg = tm.LayoutConfiguration()
cfg.k = 10
cfg.kc = 10
cfg.fme_threads = os.cpu_count()
x, y, s, t, _ = tm.layout_from_lsh_forest(lf, cfg)
t2 = time.perf_counter()

np.savez(sys.argv[2], pos=np.c_[np.asarray(x), np.asarray(y)], edges=np.c_[np.asarray(s), np.asarray(t)])
print(json.dumps({
    "index_s": t1 - t0,
    "layout_s": t2 - t1,
    "total_s": t2 - t0,
    "max_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
}))
