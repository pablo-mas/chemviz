"""Approximate k-nearest-neighbour search on MinHash vectors with an LSH Forest.

Each tree indexes a disjoint band of ``d // n_trees`` hash values, stored as
lexicographically sorted rows. A query walks prefix lengths from the full band
down to 1 until enough candidates are found, then ranks candidates by their
exact MinHash distance (1 - fraction of equal hash values).
"""

import numba as nb
import numpy as np


@nb.njit(cache=True, inline="always")
def _compare(row, query, r):
    for j in range(r):
        if row[j] < query[j]:
            return -1
        if row[j] > query[j]:
            return 1
    return 0


@nb.njit(cache=True)
def _prefix_range(keys, query, r):
    """Range [lo, hi) of sorted ``keys`` rows whose first ``r`` values equal ``query``'s."""
    n = keys.shape[0]
    lo, hi = 0, n
    while lo < hi:  # first row >= query
        mid = (lo + hi) // 2
        if _compare(keys[mid], query, r) < 0:
            lo = mid + 1
        else:
            hi = mid
    start = lo
    hi = n
    while lo < hi:  # first row > query
        mid = (lo + hi) // 2
        if _compare(keys[mid], query, r) <= 0:
            lo = mid + 1
        else:
            hi = mid
    return start, lo


@nb.njit(cache=True, parallel=True)
def _knn(hashes, sorted_keys, sorted_idx, k, n_candidates):
    n, d = hashes.shape
    n_trees, _, band = sorted_keys.shape
    indices = np.full((n, k), -1, dtype=np.int64)
    distances = np.full((n, k), np.inf, dtype=np.float32)

    for q in nb.prange(n):
        buffer = np.empty(n_trees * n_candidates, dtype=np.int64)
        uniq = buffer[:0]
        for r in range(band, 0, -1):
            m = 0
            for t in range(n_trees):
                query = hashes[q, t * band:(t + 1) * band]
                lo, hi = _prefix_range(sorted_keys[t], query, r)
                for p in range(lo, min(hi, lo + n_candidates)):
                    buffer[m] = sorted_idx[t, p]
                    m += 1
            uniq = np.unique(buffer[:m])
            if uniq.shape[0] > n_candidates:  # > because the query itself is a candidate
                break

        cand = uniq[uniq != q]
        cand_dist = np.empty(cand.shape[0], dtype=np.float32)
        for c in range(cand.shape[0]):
            equal = 0
            for j in range(d):
                if hashes[q, j] == hashes[cand[c], j]:
                    equal += 1
            cand_dist[c] = 1.0 - equal / d
        order = np.argsort(cand_dist, kind="mergesort")
        for i in range(min(k, order.shape[0])):
            indices[q, i] = cand[order[i]]
            distances[q, i] = cand_dist[order[i]]
    return indices, distances


def lsh_knn(
    hashes: np.ndarray,
    k: int = 10,
    n_trees: int = 8,
    kc: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """Approximate kNN of every row of ``hashes`` among the other rows.

    ``hashes`` is a ``(n, d)`` MinHash matrix; ``d`` must be divisible by ``n_trees``.
    ``kc`` scales the candidate pool (``k * kc``) ranked by exact MinHash distance.

    Returns ``(indices, distances)``, both ``(n, k)``, sorted by increasing distance.
    Missing neighbours (too few candidates) are marked by index -1 and distance inf.
    """
    hashes = np.ascontiguousarray(hashes, dtype=np.uint32)
    n, d = hashes.shape
    if d % n_trees:
        raise ValueError(f"Number of hash values ({d}) must be divisible by n_trees ({n_trees})")
    band = d // n_trees
    sorted_keys = np.empty((n_trees, n, band), dtype=np.uint32)
    sorted_idx = np.empty((n_trees, n), dtype=np.int64)
    for t in range(n_trees):
        keys = hashes[:, t * band:(t + 1) * band]
        order = np.lexsort(keys.T[::-1])  # lexsort uses the last key as primary
        sorted_keys[t] = keys[order]
        sorted_idx[t] = order
    return _knn(hashes, sorted_keys, sorted_idx, k, k * kc)
