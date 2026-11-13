"""Minimum spanning forest of a kNN graph (Kruskal with union-find)."""

import numba as nb
import numpy as np


@nb.njit(cache=True, inline="always")
def _find(parent, i):
    while parent[i] != i:
        parent[i] = parent[parent[i]]  # path halving
        i = parent[i]
    return i


@nb.njit(cache=True)
def _kruskal(n, src, dst):
    """Indices of the sorted edges kept in the spanning forest."""
    parent = np.arange(n)
    kept = np.empty(n - 1, dtype=np.int64)
    m = 0
    for e in range(src.shape[0]):
        a, b = _find(parent, src[e]), _find(parent, dst[e])
        if a != b:
            parent[a] = b
            kept[m] = e
            m += 1
            if m == n - 1:
                break
    return kept[:m]


def minimum_spanning_forest(
    indices: np.ndarray, distances: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Minimum spanning forest of the undirected kNN graph given by ``lsh_knn``.

    Entries with index -1 (missing neighbours) are ignored. Disconnected
    components yield one tree each.

    Returns ``(edges, weights)``: an ``(n - n_components, 2)`` int64 array and
    the matching float32 distances.
    """
    n, k = indices.shape
    src = np.repeat(np.arange(n), k)
    dst = indices.ravel()
    w = distances.ravel()
    valid = dst >= 0
    src, dst, w = src[valid], dst[valid], w[valid]
    # Duplicate (i, j) / (j, i) edges are harmless: the second one closes a cycle and is skipped.
    order = np.argsort(w, kind="stable")
    src, dst, w = src[order], dst[order], w[order]
    kept = _kruskal(n, src, dst)
    return np.stack([src[kept], dst[kept]], axis=1), w[kept].astype(np.float32)
