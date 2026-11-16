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
    """Minimum spanning forest of an undirected kNN graph (Kruskal).

    Parameters
    ----------
    indices : numpy.ndarray
        ``(n, k)`` neighbour indices, as returned by `lsh_knn`; entries of -1 are ignored.
    distances : numpy.ndarray
        ``(n, k)`` matching distances, used as edge weights.

    Returns
    -------
    edges : numpy.ndarray
        ``(n - n_components, 2)`` int64 node pairs: one tree per connected component.
    weights : numpy.ndarray
        The matching float32 distances.
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


@nb.njit(cache=True)
def _row_entries(idx, w, in_ptr, in_src, in_col, i):
    """Sorted outgoing and incoming neighbours of row ``i`` with their weights."""
    k = idx.shape[1]
    out_j = np.empty(k, dtype=np.int64)
    out_w = np.empty(k)
    m = 0
    for c in range(k):
        j = idx[i, c]
        if j >= 0 and j != i:
            out_j[m], out_w[m] = j, w[i, c]
            m += 1
    order = np.argsort(out_j[:m])
    lo, hi = in_ptr[i], in_ptr[i + 1]
    in_order = np.argsort(in_src[lo:hi])
    in_j = in_src[lo:hi][in_order]
    in_w = np.empty(hi - lo)
    for t in range(hi - lo):  # weight of the edge in_j[t] -> i, read from its kNN row
        in_w[t] = w[in_j[t], in_col[lo + in_order[t]]]
    return out_j[:m][order], out_w[:m][order], in_j, in_w


@nb.njit(cache=True, parallel=True)
def _symmetrize(idx, w, union):
    """Symmetric CSR graph ``(indptr, nbrs, weights)`` from directed kNN weights.

    Each pair gets a = w(i -> j) and b = w(j -> i) (0 if absent), combined as a + b, or
    by fuzzy union a + b - ab if ``union``; weights keep the dtype of ``w``. Rows are merged
    in parallel without forming all (i, j) pairs at once, so memory stays proportional to
    the output. At most 65,535 neighbours per row.
    """
    n, k = idx.shape
    in_ptr = np.zeros(n + 1, dtype=np.int64)
    for i in range(n):
        for c in range(k):
            j = idx[i, c]
            if j >= 0 and j != i:
                in_ptr[j + 1] += 1
    for i in range(n):
        in_ptr[i + 1] += in_ptr[i]
    in_src = np.empty(in_ptr[n], dtype=np.int32)
    in_col = np.empty(in_ptr[n], dtype=np.uint16)  # column of the edge in its source's kNN row
    fill = in_ptr[:-1].copy()
    for i in range(n):
        for c in range(k):
            j = idx[i, c]
            if j >= 0 and j != i:
                in_src[fill[j]], in_col[fill[j]] = i, c
                fill[j] += 1
    counts = np.zeros(n + 1, dtype=np.int64)
    for i in nb.prange(n):  # pass 1: distinct neighbours per row
        out_j, _, in_j, _ = _row_entries(idx, w, in_ptr, in_src, in_col, i)
        a, b, m = 0, 0, 0
        while a < len(out_j) or b < len(in_j):
            if b == len(in_j) or (a < len(out_j) and out_j[a] < in_j[b]):
                a += 1
            elif a == len(out_j) or in_j[b] < out_j[a]:
                b += 1
            else:
                a += 1
                b += 1
            m += 1
        counts[i + 1] = m
    indptr = np.cumsum(counts)
    nbrs = np.empty(indptr[n], dtype=np.int32)
    weights = np.empty(indptr[n], dtype=w.dtype)
    for i in nb.prange(n):  # pass 2: merged entries
        out_j, out_w, in_j, in_wi = _row_entries(idx, w, in_ptr, in_src, in_col, i)
        a, b, p = 0, 0, indptr[i]
        while a < len(out_j) or b < len(in_j):
            if b == len(in_j) or (a < len(out_j) and out_j[a] < in_j[b]):
                j, x, y = out_j[a], out_w[a], 0.0
                a += 1
            elif a == len(out_j) or in_j[b] < out_j[a]:
                j, x, y = in_j[b], 0.0, in_wi[b]
                b += 1
            else:
                j, x, y = out_j[a], out_w[a], in_wi[b]
                a += 1
                b += 1
            nbrs[p] = j
            weights[p] = x + y - x * y if union else x + y
            p += 1
    return indptr, nbrs, weights
