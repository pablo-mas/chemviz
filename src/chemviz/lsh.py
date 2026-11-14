"""Approximate k-nearest-neighbour search on MinHash vectors with an LSH Forest.

Each tree indexes a disjoint band of ``d // n_trees`` hash values, stored as the
lexicographic order of the rows plus the length of the common prefix (LCP) of each
pair of consecutive rows. Since every query is an indexed row, its candidates are
read off its sorted neighbourhood: walking away from it, the running minimum of the
LCP is the prefix length shared with the query. As in the LSH Forest descent, the
candidates sharing the longest prefixes (at least ``k * kc`` of them) are kept, then
ranked by MinHash distance (1 - fraction of equal hash values).

Ranking compares only the low byte of each hash value (b-bit MinHash), which reads
4 times less memory; the best ``2k`` are then re-ranked by their exact distance.
"""

import sys

import numba as nb
import numpy as np


@nb.njit(cache=True, inline="always")
def _pair_key(hashes, row, c, end):
    """Columns c and c + 1 of a row packed into one sortable uint64."""
    hi = np.uint64(hashes[row, c])
    lo = np.uint64(hashes[row, c + 1]) if c + 1 < end else np.uint64(0)
    return (hi << np.uint64(32)) | lo


@nb.njit(cache=True, inline="always")
def _equal(hashes, a, b, c0, c1):
    for c in range(c0, c1):
        if hashes[a, c] != hashes[b, c]:
            return False
    return True


@nb.njit(cache=True)
def _lexsort_band(hashes, c0, band):
    """Row order sorting ``hashes[:, c0:c0 + band]`` lexicographically."""
    n = hashes.shape[0]
    end = c0 + band
    keys = np.empty(n, dtype=np.uint64)
    for i in range(n):
        keys[i] = _pair_key(hashes, i, c0, end)
    order = np.argsort(keys).astype(np.int32)
    tmp = np.empty(n, dtype=np.int32)
    # Refine runs of rows equal on the columns sorted so far, two columns at a time.
    for c in range(c0 + 2, end, 2):
        s = 0
        while s < n:
            e = s + 1
            while e < n and _equal(hashes, order[s], order[e], c0, c):
                e += 1
            if e - s > 1:
                for i in range(s, e):
                    keys[i] = _pair_key(hashes, order[i], c, end)
                sub = np.argsort(keys[s:e])
                for i in range(e - s):
                    tmp[i] = order[s + sub[i]]
                order[s:e] = tmp[:e - s]
            s = e
    return order


@nb.njit(cache=True, parallel=True)
def _build(hashes, n_trees, band):
    """Per tree: sorted row order, rank of each row in it, and LCP with the previous sorted row."""
    n = hashes.shape[0]
    order = np.empty((n_trees, n), dtype=np.int32)
    rank = np.empty((n_trees, n), dtype=np.int32)
    lcp = np.empty((n_trees, n), dtype=np.uint8)
    for t in nb.prange(n_trees):
        c0 = t * band
        o = _lexsort_band(hashes, c0, band)
        order[t] = o
        lcp[t, 0] = 0
        for i in range(n):
            rank[t, o[i]] = i
            if i:
                m = 0
                while m < band and hashes[o[i - 1], c0 + m] == hashes[o[i], c0 + m]:
                    m += 1
                lcp[t, i] = m
    return order, rank, lcp


@nb.njit(cache=True, inline="always")
def _add(cand, level, table, stamp, mask, m, q, c, lv):
    """Add candidate ``c`` at prefix level ``lv`` (keeping its best level); returns the new count."""
    slot = np.int64((np.uint64(c) * np.uint64(0x9E3779B97F4A7C15)) >> np.uint64(40)) & mask
    while stamp[slot] == q:
        i = table[slot]
        if cand[i] == c:
            if lv > level[i]:
                level[i] = lv
            return m
        slot = (slot + 1) & mask
    stamp[slot] = q
    table[slot] = m
    cand[m] = c
    level[m] = lv
    return m + 1


@nb.njit(cache=True, parallel=True)
def _knn(hashes, sketch, order, rank, lcp, band, k, n_candidates, n_chunks):
    n, d = hashes.shape
    n_trees = order.shape[0]
    indices = np.full((n, k), -1, dtype=np.int64)
    distances = np.full((n, k), np.inf, dtype=np.float32)
    per_chunk = (n + n_chunks - 1) // n_chunks
    cap = n_trees * 2 * n_candidates
    size = 1
    while size < 2 * cap:
        size *= 2
    k2 = 2 * k

    for ch in nb.prange(n_chunks):
        cand = np.empty(cap, dtype=np.int64)
        level = np.empty(cap, dtype=np.int64)
        table = np.empty(size, dtype=np.int64)  # open-addressing set of candidates, reset by stamp
        stamp = np.full(size, -1, dtype=np.int64)
        hist = np.empty(band + 1, dtype=np.int64)
        best_c = np.empty(k2, dtype=np.int64)
        best_e = np.empty(k2, dtype=np.int64)
        for q in range(ch * per_chunk, min(n, (ch + 1) * per_chunk)):
            m = 0
            for t in range(n_trees):
                p = rank[t, q]
                lv = band
                j = p
                for _ in range(n_candidates):  # walk left while a prefix is shared
                    lv = min(lv, lcp[t, j])
                    if lv == 0:
                        break
                    j -= 1
                    m = _add(cand, level, table, stamp, size - 1, m, q, order[t, j], lv)
                lv = band
                j = p + 1
                for _ in range(n_candidates):  # walk right
                    if j >= n:
                        break
                    lv = min(lv, lcp[t, j])
                    if lv == 0:
                        break
                    m = _add(cand, level, table, stamp, size - 1, m, q, order[t, j], lv)
                    j += 1

            # Keep every candidate at or above the deepest level reaching n_candidates.
            thr = 1
            if m > n_candidates:
                hist[:] = 0
                for i in range(m):
                    hist[level[i]] += 1
                total = 0
                for lv in range(band, 0, -1):
                    total += hist[lv]
                    if total >= n_candidates:
                        thr = lv
                        break

            # Shortlist the 2k with most equal low bytes (ties by index), by insertion.
            hq = sketch[q]
            n_best = 0
            for i in range(m):
                if level[i] < thr:
                    continue
                c = cand[i]
                hc = sketch[c]
                e = 0
                for j in range(d):
                    e += hq[j] == hc[j]
                pos = n_best
                while pos > 0 and (best_e[pos - 1] < e or (best_e[pos - 1] == e and best_c[pos - 1] > c)):
                    pos -= 1
                if pos >= k2:
                    continue
                for s in range(min(n_best, k2 - 1), pos, -1):
                    best_e[s] = best_e[s - 1]
                    best_c[s] = best_c[s - 1]
                best_e[pos] = e
                best_c[pos] = c
                n_best = min(n_best + 1, k2)

            # Re-rank the shortlist by exact equal hash values and keep the k best.
            for i in range(n_best):
                c = best_c[i]
                e = 0
                for j in range(d):
                    e += hashes[q, j] == hashes[c, j]
                best_e[i] = e
            for i in range(min(k, n_best)):
                b = i
                for j in range(i + 1, n_best):
                    if best_e[j] > best_e[b] or (best_e[j] == best_e[b] and best_c[j] < best_c[b]):
                        b = j
                best_e[i], best_e[b] = best_e[b], best_e[i]
                best_c[i], best_c[b] = best_c[b], best_c[i]
                indices[q, i] = best_c[i]
                distances[q, i] = 1.0 - best_e[i] / d
    return indices, distances


def lsh_knn(
    hashes: np.ndarray,
    k: int = 10,
    n_trees: int = 8,
    kc: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """Approximate kNN of every row of ``hashes`` among the other rows.

    ``hashes`` is a ``(n, d)`` MinHash matrix; ``d`` must be divisible by ``n_trees``.
    ``kc`` scales the candidate pool (``k * kc``) ranked by MinHash distance.

    Returns ``(indices, distances)``, both ``(n, k)``, sorted by increasing exact
    distance. Missing neighbours (too few candidates) are marked by index -1 and
    distance inf.
    """
    hashes = np.ascontiguousarray(hashes, dtype=np.uint32)
    n, d = hashes.shape
    if d % n_trees:
        raise ValueError(f"Number of hash values ({d}) must be divisible by n_trees ({n_trees})")
    band = d // n_trees
    if band > 255:
        raise ValueError(f"At most 255 hash values per tree; got {band}, use more trees")
    order, rank, lcp = _build(hashes, n_trees, band)
    low_byte = 0 if sys.byteorder == "little" else 3  # low byte of each value, without a uint32 temporary
    sketch = np.ascontiguousarray(hashes.view(np.uint8).reshape(n, d, 4)[:, :, low_byte])
    n_chunks = max(1, min(n // 1000, nb.config.NUMBA_NUM_THREADS * 8))
    return _knn(hashes, sketch, order, rank, lcp, band, k, k * kc, n_chunks)
