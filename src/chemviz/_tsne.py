"""t-SNE of molecules from their approximate kNN graph (van der Maaten & Hinton, 2008).

Affinities come from the LSH Forest kNN graph (MinHash Jaccard distances, perplexity
calibrated) and the embedding is optimised as in openTSNE: early exaggeration, momentum,
gains and Barnes-Hut repulsion on an adaptive quadtree.
"""

import numba as nb
import numpy as np

from chemviz.graph import _symmetrize, minimum_spanning_forest
from chemviz.layout import _threads_for, tree_layout
from chemviz.lsh import lsh_knn
from chemviz.maps import Map, _drop_invalid, _encode


@nb.njit(cache=True)
def _bisect_row(d2, dmin, target, n_iter, row):
    """Fill ``row`` with exp(-(d2 - dmin) * beta) normalised, with entropy ``target``."""
    lo, hi, beta, total = 0.0, np.inf, 1.0, 1.0
    for _ in range(n_iter):
        total, weighted = 0.0, 0.0
        for j in range(d2.shape[0]):
            row[j] = np.exp(-(d2[j] - dmin) * beta) if np.isfinite(d2[j]) else 0.0
            total += row[j]
            weighted += row[j] * (d2[j] - dmin) if np.isfinite(d2[j]) else 0.0
        entropy = np.log(total) + beta * weighted / total
        if abs(entropy - target) < 1e-8:
            break
        if entropy > target:  # too flat: sharpen
            lo = beta
            beta = beta * 2 if hi == np.inf else (lo + hi) / 2
        else:
            hi = beta
            beta = (lo + hi) / 2
    for j in range(d2.shape[0]):
        row[j] /= total


@nb.njit(cache=True, parallel=True)
def _conditional_p(dist, perplexity, n_iter=200):
    """Gaussian conditional probabilities over each row's neighbours (squared distances),
    with the bandwidth set so that their perplexity is ``perplexity``."""
    n, k = dist.shape
    out = np.zeros((n, k), dtype=np.float32)  # enough for probabilities only summed in gradients
    target = np.log(perplexity)
    for i in nb.prange(n):
        d2 = np.empty(k)
        row = np.zeros(k)
        dmin = np.inf
        for j in range(k):
            d = np.float64(dist[i, j])
            d2[j] = d * d if np.isfinite(d) else np.inf
            dmin = min(dmin, d2[j])
        if np.isfinite(dmin):  # rows without neighbours stay zero
            _bisect_row(d2, dmin, target, n_iter, row)
            for j in range(k):
                out[i, j] = row[j]
    return out


def _affinities(idx, dist, perplexity):
    """Symmetric joint probabilities P = (P_cond + P_cond^T) / sum, as CSR ``(indptr, nbrs, p)``."""
    cond = _conditional_p(dist, float(perplexity))
    indptr, nbrs, p = _symmetrize(idx, cond, False)
    p /= p.sum()
    return indptr, nbrs, p


_DEPTH = 16  # quadtree depth limit: Morton codes hold 16 bits per axis
_LEAF = 8  # points per leaf, summed exactly
_THETA = 0.5  # Barnes-Hut opening criterion (cell width / distance), as in openTSNE


@nb.njit(cache=True, parallel=True)
def _morton_order(pos):
    """Positions sorted along a Z-order curve, their Morton codes, the permutation
    (sorted -> original index) and the side of the square bounding box."""
    n = pos.shape[0]
    lo0, lo1, hi0, hi1 = pos[0, 0], pos[0, 1], pos[0, 0], pos[0, 1]
    for i in range(n):
        lo0, hi0 = min(lo0, pos[i, 0]), max(hi0, pos[i, 0])
        lo1, hi1 = min(lo1, pos[i, 1]), max(hi1, pos[i, 1])
    span = max(hi0 - lo0, hi1 - lo1) * (1 + 1e-9) + 1e-12
    codes = np.empty(n, dtype=np.int64)
    side = 1 << _DEPTH
    for i in nb.prange(n):
        qx = min(int((pos[i, 0] - lo0) / span * side), side - 1)
        qy = min(int((pos[i, 1] - lo1) / span * side), side - 1)
        code = 0
        for bit in range(_DEPTH):
            code |= ((qx >> bit) & 1) << (2 * bit + 1) | ((qy >> bit) & 1) << (2 * bit)
        codes[i] = code
    order = np.argsort(codes, kind="mergesort")  # nearly sorted from one iteration to the next
    spos = np.empty_like(pos)
    scodes = np.empty_like(codes)
    for i in nb.prange(n):
        spos[i, 0], spos[i, 1] = pos[order[i], 0], pos[order[i], 1]
        scodes[i] = codes[order[i]]
    return spos, scodes, order, span


@nb.njit(cache=True)
def _build_tree(spos, codes, start, end, depth, child, nchild, mass, cx, cy):
    """Adaptive quadtree over Z-ordered points, filling the node arrays in place.

    A node covers a contiguous range of points; its children (consecutive nodes) split
    it by the next two bits of the Morton code. Returns the node count, or -1 if the
    arrays are too small.
    """
    n = spos.shape[0]
    cap = child.shape[0]
    start[0], end[0], depth[0] = 0, n, 0
    count, head = 1, 0
    while head < count:  # breadth first: children are created after their parent
        u = head
        head += 1
        lo, hi, d = start[u], end[u], depth[u]
        if hi - lo <= _LEAF or d == _DEPTH:
            child[u] = -1
            continue
        if count + 4 > cap:
            return -1
        child[u] = count
        shift = 2 * (_DEPTH - 1 - d)
        first = lo
        for q in range(4):
            last = hi
            if q < 3:  # first point whose quadrant is beyond q (quadrants are sorted)
                a, b = first, hi
                while a < b:
                    m = (a + b) // 2
                    if (codes[m] >> shift) & 3 > q:
                        b = m
                    else:
                        a = m + 1
                last = a
            if last > first:
                start[count], end[count], depth[count] = first, last, d + 1
                count += 1
            first = last
        nchild[u] = count - child[u]
    for u in range(count - 1, -1, -1):  # children before parents
        m, x, y = 0.0, 0.0, 0.0
        if child[u] == -1:
            for i in range(start[u], end[u]):
                x += spos[i, 0]
                y += spos[i, 1]
            m = end[u] - start[u]
        else:
            for v in range(child[u], child[u] + nchild[u]):
                m += mass[v]
                x += cx[v] * mass[v]
                y += cy[v] * mass[v]
        mass[u], cx[u], cy[u] = m, x / m, y / m
    return count


@nb.njit(cache=True, parallel=True)
def _repulsion(spos, order, span, start, end, depth, child, nchild, mass, cx, cy, rep, sum_w, n_blocks):
    """Unnormalised t-SNE repulsion sum_j w_ij^2 (y_i - y_j) and sum_j w_ij, w = 1 / (1 + d^2),
    by Barnes-Hut on the quadtree, written in original order."""
    n = spos.shape[0]
    n_blocks = min(n, n_blocks)
    theta2 = _THETA * _THETA
    for blk in nb.prange(n_blocks):
        stack = np.empty(4 * _DEPTH + 8, dtype=np.int64)
        for i in range(blk * n // n_blocks, (blk + 1) * n // n_blocks):
            xi, yi = spos[i, 0], spos[i, 1]
            fx, fy, sw = 0.0, 0.0, 0.0
            stack[0] = 0
            sp = 1
            while sp > 0:
                sp -= 1
                u = stack[sp]
                dx, dy = xi - cx[u], yi - cy[u]
                d2 = dx * dx + dy * dy
                width = span / (1 << depth[u])
                if child[u] == -1:
                    for j in range(start[u], end[u]):
                        if j != i:
                            dx, dy = xi - spos[j, 0], yi - spos[j, 1]
                            w = 1.0 / (1.0 + dx * dx + dy * dy)
                            sw += w
                            fx += w * w * dx
                            fy += w * w * dy
                elif width * width < theta2 * d2:
                    w = 1.0 / (1.0 + d2)
                    sw += mass[u] * w
                    fx += mass[u] * w * w * dx
                    fy += mass[u] * w * w * dy
                else:
                    for v in range(child[u], child[u] + nchild[u]):
                        stack[sp] = v
                        sp += 1
            rep[order[i], 0] = fx
            rep[order[i], 1] = fy
            sum_w[order[i]] = sw


@nb.njit(cache=True, parallel=True)
def _step(pos, indptr, nbrs, p, rep, z, exaggeration, lr, momentum, update, gains, max_step):
    """Gradient (attraction exaggeration * sum_j p_ij w_ij (y_i - y_j) minus repulsion / Z)
    and openTSNE's momentum + gains update, capped at ``max_step`` per point."""
    n = pos.shape[0]
    for i in nb.prange(n):
        gx, gy = -rep[i, 0] / z, -rep[i, 1] / z
        for q in range(indptr[i], indptr[i + 1]):
            j = nbrs[q]
            dx, dy = pos[i, 0] - pos[j, 0], pos[i, 1] - pos[j, 1]
            w = exaggeration * p[q] / (1.0 + dx * dx + dy * dy)
            gx += w * dx
            gy += w * dy
        norm = 0.0
        for d, g in ((0, gx), (1, gy)):
            if (update[i, d] > 0) != (g > 0):
                gains[i, d] += 0.2
            else:
                gains[i, d] = gains[i, d] * 0.8 + 0.01
            update[i, d] = momentum * update[i, d] - lr * gains[i, d] * g
            norm += update[i, d] * update[i, d]
        norm = np.sqrt(norm)
        if norm > max_step:
            update[i, 0] *= max_step / norm
            update[i, 1] *= max_step / norm


@nb.njit(cache=True)
def _optimize(pos, indptr, nbrs, p, n_iter, exaggeration, lr, momentum, update, gains, n_blocks):
    n = pos.shape[0]
    cap = 4 * n + 64
    start = np.empty(cap, dtype=np.int64)
    end = np.empty(cap, dtype=np.int64)
    depth = np.empty(cap, dtype=np.int64)
    child = np.empty(cap, dtype=np.int64)
    nchild = np.empty(cap, dtype=np.int64)
    mass = np.empty(cap)
    cx = np.empty(cap)
    cy = np.empty(cap)
    rep = np.empty_like(pos)
    sum_w = np.empty(n)
    for _ in range(n_iter):
        spos, codes, order, span = _morton_order(pos)
        n_nodes = _build_tree(spos, codes, start, end, depth, child, nchild, mass, cx, cy)
        while n_nodes < 0:  # grow the node arrays
            cap *= 2
            start = np.empty(cap, dtype=np.int64)
            end = np.empty(cap, dtype=np.int64)
            depth = np.empty(cap, dtype=np.int64)
            child = np.empty(cap, dtype=np.int64)
            nchild = np.empty(cap, dtype=np.int64)
            mass = np.empty(cap)
            cx = np.empty(cap)
            cy = np.empty(cap)
            n_nodes = _build_tree(spos, codes, start, end, depth, child, nchild, mass, cx, cy)
        _repulsion(spos, order, span, start, end, depth, child, nchild, mass, cx, cy, rep, sum_w, n_blocks)
        _step(pos, indptr, nbrs, p, rep, sum_w.sum(), exaggeration, lr, momentum, update, gains, 5.0)
        mx, my = 0.0, 0.0
        for i in range(n):
            pos[i, 0] += update[i, 0]
            pos[i, 1] += update[i, 1]
            mx += pos[i, 0]
            my += pos[i, 1]
        for i in range(n):
            pos[i, 0] -= mx / n
            pos[i, 1] -= my / n
    return pos


def tsne_layout(
    idx: np.ndarray,
    dist: np.ndarray,
    perplexity: float = 30.0,
    n_iter: int = 500,
    early_exaggeration_iter: int = 250,
    early_exaggeration: float = 12.0,
    init: np.ndarray | None = None,
    seed: int = 0,
) -> np.ndarray:
    """2D t-SNE embedding of a kNN graph (e.g. from ``lsh_knn`` with ``k = 3 * perplexity``).

    ``idx`` and ``dist`` are ``(n, k)`` neighbour indices (-1 for missing) and distances.
    The optimisation follows openTSNE's defaults: ``early_exaggeration_iter`` iterations
    with exaggerated attraction, then ``n_iter`` normal ones, learning rate
    n / exaggeration and momentum 0.8. ``init`` defaults to the tree layout of the minimum
    spanning forest of the 30 nearest neighbours (rescaled to a standard deviation of 1e-4).

    Returns an ``(n, 2)`` float32 array. Results are deterministic for a given ``seed``.
    """
    idx = np.asarray(idx, dtype=np.int64)
    dist = np.asarray(dist)
    n, k = idx.shape
    perplexity = min(perplexity, max(k / 3, 1.0))
    if init is None:
        # A full tree layout gives a better global structure than a quick one or a random start.
        # Its tree spans the 30 nearest neighbours: a better global structure than 10 or 15,
        # and as good as all 90 for a third of the memory.
        init = tree_layout(n, minimum_spanning_forest(idx[:, :30], dist[:, :30])[0], seed=seed)
    pos = np.array(init, dtype=np.float64)
    pos -= pos.mean(axis=0)
    pos *= 1e-4 / (pos[:, 0].std() or 1.0)
    indptr, nbrs, p = _affinities(idx, dist, perplexity)
    update, gains = np.zeros_like(pos), np.ones_like(pos)
    with _threads_for(n):
        for iters, exaggeration in ((early_exaggeration_iter, early_exaggeration), (n_iter, 1.0)):
            pos = _optimize(pos, indptr, nbrs, p, iters, exaggeration, max(n / exaggeration, 50.0), 0.8,
                            update, gains, 64 * nb.get_num_threads())
    return pos.astype(np.float32)


def tsne(
    data,
    smiles: str | None = None,
    *,
    perplexity: float = 30.0,
    n_iter: int = 500,
    kc: int = 3,
    n_permutations: int = 128,
    radius: int = 2,
    n_trees: int = 8,
    seed: int = 0,
    n_jobs: int | None = None,
) -> Map:
    """t-SNE map of molecules.

    ``data`` is given as for ``tmap``. Molecules are encoded as MinHash of their ECFP
    substructures, linked to their ``3 * perplexity`` approximate nearest neighbours
    (Jaccard distance; ``kc`` scales the LSH candidate pool) and embedded with t-SNE
    (see ``tsne_layout``). Like UMAP, it separates clusters well but distances between
    clusters are not meaningful, and there are no edges to follow.
    """
    hashes, frame, smiles, invalid, encoding = _encode(data, smiles, n_permutations, radius, n_jobs)
    keep, frame = _drop_invalid(frame, invalid)
    if keep is not None:
        hashes = hashes[keep]
    n = len(hashes)
    if n < 3:
        pos = np.zeros((n, 2), dtype=np.float32)
        pos[:, 0] = np.arange(n)
    else:
        idx, dist = lsh_knn(hashes, k=min(int(3 * perplexity), n - 1), n_trees=n_trees, kc=kc)
        pos = tsne_layout(idx, dist, perplexity=perplexity, n_iter=n_iter, seed=seed)
    edges = np.empty((0, 2), dtype=np.int64)
    return Map(pos[:, 0], pos[:, 1], edges, frame, np.asarray(invalid, dtype=np.int64), smiles, "tsne",
               {"encoding": encoding, "radius": radius, "n_permutations": n_permutations, "n_trees": n_trees,
                "seed": seed, "perplexity": perplexity, "n_iter": n_iter, "kc": kc, "hashes": hashes})
