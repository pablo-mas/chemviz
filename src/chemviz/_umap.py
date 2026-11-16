"""UMAP of molecules from their approximate kNN graph (McInnes, Healy & Melville, 2018).

The kNN graph comes from the LSH Forest (MinHash Jaccard distances) and the embedding is
optimised by stochastic gradient descent with negative sampling, as in umap-learn. Each
epoch updates all nodes in parallel from the positions at the start of the epoch, so
results are deterministic for a given seed whatever the number of threads.
"""

import numba as nb
import numpy as np

from chemviz.graph import _symmetrize, minimum_spanning_forest
from chemviz.layout import _threads_for, tree_layout
from chemviz.lsh import _lsh_query, lsh_knn
from chemviz.maps import Map, _drop_invalid, _encode


@nb.njit(cache=True, parallel=True)
def _smooth_knn(dist, n_iter=64, use_rho=True):
    """Per-point ``rho`` (distance to the nearest non-identical neighbour, or 0 without
    ``use_rho``) and ``sigma`` such that sum(exp(-(d - rho) / sigma)) = log2(k + 1) over
    the k neighbours."""
    n, k = dist.shape
    target = np.log2(k + 1)
    rho = np.zeros(n)
    sigma = np.ones(n)
    finite_sum, finite_n = 0.0, 0
    for i in range(n):
        for j in range(k):
            if np.isfinite(dist[i, j]):
                finite_sum += dist[i, j]
                finite_n += 1
    mean_all = finite_sum / max(finite_n, 1)
    for i in nb.prange(n):
        mean_i, cnt = 0.0, 0
        for j in range(k):
            d = dist[i, j]
            if np.isfinite(d):
                mean_i += d
                cnt += 1
                if use_rho and d > 0.0 and rho[i] == 0.0:
                    rho[i] = d
        lo, hi, mid = 0.0, np.inf, 1.0
        for _ in range(n_iter):
            s = 0.0
            for j in range(k):
                if np.isfinite(dist[i, j]):
                    s += np.exp(-max(dist[i, j] - rho[i], 0.0) / mid)
            if abs(s - target) < 1e-5:
                break
            if s > target:
                hi = mid
                mid = (lo + hi) / 2
            else:
                lo = mid
                mid = mid * 2 if hi == np.inf else (lo + hi) / 2
        floor = 1e-3 * (mean_i / cnt if rho[i] > 0.0 and cnt else mean_all)
        sigma[i] = max(mid, floor)
    return rho, sigma


def _fuzzy_graph(idx, dist, n_epochs):
    """Symmetric fuzzy kNN graph as CSR ``(indptr, nbrs, weights)``.

    Directed memberships exp(-(d - rho_i) / sigma_i) are combined by fuzzy union
    (a + b - ab); edges too weak to be sampled during ``n_epochs`` are dropped.
    """
    rho, sigma = _smooth_knn(dist.astype(np.float64))
    w = np.exp(-np.maximum(dist - rho[:, None], 0.0) / sigma[:, None])
    indptr, nbrs, weight = _symmetrize(idx, w, True)
    keep = weight >= weight.max(initial=0.0) / n_epochs
    kept_before = np.concatenate([[0], np.cumsum(keep)])
    return kept_before[indptr], nbrs[keep], weight[keep]


def _curve_params(spread, min_dist):
    """``a``, ``b`` such that 1 / (1 + a d^(2b)) fits the target membership curve in 2D
    (1 up to ``min_dist``, then exponential decay with scale ``spread``)."""
    x = np.linspace(0, 3 * spread, 300)
    y = np.where(x < min_dist, 1.0, np.exp(-(x - min_dist) / spread))
    x = x[1:]  # d = 0 gives 1 for any a, b
    y = y[1:]
    params = np.array([1.5, 0.9])
    for _ in range(100):  # Gauss-Newton on (log a, b) keeps a positive
        la, b = params
        xb = x ** (2 * b)
        f = 1 / (1 + np.exp(la) * xb)
        df = -(f**2)
        jac = np.stack([df * np.exp(la) * xb, df * np.exp(la) * xb * 2 * np.log(x)], axis=1)
        step = np.linalg.lstsq(jac, y - f, rcond=None)[0]
        params = params + step
        if np.abs(step).max() < 1e-10:
            break
    return float(np.exp(params[0])), float(params[1])


@nb.njit(cache=True)
def _rand_index(seed, epoch, entry, sample, n):
    """Deterministic pseudo-random integer in [0, n) (splitmix64 of the arguments)."""
    z = np.uint64(seed) * np.uint64(0x9E3779B97F4A7C15)
    z ^= np.uint64(epoch) * np.uint64(0xBF58476D1CE4E5B9)
    z ^= np.uint64(entry) * np.uint64(0x94D049BB133111EB)
    z ^= np.uint64(sample) * np.uint64(0xD6E8FEB86659FD93)
    z += np.uint64(0x9E3779B97F4A7C15)
    z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    z ^= z >> np.uint64(31)
    return np.int64(z % np.uint64(n))


@nb.njit(cache=True, inline="always")
def _clip(v):
    return min(max(v, -4.0), 4.0)


@nb.njit(cache=True, parallel=True)
def _epoch(pos, new, indptr, nbrs, eps, next_sample, eps_neg, next_neg, a, b, alpha, epoch, seed):
    n = pos.shape[0]
    for i in nb.prange(n):
        xi, yi = pos[i, 0], pos[i, 1]
        gx, gy = 0.0, 0.0
        for p in range(indptr[i], indptr[i + 1]):
            if next_sample[p] > epoch:
                continue
            j = nbrs[p]
            dx, dy = xi - pos[j, 0], yi - pos[j, 1]
            d2 = dx * dx + dy * dy
            if d2 > 0.0:
                coeff = -2.0 * a * b * d2 ** (b - 1.0) / (a * d2**b + 1.0)
                # umap-learn moves both ends of each directed edge, one edge after the
                # other, and the graph holds both directions. With simultaneous updates
                # from the same positions, 1.5x (not 2x) its attraction gives the same
                # balance with repulsion (same global structure on ZINC250k).
                gx += 1.5 * _clip(coeff * dx)
                gy += 1.5 * _clip(coeff * dy)
            next_sample[p] += eps[p]
            n_neg = int((epoch - next_neg[p]) / eps_neg[p])
            for s in range(n_neg):
                k = _rand_index(seed, epoch, p, s, n)
                dx, dy = xi - pos[k, 0], yi - pos[k, 1]
                d2 = dx * dx + dy * dy
                if d2 > 0.0:
                    coeff = 2.0 * b / ((0.001 + d2) * (a * d2**b + 1.0))
                    gx += _clip(coeff * dx)
                    gy += _clip(coeff * dy)
            next_neg[p] += n_neg * eps_neg[p]
        new[i, 0] = xi + alpha * gx
        new[i, 1] = yi + alpha * gy


def umap_layout(
    idx: np.ndarray,
    dist: np.ndarray,
    n_epochs: int | None = None,
    min_dist: float = 0.1,
    spread: float = 1.0,
    negative_sample_rate: int = 5,
    init: np.ndarray | None = None,
    seed: int = 0,
) -> np.ndarray:
    """2D UMAP embedding of a kNN graph (e.g. from ``lsh_knn``).

    ``idx`` and ``dist`` are ``(n, k)`` neighbour indices (-1 for missing) and distances.
    ``n_epochs`` defaults to 500 for up to 10,000 points, 200 above. ``init`` is an
    ``(n, 2)`` starting layout; by default, a quick tree layout of the graph's minimum
    spanning forest, which gives a better global structure than a random start.

    Returns an ``(n, 2)`` float32 array. Results are deterministic for a given ``seed``.
    """
    idx = np.asarray(idx, dtype=np.int64)
    dist = np.asarray(dist, dtype=np.float64)
    n = len(idx)
    if n_epochs is None:
        n_epochs = 500 if n <= 10_000 else 200
    if init is None:
        # A quick tree layout (fewer iterations than tmap's) is enough as a start.
        init = tree_layout(n, minimum_spanning_forest(idx, dist)[0], iters_coarse=150, iters_fine=10, seed=seed)
    pos = np.array(init, dtype=np.float64)
    span = pos.max(axis=0) - pos.min(axis=0)
    pos = 10 * (pos - pos.min(axis=0)) / np.where(span > 0, span, 1)  # umap-learn's [0, 10] box
    indptr, nbrs, weights = _fuzzy_graph(idx, dist, n_epochs)
    if len(nbrs) == 0:
        return pos.astype(np.float32)
    a, b = _curve_params(spread, min_dist)
    eps = weights.max() / weights
    next_sample = eps.copy()
    eps_neg = eps / negative_sample_rate
    next_neg = eps_neg.copy()
    new = np.empty_like(pos)
    with _threads_for(n):
        for epoch in range(n_epochs):
            _epoch(pos, new, indptr, nbrs, eps, next_sample, eps_neg, next_neg, a, b,
                   1.0 - epoch / n_epochs, epoch, seed)
            pos, new = new, pos
    return pos.astype(np.float32)


@nb.njit(cache=True, parallel=True)
def _transform_epochs(pos, ref_pos, idx, eps, n_epochs, a, b, negative_sample_rate, seed):
    """umap-learn's transform optimisation: each new point moves towards its reference
    neighbours and away from random reference points, which stay fixed."""
    m, k = idx.shape
    n_ref = ref_pos.shape[0]
    for i in nb.prange(m):
        next_sample = eps[i].copy()
        eps_neg = eps[i] / negative_sample_rate
        next_neg = eps_neg.copy()
        for epoch in range(n_epochs):
            alpha = 0.25 * (1.0 - epoch / n_epochs)  # umap-learn: a quarter of the fit's rate
            for c in range(k):
                if not np.isfinite(eps[i, c]) or next_sample[c] > epoch:
                    continue
                j = idx[i, c]
                dx, dy = pos[i, 0] - ref_pos[j, 0], pos[i, 1] - ref_pos[j, 1]
                d2 = dx * dx + dy * dy
                if d2 > 0.0:
                    coeff = -2.0 * a * b * d2 ** (b - 1.0) / (a * d2**b + 1.0)
                    pos[i, 0] += alpha * _clip(coeff * dx)
                    pos[i, 1] += alpha * _clip(coeff * dy)
                next_sample[c] += eps[i, c]
                n_neg = int((epoch - next_neg[c]) / eps_neg[c])
                for s in range(n_neg):
                    r = _rand_index(seed, epoch, i * k + c, s, n_ref)
                    dx, dy = pos[i, 0] - ref_pos[r, 0], pos[i, 1] - ref_pos[r, 1]
                    d2 = dx * dx + dy * dy
                    if d2 > 0.0:
                        coeff = 2.0 * b / ((0.001 + d2) * (a * d2**b + 1.0))
                        pos[i, 0] += alpha * _clip(coeff * dx)
                        pos[i, 1] += alpha * _clip(coeff * dy)
                next_neg[c] += n_neg * eps_neg[c]


def _transform(ref_pos, ref_hashes, hashes, model):
    """Positions of new molecules on a UMAP map, as umap-learn's ``transform``."""
    m, n_ref = len(hashes), len(ref_hashes)
    idx, dist = _lsh_query(ref_hashes, hashes, min(model["k"], n_ref), model["n_trees"], model["kc"])
    rho, sigma = _smooth_knn(np.where(idx >= 0, dist, np.inf).astype(np.float64), use_rho=False)
    w = np.where(idx >= 0, np.exp(-np.maximum(dist - rho[:, None], 0.0) / sigma[:, None]), 0.0)
    # Start at the membership-weighted mean of the neighbours (on an identical one if any).
    safe = np.maximum(idx, 0)
    pos = (w[:, :, None] * ref_pos[safe]).sum(axis=1) / w.sum(axis=1, keepdims=True)
    exact = (dist == 0) & (idx >= 0)
    has_exact = exact.any(axis=1)
    pos[has_exact] = ref_pos[safe[has_exact, exact[has_exact].argmax(axis=1)]]
    n_epochs = model["n_epochs"] // 3 if model["n_epochs"] else (100 if m <= 10_000 else 30)
    w[w < w.max() / n_epochs] = 0.0
    with np.errstate(divide="ignore"):
        eps = np.where(w > 0, w.max() / w, np.inf)
    a, b = _curve_params(1.0, model["min_dist"])
    pos = np.ascontiguousarray(pos, dtype=np.float64)
    _transform_epochs(pos, ref_pos.astype(np.float64), safe, eps, n_epochs, a, b, 5, model["seed"])
    return pos.astype(np.float32)


def umap(
    data,
    smiles: str | None = None,
    *,
    k: int = 15,
    min_dist: float = 0.1,
    n_epochs: int | None = None,
    kc: int = 10,
    n_permutations: int = 128,
    radius: int = 2,
    n_trees: int = 8,
    seed: int = 0,
    n_jobs: int | None = None,
) -> Map:
    """UMAP map of molecules.

    ``data`` is given as for ``tmap``. Molecules are encoded as MinHash of their ECFP
    substructures, linked to their ``k`` approximate nearest neighbours (Jaccard distance)
    and embedded with UMAP (``min_dist``, ``n_epochs``; see ``umap_layout``). Clusters are
    tighter and better separated than in a tree map, but there are no edges to follow.
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
        idx, dist = lsh_knn(hashes, k=min(k, n - 1), n_trees=n_trees, kc=kc)
        pos = umap_layout(idx, dist, n_epochs=n_epochs, min_dist=min_dist, seed=seed)
    edges = np.empty((0, 2), dtype=np.int64)
    return Map(pos[:, 0], pos[:, 1], edges, frame, np.asarray(invalid, dtype=np.int64), smiles, "umap",
               {"encoding": encoding, "radius": radius, "n_permutations": n_permutations, "n_trees": n_trees,
                "seed": seed, "k": k, "min_dist": min_dist, "n_epochs": n_epochs, "kc": kc, "hashes": hashes})
