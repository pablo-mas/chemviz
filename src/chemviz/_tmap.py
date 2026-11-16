"""TMAP: molecules in, tree map out."""

from collections import deque

import numpy as np

from chemviz.graph import _kruskal, minimum_spanning_forest
from chemviz.layout import _tree_csr, tree_layout
from chemviz.lsh import _lsh_query, lsh_knn
from chemviz.maps import Map, _drop_invalid, _encode


def tmap(
    data,
    smiles: str | None = None,
    *,
    k: int = 10,
    kc: int = 10,
    n_permutations: int = 128,
    radius: int = 2,
    n_trees: int = 8,
    seed: int = 0,
    n_jobs: int | None = None,
) -> Map:
    """Compute a tree map (TMAP) of molecules.

    ``data`` can be:

    - a list, tuple or Series of SMILES;
    - a DataFrame, with its SMILES column given by ``smiles`` (guessed if omitted);
    - a path to a .csv, .tsv, .smi or .sdf file;
    - a precomputed 2D array: uint32 MinHash vectors, or binary fingerprints (bool or 0/1).

    Molecules are encoded as MinHash of their ECFP substructures (``radius``,
    ``n_permutations``), linked to their ``k`` approximate nearest neighbours (LSH
    Forest with ``n_trees`` trees, ``k * kc`` candidates), reduced to a minimum spanning
    tree and laid out in 2D. Unparsable SMILES are skipped with a warning.
    """
    hashes, frame, smiles, invalid, encoding = _encode(data, smiles, n_permutations, radius, n_jobs)
    keep, frame = _drop_invalid(frame, invalid)
    if keep is not None:
        hashes = hashes[keep]
    n = len(hashes)
    if n == 1:
        edges = np.empty((0, 2), dtype=np.int64)
    else:
        idx, dist = lsh_knn(hashes, k=min(k, n - 1), n_trees=n_trees, kc=kc)
        edges, _ = minimum_spanning_forest(idx, dist)
    pos = tree_layout(n, edges, seed=seed)
    return Map(pos[:, 0], pos[:, 1], edges, frame, np.asarray(invalid, dtype=np.int64), smiles, "tmap",
               {"encoding": encoding, "radius": radius, "n_permutations": n_permutations, "n_trees": n_trees,
                "seed": seed, "k": k, "kc": kc, "hashes": hashes})


def _transform(ref_pos, ref_edges, ref_hashes, hashes, model):
    """Positions of new molecules on a tree map, and the tree edges linking them.

    The new edges are those of the minimum spanning forest of the reference tree plus the
    kNN edges of the new molecules (to the reference and among themselves), with the
    reference tree kept: each group of related new molecules forms its own branch, grafted
    onto its closest reference molecule.
    """
    n_ref, m = len(ref_hashes), len(hashes)
    k = min(model["k"], n_ref)
    idx, dist = _lsh_query(ref_hashes, hashes, k, model["n_trees"], model["kc"])
    src, dst, w = [np.repeat(np.arange(m), k) + n_ref], [idx.ravel()], [dist.ravel()]
    if m > 1:
        k2 = min(k, m - 1)
        idx2, dist2 = lsh_knn(hashes, k=k2, n_trees=model["n_trees"], kc=model["kc"])
        src.append(np.repeat(np.arange(m), k2) + n_ref)
        dst.append(np.where(idx2 >= 0, idx2 + n_ref, -1).ravel())
        w.append(dist2.ravel())
    src, dst, w = np.concatenate(src), np.concatenate(dst), np.concatenate(w)
    valid = dst >= 0
    order = np.argsort(w[valid], kind="stable")
    all_src = np.concatenate([ref_edges[:, 0], src[valid][order]])
    all_dst = np.concatenate([ref_edges[:, 1], dst[valid][order]])
    kept = _kruskal(n_ref + m, all_src, all_dst)  # reference edges come first: all kept
    kept = kept[kept >= len(ref_edges)]
    edges = np.stack([all_src[kept], all_dst[kept]], axis=1).astype(np.int64)
    return _place(ref_pos, ref_edges, edges, m), edges


def _place(ref_pos, ref_edges, edges, m):
    """Lay out new tree nodes outwards from the reference: the new children of a node
    share the widest angular gap between its placed neighbours, at its edge length."""
    n_ref = len(ref_pos)
    n = n_ref + m
    pos = np.zeros((n, 2))
    pos[:n_ref] = ref_pos
    placed = np.zeros(n, dtype=bool)
    placed[:n_ref] = True
    indptr, nbrs = _tree_csr(n, np.concatenate([ref_edges, edges]))
    lengths = np.linalg.norm(ref_pos[ref_edges[:, 0]] - ref_pos[ref_edges[:, 1]], axis=1)
    lengths = lengths[lengths > 0]
    default = float(np.median(lengths)) if len(lengths) else 1.0
    queue = deque(np.unique(edges[edges < n_ref]).tolist())
    while queue:
        p = queue.popleft()
        around = nbrs[indptr[p]:indptr[p + 1]]
        kids = around[~placed[around]]
        if len(kids) == 0:
            continue
        vec = pos[around[placed[around]]] - pos[p]
        norms = np.linalg.norm(vec, axis=1)
        vec, norms = vec[norms > 0], norms[norms > 0]
        length = float(np.median(norms)) if len(norms) else default
        c = len(kids)
        if len(vec) == 0:
            angles = 2 * np.pi * np.arange(c) / c
        else:
            ang = np.sort(np.arctan2(vec[:, 1], vec[:, 0]))
            gaps = np.diff(np.append(ang, ang[0] + 2 * np.pi))
            g = gaps.argmax()
            angles = ang[g] + gaps[g] * np.arange(1, c + 1) / (c + 1)
        pos[kids] = pos[p] + length * np.column_stack([np.cos(angles), np.sin(angles)])
        placed[kids] = True
        queue.extend(kids.tolist())
    return pos[n_ref:].astype(np.float32)
