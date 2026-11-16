"""TMAP: molecules in, tree map out."""

import numpy as np

from chemviz.graph import minimum_spanning_forest
from chemviz.layout import tree_layout
from chemviz.lsh import lsh_knn
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
