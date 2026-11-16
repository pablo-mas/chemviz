"""Benchmark Map.project: map part of a dataset, place the rest on it, compare with a full map.

Usage: uv run python benchmarks/compare_projection.py [fxa|zinc|chembl] [n_molecules|all] [held_out_fraction]

For each method, a map of all molecules and a map of the reference part (1 - held_out_fraction,
random) are computed; the held-out molecules are then projected onto the reference map.
kNN preservation (share of a molecule's 10 exact MinHash neighbours, among all molecules,
that are among its 10 nearest points in 2D) is measured on 2000 held-out molecules in
both maps, and on 2000 reference molecules.
"""

import sys
import time
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "prototypes"))
sys.path.insert(0, str(Path(__file__).parent))
from compare_layouts import load  # noqa: E402
from compare_tmap import exact_knn, knn_preservation  # noqa: E402

import chemviz as cv  # noqa: E402


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "zinc"
    limit = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2] != "all" else None
    held_out = float(sys.argv[3]) if len(sys.argv) > 3 else 0.1
    warnings.simplefilter("ignore")
    smiles, _, _ = load(name, limit)
    smiles = np.asarray(smiles, dtype=object)
    n = len(smiles)
    rng = np.random.default_rng(0)
    perm = rng.permutation(n)
    new_ids, ref_ids = np.sort(perm[: int(n * held_out)]), np.sort(perm[int(n * held_out):])
    hashes = cv.minhash_ecfp(list(smiles))
    q_new = rng.choice(new_ids, min(2000, len(new_ids)), replace=False)
    q_ref = rng.choice(ref_ids, min(2000, len(ref_ids)), replace=False)
    true_new, true_ref = exact_knn(hashes, q_new, 10), exact_knn(hashes, q_ref, 10)
    for f in (cv.pca, cv.tmap, cv.umap, cv.tsne):  # compile outside the timings
        f(list(smiles[:300])).project(list(smiles[300:400]))

    print(f"{name} n={n}: reference {len(ref_ids)}, projected {len(new_ids)}")
    print("  method  full map: kNN held-out / ref   projected: kNN held-out / ref   project time")
    for method in ("pca", "tmap", "umap", "tsne"):
        full = getattr(cv, method)(list(smiles))
        full_pos = np.column_stack([full.x, full.y]).astype(np.float64)
        ref = getattr(cv, method)(list(smiles[ref_ids]))
        start = time.perf_counter()
        both = ref.project(list(smiles[new_ids]))
        elapsed = time.perf_counter() - start
        pos = np.empty((n, 2))
        pos[ref_ids] = np.column_stack([both.x[: len(ref_ids)], both.y[: len(ref_ids)]])
        pos[new_ids] = np.column_stack([both.x[len(ref_ids):], both.y[len(ref_ids):]])
        print(f"  {method:5s}   {knn_preservation(full_pos, q_new, true_new):.3f} / "
              f"{knn_preservation(full_pos, q_ref, true_ref):.3f}"
              f"                     {knn_preservation(pos, q_new, true_new):.3f} / "
              f"{knn_preservation(pos, q_ref, true_ref):.3f}                    {elapsed:6.1f} s", flush=True)


if __name__ == "__main__":
    main()
