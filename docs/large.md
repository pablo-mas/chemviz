# Large datasets

chemviz is built to map millions of molecules on a laptop or a desktop. This page gives the
orders of magnitude and the settings that matter at that scale.

## Time and memory

End to end from SMILES, on a 20-thread desktop with 15 GB of RAM (see {doc}`benchmarks`):

| Method | 250,000 molecules | 1 million: time | 1 million: peak memory |
|---|---|---|---|
| TMAP | 9 s | 37 s | 2.6 GB |
| UMAP | 14 s | 58 s | 3.3 GB |
| t-SNE | 90 s | 6.5 min | 4.8 GB |
| PCA | 5 s | 19 s | 1.7 GB |

The whole of ChEMBL, 2.9 million molecules, takes under 2 minutes with TMAP and about 4 GB
of RAM. Time grows about linearly with the number of molecules, and every step runs in
parallel, so it shrinks with the number of CPU cores.

The first call in a new environment adds a few seconds for numba to compile chemviz's
functions; they are cached for the next calls.

## Where the time goes

For TMAP on ChEMBL, about half of the time is spent parsing SMILES and computing their
MinHash vectors, the rest on the neighbour search and the layout. Parsing uses every CPU by
default; `n_jobs=` limits it, for example on a shared machine:

```python
m = cv.tmap("chembl.csv", n_jobs=8)
```

## Compute once, reuse

Save the map as a `.chemviz` file as soon as it is computed: reloading it takes a second or
two, against minutes to recompute it, and it keeps everything needed to place new molecules
on it later ({doc}`saving`, {doc}`projection`).

```python
cv.tmap("chembl.csv").save("chembl.chemviz")
m = cv.load("chembl.chemviz")
```

The MinHash vectors stored in the file take 512 MB per million molecules; save with
`model=False` if you will not place new molecules on the map.

## The viewer at scale

The viewer handles millions of points: a map of a million molecules is about 40 MB and opens
in about a second in a desktop browser. Each embedded column adds to the size: embed only
the useful ones with `columns=`. Beyond a few million molecules, the file becomes heavy to
open in a browser; map a representative subset, or split the dataset.

## Lower memory use

- Prefer TMAP, UMAP or PCA to t-SNE: its affinities take the most memory.
- Keep only the needed columns in the input DataFrame: all of them are carried in the map.
- Precompute the MinHash vectors once with {func}`~chemviz.minhash_ecfp` and pass the array
  to several methods, instead of parsing the SMILES each time:

```python
hashes = cv.minhash_ecfp(smiles)
m1, m2 = cv.tmap(hashes), cv.umap(hashes)
```

The maps then have no SMILES column: add one to `m.data` (same row order) before saving the
viewer, so that it can draw structures.
