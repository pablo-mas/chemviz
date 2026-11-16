# Changelog

## Unreleased

- `Map.project()`: place new molecules on an existing map without moving it. TMAP grafts
  them onto its tree as new branches; UMAP and t-SNE follow umap-learn's and openTSNE's
  `transform`; PCA projects them on its axes. The result has a `projected` column. The CLI
  does it with `chemviz project map.chemviz new.csv`.
- `Map.save("map.chemviz")` and `chemviz.load()`: save a map and reload it without
  recomputing it. Files keep the coordinates, the data with its column types and the MinHash
  vectors (`model=False` leaves them out), and are read without pickle. The CLI writes them
  with `-o map.chemviz`.
- Maps keep their MinHash vectors (or PCA axes) in `Map.model`; t-SNE peak memory grows by
  512 bytes per molecule as a result.

## 0.1.0 (2026-11-16)

First release: from-scratch reimplementations of TMAP, UMAP, t-SNE and PCA for molecules, in
Python + numba.

- `chemviz.tmap()`: one-line tree map from SMILES lists or Series, DataFrames,
  `.csv` / `.tsv` / `.smi` / `.sdf` files, or precomputed MinHash vectors or binary
  fingerprints. Invalid SMILES are skipped with a warning.
- `chemviz.umap()`, `chemviz.tsne()`, `chemviz.pca()`: the same inputs and result for UMAP and
  t-SNE (on the LSH Forest kNN graph, close to umap-learn and openTSNE, deterministic) and PCA
  (exact, on folded ECFP fingerprints). No extra dependency.
- `Map` result: coordinates, tree edges (TMAP) and input data; `to_dataframe()`, static
  matplotlib `plot()`.
- Interactive WebGL viewer in a single HTML file (`save()`, `to_html()`, Jupyter
  `show()`): structures on hover (RDKit.js), colouring, filters, search, lasso selection
  and CSV export. `offline=True` embeds RDKit.js so the page needs no network.
- `chemviz tmap|umap|tsne|pca` command line (also `python -m chemviz`): writes the viewer, a static
  plot (`.png`, `.svg`, `.pdf`) or a `.csv` of coordinates, chosen by output extension.
- Building blocks: `minhash_ecfp`, `mhfp`, `minhash_from_fingerprints`, `lsh_knn`,
  `minimum_spanning_forest`, `tree_layout`, `umap_layout`, `tsne_layout`, `pca_layout`,
  `read_molecules`.
