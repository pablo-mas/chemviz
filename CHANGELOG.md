# Changelog

## 0.1.0 (unreleased)

First release: a from-scratch reimplementation of TMAP in Python + numba.

- `chemviz.tmap()`: one-line tree map from SMILES lists or Series, DataFrames,
  `.csv` / `.tsv` / `.smi` / `.sdf` files, or precomputed MinHash vectors or binary
  fingerprints. Invalid SMILES are skipped with a warning.
- `TMap` result: coordinates, tree edges and input data; `to_dataframe()`, static
  matplotlib `plot()`.
- Interactive WebGL viewer in a single HTML file (`save()`, `to_html()`, Jupyter
  `show()`): structures on hover (RDKit.js), colouring, filters, search, lasso selection
  and CSV export. `offline=True` embeds RDKit.js so the page needs no network.
- `chemviz tmap` command line (also `python -m chemviz`): writes the viewer, a static
  plot (`.png`, `.svg`, `.pdf`) or a `.csv` of coordinates, chosen by output extension.
- Building blocks: `minhash_ecfp`, `mhfp`, `minhash_from_fingerprints`, `lsh_knn`,
  `minimum_spanning_forest`, `tree_layout`, `read_molecules`.
