# chemviz

[![PyPI](https://img.shields.io/pypi/v/chemviz)](https://pypi.org/project/chemviz/)
[![CI](https://github.com/pablo-mas/chemviz/actions/workflows/ci.yml/badge.svg)](https://github.com/pablo-mas/chemviz/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue)](https://github.com/pablo-mas/chemviz/blob/main/LICENSE)

Fast, easy-to-install chemical space visualization.

chemviz maps molecules with [TMAP](https://doi.org/10.1186/s13321-020-0416-x) (Probst &
Reymond), UMAP, t-SNE or PCA, all reimplemented in pure Python + [numba](https://numba.pydata.org/):
molecules in, an interactive map out, in one line. There is nothing to compile and no conda
environment to set up, and it maps millions of molecules on a laptop.

![Interactive viewer](https://raw.githubusercontent.com/pablo-mas/chemviz/main/docs/images/viewer.png)

- **One line**: `cv.tmap("molecules.csv").save("map.html")`.
- **Four methods, one API**: `cv.tmap`, `cv.umap`, `cv.tsne` and `cv.pca` take the same inputs
  and return the same map object (see [methods](#methods)).
- **Fast and light**: a TMAP of 250k molecules in ~10 s from SMILES, 1M in under 40 s, the
  whole of ChEMBL (2.9M) in under 2 minutes with ~4 GB of RAM. From the same MinHash vectors,
  that is 2.5–3× faster than the original tmap, with 3–4× less memory (see
  [benchmarks](#benchmarks)).
- **Interactive viewer**: a single self-contained HTML file (WebGL) that shows the structure
  on hover, colours, filters, searches, lasso-selects and exports to CSV. It also works offline.
- **Static plots** with matplotlib, for papers and reports.
- **Command line**: `chemviz tmap molecules.csv -o map.html` (or `umap`, `tsne`, `pca`).

## Installation

```bash
pip install chemviz
# or
uv add chemviz
```

Requires Python ≥ 3.12. Every dependency (RDKit, numba, numpy, pandas, matplotlib, joblib)
installs from wheels; UMAP and t-SNE need neither umap-learn nor openTSNE.

## Quick start

```python
import chemviz as cv

m = cv.tmap("molecules.csv")    # the SMILES column is detected automatically
m.save("map.html")              # interactive viewer, open it in any browser
m.plot(color="activity")        # static matplotlib figure

cv.umap("molecules.csv").save("umap.html")   # same for cv.tsne and cv.pca
```

All methods accept:

| Input | Example |
|---|---|
| list, tuple or Series of SMILES | `cv.tmap(["CCO", "c1ccccc1O", ...])` |
| DataFrame with a SMILES column | `cv.tmap(df)` or `cv.tmap(df, smiles="canonical_smiles")` |
| file path: `.csv`, `.tsv`, `.smi`, `.sdf` | `cv.tmap("library.sdf")` (SDF properties become columns) |
| precomputed MinHash (`uint32`, not for PCA) or binary fingerprints | `cv.tmap(fps)` |

Invalid SMILES are skipped with a warning; their input positions are kept in `m.invalid`.

The result is a `Map`:

```python
m.x, m.y          # coordinates (numpy arrays), one per molecule
m.edges           # (n - 1, 2) tree edges, as row positions (TMAP only, else empty)
m.data            # the input rows (all columns, original index), minus invalid molecules
m.method          # "tmap", "umap", "tsne" or "pca"
m.to_dataframe()  # data + x, y columns
```

Save a map to reload it later without recomputing it:

```python
m.save("map.chemviz")
m = cv.load("map.chemviz")
```

A `.chemviz` file keeps the coordinates, the data (column types included) and the molecules'
MinHash vectors (512 bytes per molecule; `m.save("map.chemviz", model=False)` leaves them out
for a smaller file). It is read without pickle, so opening a file from someone else cannot
run code.

## Methods

| Method | What the map shows | ZINC250k | ChEMBL 1M |
|---|---|---|---|
| `cv.tmap` | A tree linking each molecule to a close neighbour: series and their branching are easy to follow, and nearest neighbours stay together. | 9 s | 37 s |
| `cv.umap` | Tight, well separated clusters; distances between clusters mean little. | 14 s | 58 s |
| `cv.tsne` | Clusters with fine local detail, well preserved neighbourhoods; distances between clusters mean little. | 90 s | 6.5 min |
| `cv.pca` | A linear projection of fingerprints: broad trends and gradients, but similar molecules overlap. | 5 s | 19 s |

Times are end to end from SMILES (see [benchmarks](#benchmarks)). TMAP, UMAP and t-SNE use
the same MinHash encoding and LSH Forest nearest neighbours; PCA uses folded ECFP
fingerprints.

## Command line

```bash
chemviz tmap molecules.csv                      # writes molecules.html
chemviz umap molecules.csv -o map.html -o map.png --color pKi
chemviz pca library.sdf -o coords.csv           # data + x, y coordinates
```

The subcommands are `tmap`, `umap`, `tsne` and `pca`. Each `-o` output is written according
to its extension: `.html` (interactive viewer), `.png` / `.svg` / `.pdf` (static plot),
`.csv` or `.chemviz` (the map, for `cv.load`). Options mirror the Python API (`--smiles`, `--columns`, `--offline`, `-k`, `--seed`…);
see `chemviz tmap --help`.

## Interactive viewer

```python
m.save("map.html")                      # self-contained HTML file to open or share
m.save("map.html", columns=["pKi"])     # embed only some columns
m.show()                                # inline in Jupyter / VS Code notebooks
```

The viewer:

- draws the structure under the cursor, with the molecule's SMILES and properties;
- colours by any column: numeric columns get a colour bar, categorical ones a legend where a
  click hides or shows a category;
- filters by value range, category or text, and searches by identifier;
- selects by click or lasso (Shift + drag), zooms to the selection and exports it to CSV;
- lets you adjust point size and (for TMAP) edge opacity, and follows the system light/dark
  theme.

Maps of a million molecules load in about a second. The data is compressed into the file:
250k molecules take ~12 MB and 1M ~40 MB.

**Offline use.** Structures are drawn with [RDKit.js](https://www.npmjs.com/package/@rdkit/rdkit),
loaded from a CDN when the page opens. To get a file that needs no network at all, embed it
(about 3.3 MB more):

```python
m.save("map.html", offline=True)
```

RDKit.js is downloaded once, checked against a pinned SHA-256 and cached in `~/.cache/chemviz`
(set `CHEMVIZ_CACHE` to use another folder).

## Static plots

```python
ax = m.plot(color="qed")               # numeric: colour bar (1st–99th percentile)
ax = m.plot(color="series", s=4)       # categorical: legend
ax.figure.savefig("map.png", dpi=200)
```

![Static plot](https://raw.githubusercontent.com/pablo-mas/chemviz/main/docs/images/tmap_static.png)

## Parameters

```python
cv.tmap(
    data,
    smiles=None,          # SMILES column name (guessed if omitted)
    k=10,                 # nearest neighbours per molecule
    kc=10,                # candidate pool k * kc ranked by exact distance
    n_permutations=128,   # MinHash size
    radius=2,             # ECFP radius
    n_trees=8,            # LSH Forest trees
    seed=0,               # layout seed (results are deterministic)
    n_jobs=None,          # processes for SMILES parsing (all cores by default)
)
cv.umap(data, smiles=None, k=15, min_dist=0.1, n_epochs=None, seed=0)   # + MinHash/LSH options
cv.tsne(data, smiles=None, perplexity=30, n_iter=500, seed=0)           # + MinHash/LSH options
cv.pca(data, smiles=None, n_bits=2048, radius=2)
```

## How it works

chemviz follows the TMAP pipeline:

1. **MinHash**: each molecule's unfolded ECFP substructures are MinHashed (`minhash_ecfp`).
   The Jaccard similarity of two molecules is the fraction of equal hash values.
2. **Approximate kNN with an LSH Forest** (`lsh_knn`): each tree indexes a band of hash values
   in lexicographic order, and candidates are the molecules that share the longest prefixes.
   They are ranked by MinHash distance.
3. **Minimum spanning tree** of the kNN graph (`minimum_spanning_forest`, Kruskal).
4. **Tree layout** (`tree_layout`): a multilevel force-directed layout (coarsening along the
   tree, Barnes–Hut repulsion on a quadtree), with disconnected components packed together.

Each step is available on its own:

```python
from chemviz import minhash_ecfp, lsh_knn, minimum_spanning_forest, tree_layout

hashes = minhash_ecfp(smiles)                    # (n, 128) uint32
idx, dist = lsh_knn(hashes, k=10)                # (n, k) neighbours and distances
edges, weights = minimum_spanning_forest(idx, dist)
pos = tree_layout(len(smiles), edges)            # (n, 2) float32
```

The other methods share steps 1 and 2:

- **UMAP** (`umap_layout`) builds the fuzzy kNN graph and optimises the embedding by stochastic
  gradient descent with negative sampling, as umap-learn does, but updates all molecules in
  parallel from the same positions, so results do not depend on the number of threads. It
  starts from a quick tree layout of the kNN graph.
- **t-SNE** (`tsne_layout`) calibrates affinities on 3 × perplexity neighbours and follows
  openTSNE's optimisation (early exaggeration, momentum, gains), with Barnes–Hut repulsion on
  an adaptive quadtree. It starts from the tree layout of the kNN graph.
- **PCA** (`pca_layout`) diagonalises the covariance of folded ECFP fingerprints, accumulated
  from their on-bits only, so it is exact and fast for millions of molecules.

```python
from chemviz import umap_layout, tsne_layout, pca_layout

pos = umap_layout(*lsh_knn(hashes, k=15))        # (n, 2) float32
pos = tsne_layout(*lsh_knn(hashes, k=90, kc=3))
pos = pca_layout(fingerprints)                   # (n, n_bits) binary matrix
```

`mhfp` (RDKit's MHFP6 encoder) and `minhash_from_fingerprints` (MinHash of binary fingerprints)
are alternative encoders, and `read_molecules` loads a molecule file into a DataFrame.

## Benchmarks

**Setup.** All benchmarks ran on an Intel i7-12700K (20 threads, 15 GB of RAM) under Linux
(WSL2), using every thread, with chemviz 0.1.0 on Python 3.12, numba 0.68 and RDKit 2026.09.
Each figure is a single run, with numba functions compiled beforehand. Every tool runs with
its default parameters: 128 MinHash permutations of ECFP4, an 8-tree LSH Forest, k = 10 and
kc = 10 for TMAP. ZINC250k is the usual 249,455-molecule set. The ChEMBL 1M subset is a
seeded random sample of the RDKit-parsable SMILES of ChEMBL 37, and ChEMBL 37 itself has
2.9M of them.

**TMAP vs the original tmap** (tmap-viz 1.0.16, Python 3.10). Both tools were run on the same
MinHash vectors, each in its own process. Times cover kNN, tree and layout; peak memory is for
the whole process.

| Dataset | chemviz | tmap (original) |
|---|---|---|
| ZINC250k (249k) | **5.4 s**, 0.5 GB | 12.7 s, 1.5 GB |
| ChEMBL subset (1M) | **20 s**, 1.4 GB | 62 s, 6.2 GB |
| ChEMBL 37 (2.9M), end to end | **110 s**, 4.3 GB | out of memory |

Map quality, comparing chemviz with tmap:

| Metric | ZINC250k | ChEMBL 1M |
|---|---|---|
| kNN preservation (local) | 0.320 vs 0.321 | 0.439 vs 0.448 |
| Global structure | **0.777** vs 0.707 | **0.653** vs 0.633 |

- **kNN preservation**: the share of each molecule's 10 exact MinHash neighbours that are
  among its 10 nearest points in 2D.
- **Global structure**: the Spearman correlation between distance along the tree and distance
  in 2D.

**Other methods.** End to end from SMILES (peak memory of the whole process on ChEMBL 1M):

| Method | ZINC250k: time | kNN preservation | Distance correlation | ChEMBL 1M: time | memory | kNN preservation | Distance correlation |
|---|---|---|---|---|---|---|---|
| TMAP | 9 s | 0.320 | 0.03 | 37 s | 2.6 GB | **0.439** | 0.06 |
| UMAP | 14 s | 0.072 | 0.29 | 58 s | 3.3 GB | 0.187 | 0.15 |
| t-SNE | 90 s | **0.345** | 0.31 | 388 s | 4.8 GB | 0.408 | 0.25 |
| PCA | 5 s | 0.002 | **0.44** | 19 s | 1.7 GB | 0.008 | **0.26** |

*Distance correlation* is the Spearman correlation between MinHash distance and 2D distance
over random pairs. Each method trades local neighbourhoods against overall distances
differently: PCA keeps overall distances but mixes neighbours, while TMAP keeps neighbours and
lays the rest out along its tree.

chemviz's UMAP and t-SNE were also compared with umap-learn and openTSNE embedding the same
kNN graph:

| ZINC250k, same kNN graph | Time | kNN preservation | Distance correlation |
|---|---|---|---|
| chemviz UMAP | **9.5 s** | 0.073 | 0.28 |
| umap-learn 0.6.0 | 52 s | 0.068 | 0.28 |
| chemviz t-SNE | **82 s** | **0.345** | 0.31 |
| openTSNE 1.0.4 (spectral initialisation) | 94 s | 0.337 | **0.32** |

umap-learn and openTSNE ran with their defaults on the graphs chemviz uses: 15 neighbours for
UMAP, and 90 neighbours with perplexity 30 for t-SNE.

Scripts are in [`benchmarks/`](https://github.com/pablo-mas/chemviz/tree/main/benchmarks)
(`compare_tmap.py`, `compare_methods.py`).

## Credits

chemviz is an independent reimplementation of TMAP, MHFP, UMAP and t-SNE. If you use it,
please cite the original work:

- D. Probst, J.-L. Reymond. *Visualization of very large high-dimensional data sets as minimum
  spanning trees.* J. Cheminform. 12, 12 (2020).
  [doi:10.1186/s13321-020-0416-x](https://doi.org/10.1186/s13321-020-0416-x)
- D. Probst, J.-L. Reymond. *A probabilistic molecular fingerprint for big data settings.*
  J. Cheminform. 10, 66 (2018).
  [doi:10.1186/s13321-018-0321-8](https://doi.org/10.1186/s13321-018-0321-8)
- L. McInnes, J. Healy, J. Melville. *UMAP: Uniform Manifold Approximation and Projection for
  Dimension Reduction.* arXiv:1802.03426 (2018).
  [doi:10.48550/arXiv.1802.03426](https://doi.org/10.48550/arXiv.1802.03426)
- L. van der Maaten, G. Hinton. *Visualizing Data using t-SNE.* J. Mach. Learn. Res. 9,
  2579–2605 (2008).
- P. G. Poličar, M. Stražar, B. Zupan. *openTSNE: A Modular Python Library for t-SNE
  Dimensionality Reduction and Embedding.* J. Stat. Softw. 109(3) (2024).
  [doi:10.18637/jss.v109.i03](https://doi.org/10.18637/jss.v109.i03)

## License

MIT
