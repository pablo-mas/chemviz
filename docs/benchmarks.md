# Benchmarks

Speed, memory and map quality of chemviz, compared with the original tmap, umap-learn and
openTSNE.

## Setup

All benchmarks ran on an Intel i7-12700K (20 threads, 15 GB of RAM) under Linux (WSL2), using
every thread, with Python 3.12, numba 0.68 and RDKit 2026.09; chemviz 0.1, and the
development version of 0.2 for placing new molecules. Each figure is a single run, with
numba functions compiled beforehand. Every tool runs with its default parameters: 128
MinHash permutations of ECFP4, an 8-tree LSH Forest, and k = 10, kc = 10 for TMAP.

Datasets:

- **ZINC250k**: the usual set of 249,455 drug-like molecules;
- **ChEMBL 1M**: a seeded random sample of 1 million RDKit-parsable SMILES of ChEMBL 37;
- **ChEMBL 37**: all its 2.9 million RDKit-parsable SMILES.

## Quality metrics

kNN preservation
: The share of a molecule's 10 nearest neighbours in chemical space (exact MinHash distance,
  among all molecules) that are among its 10 nearest points on the map, averaged over 2,000
  random molecules. It measures how well the map keeps local relations.

Distance correlation
: The Spearman correlation between MinHash distance and distance on the map over 20,000
  random pairs of molecules. It measures how well the map keeps large-scale relations.

Global structure (TMAP)
: The Spearman correlation between distance along the tree and distance on the map: how
  faithfully the layout draws the tree.

## TMAP against the original tmap

The original tmap (tmap-viz 1.0.16, on Python 3.10) and chemviz were run on the same MinHash
vectors, each in its own process. Times cover the neighbour search, the tree and the
layout; peak memory is for the whole process.

| Dataset | chemviz | tmap (original) |
|---|---|---|
| ZINC250k | **5.4 s**, 0.5 GB | 12.7 s, 1.5 GB |
| ChEMBL 1M | **20 s**, 1.4 GB | 62 s, 6.2 GB |
| ChEMBL 37 (2.9M), end to end from SMILES | **110 s**, 4.3 GB | out of memory |

| Metric | ZINC250k: chemviz vs tmap | ChEMBL 1M: chemviz vs tmap |
|---|---|---|
| kNN preservation | 0.320 vs 0.321 | 0.439 vs 0.448 |
| Global structure | **0.777** vs 0.707 | **0.653** vs 0.633 |

chemviz is 2.5 to 3 times faster with 3 to 4 times less memory, keeps nearest neighbours as
well, and draws the tree more faithfully.

## The four methods

End to end from SMILES; peak memory of the whole process on ChEMBL 1M.

| Method | ZINC250k: time | kNN preservation | Distance correlation | ChEMBL 1M: time | Memory | kNN preservation | Distance correlation |
|---|---|---|---|---|---|---|---|
| TMAP | 9 s | 0.320 | 0.03 | 37 s | 2.6 GB | **0.439** | 0.06 |
| UMAP | 14 s | 0.072 | 0.29 | 58 s | 3.3 GB | 0.187 | 0.15 |
| t-SNE | 90 s | **0.345** | 0.31 | 388 s | 4.8 GB | 0.408 | 0.25 |
| PCA | 5 s | 0.002 | **0.44** | 19 s | 1.7 GB | 0.008 | **0.26** |

Each method trades local neighbourhoods against large-scale distances differently: PCA keeps
large-scale distances but mixes neighbours, TMAP keeps neighbours and lays the rest out along
its tree, and t-SNE balances both (see {doc}`methods`).

## UMAP and t-SNE against umap-learn and openTSNE

umap-learn and openTSNE embedded the same kNN graphs as chemviz, with their default
parameters: 15 neighbours for UMAP, 90 neighbours and perplexity 30 for t-SNE. Times cover the
embedding only.

| ZINC250k, same kNN graph | Time | kNN preservation | Distance correlation |
|---|---|---|---|
| chemviz UMAP | **9.5 s** | 0.073 | 0.28 |
| umap-learn 0.6.0 | 52 s | 0.068 | 0.28 |
| chemviz t-SNE | **82 s** | **0.345** | 0.31 |
| openTSNE 1.0.4 (spectral initialisation) | 94 s | 0.337 | **0.32** |

chemviz's implementations match the reference libraries' quality, and are faster.

## Placing new molecules

Maps of a random 90% of ZINC250k (224,510 molecules), on which the other 24,945 were then
placed with `Map.project`, compared with maps of all molecules. kNN preservation is measured
on 2,000 of the placed molecules; that of the map's own molecules barely changes.

| Method | kNN preservation, full map | kNN preservation, placed | Time to place 24,945 | Time to place 1,000 on 1M |
|---|---|---|---|---|
| TMAP | 0.317 | 0.256 | 2.4 s | 2.0 s |
| UMAP | 0.073 | 0.049 | 1.6 s | 1.7 s |
| t-SNE | 0.338 | 0.243 | 3.8 s | 9.2 s |
| PCA | 0.003 | 0.003 | 1.1 s | 0.2 s |

Placed molecules keep two thirds (UMAP) to four fifths (TMAP) of the neighbourhood quality
they get in a map recomputed with them, since the map does not rearrange around them.
Without optimisation, starting UMAP and t-SNE molecules at their neighbours' positions
gives half the quality (0.022 and 0.124).

## Reproducing

The scripts are in
[`benchmarks/`](https://github.com/pablo-mas/chemviz/tree/main/benchmarks) of the
repository:

| Script | Measures |
|---|---|
| `compare_tmap.py` | chemviz against the original tmap, on the same MinHash vectors |
| `compare_methods.py` | the four methods end to end; with `--reference`, umap-learn and openTSNE on the same graphs |
| `compare_projection.py` | placing held-out molecules on maps |
| `prepare_chembl.py` | extracts the parsable SMILES of a ChEMBL chemreps dump |

The original tmap needs Python 3.10 in a separate environment (`.venv-tmap`), and the
reference libraries another one (`.venv-ref`); the scripts' docstrings give the commands to
create them.
