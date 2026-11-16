---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Choosing a method

chemviz offers four methods with the same inputs and the same result. They differ in what
the map preserves, and so in how to read it.

| Method | What the map shows | Read it by | Time for 250k molecules |
|---|---|---|---|
| `cv.tmap` | A tree linking each molecule to a close neighbour | following branches: series and their branching | 9 s |
| `cv.umap` | Tight, well separated clusters | looking at clusters, not at the distances between them | 14 s |
| `cv.tsne` | Clusters with fine local detail | looking at neighbourhoods, not at the distances between clusters | 90 s |
| `cv.pca` | A linear projection of fingerprints | broad trends and gradients along the axes | 5 s |

Times are end to end from SMILES on a 20-thread desktop (see {doc}`benchmarks`).

## The four maps of the same molecules

The code below maps the same 5,000 ZINC molecules with each method, coloured by logP. It
runs every time the documentation is built.

```{code-cell} python
import matplotlib.pyplot as plt
import pandas as pd

import chemviz as cv

url = ("https://raw.githubusercontent.com/the-matter-lab/chemical_vae/master/"
       "models/zinc_properties/250k_rndm_zinc_drugs_clean_3.csv")
df = pd.read_csv(url).sample(5_000, random_state=0)
df["smiles"] = df["smiles"].str.strip()

maps = {method: getattr(cv, method)(df) for method in ("tmap", "umap", "tsne", "pca")}
```

```{code-cell} python
fig, axes = plt.subplots(2, 2, figsize=(11, 11))
for ax, (method, m) in zip(axes.flat, maps.items()):
    m.plot(color="logP", ax=ax, s=1)
    ax.set_title(method)
fig.tight_layout()
```

The {doc}`gallery` has interactive versions of these maps.

## TMAP: a tree of nearest neighbours

`cv.tmap` links each molecule to its nearest neighbours, keeps the minimum spanning tree of
this graph and lays the tree out in 2D. Each molecule is next to a very similar one, and a
series of analogues forms a branch, so the map is read by following branches: where a series
starts, how it splits, which substituents lead where.

- **Strengths**: nearest neighbours stay together (the best kNN preservation of the four
  methods on a million molecules); edges make the relations explicit; fast.
- **Limits**: distances in 2D only mean something along the tree; two branches side by side
  are not necessarily related.
- **Parameters**: `k` (neighbours per molecule, default 10) sets how many relations the tree
  is chosen from; `kc` trades speed for accuracy of the neighbour search.

## UMAP: clusters

`cv.umap` builds a fuzzy graph of nearest neighbours and lays it out so that neighbours
attract and the rest repel. Groups of similar molecules become compact, well separated
clusters.

- **Strengths**: clear clusters, good for seeing how many families a library contains;
  fast.
- **Limits**: the size of a cluster and the distance between clusters mean little; within a
  cluster, the order of molecules is coarser than with TMAP or t-SNE.
- **Parameters**: `min_dist` (default 0.1) controls how tightly points pack; `k` (default
  15) balances local against global structure.

## t-SNE: neighbourhoods

`cv.tsne` matches the neighbourhood of each molecule in 2D to its neighbourhood in chemical
space. It shows clusters with more internal detail than UMAP.

- **Strengths**: the best preserved neighbourhoods on 250,000 molecules; fine structure
  inside clusters.
- **Limits**: the slowest method (about 6.5 minutes for a million molecules); as with UMAP,
  distances between clusters mean little.
- **Parameters**: `perplexity` (default 30) is the effective number of neighbours each
  molecule attends to: larger values give more global structure.

## PCA: a linear overview

`cv.pca` projects folded ECFP fingerprints on their first two principal axes. It keeps
large-scale distances best, so gradients of properties across chemical space show clearly,
but similar molecules overlap and fine clusters are not separated.

- **Strengths**: the fastest method; linear and exactly reproducible; new molecules land
  exactly where they would in a recomputed map.
- **Limits**: two axes explain little of the variance of fingerprints, so local structure
  is lost.
- **Parameters**: `n_bits` (fingerprint size, default 2048).

## Rules of thumb

- To follow series and analogues, or to find the closest molecules to a compound: **TMAP**.
- To see the families of a library and compare their sizes: **UMAP**, or **t-SNE** for more
  detail inside each family.
- To see how a property varies across chemical space, or for a reproducible overview to
  compare datasets: **PCA**.
- For millions of molecules: TMAP, UMAP and PCA take under a minute per million; t-SNE
  several minutes.

## Measured quality

On ZINC250k, end to end from SMILES (see {doc}`benchmarks` for the methodology):

| Method | kNN preservation | Distance correlation |
|---|---|---|
| TMAP | 0.320 | 0.03 |
| UMAP | 0.072 | 0.29 |
| t-SNE | **0.345** | 0.31 |
| PCA | 0.002 | **0.44** |

*kNN preservation* is the share of each molecule's 10 nearest neighbours in chemical space
that are among its 10 nearest points on the map: how well the map keeps local relations.
*Distance correlation* is the Spearman correlation between distances in chemical space and
on the map, over random pairs: how well it keeps large-scale relations. No method does both;
each one makes a different trade-off.
