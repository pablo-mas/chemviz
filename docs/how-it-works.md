# How it works

chemviz reimplements TMAP, UMAP, t-SNE and PCA in Python with [numba](https://numba.pydata.org/),
which compiles the numerical code to parallel machine code on first use. This page describes
each step and where chemviz departs from the original implementations. Each step is a public
function, so it can be used on its own (see the {doc}`api`).

## The shared steps: encoding and neighbours

TMAP, UMAP and t-SNE start the same way: encode the molecules, then find each one's nearest
neighbours.

### MinHash of ECFP substructures

{func}`~chemviz.minhash_ecfp` describes each molecule by the set of its circular
substructures: the unfolded Morgan (ECFP) identifiers up to `radius` bonds, the same
features as in [MHFP](https://doi.org/10.1186/s13321-018-0321-8). The similarity of two
molecules is the Jaccard index of their sets: shared substructures over all substructures.

Comparing sets directly would be slow, so each set is summarised by MinHash: for each of
`n_permutations` random hash functions, the smallest hash value of the set's elements. Two
molecules have the same minimum for a given function with probability equal to their
Jaccard index, so the fraction of equal values estimates it, with a standard error of about
0.04 for 128 values. A molecule thus becomes a fixed-size vector of 128 integers.

Extracting the identifiers with RDKit's Morgan generator is about 50 times faster than MHFP's
shingling. SMILES are parsed in chunks by worker processes, and the hashing itself is
vectorised with numba.

### Approximate nearest neighbours with an LSH Forest

{func}`~chemviz.lsh_knn` finds each molecule's `k` nearest neighbours without comparing all
pairs. The MinHash vector is split into `n_trees` bands of consecutive values, and each band
indexes a tree: the molecules sorted lexicographically by their band. Molecules sharing a
long prefix in a band are likely similar, so a molecule's candidates are its neighbours in
each sorted order, the ones sharing the longest prefixes first, until at least `k * kc` are
collected, as in the LSH Forest descent.

chemviz stores each tree as a sorted order and the length of the common prefix between
consecutive molecules, which makes building the forest a sort and reading candidates a walk
along an array. The candidates are first ranked by comparing only the low byte of each
hash value (b-bit MinHash, four times less memory to read), and the best `2k` are re-ranked
by their exact distance.

## TMAP

### Minimum spanning tree

{func}`~chemviz.minimum_spanning_forest` keeps, among the kNN edges, the tree of minimum
total distance (Kruskal's algorithm with a union-find). It links every molecule to the rest
by its most similar connections. When the kNN graph is disconnected, each component gets
its own tree.

### Tree layout

{func}`~chemviz.tree_layout` draws the tree in 2D with a multilevel force-directed
algorithm, in the spirit of the FM³ layout used by the original tmap:

1. **Coarsening.** Nodes are merged along tree edges, level after level, until at most
   `min_nodes` remain.
2. **Layout.** The coarsest level is laid out by forces: edges pull their ends together and
   all nodes repel each other. Repulsion is computed with a quadtree: per cell, nearby
   nodes exactly, distant cells through a first-order expansion of their field, so that an
   iteration costs time linear in the number of nodes.
3. **Expansion.** Each level is expanded into the next: a node starts near its group's
   position, shifted towards the groups it links to, then forces refine the layout. A pull of
   each node towards its group's position (`gravity`) keeps branches compact without
   collapsing the global structure.
4. **Packing.** Trees of disconnected components are packed side by side.

The original tmap uses OGDF's FM³ implementation in C++. On the same MinHash vectors,
chemviz's neighbour search, tree and layout together are 2.5 to 3 times faster, with 3 to 4
times less memory, and keep nearest neighbours as well (see {doc}`benchmarks`).

## UMAP

{func}`~chemviz.umap_layout` follows umap-learn:

- **Fuzzy graph.** Each molecule's distances to its neighbours are turned into memberships,
  exp(−(d − ρ)/σ), with ρ the distance to its nearest neighbour and σ set so that the
  memberships sum to log₂(k + 1). The directed graph is made symmetric by fuzzy union
  (a + b − ab), and edges too weak to be sampled are dropped.
- **Embedding.** Stochastic gradient descent pulls each edge's ends together, with
  frequency proportional to its membership, and pushes each molecule away from random others
  (negative sampling), with the same curve parameters (a, b) as umap-learn for a given
  `min_dist`.

Unlike umap-learn, which updates points one edge after another, chemviz updates all points
in parallel from the positions at the start of each epoch. Results thus do not depend on the
number of threads; the attraction is scaled to keep umap-learn's balance with repulsion, and
gives the same quality on the same graph. The default starting layout is a quick tree layout
of the kNN graph, instead of umap-learn's spectral embedding.

## t-SNE

{func}`~chemviz.tsne_layout` follows openTSNE:

- **Affinities.** For each molecule, a Gaussian over its `3 * perplexity` nearest neighbours,
  with the bandwidth set by bisection so that its perplexity equals `perplexity`; the
  conditional probabilities are made symmetric and normalised.
- **Optimisation.** 250 iterations of early exaggeration (attraction × 12), then `n_iter`
  normal ones, with openTSNE's learning rate (n / exaggeration), momentum (0.8), gains and
  step limit.
- **Repulsion.** Barnes–Hut on a quadtree that adapts to the density of points, built over
  their Morton (Z-order) codes at each iteration, with openTSNE's opening criterion (0.5).

The default starting layout is the tree layout of the minimum spanning forest of the 30
nearest neighbours, scaled down to a standard deviation of 10⁻⁴: it gives about the same
global structure as openTSNE's spectral initialisation.

## PCA

{func}`~chemviz.pca_layout` and {func}`~chemviz.pca` compute the exact principal axes of
folded ECFP fingerprints. The covariance matrix is accumulated from the indices of the
on-bits only, in parallel blocks, so the cost depends on the number of on-bits, not on the
fingerprint size, and millions of molecules take seconds. The two leading eigenvectors are
the axes; the sign of each is fixed (largest loading positive) so that maps are reproducible.

## Placing new molecules

{meth}`Map.project <chemviz.Map.project>` reuses the steps above with the map fixed:

- the new molecules are encoded the same way, and their neighbours are searched among the
  map's molecules by indexing them in the same LSH Forest, only keeping candidates from the
  map;
- TMAP computes the minimum spanning tree of the map's tree and the new molecules' kNN
  edges, keeping the map's tree, then lays each new branch out in the widest free direction
  around the molecule it is grafted onto;
- UMAP and t-SNE optimise the new molecules only, as umap-learn's and openTSNE's `transform`;
- PCA projects them on the stored axes.

## Determinism

Every method gives the same result for the same input and `seed`, whatever the number of
threads: random numbers come from seeded or counter-based generators, and parallel loops
write to separate outputs.

## Building custom pipelines

The building blocks can be combined directly, for example to reuse one neighbour search for
several layouts:

```python
from chemviz import (lsh_knn, minhash_ecfp, minimum_spanning_forest, tree_layout,
                     tsne_layout, umap_layout)

hashes = minhash_ecfp(smiles)                       # (n, 128) uint32
idx, dist = lsh_knn(hashes, k=10)                   # (n, k) neighbours and distances
edges, weights = minimum_spanning_forest(idx, dist)
pos = tree_layout(len(smiles), edges)               # (n, 2) float32

pos = umap_layout(*lsh_knn(hashes, k=15))
pos = tsne_layout(*lsh_knn(hashes, k=90, kc=3))
```

{func}`~chemviz.mhfp` (RDKit's MHFP6 encoder) and
{func}`~chemviz.minhash_from_fingerprints` (MinHash of binary fingerprints) are alternative
encoders.
