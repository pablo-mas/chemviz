# Placing new molecules on a map

{meth}`Map.project <chemviz.Map.project>` places new molecules on an existing map without
recomputing it or moving the molecules already on it. Typical uses: where do screening hits,
a new series or generated molecules fall among known compounds, such as ChEMBL, approved
drugs or an in-house library?

```python
ref = cv.tmap("library.csv")        # or cv.load("library.chemviz")
m = ref.project("hits.csv")         # same kinds of input as the methods
m.save("hits.html")
```

The result holds the map's molecules followed by the new ones, with a `projected` column,
True for the new molecules: colour by it in the viewer, or filter on it. The new molecules
keep their own columns; columns missing on one side are left empty. The result can receive
more molecules in turn.

The new molecules must be given the same way as the map's: SMILES (or a DataFrame or file
of them) for a map of SMILES, fingerprints of the same size for a map of fingerprints. The
map needs its model, so a map saved with `model=False` cannot receive molecules.

## How each method places molecules

Each method follows its reference implementation, with the map's molecules fixed:

**TMAP**
: The new molecules are linked to their nearest neighbours on the map and among themselves,
  and the minimum spanning tree of these links and of the map's tree is computed, keeping
  the map's tree. A group of related new molecules thus becomes its own branch, grafted onto
  its closest molecule on the map, and laid out in the widest free direction around it.

**UMAP**
: As umap-learn's `transform`: each new molecule starts at the weighted mean of its nearest
  neighbours on the map, then moves towards them and away from other molecules of the map.

**t-SNE**
: As openTSNE's `transform`: each new molecule starts at the median of its nearest
  neighbours on the map, then is optimised against the fixed map, with openTSNE's defaults
  (perplexity 5, 250 iterations).

**PCA**
: An exact projection on the map's two principal axes: a new molecule lands exactly where it
  would in a recomputed map.

## Quality and speed

A map cannot rearrange itself around molecules placed afterwards, so they keep their
neighbours less well than in a map recomputed with them. On ZINC250k, with a map of 90% of
the molecules and the other 10% placed on it, the projected molecules kept two thirds (UMAP)
to four fifths (TMAP) of the neighbourhood quality they get in a full map, while the map's
molecules were unaffected. See {doc}`benchmarks` for the numbers.

Placing 1,000 molecules on a map of a million takes about 2 seconds with TMAP or UMAP,
9 seconds with t-SNE and 0.2 seconds with PCA. To place many molecules, or once the new ones
outnumber the map's, recompute the map instead.

## From the command line

```bash
chemviz tmap library.csv -o library.chemviz
chemviz project library.chemviz hits.csv -o hits.html
chemviz project library.chemviz hits.csv -o hits.png --color projected
```
