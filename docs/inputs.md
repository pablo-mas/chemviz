# Inputs

All four methods ({func}`~chemviz.tmap`, {func}`~chemviz.umap`, {func}`~chemviz.tsne`,
{func}`~chemviz.pca`) and {meth}`Map.project <chemviz.Map.project>` accept the same kinds of
input.

## SMILES

A list, tuple or pandas Series of SMILES strings:

```python
m = cv.tmap(["CCO", "c1ccccc1O", "CC(=O)Nc1ccc(O)cc1"])
```

The map's data then has a single `smiles` column. A Series keeps its index.

## DataFrames

A DataFrame with one row per molecule and a SMILES column. All columns are kept in the map,
so they can be used to colour, filter and search in the viewer, and they come back in
`m.data` and `m.to_dataframe()`. The index is kept too.

```python
m = cv.tmap(df)                              # SMILES column found by its name
m = cv.tmap(df, smiles="canonical_smiles")   # or named explicitly
```

The SMILES column is found when exactly one column is named `smiles` (in any case), or
else when exactly one column name contains `smiles`. Otherwise, pass `smiles=`.

## Files

A path to a molecule file. Columns and SDF properties become columns of the map's data.

| Extension | Content |
|---|---|
| `.csv`, `.tsv` | A table with a SMILES column, found as for DataFrames, or given with `smiles=` |
| `.smi` | One molecule per line: a SMILES, optionally followed by a name |
| `.sdf` | Molecules with their properties; the name (title line) goes to a `name` column |

```python
m = cv.tmap("library.sdf")
```

To read a file without mapping it, use {func}`~chemviz.read_molecules`.

## Fingerprints and MinHash vectors

A 2D NumPy array with one row per molecule:

- **binary fingerprints** (bool, or integers equal to 0 or 1), such as RDKit or MACCS keys:
  TMAP, UMAP and t-SNE MinHash their on-bits, PCA uses them directly;
- **uint32 MinHash vectors**, such as from {func}`~chemviz.minhash_ecfp` or
  {func}`~chemviz.mhfp`: used as they are by TMAP, UMAP and t-SNE (PCA needs fingerprints,
  as MinHash values have no linear structure).

```python
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
fps = np.array([generator.GetFingerprintAsNumPy(Chem.MolFromSmiles(s)) for s in smiles])
m = cv.umap(fps)
```

The map's data is then empty (no SMILES to draw structures): add your own columns to
`m.data` before saving the viewer, keeping the row order.

## How molecules are encoded

From SMILES, TMAP, UMAP and t-SNE encode each molecule as the set of its circular (ECFP)
substructures, and compare molecules by the Jaccard distance between these sets, estimated
with MinHash. The `radius` parameter sets the size of the substructures (2, the default,
corresponds to ECFP4) and `n_permutations` the precision of the estimate. PCA uses folded
ECFP fingerprints of `n_bits` bits instead. See the {doc}`how-it-works` page for details.

## Invalid molecules

SMILES that RDKit cannot parse, and SDF records it cannot read, are skipped with a warning
that gives their positions in the input:

```text
UserWarning: Skipped 2 invalid molecule(s) at input positions [17, 342]
```

The map holds the valid molecules only, in input order, and `m.invalid` keeps the input
positions of the skipped ones:

```python
bad = df.iloc[m.invalid]   # the rows that could not be mapped
```

## Speed: parsing in parallel

Parsing SMILES is spread over all CPUs by default, in chunks, so that millions of molecules
are never all held in memory as RDKit objects. `n_jobs=` sets the number of processes
(`n_jobs=1` parses in the main process).
