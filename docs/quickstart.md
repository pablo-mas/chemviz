---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Quickstart

This page maps 2,000 drug-like molecules from [ZINC](https://zinc.docking.org/), explores
them in the interactive viewer and saves the result. Its code runs every time the
documentation is built, so what you see is what chemviz produces.

## Load molecules

chemviz takes molecules as a list of SMILES, a DataFrame with a SMILES column, or a file
(`.csv`, `.tsv`, `.smi`, `.sdf`). Here, a random sample of the ZINC250k dataset, with three
properties per molecule:

```{code-cell} python
import pandas as pd

url = ("https://raw.githubusercontent.com/aspuru-guzik-group/chemical_vae/master/"
       "models/zinc_properties/250k_rndm_zinc_drugs_clean_3.csv")
df = pd.read_csv(url).sample(2000, random_state=0)
df["smiles"] = df["smiles"].str.strip()  # this file ends each SMILES with a line break
df.head()
```

## Make a map

One call computes a tree map (TMAP): each molecule is linked to a close neighbour, and the
resulting tree is laid out in 2D. The SMILES column is found by its name.

```{code-cell} python
import chemviz as cv

m = cv.tmap(df)
m.to_dataframe()[["smiles", "qed", "x", "y"]].head()
```

The map keeps every column of the input and adds the coordinates. Its other attributes
include the tree edges (`m.edges`) and the method (`m.method`).

## Explore it

`m.show()` displays the interactive viewer in a notebook. Hover over a point to see the
structure, colour by a column, search, filter, and select with a click or Shift + drag:

```{code-cell} python
m.show(height=550)
```

For a static figure, `m.plot` colours the map by any column with matplotlib:

```{code-cell} python
ax = m.plot(color="qed")
```

## Save it

```{code-cell} python
m.save("zinc.html")       # the viewer, a single file to open in any browser or share
m.save("zinc.chemviz")    # the map itself, to reload with cv.load without recomputing
```

The HTML file is self-contained: it needs no server and no Python. It loads RDKit.js from a
CDN to draw structures; `m.save("zinc.html", offline=True)` embeds it so the file works
without network.

## Try another method

UMAP, t-SNE and PCA take the same inputs and return the same kind of map:

```{code-cell} python
ax = cv.umap(df).plot(color="qed")
```

## Next steps

- {doc}`methods` compares the four methods, to choose one for your data.
- The same map from the command line: `chemviz tmap molecules.csv -o map.html`.
- The {doc}`api` lists every parameter.
