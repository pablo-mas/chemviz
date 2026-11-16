# API reference

```{eval-rst}
.. currentmodule:: chemviz
```

## Making maps

Each method takes the same inputs and returns a {class}`Map`.

```{eval-rst}
.. autofunction:: tmap
.. autofunction:: umap
.. autofunction:: tsne
.. autofunction:: pca
.. autofunction:: load
```

## The map object

```{eval-rst}
.. autoclass:: Map
   :members: project, save, to_html, show, widget, plot, to_dataframe
```

## Jupyter widget

Needs the `jupyter` extra: `pip install "chemviz[jupyter]"`.

```{eval-rst}
.. autoclass:: MapWidget
```

## Building blocks

The steps of the methods, for custom pipelines (see {doc}`how-it-works`).

```{eval-rst}
.. autofunction:: minhash_ecfp
.. autofunction:: mhfp
.. autofunction:: minhash_from_fingerprints
.. autofunction:: lsh_knn
.. autofunction:: minimum_spanning_forest
.. autofunction:: tree_layout
.. autofunction:: umap_layout
.. autofunction:: tsne_layout
.. autofunction:: pca_layout
.. autofunction:: read_molecules
```
