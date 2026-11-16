from chemviz._pca import pca, pca_layout
from chemviz._store import load
from chemviz._tmap import tmap
from chemviz._tsne import tsne, tsne_layout
from chemviz._umap import umap, umap_layout
from chemviz.graph import minimum_spanning_forest
from chemviz.io import read_molecules
from chemviz.layout import tree_layout
from chemviz.lsh import lsh_knn
from chemviz.maps import Map
from chemviz.minhash import mhfp, minhash_ecfp, minhash_from_fingerprints

__all__ = [
    "Map",
    "load",
    "lsh_knn",
    "mhfp",
    "minhash_ecfp",
    "minhash_from_fingerprints",
    "minimum_spanning_forest",
    "pca",
    "pca_layout",
    "read_molecules",
    "tmap",
    "tree_layout",
    "tsne",
    "tsne_layout",
    "umap",
    "umap_layout",
]


def __getattr__(name):
    # MapWidget needs the optional anywidget dependency: import it on first access only.
    if name == "MapWidget":
        from chemviz._widget import MapWidget

        return MapWidget
    raise AttributeError(f"module 'chemviz' has no attribute {name!r}")
