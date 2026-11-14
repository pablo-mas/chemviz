"""High-level TMAP API: molecules in, tree map out."""

import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from chemviz.graph import minimum_spanning_forest
from chemviz.io import find_smiles_column, read_molecules
from chemviz.layout import tree_layout
from chemviz.lsh import lsh_knn
from chemviz.minhash import _minhash_ecfp, minhash_from_fingerprints


@dataclass
class TMap:
    """A tree map of molecules.

    ``x``, ``y`` and ``data`` rows are aligned (one per mapped molecule); ``edges``
    holds pairs of row positions. ``data`` keeps the input rows and index, minus
    ``invalid`` (positions in the input of the molecules that could not be parsed).
    ``smiles`` names the SMILES column of ``data``, if any (used to draw structures).
    """

    x: np.ndarray
    y: np.ndarray
    edges: np.ndarray
    data: pd.DataFrame
    invalid: np.ndarray
    smiles: str | None = None

    def to_dataframe(self) -> pd.DataFrame:
        """The input rows with ``x`` and ``y`` columns added."""
        return self.data.assign(x=self.x, y=self.y)

    def to_html(self, columns=None, title: str = "chemviz") -> str:
        """The interactive WebGL viewer as a self-contained HTML page.

        ``columns`` restricts the ``data`` columns embedded (all by default); they can
        be used to colour, filter, search and are shown on hover. Structures are drawn
        with RDKit.js, loaded from a CDN when the page opens.
        """
        from chemviz.viewer import to_html

        return to_html(self.x, self.y, self.edges, self.data, self.smiles, columns, title)

    def save(self, path, columns=None, title: str | None = None) -> None:
        """Write the interactive viewer to an HTML file (see ``to_html``)."""
        path = Path(path)
        path.write_text(self.to_html(columns, title or path.stem), encoding="utf-8")

    def show(self, columns=None, title: str = "chemviz", height: int = 650):
        """Display the interactive viewer inline in a Jupyter notebook (see ``to_html``)."""
        try:
            from IPython.display import HTML
        except ImportError as e:
            raise ImportError("show() needs IPython (Jupyter); use save('map.html') instead") from e
        import html

        doc = html.escape(self.to_html(columns, title), quote=True)
        return HTML(f'<iframe srcdoc="{doc}" style="width:100%;height:{int(height)}px;border:0" '
                    'sandbox="allow-scripts allow-downloads"></iframe>')

    def plot(self, color=None, ax=None, edges: bool = True, cmap=None, **scatter_kws):
        """Static matplotlib plot. ``color`` is a column of ``data`` or an array of values.

        Numeric values use a colour bar (clipped to the 1-99th percentiles), others a legend.
        Returns the matplotlib Axes.
        """
        import matplotlib.pyplot as plt
        from matplotlib.collections import LineCollection

        if ax is None:
            _, ax = plt.subplots(figsize=(8, 8))
        n = len(self.x)
        pos = np.column_stack([self.x, self.y])
        if edges and len(self.edges):
            segs = np.stack([pos[self.edges[:, 0]], pos[self.edges[:, 1]]], axis=1)
            ax.add_collection(LineCollection(segs, colors="0.6", linewidths=0.4 if n < 20_000 else 0.05, zorder=1))
        scatter_kws.setdefault("s", 6 if n < 20_000 else 0.2)
        scatter_kws.setdefault("linewidths", 0)
        values = self.data[color].to_numpy() if isinstance(color, str) else color
        if values is None:
            ax.scatter(self.x, self.y, zorder=2, **scatter_kws)
        elif pd.api.types.is_numeric_dtype(np.asarray(values)):
            values = np.asarray(values, dtype=float)
            vmin, vmax = np.nanpercentile(values, [1, 99])
            sc = ax.scatter(self.x, self.y, c=values, cmap=cmap or "viridis", vmin=vmin, vmax=vmax,
                            zorder=2, **scatter_kws)
            ax.figure.colorbar(sc, ax=ax, shrink=0.6, label=color if isinstance(color, str) else None)
        else:
            categories, codes = np.unique(np.asarray(values, dtype=str), return_inverse=True)
            palette = plt.get_cmap(cmap or ("tab10" if len(categories) <= 10 else "tab20"))
            for i, cat in enumerate(categories):
                mask = codes == i
                ax.scatter(self.x[mask], self.y[mask], color=palette(i % palette.N), label=cat,
                           zorder=2, **scatter_kws)
            ax.legend(markerscale=3, fontsize="small", loc="best", title=color if isinstance(color, str) else None)
        ax.set_aspect("equal")
        ax.axis("off")
        return ax


def tmap(
    data,
    smiles: str | None = None,
    *,
    k: int = 10,
    kc: int = 10,
    n_permutations: int = 128,
    radius: int = 2,
    n_trees: int = 8,
    seed: int = 0,
    n_jobs: int | None = None,
) -> TMap:
    """Compute a tree map (TMAP) of molecules.

    ``data`` can be:

    - a list, tuple or Series of SMILES;
    - a DataFrame, with its SMILES column given by ``smiles`` (guessed if omitted);
    - a path to a .csv, .tsv, .smi or .sdf file;
    - a precomputed 2D array: uint32 MinHash vectors, or binary fingerprints (bool or 0/1).

    Molecules are encoded as MinHash of their ECFP substructures (``radius``,
    ``n_permutations``), linked to their ``k`` approximate nearest neighbours (LSH
    Forest with ``n_trees`` trees, ``k * kc`` candidates), reduced to a minimum spanning
    tree and laid out in 2D. Unparsable SMILES are skipped with a warning.
    """
    hashes, frame, smiles, invalid = _encode(data, smiles, n_permutations, radius, n_jobs)
    if invalid:
        warnings.warn(
            f"Skipped {len(invalid)} invalid molecule(s) at input positions "
            f"{invalid[:20]}{'...' if len(invalid) > 20 else ''}",
            stacklevel=2,
        )
        keep = np.ones(len(hashes), dtype=bool)
        keep[invalid] = False
        hashes, frame = hashes[keep], frame[keep]
    n = len(hashes)
    if n == 0:
        raise ValueError("No valid molecules to map")
    if n == 1:
        edges = np.empty((0, 2), dtype=np.int64)
    else:
        idx, dist = lsh_knn(hashes, k=min(k, n - 1), n_trees=n_trees, kc=kc)
        edges, _ = minimum_spanning_forest(idx, dist)
    pos = tree_layout(n, edges, seed=seed)
    return TMap(pos[:, 0], pos[:, 1], edges, frame, np.asarray(invalid, dtype=np.int64), smiles)


def _encode(data, smiles, n_permutations, radius, n_jobs):
    """MinHash vectors, matching DataFrame, its SMILES column (or None) and invalid input positions."""
    if isinstance(data, (str, Path)):
        data, smiles = read_molecules(data, smiles)
    if isinstance(data, np.ndarray) and data.ndim == 2:
        frame = pd.DataFrame(index=pd.RangeIndex(len(data)))
        if data.dtype == np.uint32:
            return data, frame, None, []
        if data.dtype == bool or (np.issubdtype(data.dtype, np.integer) and np.isin(data, (0, 1)).all()):
            return minhash_from_fingerprints(data, n_permutations=n_permutations), frame, None, []
        raise ValueError(
            "A 2D array must hold uint32 MinHash vectors or binary (0/1) fingerprints; "
            f"got dtype {data.dtype}"
        )
    if isinstance(data, pd.DataFrame):
        frame = data
        column = smiles or find_smiles_column(data)
    else:
        column = smiles or "smiles"
        frame = pd.DataFrame({column: list(data)}, index=data.index if isinstance(data, pd.Series) else None)
    values = frame[column].fillna("").astype(str).tolist()
    hashes, invalid = _minhash_ecfp(values, n_permutations, radius, 42, 10_000, n_jobs)
    return hashes, frame, column, invalid
