"""The map object shared by all methods, and turning inputs into molecule encodings."""

import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from chemviz.io import find_smiles_column, read_molecules
from chemviz.minhash import _minhash_ecfp, minhash_from_fingerprints


@dataclass
class Map:
    """A 2D map of molecules, as returned by `tmap`, `umap`, `tsne`, `pca` and `load`.

    The coordinates and the data are aligned: row ``i`` of `data` is drawn at
    ``(x[i], y[i])``.

    Attributes
    ----------
    x, y : numpy.ndarray
        Coordinates of the mapped molecules, one per row of `data` (float32).
    edges : numpy.ndarray
        ``(n_edges, 2)`` pairs of row positions: the tree of a TMAP, empty for the other
        methods.
    data : pandas.DataFrame
        The input rows (all columns, original index), minus the molecules that could not
        be parsed.
    invalid : numpy.ndarray
        Positions in the input of the molecules that could not be parsed.
    smiles : str or None
        Name of the SMILES column of `data`, used to draw structures (None for inputs
        given as fingerprints or MinHash vectors).
    method : str
        ``"tmap"``, ``"umap"``, ``"tsne"`` or ``"pca"``.
    model : dict or None
        What `project` needs to place new molecules on the map: the encoding and method
        parameters, and the molecules' MinHash vectors (or the PCA axes). None for maps
        saved with ``model=False``.
    """

    x: np.ndarray
    y: np.ndarray
    edges: np.ndarray
    data: pd.DataFrame
    invalid: np.ndarray
    smiles: str | None = None
    method: str = "tmap"
    model: dict | None = field(default=None, repr=False, compare=False)

    def to_dataframe(self) -> pd.DataFrame:
        """The input rows with ``x`` and ``y`` columns added.

        Returns
        -------
        pandas.DataFrame
            A copy of `data` with the coordinates as two extra columns.
        """
        return self.data.assign(x=self.x, y=self.y)

    def project(self, data, smiles: str | None = None, *, n_jobs: int | None = None) -> "Map":
        """Place new molecules on this map, without moving the molecules already on it.

        Each method places molecules as its reference implementation does:

        - ``tmap``: the new molecules and the map's tree form a minimum spanning tree, so
          each group of related new molecules becomes a branch grafted onto its closest
          molecule on the map;
        - ``umap``: as umap-learn's ``transform``, from the weighted mean of the nearest
          neighbours on the map;
        - ``tsne``: as openTSNE's ``transform``, from the median of the nearest neighbours;
        - ``pca``: exact projection on the map's principal axes.

        Parameters
        ----------
        data : list, pandas.Series, pandas.DataFrame, str, pathlib.Path or numpy.ndarray
            The new molecules, given as for the method that made the map, and encoded the
            same way (SMILES for maps of SMILES, fingerprints for maps of fingerprints).
        smiles : str, optional
            SMILES column of a DataFrame or file (guessed if omitted).
        n_jobs : int, optional
            Processes used to parse SMILES (all CPUs by default).

        Returns
        -------
        Map
            This map's molecules followed by the new ones, with a ``projected`` column (True
            for the new molecules). It can itself receive more molecules.

        Raises
        ------
        ValueError
            If the map has no `model` (it was saved with ``model=False``), if the new
            molecules are not given the same way as the map's, or if none of them is valid.

        Warns
        -----
        UserWarning
            If some SMILES cannot be parsed; these molecules are skipped.

        Examples
        --------
        >>> ref = cv.tmap("library.csv")
        >>> m = ref.project("hits.csv")
        >>> m.save("hits.html")
        """
        from chemviz._project import project

        return project(self, data, smiles, n_jobs)

    def to_html(self, columns=None, title: str = "chemviz", offline: bool = False) -> str:
        """The interactive WebGL viewer as a self-contained HTML page.

        Parameters
        ----------
        columns : list of str, optional
            Columns of `data` to embed (all by default). Embedded columns can be used to
            colour, filter and search, and are shown on hover.
        title : str, default "chemviz"
            Title of the page.
        offline : bool, default False
            Embed RDKit.js, which draws the structures, in the page (about 3.3 MB more) so
            that it needs no network. By default it is loaded from a CDN when the page opens.
            RDKit.js is downloaded once, checked against a pinned SHA-256 and cached in
            ``~/.cache/chemviz`` (or ``$CHEMVIZ_CACHE``).

        Returns
        -------
        str
            The HTML page.

        See Also
        --------
        save : Write the page to a file.
        show : Display the page in a notebook.
        """
        from chemviz.viewer import to_html

        return to_html(self.x, self.y, self.edges, self.data, self.smiles, columns, title, offline)

    def save(self, path, columns=None, title: str | None = None, offline: bool = False, model: bool = True) -> None:
        """Write the map to a file, in a format chosen by the file extension.

        - ``.html``: the interactive viewer (see `to_html`).
        - ``.chemviz``: the map itself, to reload with `chemviz.load`. The file keeps the
          coordinates, the data with its column types and index, and the `model`. It is read
          without pickle, so opening a file from someone else cannot run code.

        Parameters
        ----------
        path : str or pathlib.Path
            File to write, ending in ``.html`` or ``.chemviz``.
        columns, title, offline
            For ``.html`` files: as for `to_html`. The title defaults to the file name.
        model : bool, default True
            For ``.chemviz`` files: keep the MinHash vectors (512 bytes per molecule). Without
            them the file is smaller, but new molecules can no longer be placed on the map.

        Raises
        ------
        ValueError
            If the extension is neither ``.html`` nor ``.chemviz``.

        Examples
        --------
        >>> m.save("map.html")
        >>> m.save("map.chemviz")
        >>> m = cv.load("map.chemviz")
        """
        path = Path(path)
        suffix = path.suffix.lower()
        if suffix == ".chemviz":
            from chemviz._store import save_map

            save_map(self, path, model)
        elif suffix in (".html", ".htm"):
            path.write_text(self.to_html(columns, title or path.stem, offline), encoding="utf-8")
        else:
            raise ValueError(f"Cannot save a map as {path.suffix!r}: use .html or .chemviz")

    def show(self, columns=None, title: str = "chemviz", height: int = 650, offline: bool = False):
        """Display the interactive viewer in a Jupyter notebook.

        Parameters
        ----------
        columns, title, offline
            As for `to_html`.
        height : int, default 650
            Height of the viewer, in pixels.

        Returns
        -------
        IPython.display.HTML
            The viewer, displayed when it is the last expression of a cell.

        See Also
        --------
        widget : The viewer as a widget, with its selection synced to Python.
        """
        try:
            from IPython.display import HTML
        except ImportError as e:
            raise ImportError("show() needs IPython (Jupyter); use save('map.html') instead") from e
        import html

        doc = html.escape(self.to_html(columns, title, offline), quote=True)
        return HTML(f'<iframe srcdoc="{doc}" style="width:100%;height:{int(height)}px;border:0" '
                    'sandbox="allow-scripts allow-downloads"></iframe>')

    def widget(self, columns=None, title: str = "chemviz", height: int = 650, offline: bool = False):
        """The interactive viewer as a Jupyter widget, with its selection synced to Python.

        Select molecules in the viewer (click, lasso, search) and read them back with
        ``widget.selected``, or assign index labels or a boolean mask to
        ``widget.selected`` to select molecules from Python. Needs the ``jupyter`` extra:
        ``pip install "chemviz[jupyter]"``.

        Parameters
        ----------
        columns, title, offline
            As for `to_html`.
        height : int, default 650
            Height of the viewer, in pixels.

        Returns
        -------
        chemviz.MapWidget
            The widget, displayed when it is the last expression of a cell.

        Raises
        ------
        ImportError
            If anywidget is not installed.

        Examples
        --------
        >>> w = m.widget()
        >>> w                                   # select molecules in the viewer
        >>> w.selected                          # the selected rows, as a DataFrame
        >>> w.selected = df.index[df.pKi > 8]   # select molecules from Python
        """
        try:
            from chemviz._widget import MapWidget
        except ImportError as e:
            raise ImportError("widget() needs anywidget: pip install 'chemviz[jupyter]'") from e
        return MapWidget(self, self.to_html(columns, title, offline), height)

    def plot(self, color=None, ax=None, edges: bool = True, cmap=None, **scatter_kws):
        """Static plot of the map with matplotlib.

        Parameters
        ----------
        color : str or array-like, optional
            A column of `data`, or one value per molecule. Numeric values get a colour bar
            (clipped to their 1st–99th percentiles), others a legend.
        ax : matplotlib.axes.Axes, optional
            Axes to draw on (a new 8 × 8 inch figure by default).
        edges : bool, default True
            Draw the tree edges of a TMAP.
        cmap : str or matplotlib.colors.Colormap, optional
            Colour map (default: ``"viridis"`` for numbers, ``"tab10"`` or ``"tab20"`` for
            categories).
        **scatter_kws
            Passed to `matplotlib.axes.Axes.scatter`, such as the point size ``s``.

        Returns
        -------
        matplotlib.axes.Axes
            The axes drawn on.

        Examples
        --------
        >>> ax = m.plot(color="qed")
        >>> ax.figure.savefig("map.png", dpi=200)
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


def _inputs(data, smiles):
    """Normalise an input to ``(frame, smiles_column, smiles_values, array)``.

    Either ``smiles_values`` (a list of strings) or ``array`` (a 2D array of MinHash
    vectors or fingerprints) is set; ``frame`` holds one row per input molecule.
    """
    if isinstance(data, (str, Path)):
        data, smiles = read_molecules(data, smiles)
    if isinstance(data, np.ndarray) and data.ndim == 2:
        return pd.DataFrame(index=pd.RangeIndex(len(data))), None, None, data
    if isinstance(data, pd.DataFrame):
        frame = data
        column = smiles or find_smiles_column(data)
        if column not in frame.columns:
            raise ValueError(f"No SMILES column {column!r}; available: {list(frame.columns)}")
    else:
        column = smiles or "smiles"
        frame = pd.DataFrame({column: list(data)}, index=data.index if isinstance(data, pd.Series) else None)
    return frame, column, frame[column].fillna("").astype(str).tolist(), None


def _is_binary(array):
    return array.dtype == bool or (np.issubdtype(array.dtype, np.integer) and np.isin(array, (0, 1)).all())


def _encode(data, smiles, n_permutations, radius, n_jobs):
    """MinHash vectors, matching DataFrame, its SMILES column (or None), invalid input
    positions and the kind of input encoded ("ecfp", "minhash" or "fingerprints")."""
    frame, column, values, array = _inputs(data, smiles)
    if array is not None:
        if array.dtype == np.uint32:
            return array, frame, None, [], "minhash"
        if _is_binary(array):
            return minhash_from_fingerprints(array, n_permutations=n_permutations), frame, None, [], "fingerprints"
        raise ValueError(
            "A 2D array must hold uint32 MinHash vectors or binary (0/1) fingerprints; "
            f"got dtype {array.dtype}"
        )
    hashes, invalid = _minhash_ecfp(values, n_permutations, radius, 42, 10_000, n_jobs)
    return hashes, frame, column, invalid, "ecfp"


def _drop_invalid(frame, invalid):
    """Warn about unparsable molecules; return the rows to keep (None for all) and the kept frame."""
    keep = None
    if invalid:
        warnings.warn(
            f"Skipped {len(invalid)} invalid molecule(s) at input positions "
            f"{invalid[:20]}{'...' if len(invalid) > 20 else ''}",
            stacklevel=3,
        )
        keep = np.ones(len(frame), dtype=bool)
        keep[invalid] = False
        frame = frame[keep]
    if len(frame) == 0:
        raise ValueError("No valid molecules to map")
    return keep, frame
