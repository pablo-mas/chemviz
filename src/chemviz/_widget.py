"""Jupyter widget showing the viewer, with its selection synced to Python (see ``Map.widget``)."""

import anywidget
import numpy as np
import pandas as pd
import traitlets

_ESM = """
function render({ model, el }) {
  const frame = document.createElement("iframe");
  frame.sandbox = "allow-scripts allow-downloads";
  frame.style.cssText = `width:100%;height:${model.get("height")}px;border:0`;
  frame.srcdoc = model.get("_html");
  let fromViewer = null;  // last selection received from the viewer, not to echo it back
  const send = () => frame.contentWindow.postMessage({ chemviz: "select", rows: model.get("rows") }, "*");
  const onMessage = (e) => {
    if (e.source !== frame.contentWindow || !e.data) return;
    if (e.data.chemviz === "ready") send();
    if (e.data.chemviz === "selection") {
      fromViewer = e.data.rows;
      model.set("rows", e.data.rows);
      model.save_changes();
    }
  };
  window.addEventListener("message", onMessage);
  model.on("change:rows", () => { if (model.get("rows") !== fromViewer) send(); });
  el.append(frame);
  return () => window.removeEventListener("message", onMessage);
}
export default { render };
"""


class MapWidget(anywidget.AnyWidget):
    """The interactive viewer of a map in a Jupyter notebook, with its selection in Python.

    Created by `Map.widget`. The selection is synced both ways: selecting molecules in the
    viewer updates `rows` and `selected`, and assigning them selects molecules in the
    viewer.

    Attributes
    ----------
    rows : list of int
        Row positions of the selected molecules, in increasing order. Observe it to react
        to selections: ``widget.observe(callback, names="rows")``.
    selected : pandas.DataFrame
        The selected molecules: their rows of the map's data, with ``x`` and ``y`` columns.
        Assign index labels or a boolean mask (one value per molecule) to select molecules
        from Python.
    height : int
        Height of the viewer, in pixels.
    """

    _esm = _ESM
    _html = traitlets.Unicode().tag(sync=True)
    height = traitlets.Int(650).tag(sync=True)
    rows = traitlets.List(traitlets.Int()).tag(sync=True)

    def __init__(self, m, html: str, height: int = 650):
        super().__init__(_html=html, height=int(height))
        self._map = m

    @traitlets.validate("rows")
    def _check_rows(self, proposal):
        rows = sorted({int(i) for i in proposal["value"]})
        if rows and (rows[0] < 0 or rows[-1] >= len(self._map.x)):
            raise ValueError(f"Row positions must be in [0, {len(self._map.x)})")
        return rows

    @property
    def selected(self) -> pd.DataFrame:
        """The selected molecules: the map's data rows with x and y columns."""
        return self._map.to_dataframe().iloc[self.rows]

    @selected.setter
    def selected(self, value):
        data = self._map.data
        values = np.asarray(value)
        if values.dtype == bool:
            if len(values) != len(data):
                raise ValueError(f"A boolean mask must have one value per molecule ({len(data)})")
            rows = np.flatnonzero(values)
        else:
            rows = data.index.get_indexer_for(list(np.atleast_1d(values)))
            if (rows < 0).any():
                missing = np.atleast_1d(values)[rows < 0][:5].tolist()
                raise KeyError(f"Not in the map's index: {missing}")
        self.rows = rows.tolist()
