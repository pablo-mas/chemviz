import numpy as np
import pandas as pd
import pytest

import chemviz as cv

pytest.importorskip("anywidget")

SMILES = ["C" * i + "O" for i in range(1, 11)]


@pytest.fixture
def widget():
    df = pd.DataFrame({"smiles": SMILES, "pki": np.arange(10.0)}, index=[f"m{i}" for i in range(10)])
    return cv.tmap(df).widget(columns=["pki"], height=400)


def test_widget_embeds_the_viewer(widget):
    assert "chemviz-header" in widget._html and widget.height == 400
    assert widget.rows == [] and widget.selected.empty


def test_selection_from_python(widget):
    widget.selected = ["m3", "m1"]
    assert widget.rows == [1, 3]
    assert widget.selected.index.tolist() == ["m1", "m3"]
    assert list(widget.selected.columns) == ["smiles", "pki", "x", "y"]
    widget.selected = widget._map.data["pki"].to_numpy() > 7
    assert widget.rows == [8, 9]
    with pytest.raises(KeyError, match="m42"):
        widget.selected = ["m42"]
    with pytest.raises(ValueError, match="one value per molecule"):
        widget.selected = np.array([True, False])


def test_selection_from_the_viewer(widget):
    seen = []
    widget.observe(lambda change: seen.append(change["new"]), names="rows")
    widget.rows = [5, 2, 2]  # as sent by the viewer
    assert seen == [[2, 5]] and widget.selected.index.tolist() == ["m2", "m5"]
    with pytest.raises(Exception, match="Row positions"):
        widget.rows = [10]
