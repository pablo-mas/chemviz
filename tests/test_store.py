import io
import zipfile

import numpy as np
import pandas as pd
import pytest

import chemviz as cv

SMILES = ["C" * i + "O" for i in range(1, 16)] + ["c1ccccc1" + "C" * i for i in range(1, 16)]
N = len(SMILES)


def assert_same_map(a, b):
    np.testing.assert_array_equal(a.x, b.x)
    np.testing.assert_array_equal(a.y, b.y)
    np.testing.assert_array_equal(a.edges, b.edges)
    np.testing.assert_array_equal(a.invalid, b.invalid)
    pd.testing.assert_frame_equal(a.data, b.data)
    assert (a.smiles, a.method) == (b.smiles, b.method)


@pytest.mark.parametrize("method", ["tmap", "umap", "tsne", "pca"])
def test_round_trip_keeps_map_and_model(method, tmp_path):
    m = getattr(cv, method)(SMILES)
    m.save(tmp_path / "m.chemviz")
    back = cv.load(tmp_path / "m.chemviz")
    assert_same_map(m, back)
    assert back.model.keys() == m.model.keys()
    for key, value in m.model.items():
        np.testing.assert_array_equal(back.model[key], value)


def test_round_trip_column_types(tmp_path):
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "smiles": SMILES,
        "name": [f"mol {i} é" if i % 7 else None for i in range(N)],
        "pki": rng.random(N),
        "count": np.arange(N, dtype=np.int32),
        "active": np.arange(N) % 2 == 0,
        "series": pd.Categorical(["a", "b", "c"] * 10, categories=["c", "b", "a"], ordered=True),
        "nullable": pd.array([None if i % 5 == 0 else i for i in range(N)], dtype="Int64"),
        "flag": pd.array([None if i % 4 == 0 else i % 3 == 0 for i in range(N)], dtype="boolean"),
        "label": pd.array([None if i % 6 == 0 else f"l{i}" for i in range(N)], dtype="string"),
        "date": pd.date_range("2024-01-01", periods=N, freq="D"),
        7: np.zeros(N),  # non-string column name
    }, index=pd.Index([f"id{i}" for i in range(N)], name="mol_id"))
    m = cv.tmap(df, smiles="smiles")
    m.save(tmp_path / "m.chemviz")
    assert_same_map(m, cv.load(tmp_path / "m.chemviz"))


def test_invalid_molecules_and_multiindex(tmp_path):
    df = pd.DataFrame({"smiles": [*SMILES[:10], "not_a_smiles"]},
                      index=pd.MultiIndex.from_arrays([["a"] * 11, range(11)], names=["batch", "n"]))
    with pytest.warns(UserWarning, match="invalid"):
        m = cv.pca(df)
    m.save(tmp_path / "m.chemviz")
    back = cv.load(tmp_path / "m.chemviz")
    assert_same_map(m, back)
    assert back.invalid.tolist() == [10]


def test_without_model_and_unknown_types(tmp_path):
    m = cv.umap(pd.DataFrame({"smiles": SMILES, "obj": [[i] for i in range(N)]}))
    with pytest.warns(UserWarning, match="saved as text"):
        m.save(tmp_path / "m.chemviz", model=False)
    back = cv.load(tmp_path / "m.chemviz")
    assert back.model is None
    assert back.data["obj"].iloc[3] == "[3]"
    assert (tmp_path / "m.chemviz").stat().st_size < 0.5 * N * 128 * 4 + 20_000


def test_rejects_other_files_and_pickles(tmp_path):
    with pytest.raises(ValueError, match="Cannot save"):
        cv.tmap(SMILES[:5]).save(tmp_path / "m.txt")
    path = tmp_path / "fake.chemviz"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("other.txt", "hello")
    with pytest.raises(ValueError, match="not a chemviz map"):
        cv.load(path)
    # A file whose arrays are pickled objects must not be unpickled.
    m = cv.tmap(SMILES[:5])
    m.save(path)
    with zipfile.ZipFile(path) as zf:
        entries = {name: zf.read(name) for name in zf.namelist()}
    buf = io.BytesIO()
    np.save(buf, np.array([1, "a"], dtype=object), allow_pickle=True)
    entries["x.npy"] = buf.getvalue()
    with zipfile.ZipFile(path, "w") as zf:
        for name, content in entries.items():
            zf.writestr(name, content)
    with pytest.raises(ValueError, match="pickle"):
        cv.load(path)
