import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest
from rdkit import Chem

import chemviz as cv

# Three small "series" so maps have some structure.
SMILES = (
    ["C" * i + "O" for i in range(1, 11)]
    + ["c1ccccc1" + "C" * i for i in range(1, 11)]
    + ["N" + "C" * i + "C(=O)O" for i in range(1, 11)]
)
SERIES = ["alcohol"] * 10 + ["benzene"] * 10 + ["amino_acid"] * 10


def check_map(m, n):
    assert isinstance(m, cv.TMap)
    assert m.x.shape == m.y.shape == (n,)
    assert len(m.data) == n
    assert m.edges.shape == (n - 1, 2)  # a single tree over valid molecules
    assert np.isfinite(m.x).all() and np.isfinite(m.y).all()


def test_list_of_smiles():
    m = cv.tmap(SMILES)
    check_map(m, 30)
    df = m.to_dataframe()
    assert list(df.columns) == ["smiles", "x", "y"]
    assert df["smiles"].tolist() == SMILES


def test_dataframe_keeps_columns_and_index():
    df = pd.DataFrame({"Canonical_SMILES": SMILES, "series": SERIES}, index=[f"m{i}" for i in range(30)])
    m = cv.tmap(df)  # column guessed
    check_map(m, 30)
    out = m.to_dataframe()
    assert out.index.tolist() == df.index.tolist()
    assert out["series"].tolist() == SERIES


def test_series_keeps_index():
    s = pd.Series(SMILES, index=range(100, 130))
    assert cv.tmap(s).data.index.tolist() == list(range(100, 130))


def test_invalid_smiles_are_skipped_with_warning():
    smiles = SMILES[:5] + ["not_a_smiles"] + SMILES[5:]
    with pytest.warns(UserWarning, match=r"Skipped 1 invalid molecule.*\[5\]"):
        m = cv.tmap(smiles)
    check_map(m, 30)
    assert m.invalid.tolist() == [5]
    assert "not_a_smiles" not in m.data["smiles"].tolist()


def test_missing_smiles_column_raises():
    with pytest.raises(ValueError, match="smiles="):
        cv.tmap(pd.DataFrame({"a": SMILES, "b": SERIES}))


def test_csv_smi_and_sdf_files(tmp_path):
    pd.DataFrame({"smiles": SMILES, "series": SERIES}).to_csv(tmp_path / "mols.csv", index=False)
    (tmp_path / "mols.smi").write_text("\n".join(f"{s} mol{i}" for i, s in enumerate(SMILES)))
    writer = Chem.SDWriter(str(tmp_path / "mols.sdf"))
    for s, series in zip(SMILES, SERIES):
        mol = Chem.MolFromSmiles(s)
        mol.SetProp("series", series)
        writer.write(mol)
    writer.close()

    check_map(m := cv.tmap(tmp_path / "mols.csv"), 30)
    assert m.data["series"].tolist() == SERIES
    check_map(m := cv.tmap(str(tmp_path / "mols.smi")), 30)
    assert m.data["name"].tolist() == [f"mol{i}" for i in range(30)]
    check_map(m := cv.tmap(tmp_path / "mols.sdf"), 30)
    assert m.data["series"].tolist() == SERIES


def test_precomputed_minhash_and_fingerprints():
    check_map(cv.tmap(cv.minhash_ecfp(SMILES, n_jobs=1)), 30)
    fps = np.random.default_rng(0).random((30, 512)) < 0.1
    check_map(cv.tmap(fps), 30)
    check_map(cv.tmap(fps.astype(np.uint8)), 30)
    with pytest.raises(ValueError, match="dtype"):
        cv.tmap(np.random.default_rng(0).random((30, 8)))


def test_tiny_inputs():
    check_map(cv.tmap(["CCO"]), 1)
    check_map(cv.tmap(["CCO", "CCN"]), 2)
    with pytest.raises(ValueError, match="No valid molecules"):
        with pytest.warns(UserWarning):
            cv.tmap(["not_a_smiles"])


@pytest.mark.parametrize("color", [None, "series", "value", np.arange(30)])
def test_plot(color):
    df = pd.DataFrame({"smiles": SMILES, "series": SERIES, "value": np.linspace(0, 1, 30)})
    ax = cv.tmap(df).plot(color=color)
    assert len(ax.collections) >= 2  # edges + points
