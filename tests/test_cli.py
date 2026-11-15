import subprocess
import sys

import pandas as pd
import pytest

from chemviz.cli import main

SMILES = (
    ["C" * i + "O" for i in range(1, 11)]
    + ["c1ccccc1" + "C" * i for i in range(1, 11)]
    + ["N" + "C" * i + "C(=O)O" for i in range(1, 11)]
)
SERIES = ["alcohol"] * 10 + ["benzene"] * 10 + ["amino_acid"] * 10


@pytest.fixture
def mols(tmp_path):
    path = tmp_path / "mols.csv"
    pd.DataFrame({"smiles": SMILES, "series": SERIES, "pki": range(30)}).to_csv(path, index=False)
    return path


def test_default_output_is_html_next_to_input(mols, capsys):
    assert main(["tmap", str(mols)]) == 0
    html = mols.with_suffix(".html").read_text(encoding="utf-8")
    assert "<title>mols</title>" in html
    err = capsys.readouterr().err
    assert "Mapped 30 molecules" in err and "mols.html" in err


def test_several_outputs(mols, tmp_path):
    out = [tmp_path / "map.html", tmp_path / "map.png", tmp_path / "coords.csv"]
    args = ["tmap", str(mols), "--color", "series", "--columns", "pki", "--title", "My map", "--seed", "1"]
    assert main(args + [a for p in out for a in ("-o", str(p))]) == 0
    html = out[0].read_text(encoding="utf-8")
    assert "<title>My map</title>" in html
    assert '"name": "pki"' in html and '"name": "series"' not in html
    assert out[1].read_bytes().startswith(b"\x89PNG")
    coords = pd.read_csv(out[2])
    assert list(coords.columns) == ["smiles", "series", "pki", "x", "y"] and len(coords) == 30


def test_invalid_smiles_are_reported(tmp_path, capsys):
    path = tmp_path / "mols.smi"
    path.write_text("\n".join([*SMILES, "not_a_smiles"]), encoding="utf-8")
    assert main(["tmap", str(path), "-o", str(tmp_path / "m.csv")]) == 0
    assert "chemviz: warning: Skipped 1 invalid molecule(s)" in capsys.readouterr().err
    assert len(pd.read_csv(tmp_path / "m.csv")) == 30


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        (["-o", "map.txt"], "unsupported output"),
        (["--color", "series"], "--color applies to static plots"),
    ],
)
def test_usage_errors(mols, capsys, extra, message):
    with pytest.raises(SystemExit) as e:
        main(["tmap", str(mols), *extra])
    assert e.value.code == 2
    assert message in capsys.readouterr().err


def test_missing_input(tmp_path, capsys):
    with pytest.raises(SystemExit):
        main(["tmap", str(tmp_path / "nope.csv")])
    assert "no such file" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        (["--smiles", "smi"], "No SMILES column 'smi'"),
        (["-o", "m.png", "--color", "nope"], "Unknown --color column 'nope'"),
    ],
)
def test_data_errors(mols, capsys, extra, message, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    assert main(["tmap", str(mols), *extra]) == 1
    assert message in capsys.readouterr().err


def test_module_entry_point():
    res = subprocess.run([sys.executable, "-m", "chemviz", "--version"], capture_output=True, text=True, check=True)
    assert res.stdout.startswith("chemviz ")
