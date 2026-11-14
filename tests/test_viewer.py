import base64
import json
import re
import zlib

import numpy as np
import pandas as pd
import pytest

import chemviz as cv
from chemviz.viewer import build_payload

SMILES = ["C" * i + "O" for i in range(1, 11)] + ["c1ccccc1" + "C" * i for i in range(1, 11)]


@pytest.fixture(scope="module")
def tmap():
    df = pd.DataFrame({
        "smiles": SMILES,
        "name": [f"mol{i}" for i in range(20)],
        "series": ["alcohol"] * 10 + ["benzene"] * 10,
        "value": np.r_[np.linspace(0, 1, 19), np.nan],
        "count": np.arange(20),
        "flag": [True, False] * 10,
    })
    return cv.tmap(df)


def decode(html):
    header = json.loads(re.search(r'id="chemviz-header">(.*?)</script>', html, re.S).group(1))
    blob = zlib.decompress(base64.b64decode(re.search(r'id="chemviz-data">(.*?)</script>', html, re.S).group(1)))
    return header, blob


def test_payload_round_trip(tmap):
    header, blob = decode(tmap.to_html(title="Test map"))
    assert header["n"] == 20 and header["n_edges"] == 19 and header["smiles"] == "smiles"
    off, length = header["positions"]
    pos = np.frombuffer(blob, np.float32, length // 4, off).reshape(-1, 2)
    np.testing.assert_array_equal(pos, np.column_stack([tmap.x, tmap.y]))
    off, length = header["edges"]
    np.testing.assert_array_equal(np.frombuffer(blob, np.uint32, length // 4, off).reshape(-1, 2), tmap.edges)

    cols = {c["name"]: c for c in header["columns"]}
    assert list(cols) == ["smiles", "name", "series", "value", "count", "flag"]  # SMILES first
    assert all(c["offset"] % 4 == 0 for c in cols.values())
    kinds = {name: c["kind"] for name, c in cols.items()}
    assert kinds == {"smiles": "text", "name": "text", "series": "cat", "value": "num", "count": "num", "flag": "cat"}
    c = cols["smiles"]
    assert blob[c["offset"]:c["offset"] + c["length"]].decode().split("\n") == SMILES
    c = cols["series"]
    codes = np.frombuffer(blob, np.uint32, c["length"] // 4, c["offset"])
    assert [c["categories"][k] for k in codes] == tmap.data["series"].tolist()
    c = cols["value"]
    values = np.frombuffer(blob, np.float32, c["length"] // 4, c["offset"])
    assert np.isnan(values[-1]) and c["min"] == 0 and c["max"] == 1
    assert cols["count"]["integer"] and not cols["value"]["integer"]


def test_columns_subset_and_unknown(tmap):
    header, _ = decode(tmap.to_html(columns=["value"]))
    assert [c["name"] for c in header["columns"]] == ["smiles", "value"]
    with pytest.raises(ValueError, match="Unknown column"):
        tmap.to_html(columns=["nope"])


def test_save_writes_self_contained_page(tmap, tmp_path):
    path = tmp_path / "my map.html"
    tmap.save(path)
    html = path.read_text(encoding="utf-8")
    assert "<title>my map</title>" in html
    assert "__CHEMVIZ" not in html
    assert "</script>" not in re.search(r'id="chemviz-header">(.*?)</script>', html, re.S).group(1)


def test_title_and_values_are_escaped():
    m = cv.tmap(pd.DataFrame({"smiles": SMILES, "note": ["</script><b>x"] * 20}))
    html = m.to_html(title="<b>&")
    assert "<title>&lt;b&gt;&amp;</title>" in html
    header, _ = decode(html)  # the header must survive the "</script>" inside a category
    assert header["columns"][1]["categories"] == ["</script><b>x"]


def test_without_smiles_column():
    m = cv.tmap(np.random.default_rng(0).random((30, 256)) < 0.1)
    header, _ = decode(m.to_html())
    assert header["smiles"] is None and header["columns"] == []


def test_show_returns_iframe(tmap):
    pytest.importorskip("IPython")
    out = tmap.show(height=400)
    assert out.data.startswith("<iframe srcdoc=") and "height:400px" in out.data


def test_build_payload_without_edges():
    header, blob = build_payload(np.zeros(1), np.zeros(1), np.empty((0, 2)), pd.DataFrame(index=[0]), None)
    assert header["n"] == 1 and header["edges"][1] == 0


@pytest.fixture
def fake_rdkit_js(monkeypatch, tmp_path):
    """Serve fake RDKit.js files instead of downloading them; returns the list of fetched URLs."""
    import hashlib
    import io

    from chemviz import viewer

    contents = {"RDKit_minimal.js": b"var initRDKitModule = 1;", "RDKit_minimal.wasm": b"\0asm fake"}
    monkeypatch.setattr(viewer, "RDKIT_JS_FILES", {k: hashlib.sha256(v).hexdigest() for k, v in contents.items()})
    monkeypatch.setenv("CHEMVIZ_CACHE", str(tmp_path / "cache"))
    fetched = []

    def urlopen(url, timeout):
        fetched.append(url)
        return io.BytesIO(contents[url.rsplit("/", 1)[1]])

    monkeypatch.setattr(viewer.urllib.request, "urlopen", urlopen)
    return contents, fetched


def test_offline_embeds_rdkit_js(tmap, fake_rdkit_js):
    contents, fetched = fake_rdkit_js
    html = tmap.to_html(offline=True)
    for name, element in [("RDKit_minimal.js", "rdkit_minimal-js"), ("RDKit_minimal.wasm", "rdkit_minimal-wasm")]:
        b64 = re.search(rf'id="chemviz-{element}">(.*?)</script>', html, re.S).group(1)
        assert zlib.decompress(base64.b64decode(b64)) == contents[name]
    assert len(fetched) == 2
    tmap.to_html(offline=True)
    assert len(fetched) == 2  # cached
    assert 'id="chemviz-rdkit' not in tmap.to_html()  # online by default


def test_offline_redownloads_corrupted_cache(tmap, fake_rdkit_js):
    from chemviz.viewer import RDKIT_JS_VERSION, cache_dir

    _, fetched = fake_rdkit_js
    tmap.to_html(offline=True)
    (cache_dir() / f"rdkit-js-{RDKIT_JS_VERSION}" / "RDKit_minimal.wasm").write_bytes(b"truncated")
    tmap.to_html(offline=True)
    assert len(fetched) == 3


def test_offline_rejects_bad_download_and_reports_network_errors(tmap, fake_rdkit_js, monkeypatch):
    from chemviz import viewer

    monkeypatch.setitem(viewer.RDKIT_JS_FILES, "RDKit_minimal.js", "0" * 64)
    with pytest.raises(RuntimeError, match="SHA-256"):
        tmap.to_html(offline=True)

    def no_network(url, timeout):
        raise OSError("no route to host")

    monkeypatch.setattr(viewer.urllib.request, "urlopen", no_network)
    with pytest.raises(RuntimeError, match="could not be downloaded"):
        tmap.to_html(offline=True)


def test_offline_without_smiles_embeds_nothing(fake_rdkit_js):
    _, fetched = fake_rdkit_js
    m = cv.tmap(np.random.default_rng(0).random((30, 256)) < 0.1)
    assert 'id="chemviz-rdkit' not in m.to_html(offline=True) and not fetched
