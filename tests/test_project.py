import numpy as np
import pandas as pd
import pytest

import chemviz as cv

SERIES = [
    ["C" * i + "O" for i in range(1, 16)],
    ["c1ccccc1" + "C" * i for i in range(1, 16)],
    ["N" + "C" * i + "C(=O)O" for i in range(1, 16)],
]
SMILES = [s for series in SERIES for s in series]
LABELS = np.repeat(np.arange(3), 15)
NEW = np.arange(45) % 3 == 0  # every third molecule of each series is placed afterwards
REF_SMILES = [s for s, new in zip(SMILES, NEW) if not new]
NEW_SMILES = [s for s, new in zip(SMILES, NEW) if new]


@pytest.fixture(scope="module", params=["tmap", "umap", "tsne", "pca"])
def maps(request):
    ref = getattr(cv, request.param)(REF_SMILES)
    return ref, ref.project(NEW_SMILES)


def test_new_molecules_land_near_their_series(maps):
    ref, both = maps
    n = len(ref.x)
    assert both.method == ref.method and len(both.x) == 45
    np.testing.assert_array_equal(both.x[:n], ref.x)  # the map does not move
    np.testing.assert_array_equal(both.y[:n], ref.y)
    assert both.data["projected"].tolist() == [False] * n + [True] * (45 - n)
    assert both.data["smiles"].tolist() == REF_SMILES + NEW_SMILES
    pos = np.column_stack([both.x, both.y])
    labels = np.concatenate([LABELS[~NEW], LABELS[NEW]])
    d = np.linalg.norm(pos[n:, None] - pos[None, :n], axis=2)
    assert (labels[:n][d.argmin(axis=1)] == labels[n:]).mean() >= 0.8


def test_tmap_grafts_branches_onto_the_tree(maps):
    ref, both = maps
    if ref.method != "tmap":
        pytest.skip("tree maps only")
    assert both.edges.shape == (44, 2)
    np.testing.assert_array_equal(both.edges[: len(ref.edges)], ref.edges)
    parent = np.arange(45)  # the edges form a single tree

    def find(i):
        while parent[i] != i:
            i = parent[i]
        return i

    for a, b in both.edges:
        assert find(a) != find(b)
        parent[find(a)] = find(b)


def test_projection_is_deterministic_and_chains(maps):
    ref, both = maps
    again = ref.project(NEW_SMILES)
    np.testing.assert_array_equal(both.x, again.x)
    more = both.project(["CCCCCCCCCCCCCCCCO"])
    assert len(more.x) == 46 and more.data["projected"].sum() == 16
    if ref.model.get("hashes") is not None:
        assert len(more.model["hashes"]) == 46


def test_pca_projection_is_exact():
    ref = cv.pca(SMILES)
    both = ref.project(SMILES[:5])
    np.testing.assert_allclose(both.x[45:], ref.x[:5], atol=1e-5)
    np.testing.assert_allclose(both.y[45:], ref.y[:5], atol=1e-5)


def test_inputs_are_merged_and_checked(tmp_path):
    ref = cv.tmap(REF_SMILES)
    new = pd.DataFrame({"SMILES": [*NEW_SMILES, "not_a_smiles"], "pKi": np.arange(16.0)})
    with pytest.warns(UserWarning, match="Skipped 1 invalid"):
        both = ref.project(new)
    assert len(both.x) == 45 and both.smiles == "smiles"
    assert both.data["smiles"].iloc[30:].tolist() == NEW_SMILES
    assert both.data["pKi"].iloc[:30].isna().all() and both.data["pKi"].iloc[30:].tolist() == list(range(15))
    with pytest.raises(ValueError, match="made from SMILES"):
        ref.project(cv.minhash_ecfp(NEW_SMILES))
    ref.save(tmp_path / "ref.chemviz", model=False)
    with pytest.raises(ValueError, match="no model"):
        cv.load(tmp_path / "ref.chemviz").project(NEW_SMILES)
    ref.save(tmp_path / "ref.chemviz")
    loaded = cv.load(tmp_path / "ref.chemviz").project(NEW_SMILES)
    np.testing.assert_array_equal(loaded.x, ref.project(NEW_SMILES).x)
