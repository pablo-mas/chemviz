import numba as nb
import numpy as np
import pytest

import chemviz as cv
from chemviz.umap import _curve_params, _fuzzy_graph, _smooth_knn

SERIES_SMILES = [
    ["C" * i + "O" for i in range(1, 16)],
    ["c1ccccc1" + "C" * i for i in range(1, 16)],
    ["N" + "C" * i + "C(=O)O" for i in range(1, 16)],
]
SMILES = [s for series in SERIES_SMILES for s in series]
LABELS = np.repeat(np.arange(3), 15)


def test_curve_params_match_umap_learn():
    a, b = _curve_params(1.0, 0.1)
    assert a == pytest.approx(1.577, abs=1e-3) and b == pytest.approx(0.8951, abs=1e-3)


def test_smooth_knn_calibration():
    dist = np.sort(np.random.default_rng(0).random((50, 10)), axis=1)
    dist[0, :2] = 0.0  # duplicates: rho is the first non-zero distance
    rho, sigma = _smooth_knn(dist)
    assert rho[0] == dist[0, 2] and rho[1] == dist[1, 0]
    total = np.exp(-np.maximum(dist - rho[:, None], 0) / sigma[:, None]).sum(axis=1)
    np.testing.assert_allclose(total, np.log2(11), rtol=1e-4)


def test_fuzzy_graph_is_symmetric():
    hashes = cv.minhash_ecfp(SMILES)
    idx, dist = cv.lsh_knn(hashes, k=5)
    indptr, nbrs, w = _fuzzy_graph(idx, dist, 200)
    src = np.repeat(np.arange(len(SMILES)), np.diff(indptr))
    edges = dict(zip(zip(src.tolist(), nbrs.tolist()), w.tolist()))
    assert all(edges[(j, i)] == pytest.approx(v) for (i, j), v in edges.items())
    assert (w > 0).all() and (w <= 1).all()


def test_umap_separates_series_and_is_deterministic():
    m = cv.umap(SMILES, k=5)
    assert m.method == "umap" and m.edges.shape == (0, 2) and m.x.shape == (45,)
    pos = np.column_stack([m.x, m.y])
    # Most molecules have a nearest neighbour from their own series.
    d = np.linalg.norm(pos[:, None] - pos[None], axis=2)
    np.fill_diagonal(d, np.inf)
    assert (LABELS[d.argmin(axis=1)] == LABELS).mean() > 0.8
    threads = nb.get_num_threads()
    nb.set_num_threads(1)
    try:
        again = cv.umap(SMILES, k=5)
    finally:
        nb.set_num_threads(threads)
    np.testing.assert_array_equal(m.x, again.x)


def test_umap_skips_invalid_and_handles_tiny_inputs():
    with pytest.warns(UserWarning, match="Skipped 1 invalid"):
        m = cv.umap([*SMILES[:10], "not_a_smiles"], k=5)
    assert len(m.x) == 10 and m.invalid.tolist() == [10]
    for n in (1, 2, 3):
        assert cv.umap(SMILES[:n]).x.shape == (n,)
