import numba as nb
import numpy as np
import pytest

import chemviz as cv
from chemviz._tsne import _affinities, _conditional_p

SERIES_SMILES = [
    ["C" * i + "O" for i in range(1, 16)],
    ["c1ccccc1" + "C" * i for i in range(1, 16)],
    ["N" + "C" * i + "C(=O)O" for i in range(1, 16)],
]
SMILES = [s for series in SERIES_SMILES for s in series]
LABELS = np.repeat(np.arange(3), 15)


def test_conditional_p_has_the_requested_perplexity():
    dist = np.sort(np.random.default_rng(0).random((40, 30)), axis=1)
    p = _conditional_p(dist, 5.0)
    np.testing.assert_allclose(p.sum(axis=1), 1.0, rtol=1e-6)  # stored as float32
    entropy = -(p * np.log(np.where(p > 0, p, 1))).sum(axis=1)
    np.testing.assert_allclose(np.exp(entropy), 5.0, rtol=1e-4)


def test_affinities_are_symmetric_and_normalised():
    idx, dist = cv.lsh_knn(cv.minhash_ecfp(SMILES), k=9)
    indptr, nbrs, p = _affinities(idx, dist, 3.0)
    assert p.sum() == pytest.approx(1.0)
    src = np.repeat(np.arange(len(SMILES)), np.diff(indptr))
    pairs = dict(zip(zip(src.tolist(), nbrs.tolist()), p.tolist()))
    assert all(pairs[(j, i)] == pytest.approx(v) for (i, j), v in pairs.items())


def test_tsne_separates_series_and_is_deterministic():
    m = cv.tsne(SMILES, perplexity=5)
    assert m.method == "tsne" and m.edges.shape == (0, 2) and m.x.shape == (45,)
    pos = np.column_stack([m.x, m.y])
    d = np.linalg.norm(pos[:, None] - pos[None], axis=2)
    np.fill_diagonal(d, np.inf)
    assert (LABELS[d.argmin(axis=1)] == LABELS).mean() > 0.8
    threads = nb.get_num_threads()
    nb.set_num_threads(1)
    try:
        again = cv.tsne(SMILES, perplexity=5)
    finally:
        nb.set_num_threads(threads)
    np.testing.assert_array_equal(m.x, again.x)


def test_tsne_skips_invalid_and_handles_tiny_inputs():
    with pytest.warns(UserWarning, match="Skipped 1 invalid"):
        m = cv.tsne([*SMILES[:12], "not_a_smiles"], perplexity=3)
    assert len(m.x) == 12 and m.invalid.tolist() == [12]
    for n in (1, 2, 3, 5):
        assert cv.tsne(SMILES[:n]).x.shape == (n,)
