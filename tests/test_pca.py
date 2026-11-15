import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

import chemviz as cv

SMILES = (
    ["C" * i + "O" for i in range(1, 11)]
    + ["c1ccccc1" + "C" * i for i in range(1, 11)]
    + ["N" + "C" * i + "C(=O)O" for i in range(1, 11)]
)


def test_pca_layout_matches_svd():
    rng = np.random.default_rng(0)
    x = rng.random((500, 64)) < 0.1
    x[:250, :8] |= rng.random((250, 8)) < 0.6
    pos = cv.pca_layout(x).astype(float)
    centred = x - x.mean(axis=0)
    _, _, vt = np.linalg.svd(centred, full_matrices=False)
    ref = centred @ vt[:2].T
    for c in range(2):  # principal axes are defined up to sign
        assert min(np.abs(pos[:, c] - s * ref[:, c]).max() for s in (1, -1)) < 1e-5


def test_pca_layout_rejects_non_binary():
    with pytest.raises(ValueError, match="binary"):
        cv.pca_layout(np.full((5, 4), 2))


def test_pca_of_smiles():
    df = pd.DataFrame({"smiles": SMILES + ["not_a_smiles"], "series": [i // 10 for i in range(31)]})
    with pytest.warns(UserWarning, match="Skipped 1 invalid"):
        m = cv.pca(df)
    assert m.method == "pca" and m.smiles == "smiles"
    assert m.x.shape == (30,) and len(m.data) == 30 and m.edges.shape == (0, 2)
    assert m.invalid.tolist() == [30]
    # The three series separate on the first two axes.
    centres = np.array([[m.x[m.data.series == s].mean(), m.y[m.data.series == s].mean()] for s in range(3)])
    spread = max(np.hypot(m.x[m.data.series == s] - c[0], m.y[m.data.series == s] - c[1]).mean()
                 for s, c in enumerate(centres))
    assert min(np.linalg.norm(centres[a] - centres[b]) for a, b in [(0, 1), (0, 2), (1, 2)]) > spread


def test_pca_is_deterministic_and_plots():
    a, b = cv.pca(SMILES, n_jobs=1), cv.pca(SMILES)
    np.testing.assert_array_equal(a.x, b.x)
    assert a.plot() is not None
    assert "chemviz" in a.to_html()


def test_pca_of_fingerprints_and_minhash():
    fps = np.random.default_rng(1).random((40, 128)) < 0.2
    m = cv.pca(fps)
    assert m.x.shape == (40,) and m.smiles is None
    with pytest.raises(ValueError, match="MinHash"):
        cv.pca(cv.minhash_ecfp(SMILES))
