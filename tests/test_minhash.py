import numpy as np
import pytest
from rdkit.Chem import rdMHFPFingerprint

from chemviz import mhfp, minhash_from_fingerprints

SMILES = ["CCO", "CCN", "c1ccccc1O", "CC(=O)Oc1ccccc1C(=O)O"]


def test_mhfp_shape_and_dtype():
    h = mhfp(SMILES, n_permutations=64)
    assert h.shape == (4, 64)
    assert h.dtype == np.uint32


def test_mhfp_matches_rdkit_single_encoding():
    encoder = rdMHFPFingerprint.MHFPEncoder(128, 42)
    expected = np.array([encoder.EncodeSmiles(s) for s in SMILES], dtype=np.uint32)
    np.testing.assert_array_equal(mhfp(SMILES), expected)


def test_mhfp_invalid_smiles_raises_with_index():
    with pytest.raises(ValueError, match=r"\[1\]"):
        mhfp(["CCO", "not_a_smiles"])


def test_minhash_estimates_jaccard():
    rng = np.random.default_rng(0)
    a = rng.random(4096) < 0.1
    b = a.copy()
    flip = rng.choice(4096, 300, replace=False)
    b[flip] = ~b[flip]
    true_jaccard = (a & b).sum() / (a | b).sum()

    h = minhash_from_fingerprints(np.stack([a, b]), n_permutations=2048)
    estimated = (h[0] == h[1]).mean()
    assert abs(estimated - true_jaccard) < 0.03
