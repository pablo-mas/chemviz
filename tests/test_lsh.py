import numpy as np
import pytest

from chemviz import lsh_knn, minhash_from_fingerprints


@pytest.fixture(scope="module")
def clustered_hashes():
    """MinHashes of 2000 noisy copies of 50 random sets, so neighbourhoods are meaningful."""
    rng = np.random.default_rng(0)
    n_bits, n_clusters, per_cluster = 4096, 50, 40
    centers = rng.random((n_clusters, n_bits)) < 0.03
    fps = np.repeat(centers, per_cluster, axis=0)
    noise = rng.random(fps.shape) < 0.01
    fps ^= noise
    return minhash_from_fingerprints(fps, n_permutations=128)


def brute_force_distances(hashes):
    return 1.0 - (hashes[:, None, :] == hashes[None, :, :]).mean(axis=2)


def test_knn_shapes_and_self_exclusion(clustered_hashes):
    idx, dist = lsh_knn(clustered_hashes, k=10)
    n = clustered_hashes.shape[0]
    assert idx.shape == dist.shape == (n, 10)
    assert not (idx == np.arange(n)[:, None]).any()
    assert (np.diff(dist, axis=1) >= 0).all()


def test_knn_distances_are_exact_minhash_distances(clustered_hashes):
    idx, dist = lsh_knn(clustered_hashes, k=10)
    full = brute_force_distances(clustered_hashes)
    np.testing.assert_allclose(dist, np.take_along_axis(full, idx, axis=1), atol=1e-6)


def test_knn_recall_against_brute_force(clustered_hashes):
    k = 10
    idx, dist = lsh_knn(clustered_hashes, k=k)
    full = brute_force_distances(clustered_hashes)
    np.fill_diagonal(full, np.inf)
    kth_true = np.sort(full, axis=1)[:, k - 1]
    # A found neighbour counts as correct if it is as close as the true k-th neighbour (handles ties).
    recall = (dist <= kth_true[:, None] + 1e-6).mean()
    assert recall >= 0.95


def test_d_not_divisible_by_n_trees_raises(clustered_hashes):
    with pytest.raises(ValueError):
        lsh_knn(clustered_hashes[:, :100], n_trees=8)
