import numpy as np
import pytest

from chemviz import lsh_knn, minhash_from_fingerprints, minimum_spanning_forest, tree_layout


@pytest.fixture(scope="module")
def forest():
    rng = np.random.default_rng(0)
    centers = rng.random((40, 2048)) < 0.03
    fps = np.repeat(centers, 50, axis=0) ^ (rng.random((2000, 2048)) < 0.01)
    edges, _ = minimum_spanning_forest(*lsh_knn(minhash_from_fingerprints(fps), k=10))
    return 2000, edges


def test_shape_dtype_and_finite(forest):
    n, edges = forest
    pos = tree_layout(n, edges)
    assert pos.shape == (n, 2)
    assert pos.dtype == np.float32
    assert np.isfinite(pos).all()


def test_deterministic_for_a_seed(forest):
    n, edges = forest
    np.testing.assert_array_equal(tree_layout(n, edges, seed=1), tree_layout(n, edges, seed=1))


def test_tree_edges_are_short(forest):
    n, edges = forest
    pos = tree_layout(n, edges)
    edge_len = np.linalg.norm(pos[edges[:, 0]] - pos[edges[:, 1]], axis=1)
    rng = np.random.default_rng(0)
    a, b = rng.integers(0, n, (2, 5000))
    random_len = np.linalg.norm(pos[a] - pos[b], axis=1)
    assert np.median(edge_len) < 0.05 * np.median(random_len)


def test_no_overlapping_nodes(forest):
    n, edges = forest
    pos = tree_layout(n, edges)
    assert len(np.unique(np.round(pos, 4), axis=0)) == n


@pytest.mark.parametrize(
    "n, edges",
    [
        (1, np.empty((0, 2), dtype=np.int64)),
        (2, np.array([[0, 1]])),
        (5, np.empty((0, 2), dtype=np.int64)),  # only isolated nodes
        (6, np.array([[0, 1], [1, 2], [3, 4]])),  # several components
    ],
)
def test_small_and_disconnected_inputs(n, edges):
    pos = tree_layout(n, edges)
    assert pos.shape == (n, 2)
    assert np.isfinite(pos).all()
