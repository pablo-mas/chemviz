import numpy as np
import pytest
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, minimum_spanning_tree

from chemviz import lsh_knn, minhash_from_fingerprints, minimum_spanning_forest


def knn_graph(n_clusters=30, per_cluster=30, seed=0):
    rng = np.random.default_rng(seed)
    centers = rng.random((n_clusters, 2048)) < 0.03
    fps = np.repeat(centers, per_cluster, axis=0) ^ (rng.random((n_clusters * per_cluster, 2048)) < 0.01)
    return lsh_knn(minhash_from_fingerprints(fps), k=5)


def scipy_graph(indices, distances, offset=1.0):
    """Symmetric sparse graph; ``offset`` avoids scipy treating zero distances as missing edges."""
    n, k = indices.shape
    valid = indices.ravel() >= 0
    rows = np.repeat(np.arange(n), k)[valid]
    cols = indices.ravel()[valid]
    g = coo_matrix((distances.ravel()[valid] + offset, (rows, cols)), shape=(n, n)).tocsr()
    return g.maximum(g.T)  # MinHash distances are symmetric, so this just fills in reverse edges


def test_forest_has_one_tree_per_component():
    indices, distances = knn_graph()
    edges, _ = minimum_spanning_forest(indices, distances)
    n = indices.shape[0]
    n_comp, labels = connected_components(scipy_graph(indices, distances), directed=False)
    assert edges.shape == (n - n_comp, 2)
    # Same components as the input graph => the forest spans each one without cycles.
    forest = coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(n, n))
    f_comp, f_labels = connected_components(forest, directed=False)
    assert f_comp == n_comp
    assert len(set(zip(labels, f_labels))) == n_comp


def test_total_weight_matches_scipy():
    indices, distances = knn_graph()
    edges, weights = minimum_spanning_forest(indices, distances)
    # Shifting all weights by a constant does not change which forests are minimal.
    expected = minimum_spanning_tree(scipy_graph(indices, distances)).sum() - len(edges)
    np.testing.assert_allclose(weights.sum(), expected, rtol=1e-5)


def test_missing_neighbours_are_ignored():
    indices = np.array([[1, -1], [0, -1], [-1, -1]])
    distances = np.array([[0.2, np.inf], [0.2, np.inf], [np.inf, np.inf]], dtype=np.float32)
    edges, weights = minimum_spanning_forest(indices, distances)
    assert edges.tolist() == [[0, 1]]
    assert weights.tolist() == [np.float32(0.2)]


@pytest.mark.parametrize("union", [False, True])
def test_symmetrize_matches_dense(union):
    from chemviz.graph import _symmetrize

    rng = np.random.default_rng(0)
    n, k = 30, 5
    idx = np.array([rng.choice(np.delete(np.arange(n), i), k, replace=False) for i in range(n)])
    idx[0, 0] = -1  # a missing neighbour
    w = rng.random((n, k))
    dense = np.zeros((n, n))
    for i in range(n):
        for c in range(k):
            if idx[i, c] >= 0:
                dense[i, idx[i, c]] = w[i, c]
    expected = dense + dense.T - dense * dense.T if union else dense + dense.T
    indptr, nbrs, weights = _symmetrize(idx, w, union)
    got = np.zeros((n, n))
    rows = np.repeat(np.arange(n), np.diff(indptr))
    got[rows, nbrs] = weights
    np.testing.assert_allclose(got, expected)
    assert all((np.diff(nbrs[indptr[i]:indptr[i + 1]]) > 0).all() for i in range(n))
