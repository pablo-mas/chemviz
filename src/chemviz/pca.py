"""PCA of binary fingerprints (folded ECFP by default), exact and sparse."""

import numba as nb
import numpy as np

from chemviz.maps import Map, _drop_invalid, _inputs, _is_binary
from chemviz.minhash import _folded_ecfp


@nb.njit(cache=True, parallel=True)
def _gram(indptr, bits, d, n_blocks):
    """Upper triangle of X^T X for a binary matrix X given as sorted CSR on-bit indices.

    Rows are split into ``n_blocks`` blocks accumulated in parallel into separate counts.
    """
    n = indptr.shape[0] - 1
    out = np.zeros((n_blocks, d, d), dtype=np.uint32)
    for blk in nb.prange(n_blocks):
        for i in range(blk * n // n_blocks, (blk + 1) * n // n_blocks):
            for p in range(indptr[i], indptr[i + 1]):
                u = bits[p]
                for q in range(p, indptr[i + 1]):
                    out[blk, u, bits[q]] += 1
    return out


@nb.njit(cache=True, parallel=True)
def _project(indptr, bits, components, shift):
    """Scores ``X @ components - shift`` of a CSR binary matrix."""
    n, k = indptr.shape[0] - 1, components.shape[1]
    out = np.empty((n, k))
    for i in nb.prange(n):
        for c in range(k):
            s = -shift[c]
            for p in range(indptr[i], indptr[i + 1]):
                s += components[bits[p], c]
            out[i, c] = s
    return out


def _pca_csr(indptr, bits, d, n_components=2):
    """PCA scores of a binary matrix given as CSR on-bit indices (sorted within rows)."""
    n = len(indptr) - 1
    # Each block holds a d x d count matrix: cap their total size at ~512 MB.
    n_blocks = int(max(1, min(nb.get_num_threads(), 8, 512e6 // (4 * d * d), n // 1000)))
    gram = _gram(indptr, bits, d, n_blocks).sum(axis=0, dtype=np.float64)
    gram = gram + gram.T - np.diag(np.diag(gram))
    mean = np.diag(gram) / n
    cov = gram / n - np.outer(mean, mean)
    _, vectors = np.linalg.eigh(cov)  # ascending eigenvalues
    components = vectors[:, ::-1][:, :n_components].copy()
    # Fix the sign of each axis (its largest loading is positive) for reproducible plots.
    signs = np.sign(components[np.abs(components).argmax(axis=0), np.arange(n_components)])
    components *= np.where(signs == 0, 1, signs)
    return _project(indptr, bits, components, mean @ components)


def _dense_to_csr(fingerprints):
    rows, cols = np.nonzero(fingerprints)
    indptr = np.zeros(len(fingerprints) + 1, dtype=np.int64)
    np.cumsum(np.bincount(rows, minlength=len(fingerprints)), out=indptr[1:])
    return indptr, cols.astype(np.int32)


def pca_layout(fingerprints: np.ndarray) -> np.ndarray:
    """2D PCA of a ``(n, n_bits)`` binary fingerprint matrix.

    The covariance is accumulated from the on-bits only, so it is exact and fast for
    sparse fingerprints. Returns an ``(n, 2)`` float32 array of scores on the first two
    principal axes.
    """
    fingerprints = np.asarray(fingerprints)
    if fingerprints.ndim != 2 or not _is_binary(fingerprints):
        raise ValueError("pca_layout expects a 2D binary (0/1 or bool) fingerprint matrix")
    indptr, bits = _dense_to_csr(fingerprints)
    return _pca_csr(indptr, bits, fingerprints.shape[1]).astype(np.float32)


def pca(
    data,
    smiles: str | None = None,
    *,
    n_bits: int = 2048,
    radius: int = 2,
    n_jobs: int | None = None,
) -> Map:
    """Principal component analysis map of molecules.

    ``data`` is given as for ``tmap`` (SMILES, DataFrame or file), or as a 2D binary
    fingerprint array. Molecules are encoded as folded ECFP fingerprints (``radius``,
    ``n_bits``) and projected on their first two principal axes: a fast, linear and
    deterministic overview, which separates broad chemotypes but not fine clusters.
    """
    frame, column, values, array = _inputs(data, smiles)
    if array is not None:
        if not _is_binary(array):
            raise ValueError("pca needs SMILES or binary (0/1) fingerprints; MinHash vectors have no linear structure")
        indptr, bits = _dense_to_csr(array)
        n_bits, invalid = array.shape[1], []
    else:
        indptr, bits, invalid = _folded_ecfp(values, radius, n_bits, n_jobs=n_jobs)
    keep, frame = _drop_invalid(frame, invalid)
    if keep is not None:  # invalid molecules have empty rows: drop them from the CSR offsets
        lengths = np.diff(indptr)[keep]
        indptr = np.zeros(len(lengths) + 1, dtype=np.int64)
        np.cumsum(lengths, out=indptr[1:])
    pos = _pca_csr(indptr, bits, n_bits).astype(np.float32)
    edges = np.empty((0, 2), dtype=np.int64)
    return Map(pos[:, 0], pos[:, 1], edges, frame, np.asarray(invalid, dtype=np.int64), column, "pca")
