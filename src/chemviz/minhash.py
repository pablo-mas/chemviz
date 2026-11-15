"""Molecule encodings: MinHash of ECFP substructures (MHFP-like), of precomputed binary
fingerprints, and folded ECFP fingerprints."""

from collections.abc import Sequence

import numba as nb
import numpy as np
from joblib import Parallel, delayed
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator, rdMHFPFingerprint

_MERSENNE_PRIME = np.uint64((1 << 61) - 1)
_MAX_HASH = np.uint64(0xFFFFFFFF)


@nb.njit(cache=True, parallel=True)
def _minhash_csr(indptr, features, a, b):
    """MinHash of each set ``features[indptr[i]:indptr[i+1]]`` with hashes ((a*x + b) mod p) & 0xFFFFFFFF."""
    n, d = indptr.shape[0] - 1, a.shape[0]
    out = np.full((n, d), 0xFFFFFFFF, dtype=np.uint32)
    for i in nb.prange(n):
        for p in range(indptr[i], indptr[i + 1]):
            x = features[p]
            for j in range(d):
                h = ((a[j] * x + b[j]) % _MERSENNE_PRIME) & _MAX_HASH
                if h < out[i, j]:
                    out[i, j] = h
    return out


def _ecfp_features(smiles: Sequence[str], radius: int):
    """Morgan feature ids of a chunk of SMILES as CSR arrays, plus local indices of invalid SMILES."""
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius)
    lengths, feats, invalid = [], [], []
    for i, s in enumerate(smiles):
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            invalid.append(i)
            lengths.append(0)
            continue
        bits = generator.GetSparseFingerprint(mol).GetOnBits()
        feats.extend(bits)
        lengths.append(len(bits))
    indptr = np.zeros(len(smiles) + 1, dtype=np.int64)
    np.cumsum(lengths, out=indptr[1:])
    # RDKit reports signed 32-bit ids; reinterpret them as unsigned.
    features = (np.array(feats, dtype=np.int64) & 0xFFFFFFFF).astype(np.uint64)
    return indptr, features, invalid


def _ecfp_bits(smiles: Sequence[str], radius: int, n_bits: int):
    """On-bit indices of the folded ECFP (Morgan) fingerprints of a chunk of SMILES as CSR
    arrays, plus local indices of invalid SMILES."""
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    lengths, bits, invalid = [], [], []
    for i, s in enumerate(smiles):
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            invalid.append(i)
            lengths.append(0)
            continue
        on = generator.GetFingerprint(mol).GetOnBits()
        bits.extend(on)
        lengths.append(len(on))
    indptr = np.zeros(len(smiles) + 1, dtype=np.int64)
    np.cumsum(lengths, out=indptr[1:])
    return indptr, np.array(bits, dtype=np.int32), invalid


def _map_chunks(func, smiles, chunk_size, n_jobs, *args):
    """``func(chunk, *args)`` over consecutive chunks of SMILES, in worker processes.

    Yields ``(start, result)`` in order, so results can be consumed as they come.
    """
    starts = range(0, len(smiles), chunk_size)
    chunks = [smiles[i:i + chunk_size] for i in starts]
    if n_jobs == 1 or len(chunks) <= 1:
        results = (func(c, *args) for c in chunks)
    else:
        # RDKit holds the GIL, so parse in processes. joblib's loky workers neither re-run the
        # caller's __main__ (unlike spawn) nor fork a process already running numba threads.
        results = Parallel(n_jobs=-1 if n_jobs is None else n_jobs, return_as="generator")(
            delayed(func)(c, *args) for c in chunks
        )
    return zip(starts, results)


def _folded_ecfp(smiles, radius, n_bits, chunk_size=10_000, n_jobs=None):
    """Folded ECFP fingerprints of all SMILES as CSR on-bit indices ``(indptr, bits)``,
    plus the indices of invalid SMILES (which get empty rows)."""
    smiles = list(smiles)
    indptrs, bits, invalid = [np.zeros(1, dtype=np.int64)], [], []
    for start, (indptr, chunk_bits, bad) in _map_chunks(_ecfp_bits, smiles, chunk_size, n_jobs, radius, n_bits):
        indptrs.append(indptr[1:] + indptrs[-1][-1])
        bits.append(chunk_bits)
        invalid.extend(start + i for i in bad)
    return np.concatenate(indptrs), np.concatenate(bits) if bits else np.zeros(0, np.int32), invalid


def minhash_ecfp(
    smiles: Sequence[str],
    n_permutations: int = 128,
    radius: int = 2,
    seed: int = 42,
    chunk_size: int = 10_000,
    n_jobs: int | None = None,
) -> np.ndarray:
    """MinHash of the unfolded ECFP (Morgan) substructure identifiers of each molecule.

    Like MHFP, the MinHash distance estimates the Jaccard distance between sets of
    circular substructures, but feature extraction is ~50x faster than MHFP shingling.
    SMILES are parsed in chunks by ``n_jobs`` worker processes (default: all CPUs;
    1 disables multiprocessing), so molecules are never all held in memory.

    Returns a ``(n_molecules, n_permutations)`` uint32 array.
    Raises ``ValueError`` listing the indices of unparsable SMILES.
    """
    out, invalid = _minhash_ecfp(smiles, n_permutations, radius, seed, chunk_size, n_jobs)
    if invalid:
        raise ValueError(f"Invalid SMILES at indices {invalid[:20]}{'...' if len(invalid) > 20 else ''}")
    return out


def _minhash_ecfp(smiles, n_permutations, radius, seed, chunk_size, n_jobs):
    """``minhash_ecfp`` returning ``(hashes, invalid_indices)`` instead of raising."""
    smiles = list(smiles)
    rng = np.random.default_rng(seed)
    # a, b < 2**32 keep a * x + b within uint64 for 32-bit feature ids.
    a = rng.integers(1, 1 << 32, n_permutations, dtype=np.uint64)
    b = rng.integers(0, 1 << 32, n_permutations, dtype=np.uint64)
    out = np.empty((len(smiles), n_permutations), dtype=np.uint32)
    invalid = []
    for start, (indptr, features, bad) in _map_chunks(_ecfp_features, smiles, chunk_size, n_jobs, radius):
        out[start:start + len(indptr) - 1] = _minhash_csr(indptr, features, a, b)
        invalid.extend(start + i for i in bad)
    return out, invalid


def mhfp(
    smiles: Sequence[str],
    n_permutations: int = 128,
    radius: int = 3,
    seed: int = 42,
) -> np.ndarray:
    """Encode SMILES as MHFP MinHash vectors.

    Returns a ``(n_molecules, n_permutations)`` uint32 array.
    Raises ``ValueError`` listing the indices of unparsable SMILES.
    """
    mols = [Chem.MolFromSmiles(s) for s in smiles]
    invalid = [i for i, m in enumerate(mols) if m is None]
    if invalid:
        raise ValueError(f"Invalid SMILES at indices {invalid[:20]}{'...' if len(invalid) > 20 else ''}")
    encoder = rdMHFPFingerprint.MHFPEncoder(n_permutations, seed)
    hashes = encoder.EncodeMolsBulk(mols, radius=radius)
    return np.asarray(hashes, dtype=np.uint32).reshape(len(mols), n_permutations)


def minhash_from_fingerprints(
    fingerprints: np.ndarray,
    n_permutations: int = 128,
    seed: int = 42,
) -> np.ndarray:
    """MinHash a ``(n, n_bits)`` binary fingerprint matrix (set = indices of on-bits).

    Returns a ``(n, n_permutations)`` uint32 array.
    """
    fingerprints = np.asarray(fingerprints, dtype=bool)
    encoder = rdMHFPFingerprint.MHFPEncoder(n_permutations, seed)
    out = np.empty((fingerprints.shape[0], n_permutations), dtype=np.uint32)
    for i, row in enumerate(fingerprints):
        out[i] = encoder.FromArray(np.flatnonzero(row).tolist())
    return out
