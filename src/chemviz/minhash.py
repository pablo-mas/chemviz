"""MinHash encodings of molecules (MHFP) and of precomputed binary fingerprints."""

from collections.abc import Sequence

import numba as nb
import numpy as np
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


def minhash_ecfp(
    smiles: Sequence[str],
    n_permutations: int = 128,
    radius: int = 2,
    seed: int = 42,
    chunk_size: int = 50_000,
) -> np.ndarray:
    """MinHash of the unfolded ECFP (Morgan) substructure identifiers of each molecule.

    Like MHFP, the MinHash distance estimates the Jaccard distance between sets of
    circular substructures, but feature extraction is ~50x faster than MHFP shingling.
    Molecules are processed in chunks so they are never all held in memory.

    Returns a ``(n_molecules, n_permutations)`` uint32 array.
    Raises ``ValueError`` listing the indices of unparsable SMILES.
    """
    rng = np.random.default_rng(seed)
    # a, b < 2**32 keep a * x + b within uint64 for 32-bit feature ids.
    a = rng.integers(1, 1 << 32, n_permutations, dtype=np.uint64)
    b = rng.integers(0, 1 << 32, n_permutations, dtype=np.uint64)
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius)
    out = np.empty((len(smiles), n_permutations), dtype=np.uint32)
    invalid = []
    for start in range(0, len(smiles), chunk_size):
        feats = []
        for i, s in enumerate(smiles[start:start + chunk_size], start):
            mol = Chem.MolFromSmiles(s)
            if mol is None:
                invalid.append(i)
                feats.append(())
            else:
                feats.append(generator.GetSparseFingerprint(mol).GetOnBits())
        indptr = np.zeros(len(feats) + 1, dtype=np.int64)
        np.cumsum([len(f) for f in feats], out=indptr[1:])
        # RDKit reports signed 32-bit ids; reinterpret them as unsigned.
        features = np.fromiter((x for f in feats for x in f), dtype=np.int64, count=indptr[-1])
        features = (features & 0xFFFFFFFF).astype(np.uint64)
        out[start:start + len(feats)] = _minhash_csr(indptr, features, a, b)
    if invalid:
        raise ValueError(f"Invalid SMILES at indices {invalid[:20]}{'...' if len(invalid) > 20 else ''}")
    return out


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
