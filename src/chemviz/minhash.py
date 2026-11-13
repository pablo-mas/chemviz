"""MinHash encodings of molecules (MHFP) and of precomputed binary fingerprints."""

from collections.abc import Sequence

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMHFPFingerprint


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
