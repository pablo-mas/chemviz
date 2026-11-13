"""Extract the RDKit-parsable SMILES of a ChEMBL chemreps dump into a .smi file.

Usage: uv run python benchmarks/prepare_chembl.py data/chembl/chembl_37_chemreps.txt.gz data/chembl/chembl_37_valid.smi
"""

import gzip
import sys
from multiprocessing import Pool

from rdkit import Chem, RDLogger


def is_valid(smiles):
    RDLogger.DisableLog("rdApp.*")
    return Chem.MolFromSmiles(smiles) is not None


def main():
    src, dst = sys.argv[1], sys.argv[2]
    with gzip.open(src, "rt") as f:
        next(f)  # header
        smiles = [line.split("\t", 2)[1] for line in f]
    with Pool() as pool:
        valid = pool.map(is_valid, smiles, chunksize=2000)
    kept = [s for s, ok in zip(smiles, valid) if ok]
    with open(dst, "w") as f:
        f.write("\n".join(kept) + "\n")
    print(f"{len(kept)} / {len(smiles)} SMILES kept")


if __name__ == "__main__":
    main()
