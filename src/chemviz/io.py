"""Reading molecule tables from files and locating their SMILES column."""

from pathlib import Path

import pandas as pd
from rdkit import Chem


def find_smiles_column(df: pd.DataFrame) -> str:
    """Return the single column named like "smiles" (case-insensitive)."""
    candidates = [c for c in df.columns if isinstance(c, str) and "smiles" in c.lower()]
    exact = [c for c in candidates if c.lower() == "smiles"]
    if len(exact) == 1:
        return exact[0]
    if len(candidates) == 1:
        return candidates[0]
    raise ValueError(
        f"Cannot guess the SMILES column among {list(df.columns)}; pass it explicitly with smiles=..."
    )


def read_molecules(path: str | Path, smiles: str | None = None) -> tuple[pd.DataFrame, str]:
    """Read a molecule file into a DataFrame.

    Parameters
    ----------
    path : str or pathlib.Path
        A ``.csv``, ``.tsv``, ``.smi`` or ``.sdf`` file. SDF properties become columns.
    smiles : str, optional
        SMILES column of a ``.csv`` or ``.tsv`` file (guessed if omitted).

    Returns
    -------
    data : pandas.DataFrame
        One row per molecule. SDF records RDKit cannot read are kept with an empty SMILES,
        so that they are reported as invalid like unparsable SMILES.
    smiles : str
        Name of the SMILES column of ``data``.
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in (".csv", ".tsv"):
        df = pd.read_csv(path, sep="\t" if suffix == ".tsv" else ",")
        return df, smiles or find_smiles_column(df)
    if suffix == ".smi":
        rows = []
        for line in path.read_text(encoding="utf-8").splitlines():
            parts = line.split(maxsplit=1)
            if parts:
                rows.append({"smiles": parts[0], "name": parts[1] if len(parts) > 1 else None})
        return pd.DataFrame(rows, columns=["smiles", "name"]), "smiles"
    if suffix == ".sdf":
        rows = []
        for mol in Chem.SDMolSupplier(str(path)):
            row = {"smiles": Chem.MolToSmiles(mol) if mol is not None else ""}
            if mol is not None:
                row.update(mol.GetPropsAsDict())
                if mol.HasProp("_Name"):
                    row["name"] = mol.GetProp("_Name")
            rows.append(row)
        return pd.DataFrame(rows), "smiles"
    raise ValueError(f"Unsupported file type {suffix!r}; expected .csv, .tsv, .smi or .sdf")
