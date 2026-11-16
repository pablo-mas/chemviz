"""Generate the interactive maps of the gallery page (run by conf.py at every build).

The maps are written to docs/_gallery/maps/, which Sphinx copies to the site as maps/; the
downloaded dataset and the cache stamp stay in docs/_gallery_data/. The maps are regenerated
only when this script or the chemviz version changes.
"""

import hashlib
import sys
import warnings
from importlib.metadata import version
from pathlib import Path

import pandas as pd

import chemviz as cv

URL = ("https://raw.githubusercontent.com/the-matter-lab/chemical_vae/master/"
       "models/zinc_properties/250k_rndm_zinc_drugs_clean_3.csv")
DATA = Path(__file__).parent / "_gallery_data"
MAPS = Path(__file__).parent / "_gallery" / "maps"
N = 50_000
TITLES = {"tmap": "TMAP", "umap": "UMAP", "tsne": "t-SNE", "pca": "PCA"}


def main():
    stamp = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16] + "-" + version("chemviz")
    names = [*TITLES, "projection"]
    if (DATA / "stamp").exists() and (DATA / "stamp").read_text() == stamp \
            and all((MAPS / f"{name}.html").exists() for name in names):
        return
    MAPS.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(exist_ok=True)
    csv = DATA / "zinc250k.csv"
    if not csv.exists():
        pd.read_csv(URL).to_csv(csv, index=False)
    df = pd.read_csv(csv).sample(N, random_state=0)
    df["ZINC row"] = "ZINC250k-" + df.index.astype(str)  # row in the original file
    df = df.reset_index(drop=True)
    df["smiles"] = df["smiles"].str.strip()
    df["QED class"] = pd.cut(df["qed"], [0, 0.5, 0.7, 1], labels=["low (<0.5)", "medium", "high (>0.7)"])
    df = df[["smiles", "ZINC row", "QED class", "qed", "logP", "SAS"]]
    warnings.simplefilter("ignore")
    for method, title in TITLES.items():
        print(f"gallery: {method}", file=sys.stderr, flush=True)
        getattr(cv, method)(df).save(MAPS / f"{method}.html", title=f"{title} of 50,000 ZINC molecules")
    # Placing the last 5,000 molecules on a tree map of the first 45,000.
    print("gallery: projection", file=sys.stderr, flush=True)
    ref = cv.tmap(df.iloc[:45_000])
    both = ref.project(df.iloc[45_000:])
    columns = ["projected", *df.columns]  # first categorical column: the default colouring
    both.save(MAPS / "projection.html", columns=columns, title="5,000 molecules placed on a TMAP of 45,000")
    (DATA / "stamp").write_text(stamp)


if __name__ == "__main__":
    main()
