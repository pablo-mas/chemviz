# Installation

chemviz requires Python 3.12 or later. Install it from PyPI with pip or uv:

```bash
pip install chemviz
```

```bash
uv add chemviz
```

Every dependency (RDKit, numba, NumPy, pandas, matplotlib, joblib) installs from pre-built
wheels on Linux, macOS and Windows: there is nothing to compile, and no conda environment is
needed. UMAP and t-SNE are part of chemviz itself, so neither umap-learn nor openTSNE is
required.

## Jupyter widget

To show maps in Jupyter with their selection synced to Python (`Map.widget`), install the
`jupyter` extra, which adds [anywidget](https://anywidget.dev):

```bash
pip install "chemviz[jupyter]"
```

`Map.show` works in notebooks without this extra.

## Check the installation

```bash
chemviz --version
```

or, in Python:

```python
import chemviz as cv

m = cv.tmap(["CCO", "CCCO", "CCCCO", "c1ccccc1O", "c1ccccc1N"])
print(m.to_dataframe())
```

The first call takes a few seconds longer than the next ones: numba compiles chemviz's
functions once, then keeps them in a cache.

## Development version

To work on chemviz itself, clone the repository and let uv create the environment:

```bash
git clone https://github.com/pablo-mas/chemviz
cd chemviz
uv sync
uv run pytest
```
