# Command line

The `chemviz` command maps a molecule file without writing Python. It is installed with the
package.

```bash
chemviz tmap molecules.csv                         # writes molecules.html
chemviz umap molecules.csv -o map.html -o map.png --color pKi
chemviz pca library.sdf -o coords.csv              # data + x, y coordinates
```

## Subcommands

| Subcommand | Does |
|---|---|
| `chemviz tmap FILE` | Tree map (TMAP) |
| `chemviz umap FILE` | UMAP map |
| `chemviz tsne FILE` | t-SNE map |
| `chemviz pca FILE` | PCA map |
| `chemviz project MAP FILE` | Places the molecules of `FILE` on a map saved as `.chemviz` |

`FILE` is a `.csv`, `.tsv`, `.smi` or `.sdf` file (see {doc}`inputs`). `chemviz --version`
prints the version, and `chemviz SUBCOMMAND --help` lists every option.

## Outputs

`-o FILE` writes an output, in a format chosen by its extension; repeat it for several
outputs. Without `-o`, the viewer is written next to the input, as `INPUT.html`.

| Extension | Output |
|---|---|
| `.html` | The interactive viewer |
| `.png`, `.svg`, `.pdf` | A static plot; `--color COL` colours it by a column |
| `.csv` | The data with `x` and `y` columns |
| `.chemviz` | The map, to reload with `cv.load` or to place molecules on with `chemviz project` |

## Options

Common to every subcommand:

| Option | Meaning |
|---|---|
| `--smiles COL` | SMILES column (guessed if omitted) |
| `--columns A,B` | Columns to embed in the viewer (all by default) |
| `--title TEXT` | Viewer title (the output file name by default) |
| `--offline` | Embed RDKit.js so the viewer needs no network |
| `--color COL` | Column to colour static plots by |
| `--n-jobs N` | Processes for SMILES parsing (all cores by default) |

The method subcommands also take the parameters of their Python function, with dashes:
`--radius` for all, `--n-bits` for `pca`; `-k`, `--kc`, `--n-permutations`, `--n-trees` and
`--seed` for `tmap`, `umap` and `tsne`; `--min-dist` and `--n-epochs` for `umap`;
`--perplexity` and `--n-iter` for `tsne`. See the {doc}`api` for their meaning.

## Placing molecules on a saved map

```bash
chemviz tmap library.csv -o library.chemviz
chemviz project library.chemviz hits.csv -o hits.html
chemviz project library.chemviz hits.csv -o hits.png --color projected
```

The output holds the map's molecules and the new ones, with a `projected` column marking the
new ones (see {doc}`projection`).

## Messages and exit codes

Progress goes to the standard error: the number of molecules mapped and the time taken, each
file written, and warnings such as skipped invalid SMILES. The exit code is 0 on success, 1
if the data cannot be mapped (no valid molecule, unknown column) and 2 for invalid arguments.
