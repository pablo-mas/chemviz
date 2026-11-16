"""Command-line interface: ``chemviz tmap molecules.csv -o map.html`` (or umap, tsne, pca)."""

import argparse
import sys
import time
import warnings
from importlib.metadata import version
from pathlib import Path

IMAGE_SUFFIXES = (".png", ".svg", ".pdf")
SUFFIXES = (".html", ".csv", ".chemviz", *IMAGE_SUFFIXES)


METHODS = {
    "tmap": ("tree map (TMAP): a tree of nearest neighbours, laid out in 2D",
             "Compute a tree map (TMAP) of the molecules in a .csv, .tsv, .smi or .sdf file."),
    "umap": ("UMAP: tight, well separated clusters",
             "Compute a UMAP map of the molecules in a .csv, .tsv, .smi or .sdf file."),
    "tsne": ("t-SNE: clusters with fine local detail",
             "Compute a t-SNE map of the molecules in a .csv, .tsv, .smi or .sdf file."),
    "pca": ("PCA: fast linear overview",
            "Compute a principal component analysis map of the molecules in a .csv, .tsv, .smi or .sdf file."),
}


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("input", type=Path, help="molecule file (.csv, .tsv, .smi or .sdf)")
    p.add_argument(
        "-o", "--output", type=Path, action="append", metavar="FILE",
        help="output file, by extension: .html (interactive viewer), .png/.svg/.pdf (static plot) "
        "or .csv (data with x, y coordinates) or .chemviz (the map, to reload in Python); repeat for "
        "several outputs (default: INPUT.html)",
    )
    p.add_argument("--smiles", metavar="COL", help="SMILES column (guessed if omitted)")
    p.add_argument("--columns", metavar="COLS", help="comma-separated columns to embed in the viewer (default: all)")
    p.add_argument("--title", metavar="TEXT", help="viewer title (default: output file name)")
    p.add_argument("--offline", action="store_true", help="embed RDKit.js so the viewer needs no network")
    p.add_argument("--color", metavar="COL", help="column to colour the static plot by")
    p.add_argument("--radius", type=int, metavar="N", default=2, help="ECFP radius (default: 2)")
    p.add_argument("--n-jobs", type=int, metavar="N", help="processes for SMILES parsing (default: all cores)")


def _add_method(p: argparse.ArgumentParser, method: str) -> list[str]:
    """Add the method's own options; returns the names passed on to the method."""
    def opt(flag, default, help, type=int, metavar="N"):
        suffix = "" if "default" in help else f" (default: {default})"
        p.add_argument(flag, type=type, metavar=metavar, default=default, help=help + suffix)

    if method == "pca":
        opt("--n-bits", 2048, "fingerprint size")
        return ["n_bits", "radius", "n_jobs"]
    if method == "tmap":
        opt("-k", 10, "nearest neighbours per molecule")
        opt("--kc", 10, "LSH candidate pool factor")
        names = ["k", "kc"]
    elif method == "umap":
        opt("-k", 15, "nearest neighbours per molecule")
        opt("--min-dist", 0.1, "minimum distance between embedded points", type=float, metavar="X")
        opt("--n-epochs", None, "optimisation epochs (default: 500 up to 10,000 molecules, else 200)")
        opt("--kc", 10, "LSH candidate pool factor")
        names = ["k", "min_dist", "n_epochs", "kc"]
    else:
        opt("--perplexity", 30.0, "perplexity (neighbours: 3 x perplexity)", type=float, metavar="X")
        opt("--n-iter", 500, "iterations after early exaggeration")
        opt("--kc", 3, "LSH candidate pool factor")
        names = ["perplexity", "n_iter", "kc"]
    opt("--n-permutations", 128, "MinHash size")
    opt("--n-trees", 8, "LSH Forest trees")
    opt("--seed", 0, "random seed")
    return [*names, "n_permutations", "n_trees", "seed", "radius", "n_jobs"]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="chemviz", description="Fast chemical space visualization.")
    parser.add_argument("--version", action="version", version=f"%(prog)s {version('chemviz')}")
    commands = parser.add_subparsers(dest="command", required=True, metavar="command")
    for method, (summary, description) in METHODS.items():
        p = commands.add_parser(
            method,
            help=summary,
            description=description,
            epilog="examples:\n"
            f"  chemviz {method} molecules.csv\n"
            f"  chemviz {method} molecules.csv -o map.html -o map.png --color pKi\n"
            f"  chemviz {method} library.sdf -o map.html --offline --columns name,MW",
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )
        _add_common(p)
        p.set_defaults(parser=p, params=_add_method(p, method))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    outputs = args.output or [args.input.with_suffix(".html")]
    for path in outputs:
        if path.suffix.lower() not in SUFFIXES:
            args.parser.error(f"unsupported output {str(path)!r}; expected one of {', '.join(SUFFIXES)}")
    if args.color and not any(p.suffix.lower() in IMAGE_SUFFIXES for p in outputs):
        args.parser.error("--color applies to static plots: add an output such as -o map.png")
    if not args.input.is_file():
        args.parser.error(f"no such file: {str(args.input)!r}")
    columns = [c.strip() for c in args.columns.split(",")] if args.columns else None

    import chemviz  # deferred: importing numba and RDKit takes a moment

    method = getattr(chemviz, args.command)
    try:
        start = time.perf_counter()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            m = method(args.input, args.smiles, **{name: getattr(args, name) for name in args.params})
        for w in caught:
            print(f"chemviz: warning: {w.message}", file=sys.stderr)
        print(f"Mapped {len(m.x)} molecules in {time.perf_counter() - start:.1f} s", file=sys.stderr)
        if args.color and args.color not in m.data.columns:
            raise ValueError(f"Unknown --color column {args.color!r}; available: {list(m.data.columns)}")
        for path in outputs:
            suffix = path.suffix.lower()
            if suffix == ".html":
                m.save(path, columns=columns, title=args.title, offline=args.offline)
            elif suffix == ".csv":
                m.to_dataframe().to_csv(path, index=False)
            elif suffix == ".chemviz":
                m.save(path)
            else:
                import matplotlib

                matplotlib.use("Agg")  # no display needed
                import matplotlib.pyplot as plt

                ax = m.plot(color=args.color)
                ax.figure.savefig(path, dpi=200, bbox_inches="tight")
                plt.close(ax.figure)
            print(f"Wrote {path}", file=sys.stderr)
    except (OSError, ValueError, RuntimeError) as e:
        print(f"chemviz: error: {e}", file=sys.stderr)
        return 1
    return 0
