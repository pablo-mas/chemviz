"""Command-line interface: ``chemviz tmap molecules.csv -o map.html``."""

import argparse
import sys
import time
import warnings
from importlib.metadata import version
from pathlib import Path

IMAGE_SUFFIXES = (".png", ".svg", ".pdf")
SUFFIXES = (".html", ".csv", *IMAGE_SUFFIXES)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="chemviz", description="Fast chemical space visualization.")
    parser.add_argument("--version", action="version", version=f"%(prog)s {version('chemviz')}")
    commands = parser.add_subparsers(dest="command", required=True, metavar="command")

    p = commands.add_parser(
        "tmap",
        help="tree map (TMAP) of a molecule file",
        description="Compute a tree map (TMAP) of the molecules in a .csv, .tsv, .smi or .sdf file.",
        epilog="examples:\n"
        "  chemviz tmap molecules.csv\n"
        "  chemviz tmap molecules.csv -o map.html -o map.png --color pKi\n"
        "  chemviz tmap library.sdf -o map.html --offline --columns name,MW",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.set_defaults(parser=p)
    p.add_argument("input", type=Path, help="molecule file (.csv, .tsv, .smi or .sdf)")
    p.add_argument(
        "-o", "--output", type=Path, action="append", metavar="FILE",
        help="output file, by extension: .html (interactive viewer), .png/.svg/.pdf (static plot) "
        "or .csv (data with x, y coordinates); repeat for several outputs (default: INPUT.html)",
    )
    p.add_argument("--smiles", metavar="COL", help="SMILES column (guessed if omitted)")
    p.add_argument("--columns", metavar="COLS", help="comma-separated columns to embed in the viewer (default: all)")
    p.add_argument("--title", metavar="TEXT", help="viewer title (default: output file name)")
    p.add_argument("--offline", action="store_true", help="embed RDKit.js so the viewer needs no network")
    p.add_argument("--color", metavar="COL", help="column to colour the static plot by")
    p.add_argument("-k", type=int, metavar="N", default=10, help="nearest neighbours per molecule (default: 10)")
    p.add_argument("--kc", type=int, metavar="N", default=10, help="candidate pool factor (default: 10)")
    p.add_argument("--n-permutations", type=int, metavar="N", default=128, help="MinHash size (default: 128)")
    p.add_argument("--radius", type=int, metavar="N", default=2, help="ECFP radius (default: 2)")
    p.add_argument("--n-trees", type=int, metavar="N", default=8, help="LSH Forest trees (default: 8)")
    p.add_argument("--seed", type=int, metavar="N", default=0, help="layout seed (default: 0)")
    p.add_argument("--n-jobs", type=int, metavar="N", help="processes for SMILES parsing (default: all cores)")
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

    from chemviz import tmap  # deferred: importing numba and RDKit takes a moment

    try:
        start = time.perf_counter()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            m = tmap(args.input, args.smiles, k=args.k, kc=args.kc, n_permutations=args.n_permutations,
                     radius=args.radius, n_trees=args.n_trees, seed=args.seed, n_jobs=args.n_jobs)
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
