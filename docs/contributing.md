# Contributing

Bug reports, questions and suggestions are welcome as
[GitHub issues](https://github.com/pablo-mas/chemviz/issues). For a bug, include the chemviz
version (`chemviz --version`), your operating system and, if possible, a few SMILES that
reproduce it.

## Development setup

chemviz uses [uv](https://docs.astral.sh/uv/). Clone the repository, then:

```bash
uv sync               # create the environment, with the test dependencies
uv run pytest         # run the tests
```

The tests take a few seconds once numba has compiled chemviz's functions; the first run is
longer.

## Documentation

The documentation is built with Sphinx from `docs/`:

```bash
uv run --group docs sphinx-build -W -b html docs docs/_build/html
```

The first build downloads ZINC250k (23 MB), runs the executed pages and makes the gallery
maps, which takes a few minutes; later builds reuse the cached results. Warnings are errors,
and every cross-reference must resolve.

## Pull requests

Keep each pull request to one change, with tests for new behaviour, and update the
documentation and `CHANGELOG.md` when the change is visible to users. The continuous
integration runs the tests on Linux, macOS and Windows with Python 3.12 to 3.15, and builds
the documentation.
