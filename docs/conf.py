"""Sphinx configuration for the chemviz documentation."""

import subprocess
import sys
from importlib.metadata import version as _version
from pathlib import Path

project = "chemviz"
author = "Pablo Mas"
copyright = "2026, Pablo Mas"
release = _version("chemviz")
version = ".".join(release.split(".")[:2])

extensions = [
    "myst_nb",  # Markdown pages (MyST) and notebooks executed at build
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",  # NumPy-style docstrings
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
]

default_role = "py:obj"  # `name` in docstrings links to the API reference
nitpicky = True  # every reference must resolve (the build treats warnings as errors)
exclude_patterns = ["_build", "_gallery_data", "jupyter_execute", "Thumbs.db", ".DS_Store"]
myst_enable_extensions = ["colon_fence", "deflist"]
myst_heading_anchors = 3

# Executable pages (option B): run at every build, so their examples always work.
nb_execution_mode = "cache"
nb_execution_timeout = 600
nb_execution_raise_on_error = True
nb_execution_in_temp = True  # files the examples write do not land in docs/

autodoc_typehints = "description"
autodoc_member_order = "bysource"
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_preprocess_types = True  # "int, default 10" and "optional" are not class names
napoleon_type_aliases = {"array-like": ":term:`array-like <numpy:array_like>`", "sequence": ":term:`sequence`"}

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "pandas": ("https://pandas.pydata.org/docs", None),
    "matplotlib": ("https://matplotlib.org/stable", None),
    "IPython": ("https://ipython.readthedocs.io/en/stable", None),
}

linkcheck_ignore = [r"https://www\.npmjs\.com/"]  # answers 403 to non-browsers

html_theme = "pydata_sphinx_theme"
html_title = "chemviz"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_favicon = "_static/favicon.svg"
html_extra_path = ["_gallery"]  # gallery maps (made by make_gallery.py), served as maps/...
html_theme_options = {
    "logo": {"image_light": "_static/wordmark.svg", "image_dark": "_static/wordmark-dark.svg", "alt_text": "chemviz"},
    "github_url": "https://github.com/pablo-mas/chemviz",
    "icon_links": [{"name": "PyPI", "url": "https://pypi.org/project/chemviz/", "icon": "fa-brands fa-python"}],
    "navbar_align": "left",
    "header_links_before_dropdown": 7,
    "show_toc_level": 2,
    "use_edit_page_button": True,
}
# Pages without sub-pages have nothing to show in the left sidebar: give their content the room.
html_sidebars = {page: [] for page in ("index", "gallery", "how-it-works", "benchmarks", "api")}
html_context = {
    "github_user": "pablo-mas",
    "github_repo": "chemviz",
    "github_version": "main",
    "doc_path": "docs",
}


def _make_gallery(app, config):
    """Generate the gallery's interactive maps (cached until the script or chemviz changes)."""
    subprocess.run([sys.executable, str(Path(__file__).parent / "make_gallery.py")], check=True)


def setup(app):
    # Before Sphinx checks that html_extra_path exists (priority 800), so a fresh build works.
    app.connect("config-inited", _make_gallery, priority=100)
