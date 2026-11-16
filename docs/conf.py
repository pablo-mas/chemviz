"""Sphinx configuration for the chemviz documentation."""

from importlib.metadata import version as _version

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

exclude_patterns = ["_build", "jupyter_execute", "Thumbs.db", ".DS_Store"]
myst_enable_extensions = ["colon_fence", "deflist"]
myst_heading_anchors = 3

# Executable pages (option B): run at every build, so their examples always work.
nb_execution_mode = "cache"
nb_execution_timeout = 600
nb_execution_raise_on_error = True

autodoc_typehints = "description"
autodoc_member_order = "bysource"
napoleon_google_docstring = False
napoleon_numpy_docstring = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "pandas": ("https://pandas.pydata.org/docs", None),
    "matplotlib": ("https://matplotlib.org/stable", None),
}

html_theme = "pydata_sphinx_theme"
html_title = "chemviz"
html_static_path = ["_static"]
html_theme_options = {
    "github_url": "https://github.com/pablo-mas/chemviz",
    "icon_links": [{"name": "PyPI", "url": "https://pypi.org/project/chemviz/", "icon": "fa-brands fa-python"}],
    "navbar_align": "left",
    "show_toc_level": 2,
    "use_edit_page_button": True,
}
html_context = {
    "github_user": "pablo-mas",
    "github_repo": "chemviz",
    "github_version": "main",
    "doc_path": "docs",
}
