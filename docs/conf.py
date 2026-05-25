"""Sphinx configuration for neural-cca.

All project metadata (name, version, author, URLs) is read from
``pyproject.toml`` so there is a single source of truth.
"""

import sys
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib  # Python < 3.11 backport

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root))

# ---- Read metadata from pyproject.toml -------------------------------------
with open(_root / "pyproject.toml", "rb") as _f:
    _pyproject = tomllib.load(_f)["project"]

project = "Neural CCA"
release = _pyproject["version"]
version = release
author = _pyproject["authors"][0]["name"]
copyright = f"2026, {author}"

_repo_url = _pyproject.get("urls", {}).get("Repository", "")

# ---- Extensions -------------------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_design",
    "numpydoc",
    "myst_parser",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "**.ipynb_checkpoints"]
autosummary_generate = True

# Don't try to document inherited members (avoids autosummary stub spam for
# TypedDict subclasses such as ``OsMetricsResult``, which would otherwise
# pull in every inherited ``dict`` method — clear, copy, get, items, …).
numpydoc_show_inherited_class_members = False
numpydoc_class_members_toctree = False
autodoc_default_options = {
    "inherited-members": False,
}

# ---- HTML theme --------------------------------------------------------------
html_theme = "pydata_sphinx_theme"
html_theme_options = {
    "github_url": _repo_url,
    "navbar_align": "left",
    "show_toc_level": 2,
    "logo": {
        "image_light": "_static/logo_neural_cca.svg",
        "image_dark": "_static/logo_neural_cca.svg",
        "text": "Neural CCA",
    },
}
html_static_path = ["_static"]
html_favicon = "_static/favicons/favicon.ico"

# ---- Intersphinx -------------------------------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
}
