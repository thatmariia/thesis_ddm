import os
import sys
from pathlib import Path
import tomllib

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src"))
)

pyproject = Path(__file__).parent.parent.parent / "pyproject.toml"
with open(pyproject, "rb") as f:
    data = tomllib.load(f)

# extract metadata
project_meta = data.get("project", {})

project = project_meta.get("name", "Unknown Project")
author = ", ".join(
    a.get("name") for a in project_meta.get("authors", []) if a.get("name")
)
version = project_meta.get("version", "0.0.0")
copyright = "2026, " + author

# -- General configuration ---------------------------------------------------

extensions = [
    "myst_parser",  # Markdown parser
]

templates_path = ['_templates']
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------

html_theme = "sphinx_book_theme"
html_static_path = ["_static"]

html_theme_options = {
    "repository_url": "https://github.com/brunykrijgsman/thesis_ddm",
    "use_repository_button": True,
    "use_issues_button": True,
    "path_to_docs": "docs",
    "navigation_depth": 10,
}
