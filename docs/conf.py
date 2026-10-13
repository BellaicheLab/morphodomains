"""Sphinx configuration for morphodomains documentation."""

# -- Project information -----------------------------------------------------
project = "morphodomains"
author = "Adrien Leroy, Lorette Noiret"
copyright = "2026, Bellaiche Lab — Institut Curie"
release = "0.9.0"
version = release

# -- Extensions --------------------------------------------------------------
extensions = [
    "sphinx.ext.intersphinx",   # Cross-link to numpy/pandas/scipy docs
    "sphinx.ext.mathjax",       # Render LaTeX math in HTML
    "myst_parser",              # Write pages in Markdown (not just RST)
    "nbsphinx",                 # Render Jupyter notebooks as doc pages
    "sphinx_copybutton",        # Copy button on code blocks
    "sphinx_design",            # Cards, tabs, badges, grid layouts
]

source_suffix = {".rst": "restructuredtext", ".md": "markdown"}
exclude_patterns = [
    "_build", "Thumbs.db", ".DS_Store", "**.ipynb_checkpoints",
]
templates_path = ["_templates"]

# -- Intersphinx (ecosystem cross-linking) -----------------------------------
intersphinx_mapping = {
    "python":     ("https://docs.python.org/3", None),
    "numpy":      ("https://numpy.org/doc/stable/", None),
    "pandas":     ("https://pandas.pydata.org/docs/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "anndata":    ("https://anndata.readthedocs.io/en/latest/", None),
    "scanpy":     ("https://scanpy.readthedocs.io/en/stable/", None),
    "napari":     ("https://napari.org/stable/", None),
    # Cross-repo links (uncomment once each site is deployed on RTD):
    # "vgep":               ("https://vgep.readthedocs.io/en/latest/", None),
    # "pyanimalprocessing": ("https://pyanimalprocessing.readthedocs.io/en/latest/", None),
}

# -- HTML output -------------------------------------------------------------
html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_css_files = ["custom.css"]

# Navbar brand: show just the project name; version is rendered as a small
# secondary line below via CSS (.logo__title::after in custom.css).
html_title = project

html_theme_options = {
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/BellaicheLab/morphodomains",
            "icon": "fa-brands fa-github",
        },
    ],
    "navbar_align": "left",
    "show_prev_next": True,
    "navigation_depth": 3,
    "show_nav_level": 2,
    "secondary_sidebar_items": ["page-toc"],
    "footer_start": ["copyright"],
    "footer_end": [],
}

html_context = {
    "github_user": "BellaicheLab",
    "github_repo": "morphodomains",
    "github_version": "main",
    "doc_path": "docs",
}

# -- Notebooks ---------------------------------------------------------------
nbsphinx_execute = "never"
nbsphinx_allow_errors = False

# nbsphinx resolves in-notebook anchor links (e.g. "#1-setup" in a TOC) as Sphinx
# cross-references and warns; they are harmless in the rendered HTML.
suppress_warnings = ["ref.ref", "nbsphinx.localfile"]


def _sync_tutorial_notebooks(app, config):
    """Mirror ../notebooks/*.ipynb into docs/notebooks/ so Sphinx can render them.

    The source of truth lives at the repo root. Copying keeps nbsphinx happy
    without duplicating files in git (docs/notebooks/ is .gitignored).
    """
    import shutil
    from pathlib import Path

    src = Path(app.srcdir).parent / "notebooks"
    dst = Path(app.srcdir) / "notebooks"
    if not src.exists():
        return
    dst.mkdir(exist_ok=True)
    for nb in src.glob("*.ipynb"):
        shutil.copy2(nb, dst / nb.name)


def setup(app):
    app.connect("config-inited", _sync_tutorial_notebooks)

# -- MyST Markdown extensions ------------------------------------------------
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "dollarmath",
    "fieldlist",
    "tasklist",
]
myst_heading_anchors = 3