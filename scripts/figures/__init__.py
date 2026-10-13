"""Figure generation framework.

Provides a registry of figure modules for the dispatcher (generate.py),
plus shared utilities used by all figure and panel modules.

Public API
----------
is_panel_enabled() – check whether a panel is enabled (priority logic)
create_viewer()    – create a napari Viewer with config defaults
save_and_close()   – save a matplotlib figure to disk
apply_style()      – apply rcParams from the YAML style section
figsize()          – look up a named figure-size preset (inches)
get_figure_module  – load a figure module by name
list_registered_figures – list all registered figure names
"""

import importlib
import os

import matplotlib.pyplot as plt


# ── Shared helpers ───────────────────────────────────────────────────────────


def is_panel_enabled(fig_cfg, panel_name):
    """Whether *panel_name* is enabled in ``fig_cfg["enabled_panels"]`` (default True)."""
    return fig_cfg.get("enabled_panels", {}).get(panel_name, True)


def create_viewer(config=None):
    """Create a napari Viewer with defaults from config.

    Parameters
    ----------
    config : dict or None
        Full config dict.  The ``napari`` section is used for
        ``window_size`` and ``theme``.

    Returns
    -------
    napari.Viewer
    """
    import napari

    napari_cfg = (config or {}).get("napari", {})
    viewer = napari.Viewer()
    w, h = napari_cfg.get("window_size", [900, 554])
    viewer.window.resize(w, h)
    viewer.theme = napari_cfg.get("theme", "light")
    return viewer


def save_and_close(fig, output_path, *, formats=("svg",), dpi=1200, close=True):
    """Save a matplotlib figure and optionally close it.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure to save.
    output_path : str or Path
        Output path without extension (extensions added per format).
    formats : tuple of str
        File formats to save (e.g. ("svg",) or ("svg", "png")).
    dpi : int
        Resolution for raster formats.
    close : bool
        Whether to close the figure after saving.
    """
    output_path = str(output_path)
    # Ensure parent directory exists
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    for fmt in formats:
        # If output_path already has the extension, use as-is
        if output_path.endswith(f".{fmt}"):
            save_path = output_path
        else:
            save_path = f"{output_path}.{fmt}"
        fig.savefig(save_path, format=fmt, dpi=dpi, bbox_inches="tight")

    if close:
        plt.close(fig)


# Named figure-size presets (inches), populated by apply_style() from the
# YAML style section; read by figsize().
_FIGSIZE_PRESETS = {}


def apply_style(style_config):
    """Apply matplotlib rcParams from the style config section.

    Parameters
    ----------
    style_config : dict
        Keys are rcParam names, values are the settings. The two
        non-rcParam keys ``unit_mm`` (documents how the presets were
        derived) and ``figsize_presets`` (stored for figsize()) are
        consumed here rather than pushed to rcParams.
    """
    global _FIGSIZE_PRESETS
    _FIGSIZE_PRESETS = style_config.get("figsize_presets", {})
    for key, value in style_config.items():
        if key in ("unit_mm", "figsize_presets"):
            continue
        try:
            plt.rcParams[key] = value
        except (KeyError, ValueError):
            pass  # Skip unknown rcParams


def figsize(preset):
    """Return (width, height) in inches for a named preset from the YAML style.

    Presets (e.g. ``"1x3"`` -> ``[7.09, 2.36]``) are defined under
    ``style.figsize_presets`` and loaded by apply_style(). New figures should
    size panels with ``figsize("...")`` rather than hardcoded inches.
    """
    try:
        return tuple(_FIGSIZE_PRESETS[preset])
    except KeyError:
        raise KeyError(
            f"Unknown figsize preset {preset!r}. "
            f"Available: {sorted(_FIGSIZE_PRESETS)}"
        )


# ── Figure registry ──────────────────────────────────────────────────────────

# Lazy figure registry: maps figure name -> module path
_FIGURE_MODULE_PATHS = {
    "figure_flow_deformation": "scripts.figures.figure_flow_deformation",
    "figure_morphodynamic_domains": "scripts.figures.figure_morphodynamic_domains",
    "figure_spatial_transcriptome": "scripts.figures.figure_spatial_transcriptome",
    "figure_grn": "scripts.figures.figure_grn",
    "figure_morphodynamics_comparison": "scripts.figures.figure_morphodynamics_comparison",
    "figure_apoptosis_compression": "scripts.figures.figure_apoptosis_compression",
    "figure_delamination_analysis": "scripts.figures.figure_delamination_analysis",
    "figure_tollo_gfp": "scripts.figures.figure_tollo_gfp",
    "figure_clustering_stability": "scripts.figures.figure_clustering_stability",
    "figure_clustering_temporal": "scripts.figures.figure_clustering_temporal",
    "figure_clustering_spatiotemporal": "scripts.figures.figure_clustering_spatiotemporal",
    "figure_atlas_size": "scripts.figures.figure_atlas_size",
    "figure_gene_atlas": "scripts.figures.figure_gene_atlas",
    "figure_neck_archetype_spread": "scripts.figures.figure_neck_archetype_spread",
}


def _load_module(module_path):
    """Import a module by its dotted path."""
    return importlib.import_module(module_path)


def get_figure_module(figure_name):
    """Load and return a figure module by name.

    Parameters
    ----------
    figure_name : str
        e.g. "figure_grn"

    Returns
    -------
    module
        The figure module with PANELS dict and preprocess().

    Raises
    ------
    KeyError
        If the figure is not registered.
    """
    if figure_name not in _FIGURE_MODULE_PATHS:
        raise KeyError(
            f"Figure '{figure_name}' not registered. "
            f"Available: {list(_FIGURE_MODULE_PATHS)}"
        )
    return _load_module(_FIGURE_MODULE_PATHS[figure_name])


def list_registered_figures():
    """Return list of registered figure names."""
    return list(_FIGURE_MODULE_PATHS)
