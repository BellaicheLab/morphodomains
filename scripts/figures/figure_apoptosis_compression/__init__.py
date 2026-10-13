"""figure_apoptosis_compression: Apoptosis and compression analysis (vi vs Tolloi).

Panels (matplotlib):
- plot_instantaneous: Instantaneous rates for tensor quantities in ROI
- plot_cumulative: Cumulative rates for tensor quantities in ROI
- correlation_mean: Apoptosis-compression correlation (per-condition, ROI)

Scenes (napari): none
"""

from .preprocessing import load_apoptosis_compression_data
from .panels_timeseries import plot_instantaneous, plot_cumulative
from .panels_correlation import correlation_mean

FIGURE_NAME = "figure_apoptosis_compression"

PANELS = {
    "plot_instantaneous": plot_instantaneous,
    "plot_cumulative": plot_cumulative,
    "correlation_mean": correlation_mean,
}

SCENES = {}


def preprocess(paths, fig_config):
    """Load all data needed by figure_apoptosis_compression panels.

    Parameters
    ----------
    paths : dict
        Resolved paths from config (uppercase keys).
    fig_config : dict
        The figure_apoptosis_compression config section from figure_config.yaml.

    Returns
    -------
    dict
        Shared data for all figure_apoptosis_compression panels.
    """
    return load_apoptosis_compression_data(paths, fig_config)
