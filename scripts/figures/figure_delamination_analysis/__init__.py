"""Figure delamination analysis: Lagrangian apoptosis quantification (vi vs Tolloi).

Panels (matplotlib):
- cumulative_apoptosis: Cumulative normalized apoptosis (grid mean)
- apoptosis_2h_rate: 2h average normalized apoptosis rate (grid mean)

Scenes (napari): none
"""

from .panels import apoptosis_2h_rate, cumulative_apoptosis
from .preprocessing import load_delamination_data

FIGURE_NAME = "figure_delamination_analysis"

PANELS = {
    "cumulative_apoptosis": cumulative_apoptosis,
    "apoptosis_2h_rate": apoptosis_2h_rate,
}

SCENES = {}


def preprocess(paths, fig_config):
    """Load all data needed by delamination analysis panels.

    Parameters
    ----------
    paths : dict
        Resolved paths from config (uppercase keys).
    fig_config : dict
        The figure_delamination_analysis config section.

    Returns
    -------
    dict
        Shared data for all panels.
    """
    return load_delamination_data(paths, fig_config)
