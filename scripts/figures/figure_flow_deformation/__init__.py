"""figure_flow_deformation: Velocity and deformation tensor maps.

Panels:
- velocity_profile: Average tissue flow velocity profile (matplotlib)
- velocity_maps: Velocity tensor fields — scene + YAML exports (napari)
- deformation_maps: Deformation tensor fields — scene + YAML exports (napari)
"""

from .preprocessing import load_flow_deformation_data
from .panels_velocity_profile import velocity_profile
from .panels_velocity_maps import build_velocity_scene
from .panels_deformation_maps import build_deformation_scene

FIGURE_NAME = "figure_flow_deformation"

PANELS = {
    "velocity_profile": velocity_profile,
}

SCENES = {
    "build_velocity_scene": build_velocity_scene,
    "build_deformation_scene": build_deformation_scene,
}


def preprocess(paths, fig_config):
    """Load all data needed by figure_flow_deformation panels.

    Parameters
    ----------
    paths : dict
        Resolved paths from config (uppercase keys).
    fig_config : dict
        The figure_flow_deformation config section from figure_config.yaml.

    Returns
    -------
    dict
        Shared data for all figure_flow_deformation panels.
    """
    return load_flow_deformation_data(paths, fig_config)
