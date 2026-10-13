"""figure_morphodynamics_comparison: Spatial statistics and Moran analysis.

Panels (matplotlib):
- moran_statistics: Moran's I time-series for deformation rate and velocity
- autocorrelation_scatterplot: Moran scatterplots at peak indices
- moran_colorbars: Diverging colorbar legends for LISA maps
- save_lisa_maps: Save LISA maps to disk
- contraction_analysis: Contraction rate comparison (vi vs Tolloi)
- velocity_analysis: Velocity vector polar plot
- velocity_amplitude: Velocity amplitude comparison (barplot)
- rearrangement_anisotropy: Rearrangement anisotropy (ER eigenvalue difference) barplot
- cell_shape_trace: Cell shape isotropic change (ES trace) barplot
- cell_shape_anisotropy: Cell shape anisotropic change (ES eigenvalue difference) barplot
- apoptosis_rate: Apoptosis rate (dnA/nCoreRNs) barplot

Scenes (napari):
- build_deformation_vi_scene: Vi control deformation field
- build_deformation_tolloi_scene: Tolloi deformation field
- build_deformation_difference_scene: Deformation difference with LISA overlays
- build_velocity_vi_scene: Vi control velocity field
- build_velocity_tolloi_scene: Tolloi velocity field
- build_velocity_difference_scene: Velocity difference with LISA overlays
"""

from .preprocessing import load_morphodynamics_comparison_data
from .panels_moran import (
    moran_statistics,
    autocorrelation_scatterplot,
    moran_colorbars,
    save_lisa_maps,
)
from .panels_comparison import (
    contraction_analysis,
    velocity_analysis,
    velocity_amplitude,
    rearrangement_anisotropy,
    cell_shape_trace,
    cell_shape_anisotropy,
    apoptosis_rate,
)
from .panels_deformation import (
    build_deformation_vi_scene,
    build_deformation_tolloi_scene,
    build_deformation_difference_scene,
)
from .panels_velocity import (
    build_velocity_vi_scene,
    build_velocity_tolloi_scene,
    build_velocity_difference_scene,
)

FIGURE_NAME = "figure_morphodynamics_comparison"

PANELS = {
    # Moran statistics (matplotlib)
    "moran_statistics": moran_statistics,
    "autocorrelation_scatterplot": autocorrelation_scatterplot,
    "moran_colorbars": moran_colorbars,
    "save_lisa_maps": save_lisa_maps,
    # Comparison plots (matplotlib)
    "contraction_analysis": contraction_analysis,
    "velocity_analysis": velocity_analysis,
    "velocity_amplitude": velocity_amplitude,
    # Lagrangian barplots (velocity ROI)
    "rearrangement_anisotropy": rearrangement_anisotropy,
    "cell_shape_trace": cell_shape_trace,
    "cell_shape_anisotropy": cell_shape_anisotropy,
    "apoptosis_rate": apoptosis_rate,
}

SCENES = {
    # Deformation maps (napari)
    "build_deformation_vi_scene": build_deformation_vi_scene,
    "build_deformation_tolloi_scene": build_deformation_tolloi_scene,
    "build_deformation_difference_scene": build_deformation_difference_scene,
    # Velocity maps (napari)
    "build_velocity_vi_scene": build_velocity_vi_scene,
    "build_velocity_tolloi_scene": build_velocity_tolloi_scene,
    "build_velocity_difference_scene": build_velocity_difference_scene,
}


def preprocess(paths, fig_config):
    """Load all data needed by figure_morphodynamics_comparison panels.

    Parameters
    ----------
    paths : dict
        Resolved paths from config (uppercase keys).
    fig_config : dict
        The figure_morphodynamics_comparison config section from figure_config.yaml.

    Returns
    -------
    dict
        Shared data for all figure_morphodynamics_comparison panels.
    """
    return load_morphodynamics_comparison_data(paths, fig_config)
