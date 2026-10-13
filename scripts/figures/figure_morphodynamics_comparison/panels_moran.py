"""Moran statistics panels for figure_morphodynamics_comparison.

Panels:
- moran_statistics: Time-series of global Moran's I (multi-figure, self-managed)
- autocorrelation_scatterplot: Moran scatterplots at max index
- moran_colorbars: Diverging colorbar legends (multi-figure, self-managed)
- save_lisa_maps: Save LISA maps to disk (self-managed)
"""

import logging
import os

import numpy as np

from pyanimalprocessing.spatial_statistics import compute_moran, plot_moran
from scripts.figures import save_and_close

from ._helpers import generate_moran_colorbars, plot_moran_figures

logger = logging.getLogger(__name__)


def moran_statistics(shared_data, output_base):
    """Generate Moran's I time-series figures. Saves U and epsilon plots."""
    moran_data = shared_data["moran_data"]
    fig_U, fig_epsilon = plot_moran_figures(moran_data)

    save_and_close(fig_U, os.path.join(output_base, "moran_plots/global_moran_I_dba_UPIV"))
    save_and_close(fig_epsilon, os.path.join(output_base, "moran_plots/global_moran_I_dba_Epsilon"))
    return None


def autocorrelation_scatterplot(shared_data, output_base):
    """Generate autocorrelation scatterplots for deformation rate and velocity.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    logger.info("Generating autocorrelation scatterplots...")

    moran_data = shared_data["moran_data"]
    max_idx_epsilon = moran_data["max_idx_epsilon"]
    max_idx_u = moran_data["max_idx_u"]
    data_epsilon_dba_iso = moran_data["data_epsilon_dba_iso"]
    data_u_dba = moran_data["data_u_dba"]
    median_times = list(moran_data["median_times"])

    fig, (ax_epsilon, ax_U) = plt.subplots(1, 2, figsize=(6, 2.5))

    # Deformation rate scatterplot
    plot_moran(
        compute_moran(data_epsilon_dba_iso[max_idx_epsilon]),
        ax=ax_epsilon,
        scatter_kwds={"color": "#1f77b4", "marker": "o", "alpha": 0.8, "s": 2},
        fit_kwds={"color": "red", "linestyle": "--", "linewidth": 1},
    )
    ax_epsilon.set_title(
        "Spatial Autocorrelation\nDeformation Rate", fontsize=7, fontweight="bold"
    )
    ax_epsilon.set_xlabel("Deformation Rate (h$^{-1}$)", fontsize=7)
    ax_epsilon.set_ylabel("Spatially Lagged Deformation Rate", fontsize=7)
    ax_epsilon.annotate(
        f"t = {median_times[max_idx_epsilon]}",
        xy=(0.05, 0.95), xycoords="axes fraction", fontsize=6,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
    )

    # Velocity scatterplot
    plot_moran(
        compute_moran(data_u_dba[max_idx_u]),
        ax=ax_U,
        scatter_kwds={"color": "#ff7f0e", "marker": "o", "alpha": 0.8, "s": 2},
        fit_kwds={"color": "red", "linestyle": "--", "linewidth": 1},
    )
    ax_U.set_title(
        "Spatial Autocorrelation\nVelocity", fontsize=7, fontweight="bold"
    )
    ax_U.set_xlabel("Velocity (µm·h$^{-1}$)", fontsize=7)
    ax_U.set_ylabel("Spatially Lagged Velocity", fontsize=7)
    ax_U.annotate(
        f"t = {median_times[max_idx_u]}",
        xy=(0.05, 0.95), xycoords="axes fraction", fontsize=6,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
    )

    plt.tight_layout()
    return fig


def moran_colorbars(shared_data, output_base):
    """Generate diverging colorbar legends for Local Moran statistics.

    Self-managed: saves directly to ``moran_plots/`` and returns None.
    """
    return generate_moran_colorbars(output_base)


def save_lisa_maps(shared_data, output_base):
    """Save LISA maps and p-values to disk.

    Returns
    -------
    None
    """
    moran_data = shared_data["moran_data"]
    output_dir = os.path.join(output_base, "moran_plots")
    os.makedirs(output_dir, exist_ok=True)

    logger.info(f"Saving LISA maps and p-values to {output_dir}")

    for key in [
        "p_values_epsilon", "p_values_u",
        "hh_map_epsilon", "ll_map_epsilon", "hl_map_epsilon", "lh_map_epsilon",
        "hh_map_u", "ll_map_u", "hl_map_u", "lh_map_u",
        "data_epsilon_dba_iso", "data_u_dba",
    ]:
        np.save(os.path.join(output_dir, f"{key}.npy"), moran_data[key])

    logger.info("LISA maps and p-values saved successfully")
    return None
