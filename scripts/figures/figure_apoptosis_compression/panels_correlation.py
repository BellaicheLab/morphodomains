"""Correlation panel for figure_apoptosis_compression.

Computes per-condition (vi vs Tolloi) correlation between cumulative
apoptosis and cumulative tissue compression in the LISA-defined ROI.
"""

import logging

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats

from pyanimalprocessing.rescaling.timings import find_closest_time_index, median_time
from scripts.figures.config import aoa_backup

logger = logging.getLogger(__name__)


def correlation_mean(shared_data, output_base):
    """Apoptosis-compression correlation (per-condition: vi and Tolloi) in ROI.

    Parameters
    ----------
    shared_data : dict
        Shared data from preprocessing.
    output_base : str
        Output directory path.

    Returns
    -------
    matplotlib.figure.Figure
        Scatter plot with per-condition regression lines.
    """
    indices = shared_data["indices"]
    avgAnimal_lagrangian = shared_data["avgAnimal_lagrangian"]
    time_array = shared_data["time_array"]
    time_array_instantaneous = aoa_backup(
        avgAnimal_lagrangian, "AOA_avgAnimals_tolloi_0h_0olap"
    ).attrs["TimeArray"]

    median_times_instantaneous = [
        median_time(frame[0], frame[1]) for frame in time_array_instantaneous
    ]
    median_times = [median_time(frame[0], frame[1]) for frame in time_array]

    dt = 5 / 60  # 5 min in hours
    start_time = "16h47"
    end_time = "30h01"

    closest_idx_norm = find_closest_time_index(
        start_time, median_times_instantaneous
    )
    closest_idx_start = find_closest_time_index(start_time, median_times)
    closest_idx_final = find_closest_time_index(end_time, median_times)

    # Load tensors
    # Animal-averaged fields (index 0 of the animal axis), (T, Ny, Nx[, comp]).
    window = slice(closest_idx_start, closest_idx_final)
    backup_vi = aoa_backup(avgAnimal_lagrangian, "AOA_avgAnimals_vi_2h_0.96olap")
    backup_tolloi = aoa_backup(avgAnimal_lagrangian, "AOA_avgAnimals_tolloi_2h_0.96olap")
    avg_tensor_epsilon_vi = backup_vi["Epsilon"][window, 0, :, :, :, 0]
    avg_tensor_epsilon_tolloi = backup_tolloi["Epsilon"][window, 0, :, :, :, 0]
    avg_tensor_dna_vi = backup_vi["dnA"][window, 0, :, :, 0, 0]
    avg_tensor_dna_tolloi = backup_tolloi["dnA"][window, 0, :, :, 0, 0]
    avg_tensor_dnd_vi = backup_vi["dnD"][window, 0, :, :, 0, 0]
    avg_tensor_dnd_tolloi = backup_tolloi["dnD"][window, 0, :, :, 0, 0]
    n_cells_reference_control = aoa_backup(
        avgAnimal_lagrangian, "AOA_avgAnimals_vi_0h_0olap"
    )["nCoreRNs"][closest_idx_norm, 0, :, :, 0, 0]
    n_cells_reference_tolloi = aoa_backup(
        avgAnimal_lagrangian, "AOA_avgAnimals_tolloi_0h_0olap"
    )["nCoreRNs"][closest_idx_norm, 0, :, :, 0, 0]

    # Compute epsilon traces
    epsilon_traces_vi = np.trace(
        avg_tensor_epsilon_vi.reshape(*avg_tensor_epsilon_vi.shape[:-1], 2, 2),
        axis1=-2,
        axis2=-1,
    )
    epsilon_traces_tolloi = np.trace(
        avg_tensor_epsilon_tolloi.reshape(*avg_tensor_epsilon_tolloi.shape[:-1], 2, 2),
        axis1=-2,
        axis2=-1,
    )

    # Normalize apoptosis by cell count
    normalizing_factor_control = np.broadcast_to(
        n_cells_reference_control,
        avg_tensor_dna_vi.shape,
    )
    normalizing_factor_tolloi = np.broadcast_to(
        n_cells_reference_tolloi,
        avg_tensor_dna_tolloi.shape,
    )
    tensor_dna_vi_normalized = avg_tensor_dna_vi / (
        normalizing_factor_control + np.nancumsum(avg_tensor_dnd_vi, axis=0)
    )
    tensor_dna_tolloi_normalized = avg_tensor_dna_tolloi / (
        normalizing_factor_tolloi + np.nancumsum(avg_tensor_dnd_tolloi, axis=0)
    )

    # Extract ROI values
    time_points = epsilon_traces_vi.shape[0]
    n_indices_roi = len(indices)

    epsilon_vi_roi = np.zeros((time_points, n_indices_roi))
    dna_vi_roi = np.zeros((time_points, n_indices_roi))
    epsilon_tolloi_roi = np.zeros((time_points, n_indices_roi))
    dna_tolloi_roi = np.zeros((time_points, n_indices_roi))

    for t in range(time_points):
        for i, (row, col) in enumerate(indices):
            epsilon_vi_roi[t, i] = epsilon_traces_vi[t, row, col]
            dna_vi_roi[t, i] = tensor_dna_vi_normalized[t, row, col]
            epsilon_tolloi_roi[t, i] = epsilon_traces_tolloi[t, row, col]
            dna_tolloi_roi[t, i] = tensor_dna_tolloi_normalized[t, row, col]

    # Cumulative integration
    cum_epsilon_vi_roi = np.nancumsum(epsilon_vi_roi, axis=0) * dt
    cum_dna_vi_roi = np.nancumsum(dna_vi_roi, axis=0) * dt
    cum_epsilon_tolloi_roi = np.nancumsum(epsilon_tolloi_roi, axis=0) * dt
    cum_dna_tolloi_roi = np.nancumsum(dna_tolloi_roi, axis=0) * dt

    # Final timepoint values
    final_epsilon_vi = cum_epsilon_vi_roi[-1, :] if cum_epsilon_vi_roi.shape[0] > 0 else np.array([])
    final_dna_vi = cum_dna_vi_roi[-1, :] if cum_dna_vi_roi.shape[0] > 0 else np.array([])
    final_epsilon_tolloi = cum_epsilon_tolloi_roi[-1, :] if cum_epsilon_tolloi_roi.shape[0] > 0 else np.array([])
    final_dna_tolloi = cum_dna_tolloi_roi[-1, :] if cum_dna_tolloi_roi.shape[0] > 0 else np.array([])

    # Filter NaNs
    valid_vi = ~np.isnan(final_epsilon_vi) & ~np.isnan(final_dna_vi)
    x_vi, y_vi = final_epsilon_vi[valid_vi], final_dna_vi[valid_vi]
    valid_tolloi = ~np.isnan(final_epsilon_tolloi) & ~np.isnan(final_dna_tolloi)
    x_tolloi, y_tolloi = final_epsilon_tolloi[valid_tolloi], final_dna_tolloi[valid_tolloi]

    x_combined = np.concatenate((x_vi, x_tolloi))
    y_combined = np.concatenate((y_vi, y_tolloi))
    valid_combined = ~np.isnan(x_combined) & ~np.isnan(y_combined)
    x_combined_valid = x_combined[valid_combined]
    y_combined_valid = y_combined[valid_combined]

    def get_stats(x, y):
        if len(x) > 1 and len(y) > 1:
            try:
                r, p = stats.pearsonr(x, y)
                return r, p, len(x)
            except ValueError:
                return np.nan, np.nan, len(x)
        return np.nan, np.nan, len(x)

    r_vi, p_vi, n_vi = get_stats(x_vi, y_vi)
    r_tolloi, p_tolloi, n_tolloi = get_stats(x_tolloi, y_tolloi)
    r_pooled, p_pooled, n_pooled = get_stats(x_combined_valid, y_combined_valid)

    # Plot
    plt.rcParams.update(
        {
            "font.size": 6,
            "font.family": "Arial",
            "axes.titlesize": 8,
            "axes.labelsize": 7,
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
            "legend.fontsize": 5,
            "figure.dpi": 300,
            "axes.linewidth": 0.5,
        }
    )

    fig, ax = plt.subplots(figsize=(2.5, 2.2))
    legend_elements = []

    if n_vi > 1:
        sns.regplot(
            x=x_vi, y=y_vi, ax=ax,
            color="cornflowerblue",
            scatter_kws={"s": 5, "alpha": 0.6},
            line_kws={"lw": 0.5, "linestyle": "--", "alpha": 0.4},
        )
        legend_elements.append(
            plt.Line2D([0], [0], marker="o", color="cornflowerblue",
                       linewidth=0.5, linestyle="--", markersize=3, alpha=0.6,
                       label=f"Ctrl ($r$={r_vi:.2f}, p={p_vi:.1e}, n={n_vi})")
        )

    if n_tolloi > 1:
        sns.regplot(
            x=x_tolloi, y=y_tolloi, ax=ax,
            color="salmon",
            scatter_kws={"s": 5, "alpha": 0.6},
            line_kws={"lw": 0.5, "linestyle": "--", "alpha": 0.4},
        )
        legend_elements.append(
            plt.Line2D([0], [0], marker="o", color="salmon",
                       linewidth=0.5, linestyle="--", markersize=3, alpha=0.6,
                       label=f"Tolloi ($r$={r_tolloi:.2f}, p={p_tolloi:.1e}, n={n_tolloi})")
        )

    if len(x_vi) > 0 and len(x_tolloi) > 0:
        min_x = max(np.nanmin(x_vi), np.nanmin(x_tolloi))
        max_x = min(np.nanmax(x_vi), np.nanmax(x_tolloi))
        if not (np.isnan(min_x) or np.isnan(max_x)) and min_x < max_x:
            ax.axvspan(min_x, max_x, alpha=0.1, color="lightgrey", zorder=0)

    if len(x_combined_valid) > 1:
        sns.regplot(
            x=x_combined_valid, y=y_combined_valid, ax=ax,
            color="grey", scatter=False,
            line_kws={"lw": 0.8, "linestyle": "--", "zorder": 1},
        )
        if not np.isnan(r_pooled):
            legend_elements.append(
                plt.Line2D([0], [0], color="grey", linewidth=0.8, linestyle="--",
                           label=f"Pooled Fit ($r$={r_pooled:.2f}, p={p_pooled:.1e}, n={n_pooled})")
            )

    ax.set_title("ROI")
    ax.set_xlabel("Cumulative \u03b5 trace")
    ax.set_ylabel("Cumulative norm. apoptosis")

    if legend_elements:
        ax.legend(handles=legend_elements, loc="upper right", fontsize=5)

    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
    ax.grid(True, linestyle="--", alpha=0.3, linewidth=0.5)

    if len(x_combined_valid) > 0:
        x_min, x_max = np.nanmin(x_combined_valid), np.nanmax(x_combined_valid)
        x_pad = (x_max - x_min) * 0.1 if (x_max - x_min) > 0 else 0.1
        ax.set_xlim(x_min - x_pad, x_max + x_pad)
    if len(y_combined_valid) > 0:
        y_min, y_max = np.nanmin(y_combined_valid), np.nanmax(y_combined_valid)
        y_pad = (y_max - y_min) * 0.1 if (y_max - y_min) > 0 else 0.1
        ax.set_ylim(y_min - y_pad, y_max + y_pad)

    fig.suptitle("ROI: Apoptosis vs. Tissue Contraction", fontsize=8, y=1.03)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    logger.info("ROI Ctrl: r=%.2f, p=%.2e, n=%d", r_vi, p_vi, n_vi)
    logger.info("ROI Tolloi: r=%.2f, p=%.2e, n=%d", r_tolloi, p_tolloi, n_tolloi)

    return fig