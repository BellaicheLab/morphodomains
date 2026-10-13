"""Panel functions for delamination analysis.

Self-managed panels (return ``None``) that produce figures + CSV data,
saving outputs directly via :func:`~scripts.figures.save_and_close`.
"""

import logging
import os

import numpy as np
import pandas as pd
from scipy import stats

from scripts.figures import save_and_close

from pyanimalprocessing.rescaling.timings import find_closest_time_index

from ._helpers import (
    END_TIME_STR,
    START_TIME_STR,
    get_median_time_strings_from_pairs,
    nancumsum_with_trailing_nans,
    plot_comparison_traces,
)

logger = logging.getLogger(__name__)


# ── Shared data loading helpers ──────────────────────────────────────────────


def _decode_time_pairs(aoa, group_path):
    """Load and decode time array pairs from a zarr group.

    Returns a list of ``[start_str, end_str]`` pairs.
    """
    raw = aoa[group_path]["time_array"]
    try:
        return [[s.decode("utf-8"), e.decode("utf-8")] for s, e in raw]
    except AttributeError:
        # Already decoded
        return [[str(s), str(e)] for s, e in raw]


def _compute_condition(
    condition_key,
    aoa,
    *,
    mode="cumulative",
):
    """Compute per-condition results for one zarr archive.

    Parameters
    ----------
    condition_key : str
        ``"vi"`` or ``"tolloi"`` (for log messages only).
    aoa : zarr.Group
        Opened averaged zarr archive.
    mode : str
        ``"cumulative"`` or ``"2h_rate"``.

    Returns
    -------
    dict or None
        ``{"mean_trace", "sem_trace", "individual_traces_grid_mean"}``
        or ``None`` on failure.
    list of str or None
        Time labels for the x-axis (from the first condition processed).
    """
    logger.info("Processing %s for condition: %s", mode, condition_key)

    time_2h_pairs = _decode_time_pairs(aoa, "LAGRANGIAN_TRACKING/CELL/D2/average_02h00")
    time_inst_pairs = _decode_time_pairs(aoa, "LAGRANGIAN_TRACKING/CELL/D2/instantaneous")

    median_inst = get_median_time_strings_from_pairs(time_inst_pairs)
    median_2h = get_median_time_strings_from_pairs(time_2h_pairs)

    idx_start_2h = find_closest_time_index(START_TIME_STR, median_2h)
    idx_end_2h = find_closest_time_index(END_TIME_STR, median_2h)

    sliced_2h_labels = get_median_time_strings_from_pairs(
        time_2h_pairs[idx_start_2h : idx_end_2h + 1]
    )

    if mode == "cumulative":
        return _compute_cumulative(
            condition_key, aoa, median_inst, idx_start_2h, idx_end_2h
        ), sliced_2h_labels
    else:
        return _compute_2h_rate(
            condition_key, aoa, median_inst, median_2h,
            idx_start_2h, idx_end_2h, sliced_2h_labels,
        ), sliced_2h_labels


def _compute_cumulative(condition_key, aoa, median_inst, idx_start_2h, idx_end_2h):
    """Compute cumulative normalised apoptosis for one condition."""
    idx_start_inst = find_closest_time_index(START_TIME_STR, median_inst)

    cell_number_all = aoa["LAGRANGIAN_TRACKING/CELL/D2/instantaneous/cell_number"][:]
    initial_cells = cell_number_all[idx_start_inst, 0, :, :, 0, 1:].astype(float)

    dnD_all = aoa["LAGRANGIAN_TRACKING/CELL/D2/average_02h00/divisions_sum_per_cell"][:]
    dnD_sliced = dnD_all[idx_start_2h : idx_end_2h + 1, 0, :, :, 0, 1:]
    cum_div = nancumsum_with_trailing_nans(dnD_sliced, axis=0)

    dnA_all = aoa["LAGRANGIAN_TRACKING/CELL/D2/average_02h00/apoptosis_sum_per_cell"][:]
    dnA_sliced = dnA_all[idx_start_2h : idx_end_2h + 1, 0, :, :, 0, 1:]
    cum_apop = nancumsum_with_trailing_nans(dnA_sliced, axis=0)

    denom = initial_cells[np.newaxis, :, :, :] + cum_div
    denom = np.where(denom == 0, np.nan, denom)

    normalised = np.full_like(cum_apop, np.nan, dtype=float)
    valid = ~np.isnan(denom)
    normalised[valid] = cum_apop[valid] / denom[valid]

    grid_mean = np.nanmean(normalised, axis=(1, 2))  # (T, samples)
    mean_trace = np.nanmean(grid_mean, axis=1)
    sem_trace = stats.sem(grid_mean, axis=1, nan_policy="omit")
    sem_trace[np.isnan(mean_trace)] = np.nan

    return {
        "mean_trace": mean_trace,
        "sem_trace": sem_trace,
        "individual_traces_grid_mean": grid_mean,
    }


def _compute_2h_rate(condition_key, aoa, median_inst, median_2h,
                     idx_start_2h, idx_end_2h, sliced_labels):
    """Compute 2h-average normalised apoptosis rate for one condition."""
    cell_number_all = aoa["LAGRANGIAN_TRACKING/CELL/D2/instantaneous/cell_number"][:]

    dnA_2h_all = aoa["LAGRANGIAN_TRACKING/CELL/D2/average_02h00/apoptosis_sum_per_cell"][:]
    dnA_2h = dnA_2h_all[idx_start_2h : idx_end_2h + 1, 0, :, :, 0, 1:].astype(float)

    num_tp = len(sliced_labels)
    ny, nx, ns = dnA_2h.shape[1], dnA_2h.shape[2], dnA_2h.shape[3]
    cell_counts = np.full((num_tp, ny, nx, ns), np.nan, dtype=float)

    for t in range(num_tp):
        idx_inst = find_closest_time_index(sliced_labels[t], median_inst)
        try:
            cell_counts[t] = cell_number_all[idx_inst, 0, :, :, 0, 1:].astype(float)
        except IndexError:
            logger.warning(
                "IndexError fetching cell count for %s (inst idx %d) for %s",
                sliced_labels[t], idx_inst, condition_key,
            )

    denom = np.where(cell_counts == 0, np.nan, cell_counts)
    normalised = np.full_like(dnA_2h, np.nan, dtype=float)
    valid = ~np.isnan(denom) & ~np.isnan(dnA_2h)
    normalised[valid] = dnA_2h[valid] / denom[valid]

    grid_mean = np.nanmean(normalised, axis=(1, 2))
    mean_trace = np.nanmean(grid_mean, axis=1)
    sem_trace = stats.sem(grid_mean, axis=1, nan_policy="omit")
    sem_trace[np.isnan(mean_trace)] = np.nan

    return {
        "mean_trace": mean_trace,
        "sem_trace": sem_trace,
        "individual_traces_grid_mean": grid_mean,
    }


# ── Panel functions ──────────────────────────────────────────────────────────


def cumulative_apoptosis(shared_data, output_base):
    """Cumulative normalised apoptosis: vi vs Tolloi.

    Self-managed panel: saves SVG, PNG, and CSV directly, returns ``None``.
    """
    aoa_vi = shared_data["aoa_vi"]
    aoa_tolloi = shared_data["aoa_tolloi"]

    results_vi, time_labels = _compute_condition("vi", aoa_vi, mode="cumulative")
    results_tolloi, _ = _compute_condition("tolloi", aoa_tolloi, mode="cumulative")

    if results_vi is None or results_tolloi is None or time_labels is None:
        logger.error("Failed to compute cumulative apoptosis for one or both conditions.")
        return None

    n_vi = results_vi["individual_traces_grid_mean"].shape[1]
    n_tolloi = results_tolloi["individual_traces_grid_mean"].shape[1]
    logger.info("Cumulative apoptosis — vi samples: %d, Tolloi samples: %d", n_vi, n_tolloi)

    traces_vi = np.hstack((
        results_vi["mean_trace"][:, np.newaxis],
        results_vi["individual_traces_grid_mean"],
    ))
    traces_tolloi = np.hstack((
        results_tolloi["mean_trace"][:, np.newaxis],
        results_tolloi["individual_traces_grid_mean"],
    ))

    fig = plot_comparison_traces(
        traces_cond1=traces_vi,
        sem_cond1=results_vi["sem_trace"],
        label_cond1=f"$v^{{RNAi}}$ (n={n_vi})",
        traces_cond2=traces_tolloi,
        sem_cond2=results_tolloi["sem_trace"],
        label_cond2=f"$Tollo^{{RNAi}}$ (n={n_tolloi})",
        time_points_median_str=time_labels,
        title="Cumulative Normalized Apoptosis (Grid Mean)",
        y_label="Normalized Cumulative Apoptosis Events / Cell",
        plot_individuals=True,
    )

    out_dir = os.path.join(output_base, "delamination")
    os.makedirs(out_dir, exist_ok=True)

    save_and_close(
        fig,
        os.path.join(out_dir, "cumulative_norm_apoptosis"),
        formats=("svg", "png"),
        dpi=1200,
    )

    # Save CSV
    try:
        pd.DataFrame({
            "Time": time_labels,
            "Vi_mean": results_vi["mean_trace"],
            "Vi_sem": results_vi["sem_trace"],
            "Tolloi_mean": results_tolloi["mean_trace"],
            "Tolloi_sem": results_tolloi["sem_trace"],
        }).to_csv(os.path.join(out_dir, "pyap_apoptosis_cumul.csv"), index=False)
        logger.info("Saved cumulative apoptosis CSV")
    except Exception as e:
        logger.error("Error saving cumulative CSV: %s", e)

    return None


def apoptosis_2h_rate(shared_data, output_base):
    """2h-average normalised apoptosis rate: vi vs Tolloi.

    Self-managed panel: saves SVG, PNG, and CSV directly, returns ``None``.
    """
    aoa_vi = shared_data["aoa_vi"]
    aoa_tolloi = shared_data["aoa_tolloi"]

    results_vi, time_labels = _compute_condition("vi", aoa_vi, mode="2h_rate")
    results_tolloi, _ = _compute_condition("tolloi", aoa_tolloi, mode="2h_rate")

    if results_vi is None or results_tolloi is None or time_labels is None:
        logger.error("Failed to compute 2h rate apoptosis for one or both conditions.")
        return None

    n_vi = results_vi["individual_traces_grid_mean"].shape[1]
    n_tolloi = results_tolloi["individual_traces_grid_mean"].shape[1]
    logger.info("2h rate apoptosis — vi samples: %d, Tolloi samples: %d", n_vi, n_tolloi)

    traces_vi = np.hstack((
        results_vi["mean_trace"][:, np.newaxis],
        results_vi["individual_traces_grid_mean"],
    ))
    traces_tolloi = np.hstack((
        results_tolloi["mean_trace"][:, np.newaxis],
        results_tolloi["individual_traces_grid_mean"],
    ))

    fig = plot_comparison_traces(
        traces_cond1=traces_vi,
        sem_cond1=results_vi["sem_trace"],
        label_cond1=f"$v^{{RNAi}}$ (n={n_vi})",
        traces_cond2=traces_tolloi,
        sem_cond2=results_tolloi["sem_trace"],
        label_cond2=f"$Tollo^{{RNAi}}$ (n={n_tolloi})",
        time_points_median_str=time_labels,
        title="2h Average Normalized Apoptosis Rate (Grid Mean)",
        y_label="Normalized 2h Apoptosis Events / Cell",
        plot_individuals=True,
    )

    out_dir = os.path.join(output_base, "delamination")
    os.makedirs(out_dir, exist_ok=True)

    save_and_close(
        fig,
        os.path.join(out_dir, "2h_avg_norm_apoptosis"),
        formats=("svg", "png"),
        dpi=1200,
    )

    # Save CSV
    try:
        pd.DataFrame({
            "Time": time_labels,
            "Vi_mean": results_vi["mean_trace"],
            "Vi_sem": results_vi["sem_trace"],
            "Tolloi_mean": results_tolloi["mean_trace"],
            "Tolloi_sem": results_tolloi["sem_trace"],
        }).to_csv(os.path.join(out_dir, "pyap_apoptosis_instant.csv"), index=False)
        logger.info("Saved 2h rate apoptosis CSV")
    except Exception as e:
        logger.error("Error saving 2h rate CSV: %s", e)

    return None
