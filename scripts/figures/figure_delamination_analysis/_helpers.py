"""Constants and utilities for delamination analysis panels.

Provides helper functions shared across the cumulative apoptosis and
2h-rate apoptosis panels, plus the reusable comparison trace plotter.
"""

import logging

import matplotlib.colors
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from scipy import stats

logger = logging.getLogger(__name__)

# ── Plot colours ─────────────────────────────────────────────────────────────
COLOR_VI = "#98C3E4"
COLOR_TOLLOI = "#F8AE8B"

# ── Time cropping / axis tick configuration ──────────────────────────────────
CROP_TIME_LABEL = "26h47"
DESIRED_TICK_LABELS = [
    "16h47",
    "18h27",
    "20h07",
    "21h47",
    "23h27",
    "25h07",
    "26h47",
]
START_TIME_STR = "16h47"
END_TIME_STR = "30h01"

# ── P-value bar colour boundaries ────────────────────────────────────────────
PVAL_BOUNDS = [0, 0.001, 0.01, 0.05, 1.0]


# ── Utilities ────────────────────────────────────────────────────────────────


def get_median_time_strings_from_pairs(time_array_pairs):
    """Extract the start-time string from each ``[start, end]`` pair.

    Parameters
    ----------
    time_array_pairs : list of [str, str]
        E.g. ``[['20h00', '20h05'], ...]``.

    Returns
    -------
    list of str
    """
    return [pair[0] for pair in time_array_pairs]


def nancumsum_with_trailing_nans(arr, axis=0):
    """Cumulative sum treating NaN as zero, but preserving trailing NaNs.

    After the last valid (non-NaN) entry along a slice, trailing positions
    remain NaN rather than carrying over the last cumulative value.
    """
    arr = np.asarray(arr)

    def _process_1d(s):
        if np.all(np.isnan(s)):
            return np.full_like(s, np.nan)
        cs = np.nancumsum(s)
        nan_mask = np.isnan(s)
        if not np.any(nan_mask):
            return cs
        non_nan = np.where(~nan_mask)[0]
        if len(non_nan) == 0:
            return cs
        last_valid = non_nan[-1]
        if last_valid < len(s) - 1:
            cs[last_valid + 1 :] = np.nan
        return cs

    if arr.ndim == 0:
        return np.nancumsum(arr)
    if arr.ndim == 1:
        return _process_1d(arr)
    return np.apply_along_axis(_process_1d, axis, arr)


def plot_comparison_traces(
    traces_cond1,
    sem_cond1,
    label_cond1,
    traces_cond2,
    sem_cond2,
    label_cond2,
    time_points_median_str,
    title,
    y_label,
    *,
    plot_individuals=True,
    crop_time_label=CROP_TIME_LABEL,
    desired_tick_labels=None,
):
    """Plot two-condition comparison with mean, SEM, individual traces and p-value bar.

    Returns the ``Figure`` for the caller to save/close.

    Parameters
    ----------
    traces_cond1, traces_cond2 : ndarray, shape (T, 1+N)
        Column 0 = mean trace, columns 1..N = individual traces.
    sem_cond1, sem_cond2 : ndarray or None
        SEM arrays of length T.
    label_cond1, label_cond2 : str
        Legend labels.
    time_points_median_str : list of str
        ``'HHhMM'`` time labels for each time point.
    title, y_label : str
        Plot title and y-axis label.
    plot_individuals : bool
        Whether to show individual traces.
    crop_time_label : str
        Last time label to include (data is cropped here).
    desired_tick_labels : list of str or None
        Explicit x-axis tick labels to show. Defaults to
        :data:`DESIRED_TICK_LABELS`.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if desired_tick_labels is None:
        desired_tick_labels = DESIRED_TICK_LABELS

    processed_time_labels = [str(lbl) for lbl in time_points_median_str]

    # ── Crop at crop_time_label ──────────────────────────────────────────
    try:
        crop_idx = processed_time_labels.index(crop_time_label)
        traces_cond1 = traces_cond1[: crop_idx + 1]
        if sem_cond1 is not None and hasattr(sem_cond1, "shape") and sem_cond1.ndim > 0:
            sem_cond1 = sem_cond1[: crop_idx + 1]
        traces_cond2 = traces_cond2[: crop_idx + 1]
        if sem_cond2 is not None and hasattr(sem_cond2, "shape") and sem_cond2.ndim > 0:
            sem_cond2 = sem_cond2[: crop_idx + 1]
        processed_time_labels = processed_time_labels[: crop_idx + 1]
    except ValueError:
        raise ValueError(
            f"Crop time label '{crop_time_label}' not found. "
            "Cannot proceed with specified cropping."
        ) from None

    num_tp = traces_cond1.shape[0]
    t_idx = np.arange(num_tp)

    mean1 = traces_cond1[:, 0]
    indiv1 = traces_cond1[:, 1:]
    mean2 = traces_cond2[:, 0]
    indiv2 = traces_cond2[:, 1:]

    # ── P-values (Mann-Whitney U) ────────────────────────────────────────
    p_values = np.ones(num_tp)
    if plot_individuals and indiv1.shape[1] > 0 and indiv2.shape[1] > 0 and num_tp > 0:
        for t in range(num_tp):
            v1 = indiv1[t, ~np.isnan(indiv1[t, :])]
            v2 = indiv2[t, ~np.isnan(indiv2[t, :])]
            if len(v1) >= 2 and len(v2) >= 2:
                try:
                    _, pv = stats.mannwhitneyu(v1, v2, alternative="two-sided")
                    p_values[t] = pv
                except ValueError:
                    p_values[t] = 1.0

    # ── Figure layout ────────────────────────────────────────────────────
    fig = plt.figure(figsize=(5, 3.5))
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 10], hspace=0.05)
    ax_main = fig.add_subplot(gs[1])
    ax_pval = fig.add_subplot(gs[0], sharex=ax_main)

    if num_tp > 0:
        # Condition 1
        if plot_individuals and indiv1.shape[1] > 0:
            for i in range(indiv1.shape[1]):
                ax_main.plot(t_idx, indiv1[:, i], color=COLOR_VI, alpha=0.2, linewidth=1)
        ax_main.fill_between(
            t_idx,
            mean1 - (sem_cond1 if sem_cond1 is not None else 0),
            mean1 + (sem_cond1 if sem_cond1 is not None else 0),
            color=COLOR_VI,
            alpha=0.5,
        )
        ax_main.plot(t_idx, mean1, color=COLOR_VI, label=label_cond1, linewidth=2)

        # Condition 2
        if plot_individuals and indiv2.shape[1] > 0:
            for i in range(indiv2.shape[1]):
                ax_main.plot(
                    t_idx, indiv2[:, i], color=COLOR_TOLLOI, alpha=0.2, linewidth=1
                )
        ax_main.fill_between(
            t_idx,
            mean2 - (sem_cond2 if sem_cond2 is not None else 0),
            mean2 + (sem_cond2 if sem_cond2 is not None else 0),
            color=COLOR_TOLLOI,
            alpha=0.5,
        )
        ax_main.plot(t_idx, mean2, color=COLOR_TOLLOI, label=label_cond2, linewidth=2)

    ax_main.set_ylabel(y_label)
    ax_main.set_xlabel("Time (h)")

    # X-limits and p-value extent
    if num_tp > 1:
        ax_main.set_xlim(t_idx[0] - 0.5, t_idx[-1] + 0.5)
        pval_extent = [t_idx[0] - 0.5, t_idx[-1] + 0.5, 0, 1]
    elif num_tp == 1:
        ax_main.set_xlim(-0.5, 0.5)
        pval_extent = [-0.5, 0.5, 0, 1]
    else:
        pval_extent = [-0.5, 0.5, 0, 1]

    # P-value colourbar
    cmap_pval = plt.cm.Reds_r
    norm_pval = matplotlib.colors.BoundaryNorm(PVAL_BOUNDS, cmap_pval.N)

    if num_tp > 0:
        ax_pval.imshow(
            p_values.reshape(1, -1),
            aspect="auto",
            cmap=cmap_pval,
            norm=norm_pval,
            extent=pval_extent,
        )
    ax_pval.set_yticks([])
    plt.setp(ax_pval.get_xticklabels(), visible=False)
    ax_pval.set_title(title)

    # Legend
    handles, labels = ax_main.get_legend_handles_labels()
    sig_patches = [
        Patch(facecolor=cmap_pval(norm_pval(0.0005)), edgecolor="black", label="p < 0.001 (***)"),
        Patch(facecolor=cmap_pval(norm_pval(0.005)), edgecolor="black", label="p < 0.01 (**)"),
        Patch(facecolor=cmap_pval(norm_pval(0.03)), edgecolor="black", label="p < 0.05 (*)"),
        Patch(facecolor=cmap_pval(norm_pval(0.5)), edgecolor="black", label="p >= 0.05 (ns)"),
    ]
    handles.extend(sig_patches)
    legend_labels = labels + [p.get_label() for p in sig_patches]
    ax_main.legend(handles, legend_labels, loc="best", frameon=True, fontsize="small")

    # X-axis ticks
    if num_tp > 0:
        tick_indices = []
        tick_labels = []
        for lbl in desired_tick_labels:
            try:
                idx = processed_time_labels.index(lbl)
                tick_indices.append(idx)
                tick_labels.append(lbl)
            except ValueError:
                logger.warning(
                    "Desired tick label %s not found in cropped time labels. Skipping.",
                    lbl,
                )
        if tick_indices:
            ax_main.set_xticks(tick_indices)
            ax_main.set_xticklabels(tick_labels, rotation=0, ha="center")
        elif processed_time_labels:
            sub = max(1, len(processed_time_labels) // 8)
            fb_idx = t_idx[::sub]
            fb_lbl = [
                processed_time_labels[i] for i in fb_idx if i < len(processed_time_labels)
            ]
            if fb_idx.size > 0 and fb_lbl:
                ax_main.set_xticks(fb_idx)
                ax_main.set_xticklabels(fb_lbl, rotation=45, ha="right")

    ax_main.grid(False)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    return fig
