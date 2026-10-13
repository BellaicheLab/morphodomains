"""Internal helpers for figure_apoptosis_compression timeseries panels.

Aggregate per-animal Lagrangian tensor traces inside a spatial ROI and render
the vi vs Tolloi time-series plots.
"""

from __future__ import annotations

import json
import os
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from pyanimalprocessing.rescaling.timings import median_time
from pyanimalprocessing.transformation.tensors import (
    get_unbiased_weighted_std_matrix_aoa as get_unbiased_weighted_std_matrix,
)


# ---------------------------------------------------------------------------
# Per-animal spatial aggregation inside an ROI
# ---------------------------------------------------------------------------

def average_matrix_traces(
    tensor, indices, time_points, y_dim, x_dim, area_ratios, matrix_op="trace"
):
    """Weighted average over ROI of a scalar derived from a 2x2 matrix tensor.

    ``matrix_op='trace'`` returns Tr = m00 + m11;
    ``matrix_op='anisotropy'`` returns the eigenvalue difference sqrt((m00 - m11)^2 + 4 m01^2).
    """
    n_animals = tensor.shape[-1]
    traces = np.zeros((time_points, n_animals))
    weights = np.zeros((time_points, n_animals))

    for t in range(time_points):
        for animal in range(n_animals):
            matrices = tensor[t, indices[:, 0], indices[:, 1], :, animal]
            current_weights = area_ratios[t, indices[:, 0], indices[:, 1], animal]

            if matrix_op == "trace":
                matrix_values = matrices[:, 0] + matrices[:, 3]
            elif matrix_op == "anisotropy":
                matrix_values = np.sqrt(
                    (matrices[:, 0] - matrices[:, 3]) ** 2 + 4 * matrices[:, 1] ** 2
                )
            else:
                raise ValueError("matrix_op must be either 'trace' or 'anisotropy'")

            if np.all(current_weights == 0):
                traces[t, animal] = np.nan
            else:
                valid_mask = ~np.isnan(matrix_values)
                if np.any(valid_mask):
                    traces[t, animal] = np.average(
                        matrix_values[valid_mask],
                        weights=current_weights[valid_mask],
                    )
                else:
                    traces[t, animal] = np.nan
            weights[t, animal] = np.mean(current_weights)

    return traces, weights


def average_vector_norms(tensor, indices, time_points, y_dim, x_dim, area_ratios):
    """Weighted average over ROI of the L2 norm of a 2D vector tensor."""
    n_animals = tensor.shape[-1]
    norms = np.zeros((time_points, n_animals))
    weights = np.zeros((time_points, n_animals))

    for t in range(time_points):
        for animal in range(n_animals):
            vectors = tensor[t, indices[:, 0], indices[:, 1], :, animal]
            current_weights = area_ratios[t, indices[:, 0], indices[:, 1], animal]

            vector_norms = np.linalg.norm(vectors, axis=1)

            if np.all(current_weights == 0):
                norms[t, animal] = np.nan
            else:
                valid_mask = ~np.isnan(vector_norms)
                if np.any(valid_mask):
                    norms[t, animal] = np.average(
                        vector_norms[valid_mask],
                        weights=current_weights[valid_mask],
                    )
                else:
                    norms[t, animal] = np.nan
            weights[t, animal] = np.mean(current_weights)

    return norms, weights


def average_scalar_values(tensor, indices, time_points, y_dim, x_dim, area_ratios):
    """Weighted average over ROI of a scalar-valued tensor."""
    n_animals = tensor.shape[-1]
    values = np.zeros((time_points, n_animals))
    weights = np.zeros((time_points, n_animals))

    for t in range(time_points):
        for animal in range(n_animals):
            scalars = tensor[t, indices[:, 0], indices[:, 1], animal]
            current_weights = area_ratios[t, indices[:, 0], indices[:, 1], animal]

            if np.all(current_weights == 0):
                values[t, animal] = np.nan
            else:
                valid_mask = ~np.isnan(scalars)
                if np.any(valid_mask):
                    values[t, animal] = np.average(
                        scalars[valid_mask],
                        weights=current_weights[valid_mask],
                    )
                else:
                    values[t, animal] = np.nan
            weights[t, animal] = np.mean(current_weights)

    return values, weights


# ---------------------------------------------------------------------------
# Top-level tensor → per-animal trace + SEM
# ---------------------------------------------------------------------------

def process_tensor(
    tensor,
    indices,
    area_ratios,
    matrix_op: Literal["trace", "anisotropy"] = "trace",
    dt: float = 5 / 60,
    dynamic_op: Literal["average", "cumulative"] = "average",
) -> tuple[np.ndarray, np.ndarray]:
    """Spatially average a Lagrangian tensor inside an ROI, per animal.

    Dispatches on tensor rank:
    - 5D (T, Y, X, 4, N) matrix → :func:`average_matrix_traces`
    - 5D (T, Y, X, 2, N) vector → :func:`average_vector_norms`
    - 4D (T, Y, X, N) scalar → :func:`average_scalar_values`

    ``dynamic_op='cumulative'`` integrates in time with step ``dt`` (hours).

    Returns
    -------
    all_traces : np.ndarray, shape (T, N+1)
        Column 0 = weighted mean across animals; columns 1..N = per-animal.
    sem : np.ndarray, shape (T,)
        Standard error of the mean across animals.
    """
    tensor_shape = tensor.shape
    num_dims = len(tensor_shape)

    time_points = tensor_shape[0]
    y_dim = tensor_shape[1]
    x_dim = tensor_shape[2]

    if num_dims == 5:
        tensor_elements = tensor_shape[3]

        if tensor_elements == 4:
            traces, weights = average_matrix_traces(
                tensor, indices, time_points, y_dim, x_dim, area_ratios,
                matrix_op=matrix_op,
            )

            if dynamic_op == "cumulative":
                traces = np.nancumsum(traces, axis=0) * dt

            valid_mask = ~np.isnan(traces)
            averaged_trace = np.where(
                np.any(valid_mask, axis=1),
                np.nansum(traces * weights, axis=1) / np.sum(weights, axis=1),
                np.nan,
            )

            std = get_unbiased_weighted_std_matrix(traces, averaged_trace, weights)
            sem = std / np.sqrt(tensor_shape[-1])

            all_traces = np.zeros((time_points, traces.shape[1] + 1))
            all_traces[:, 0] = averaged_trace
            all_traces[:, 1:] = traces
            return all_traces, sem

        elif tensor_elements == 2:
            norms, weights = average_vector_norms(
                tensor, indices, time_points, y_dim, x_dim, area_ratios,
            )
            if dynamic_op == "cumulative":
                norms = np.nancumsum(norms, axis=0) * dt

            averaged_trace = np.average(norms, weights=weights, axis=1)
            std = get_unbiased_weighted_std_matrix(norms, averaged_trace, weights)
            sem = std / np.sqrt(tensor_shape[-1])

            all_traces = np.zeros((time_points, norms.shape[1] + 1))
            all_traces[:, 0] = averaged_trace
            all_traces[:, 1:] = norms
            return all_traces, sem

    elif num_dims == 4:
        values, weights = average_scalar_values(
            tensor, indices, time_points, y_dim, x_dim, area_ratios,
        )

        if dynamic_op == "cumulative":
            values = np.nancumsum(values, axis=0) * dt

        averaged_trace = np.nansum(values * weights, axis=1) / np.nansum(
            weights, axis=1,
        )

        std = get_unbiased_weighted_std_matrix(values, averaged_trace, weights)
        sem = std / np.sqrt(tensor_shape[-1])

        all_traces = np.zeros((time_points, values.shape[1] + 1))
        all_traces[:, 0] = averaged_trace
        all_traces[:, 1:] = values
        return all_traces, sem

    raise ValueError(f"Unsupported tensor shape: {tensor_shape}")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_results(
    traces_vi,
    sem_vi,
    traces_tolloi,
    sem_tolloi,
    time_array,
    start,
    stop,
    item,
    plot_individuals=False,
    ylims=None,
    output_json_path=None,
):
    """Plot vi vs Tolloi time-series with per-timepoint Mann-Whitney p-values.

    Two-panel figure: top strip shows the p-value color-map, main panel
    shows mean ± SEM bands plus thin individual traces. p-values are
    serialized to ``output_json_path`` when provided.
    """
    subsample_rate = 20

    mean_vi = traces_vi[:, 0]
    mean_tolloi = traces_tolloi[:, 0]
    individual_vi = traces_vi[:, 1:]
    individual_tolloi = traces_tolloi[:, 1:]

    median_times = [median_time(frame[0], frame[1]) for frame in time_array]
    start_index = median_times.index(start)
    stop_index = median_times.index(stop)
    median_times = median_times[start_index : stop_index + 1]
    median_times_subsampled = median_times[::subsample_rate]
    adjusted_time = np.arange(len(median_times))

    p_values = np.zeros(len(adjusted_time))
    for t in range(len(adjusted_time)):
        idx = t + start_index
        vi_vals = individual_vi[idx, ~np.isnan(individual_vi[idx])]
        tolloi_vals = individual_tolloi[idx, ~np.isnan(individual_tolloi[idx])]
        if len(vi_vals) < 2 or len(tolloi_vals) < 2:
            p_values[t] = 1.0
            continue
        try:
            _, p_val = stats.mannwhitneyu(vi_vals, tolloi_vals)
            p_values[t] = p_val
        except Exception:
            p_values[t] = 1.0

    if output_json_path is not None:
        os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
        with open(output_json_path, "w") as json_file:
            json.dump(
                {
                    "item": item,
                    "time_points": median_times,
                    "p_values": p_values.tolist(),
                },
                json_file,
                indent=4,
            )

    cmap = plt.cm.Reds_r
    bounds = [0, 0.001, 0.01, 0.05, 1.0]
    norm = plt.matplotlib.colors.BoundaryNorm(bounds, cmap.N)

    fig = plt.figure(figsize=(5, 3.5))
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 10], hspace=0.05)
    ax = fig.add_subplot(gs[1])
    pval_ax = fig.add_subplot(gs[0], sharex=ax)

    if plot_individuals:
        for i in range(individual_vi.shape[1]):
            ax.plot(
                adjusted_time,
                individual_vi[start_index : stop_index + 1, i],
                color="#98C3E4", alpha=0.2, linewidth=1,
            )
        for i in range(individual_tolloi.shape[1]):
            ax.plot(
                adjusted_time,
                individual_tolloi[start_index : stop_index + 1, i],
                color="#F8AE8B", alpha=0.2, linewidth=1,
            )

    ax.fill_between(
        adjusted_time,
        mean_vi[start_index : stop_index + 1] - sem_vi[start_index : stop_index + 1],
        mean_vi[start_index : stop_index + 1] + sem_vi[start_index : stop_index + 1],
        color="#98C3E4", alpha=0.5,
    )
    ax.plot(
        adjusted_time,
        mean_vi[start_index : stop_index + 1],
        color="#98C3E4", label="$V^{RNAi}$ Mean", linewidth=2,
    )

    ax.fill_between(
        adjusted_time,
        mean_tolloi[start_index : stop_index + 1]
        - sem_tolloi[start_index : stop_index + 1],
        mean_tolloi[start_index : stop_index + 1]
        + sem_tolloi[start_index : stop_index + 1],
        color="#F8AE8B", alpha=0.5,
    )
    ax.plot(
        adjusted_time,
        mean_tolloi[start_index : stop_index + 1],
        color="#F8AE8B", label="$Tollo^{RNAi}$ Mean", linewidth=2,
    )

    if ylims is not None:
        ax.set_ylim(ylims)

    ax.set_ylabel(item)
    pval_ax.set_title(
        f"Temporal Evolution of {item} in $V^{{RNAi}}$ and $Tollo^{{RNAi}}$ Genotypes"
    )
    ax.set_xlabel("Time (h)")

    handles, labels = ax.get_legend_handles_labels()

    pval_ax.imshow(
        p_values.reshape(1, -1),
        aspect="auto",
        cmap=cmap,
        norm=norm,
        extent=[0, len(adjusted_time) - 1, 0, 1],
    )

    from matplotlib.patches import Patch

    significance_elements = [
        Patch(facecolor=cmap(norm(0.0005)), edgecolor="black", label="p < 0.001 (***)"),
        Patch(facecolor=cmap(norm(0.005)), edgecolor="black", label="p < 0.01 (**)"),
        Patch(facecolor=cmap(norm(0.03)), edgecolor="black", label="p < 0.05 (*)"),
        Patch(facecolor=cmap(norm(0.5)), edgecolor="black", label="p ≥ 0.05 (ns)"),
    ]
    handles.extend(significance_elements)
    labels.extend(["p < 0.001 (***)", "p < 0.01 (**)", "p < 0.05 (*)", "p ≥ 0.05 (ns)"])
    ax.legend(handles, labels, loc="best", frameon=True, fontsize="small")

    pval_ax.set_yticks([])
    pval_ax.set_xticks(np.arange(0, len(median_times), subsample_rate))
    pval_ax.set_xticklabels(median_times_subsampled, rotation=90)
    plt.setp(pval_ax.get_xticklabels(), visible=False)

    fig.tight_layout()
    return fig