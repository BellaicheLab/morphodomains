"""Timeseries panels for figure_apoptosis_compression.

Self-managed panels that iterate over tensor quantities and produce
per-quantity figures for each ROI defined in the ``rois:`` section of
figure_config.yaml.
"""

import logging
import os

import numpy as np

from pyanimalprocessing.rescaling.timings import (
    find_closest_time_index,
    median_time,
)
from scripts.figures import save_and_close
from scripts.figures.config import aoa_backup

from ._helpers import plot_results, process_tensor

logger = logging.getLogger(__name__)


def _per_animal(avgAnimal_lagrangian, backup_name, item):
    """Per-animal array as (T, Ny, Nx, comp, animals), or (T, Ny, Nx, animals) for scalars."""
    tensor = aoa_backup(avgAnimal_lagrangian, backup_name)[item][:][:, 0, ..., 1:]
    return tensor[..., 0, :] if tensor.shape[-2] == 1 else tensor


def _process_matrix_item(
    item, tensor_control, tensor_tolloi, indices,
    area_ratios_vi, area_ratios_tolloi, time_array,
    dynamic_op, out_dir, pval_dir,
):
    """Process a 5D matrix tensor item (trace + anisotropy)."""
    for matrix_op in ("trace", "anisotropy"):
        traces_vi, sem_vi = process_tensor(
            tensor_control, indices, area_ratios_vi,
            matrix_op=matrix_op, dynamic_op=dynamic_op,
        )
        traces_tolloi, sem_tolloi = process_tensor(
            tensor_tolloi, indices, area_ratios_tolloi,
            matrix_op=matrix_op, dynamic_op=dynamic_op,
        )

        # Negate Epsilon trace for instantaneous to show compression
        if dynamic_op == "average" and matrix_op == "trace" and item == "Epsilon":
            traces_vi = -traces_vi
            traces_tolloi = -traces_tolloi

        ylims = None
        if dynamic_op == "average":
            if item in ("ES", "EA", "ER", "ED", "EG", "EJ"):
                ylims = (-0.5, 0.5) if matrix_op == "trace" else (0, 0.2)

        label = f"{item}_{matrix_op}"
        fig = plot_results(
            traces_vi, sem_vi, traces_tolloi, sem_tolloi,
            time_array, "16h47", "26h47", label,
            plot_individuals=True,
            ylims=ylims,
            output_json_path=_pval_path(pval_dir, item, dynamic_op, matrix_op),
        )
        save_and_close(fig, os.path.join(out_dir, label), formats=("svg", "png"))


def _process_dn_item(
    item, tensor_control, tensor_tolloi, indices,
    area_ratios_vi, area_ratios_tolloi, time_array,
    dynamic_op, out_dir, pval_dir, avgAnimal_lagrangian,
):
    """Process dnA/dnD items with nCoreRNs normalization."""
    n_cells_reference_control = _per_animal(
        avgAnimal_lagrangian, "AOA_avgAnimals_vi_0h_0olap", "nCoreRNs"
    )
    n_cells_reference_tolloi = _per_animal(
        avgAnimal_lagrangian, "AOA_avgAnimals_tolloi_0h_0olap", "nCoreRNs"
    )
    time_array_instantaneous = aoa_backup(
        avgAnimal_lagrangian, "AOA_avgAnimals_tolloi_0h_0olap"
    ).attrs["TimeArray"]
    median_times = [
        median_time(frame[0], frame[1])
        for frame in time_array_instantaneous
    ]
    closest_idx = find_closest_time_index("16h47", median_times)
    normalizing_factor_control = np.broadcast_to(
        n_cells_reference_control[closest_idx][np.newaxis],
        tensor_control.shape,
    )
    normalizing_factor_tolloi = np.broadcast_to(
        n_cells_reference_tolloi[closest_idx][np.newaxis],
        tensor_tolloi.shape,
    )
    tensor_control_normalized = tensor_control / normalizing_factor_control
    tensor_tolloi_normalized = tensor_tolloi / normalizing_factor_tolloi

    traces_vi, sem_vi = process_tensor(
        tensor_control_normalized, indices, area_ratios_vi,
        dynamic_op=dynamic_op,
    )
    traces_tolloi, sem_tolloi = process_tensor(
        tensor_tolloi_normalized, indices, area_ratios_tolloi,
        dynamic_op=dynamic_op,
    )
    fig = plot_results(
        traces_vi, sem_vi, traces_tolloi, sem_tolloi,
        time_array, "16h47", "26h47", item,
        plot_individuals=True,
        output_json_path=_pval_path(pval_dir, item, dynamic_op, None),
    )
    save_and_close(fig, os.path.join(out_dir, item), formats=("svg", "png"))


def _process_scalar_item(
    item, tensor_control, tensor_tolloi, indices,
    area_ratios_vi, area_ratios_tolloi, time_array,
    dynamic_op, out_dir, pval_dir,
):
    """Process plain scalar items."""
    traces_vi, sem_vi = process_tensor(
        tensor_control, indices, area_ratios_vi,
        dynamic_op=dynamic_op,
    )
    traces_tolloi, sem_tolloi = process_tensor(
        tensor_tolloi, indices, area_ratios_tolloi,
        dynamic_op=dynamic_op,
    )
    fig = plot_results(
        traces_vi, sem_vi, traces_tolloi, sem_tolloi,
        time_array, "16h47", "26h47", item,
        plot_individuals=True,
        output_json_path=_pval_path(pval_dir, item, dynamic_op, None),
    )
    save_and_close(fig, os.path.join(out_dir, item), formats=("svg", "png"))


def _pval_path(pval_dir, item, dynamic_op, matrix_op):
    """Return JSON p-value path for cumulative mode, None otherwise."""
    if dynamic_op != "cumulative":
        return None
    suffix = f"_{matrix_op}" if matrix_op else ""
    return os.path.join(pval_dir, f"{item}_cumulative{suffix}_pvalues.json")


def _plot_rates(shared_data, output_base, dynamic_op):
    """Plot rates for tensor quantities across YAML-configured ROIs.

    Parameters
    ----------
    shared_data : dict
        Shared data from preprocessing.
    output_base : str
        Output directory path.
    dynamic_op : str
        "average" for instantaneous, "cumulative" for cumulative.
    """
    common_items = shared_data["common_items"]
    avgAnimal_lagrangian = shared_data["avgAnimal_lagrangian"]
    area_ratios_vi = shared_data["area_ratios_vi"]
    area_ratios_tolloi = shared_data["area_ratios_tolloi"]
    time_array = shared_data["time_array"]

    op_label = "instantaneous" if dynamic_op == "average" else "cumulative"

    # Read ROI configs from YAML (source of truth)
    fig_cfg = shared_data.get("_fig_config", {})
    roi_configs = fig_cfg.get("rois", {})

    for roi_name, roi_cfg in roi_configs.items():
        indices_key = roi_cfg["indices_key"]
        roi_path = roi_cfg["path"]
        indices = shared_data.get(indices_key)
        if indices is None:
            logger.warning("Skipping ROI %s: %s not available", roi_name, indices_key)
            continue

        out_dir = os.path.join(output_base, roi_path, op_label)
        pval_dir = os.path.join(output_base, "rates", "pvalues")
        os.makedirs(out_dir, exist_ok=True)
        os.makedirs(pval_dir, exist_ok=True)

        logger.info(
            "Processing %s rates for ROI=%s (%d grid positions)",
            op_label, roi_name, len(indices),
        )

        printed_sample_count = False
        for item in common_items:
            logger.info("Processing tensor (%s, %s): %s", op_label, roi_name, item)
            tensor_control = _per_animal(
                avgAnimal_lagrangian, "AOA_avgAnimals_vi_2h_0.96olap", item
            )
            tensor_tolloi = _per_animal(
                avgAnimal_lagrangian, "AOA_avgAnimals_tolloi_2h_0.96olap", item
            )

            if not printed_sample_count:
                logger.info(
                    "Number of samples — control: %d, tolloi: %d",
                    tensor_control.shape[-1], tensor_tolloi.shape[-1],
                )
                printed_sample_count = True

            if len(tensor_control.shape) == 5 and tensor_control.shape[3] == 4:
                _process_matrix_item(
                    item, tensor_control, tensor_tolloi, indices,
                    area_ratios_vi, area_ratios_tolloi, time_array,
                    dynamic_op, out_dir, pval_dir,
                )
            elif item in ("dnA", "dnD"):
                _process_dn_item(
                    item, tensor_control, tensor_tolloi, indices,
                    area_ratios_vi, area_ratios_tolloi, time_array,
                    dynamic_op, out_dir, pval_dir, avgAnimal_lagrangian,
                )
            else:
                _process_scalar_item(
                    item, tensor_control, tensor_tolloi, indices,
                    area_ratios_vi, area_ratios_tolloi, time_array,
                    dynamic_op, out_dir, pval_dir,
                )

    return None


def plot_instantaneous(shared_data, output_base):
    """Plot instantaneous rates for tensor quantities across both ROIs.

    Self-managed panel: saves figures directly and returns None.
    """
    return _plot_rates(shared_data, output_base, dynamic_op="average")


def plot_cumulative(shared_data, output_base):
    """Plot cumulative rates for tensor quantities across both ROIs.

    Self-managed panel: saves figures directly and returns None.
    """
    return _plot_rates(shared_data, output_base, dynamic_op="cumulative")
