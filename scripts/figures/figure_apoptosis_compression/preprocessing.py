"""Preprocessing for figure_apoptosis_compression.

Opens zarr groups, loads tensor backups, figure_morphodynamics_comparison cache data, and
assembles the shared_data dict consumed by panel functions.
"""

import logging
import os
from pathlib import Path

import numpy as np
import zarr

from scripts.figures.config import aoa_backup

logger = logging.getLogger(__name__)


def load_apoptosis_compression_data(paths, fig_config):
    """Load and preprocess all data needed by figure_apoptosis_compression panels.

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
    output_base = Path(paths["OUTPUT_BASE"]) / fig_config["output_dir"]

    avgAnimal_lagrangian = zarr.open(paths["AVERAGES_240920_LAGRANGIAN"])

    # ── figure_morphodynamics_comparison cache (ROI masks and Moran data) ──
    cache_dir = str(output_base.parent / fig_config["comparison_cache_dir"])

    moran_roi_path = os.path.join(cache_dir, "moran_roi.npy")
    restrictive_roi_path = os.path.join(cache_dir, "restrictive_roi.npy")
    if not all(os.path.exists(p) for p in [moran_roi_path, restrictive_roi_path]):
        raise FileNotFoundError(
            f"ROI files not found in {cache_dir}. "
            "Please run figure_morphodynamics_comparison with deformation_difference panel enabled first."
        )
    moran_roi = np.load(moran_roi_path)
    restrictive_roi = np.load(restrictive_roi_path)

    # ── Flow ROI (optional, from velocity_difference scene) ──────────────
    flow_roi_path = os.path.join(cache_dir, "flow_roi.npy")
    if os.path.exists(flow_roi_path):
        indices_flow = np.array(np.where(np.load(flow_roi_path))).T
    else:
        indices_flow = None
        logger.warning(
            "Flow ROI not found at %s. Run figure_morphodynamics_comparison velocity_difference first. "
            "Skipping flow ROI timeseries.", flow_roi_path,
        )

    # ── Tensor quantities present for both genotypes ────────────────────────
    # (T, 1, Ny, Nx, comp, animals) arrays only: skips *_std, weights, time/frame arrays.
    backup_vi = aoa_backup(avgAnimal_lagrangian, "AOA_avgAnimals_vi_2h_0.96olap")
    backup_tolloi = aoa_backup(avgAnimal_lagrangian, "AOA_avgAnimals_tolloi_2h_0.96olap")
    common_items = sorted(
        name for name in set(backup_vi.array_keys()) & set(backup_tolloi.array_keys())
        if backup_vi[name].ndim == 6
        and name not in {"AreaRatios", "weights", "Macrocaetes", "CDCad"}
    )

    time_array = backup_vi.attrs["TimeArray"]

    area_ratios_vi = backup_vi["AreaRatios"][:][:, 0, :, :, 0, 1:]  # (T, Ny, Nx, animals)
    area_ratios_tolloi = backup_tolloi["AreaRatios"][:][:, 0, :, :, 0, 1:]

    # ── Pre-compute ROI indices ─────────────────────────────────────────────
    indices = np.array(np.where(restrictive_roi)).T
    indices_moran = np.array(np.where(moran_roi)).T
    logger.info(
        "ROI indices: restrictive=%d, moran=%d, flow=%s",
        len(indices), len(indices_moran),
        len(indices_flow) if indices_flow is not None else "N/A",
    )

    # ── Assemble shared_data ────────────────────────────────────────────────
    return {
        "avgAnimal_lagrangian": avgAnimal_lagrangian,
        # Data arrays
        "common_items": common_items,
        "time_array": time_array,
        "area_ratios_vi": area_ratios_vi,
        "area_ratios_tolloi": area_ratios_tolloi,
        # ROI indices (pre-computed)
        "indices": indices,
        "indices_moran": indices_moran,
        "indices_flow": indices_flow,
    }
