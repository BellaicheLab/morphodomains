"""Preprocessing for figure_morphodynamics_comparison.

Opens zarr groups, loads tensor backups, images, spatial transcriptome,
computes Moran data (or loads from cache), and assembles the shared_data
dict consumed by panel functions.
"""

import logging
from pathlib import Path

import numpy as np
import zarr
from pyanimalprocessing.rescaling.geometry import movie_frame_coordinate_system
from pyanimalprocessing.rescaling.timings import find_frame

from scripts.figures import is_panel_enabled
from scripts.figures.config import aoa_backup

from ._helpers import (
    compute_moran_data,
    generate_atlas_advection_pattern,
    generate_cluster_advection_pattern,
    generate_gene_advection_pattern,
    load_moran_cache,
    save_moran_cache,
)

logger = logging.getLogger(__name__)


def load_morphodynamics_comparison_data(paths, fig_config):
    """Load and preprocess all data needed by figure_morphodynamics_comparison panels.

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
    output_base = Path(paths["OUTPUT_BASE"]) / fig_config["output_dir"]

    # ── Open zarr groups ────────────────────────────────────────────────────
    animal = zarr.open_group(paths["MOVIE_ANIMAL_PATH"], "a")
    avgAnimal = zarr.open(paths["AVERAGES_241002_EULERIAN"])
    avgAnimal_lagrangian = zarr.open(paths["AVERAGES_240920_LAGRANGIAN"])

    # ── Tensor backups ──────────────────────────────────────────────────────
    AOA_vi = aoa_backup(avgAnimal, "AOA_avgAnimals_vi_2h_0.96olap")
    AOA_Tolloi = aoa_backup(avgAnimal, "AOA_avgAnimals_tolloi_2h_0.96olap")
    DBA_Tolloi_vi = aoa_backup(avgAnimal, "DBA_avgAnimals_Tolloi-avgAnimals_vi_2h_0.96olap")

    # ── Images ──────────────────────────────────────────────────────────────
    outlines = animal.IMAGE.D2.outlines

    # ── Spatial transcriptome ───────────────────────────────────────────────
    wt15h_zarr = animal.GEP.SPATIAL_TRANSCRIPTOME_15APF

    # ── Geometric parameters ────────────────────────────────────────────────
    overlap = np.array([0.5, 0.5, 0])
    # Archetype box size in movie pixels (boxWidth/xFactor, boxHeight/yFactor):
    # the same SAP rescaling read_property applies to place the average grid on this movie.
    _, _, box_xyz, _ = movie_frame_coordinate_system(DBA_Tolloi_vi, reference_animal=animal)
    boxsizes = np.array([box_xyz[:2]])
    xy_starts_grid = np.array(
        [wt15h_zarr.attrs["grid_xStart"], wt15h_zarr.attrs["grid_yStart"]]
    )
    xy_starts = np.array([[animal.attrs["xStart"], animal.attrs["yStart"]]])

    dt = animal.attrs["dt"]
    temperature = animal.attrs["temperature"]
    frame_ref_old = animal.attrs["frameRef"]
    time_ref = animal.attrs["timeRef"]

    # ── Thresholds from config ──────────────────────────────────────────────
    thresholds = fig_config.get("thresholds", {})
    pval_threshold_roi = thresholds.get("pval_threshold_roi", 0.05)
    contraction_threshold = thresholds.get("contraction_threshold", -0.02)
    roi_component_deformation = thresholds.get("roi_component_deformation", "largest")
    roi_component_velocity = thresholds.get("roi_component_velocity", "largest")

    # ── Moran computation (or cache load) ───────────────────────────────────
    cache_config = fig_config.get("cache", {})
    use_cache = cache_config.get("use_moran_cache", True)
    regenerate = cache_config.get("regenerate_cache", False)
    cache_path = str(output_base / "cache" / "moran_data.npz")

    moran_data = None
    if use_cache and not regenerate:
        moran_data = load_moran_cache(cache_path)

    if moran_data is None:
        moran_data = compute_moran_data(avgAnimal, animal)
        if use_cache:
            save_moran_cache(moran_data, cache_path)

    # ── Pre-compute advection patterns if needed ────────────────────────────
    panels_cfg = fig_config.get("panels", {})
    tollo_pattern = None
    atlas_tollo_pattern = None
    cluster_pattern = None

    # Check if any overlay-requiring exports are enabled.
    # In the scene/exports model, overlays are controlled by enabling
    # specific export entries whose ``layers:`` list includes the overlay.
    def _has_enabled_export_with_layer(panel_name, layer_name):
        """Check if a panel has an enabled export referencing *layer_name*."""
        pcfg = panels_cfg.get(panel_name, {})
        if not is_panel_enabled(fig_config, panel_name):
            return False
        for export_cfg in pcfg.get("exports", {}).values():
            if not export_cfg.get("enabled", True):
                continue
            export_layers = export_cfg.get("layers") or []
            if layer_name in export_layers:
                return True
        return False

    needs_tollo = _has_enabled_export_with_layer(
        "deformation_difference", "Tollo Expression"
    ) or _has_enabled_export_with_layer("velocity_difference", "Tollo Expression")
    needs_atlas_tollo = _has_enabled_export_with_layer(
        "deformation_difference", "Atlas Tollo Expression"
    ) or _has_enabled_export_with_layer("velocity_difference", "Atlas Tollo Expression")
    needs_clusters = _has_enabled_export_with_layer(
        "deformation_difference", "Morphogenetic Clusters"
    ) or _has_enabled_export_with_layer("velocity_difference", "Morphogenetic Clusters")

    if needs_tollo or needs_atlas_tollo or needs_clusters:
        # Advection reference frame (16h00)
        advection_frame_ref = int(
            np.array(find_frame(frame_ref_old, time_ref, temperature, "16h00", dt))[0]
        )
        tracking_mask = np.array(animal["TRACKING"]["masks_division_agnostic"])

        # Common geometry args shared by all advection helpers
        _adv_kw = dict(
            xy_starts_grid=xy_starts_grid,
            xy_start=xy_starts[0],
            overlap=0.5,
            boxsize=boxsizes[0],
            tracking_mask=tracking_mask,
            frame_ref=advection_frame_ref,
        )

        if needs_tollo:
            logger.info("Pre-computing Tollo advection pattern...")
            tollo_pattern = generate_gene_advection_pattern(
                wt15h_zarr,
                "Tollo",
                **_adv_kw,
            )

        if needs_atlas_tollo:
            logger.info("Pre-computing Atlas Tollo advection pattern...")
            atlas_tollo_pattern = generate_atlas_advection_pattern(
                paths["ATLAS_VALIDATION_TXT"],
                "Tollo",
                **_adv_kw,
            )

        if needs_clusters:
            logger.info("Pre-computing cluster advection pattern...")
            cluster_labels_path = (
                Path(paths["OUTPUT_BASE"])
                / "fig_morphodynamic_domains"
                / "clustering"
                / "cluster_labels_5.csv"
            )
            cluster_pattern = generate_cluster_advection_pattern(
                cluster_labels_path,
                **_adv_kw,
            )

    # ── Assemble shared_data ────────────────────────────────────────────────
    return {
        # Zarr groups
        "animal": animal,
        "avgAnimal_lagrangian": avgAnimal_lagrangian,
        # Tensor backups
        "AOA_vi": AOA_vi,
        "AOA_Tolloi": AOA_Tolloi,
        "DBA_Tolloi_vi": DBA_Tolloi_vi,
        # Images
        "outlines": outlines,
        # Geometric parameters
        "boxsizes": boxsizes,
        "xy_starts_grid": xy_starts_grid,
        "xy_starts": xy_starts,
        "overlap": overlap,
        # Moran data
        "moran_data": moran_data,
        # Advection patterns
        "tollo_pattern": tollo_pattern,
        "atlas_tollo_pattern": atlas_tollo_pattern,
        "cluster_pattern": cluster_pattern,
        # Thresholds
        "pval_threshold_roi": pval_threshold_roi,
        "contraction_threshold": contraction_threshold,
        "roi_component_deformation": roi_component_deformation,
        "roi_component_velocity": roi_component_velocity,
    }
