#!/usr/bin/env python
"""Draw ROIs on the reference animal and advect them to each Tollo:GFP timepoint.

The ROI is defined once at a chosen timepoint on the reference animal, then
**advected** to each target timepoint via cell tracking.  Two contour variants
are stored in the reference zarr (``SPACE_REGISTRATION/``):

1. **Cell-based** (``tollo_gfp_<tp>_<roi_name>/rois``) — the raw contour
   following individual cell edges (jagged).
2. **Grid-snapped** (``tollo_gfp_<tp>_<roi_name>_grid/rois``) — the same ROI
   projected onto the figure-1 Eulerian grid, producing clean rectangular
   boundaries.  The grid coordinate system is loaded from the movie's time-averaged
   Eulerian backup via ``get_coordinate_system_from_backup`` (the transform of
   ``map_points_to_movie``, inverted).

Usage
-----
    python scripts/registration/draw_tollo_gfp_rois.py
"""

import logging
import os
import sys
from pathlib import Path

import numpy as np
import zarr
from pyanimalprocessing.rescaling.geometry import get_coordinate_system_from_backup
from pyanimalprocessing.rescaling.timings import find_frame
from skimage import measure, morphology

_REPO_ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MORPHODOMAINS_DATA", str(_REPO_ROOT.parents[1]))
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.figures.config import get_paths, load_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Time-averaged Eulerian backup providing the grid (same grid as figure_flow_deformation) ──
GRID_NAME = "EGrid_xy_1259.5_612.3_wh_200.2x228.1_nynx_16x36_olap_0.5"
GRID_BACKUP = "Average_2h_12h00_to_34h00_olap_0.96"

# ── ROI configs ──────────────────────────────────────────────────────────────

ROI_CONFIGS = {
    "flow": {
        "roi_image_filename": "flow_roi_image.npy",
        "definition_frame": 67,
        "closing_radius": 3,
    },
    "extended_compression": {
        "roi_image_filename": "moran_roi_image.npy",
        "definition_frame": 67,
        "closing_radius": 3,
    },
    "restrictive_compression": {
        "roi_image_filename": "restrictive_roi_image.npy",
        "definition_frame": 67,
        "closing_radius": 3,
    },
}

TARGET_TIMES = ["15h00", "20h00", "25h00", "28h00"]


# ── Cell-based ROI ─────────────────────────────────────────────────────────


def _identify_cells_in_roi(animal, roi_mask, frame):
    """Return the tracked cell labels falling within *roi_mask* at *frame*."""
    masks = animal["TRACKING"]["masks_division_agnostic"][frame]
    cells = np.unique(masks[roi_mask])
    cells = cells[cells != 0]
    logger.info("Found %d cells within ROI at frame %d", len(cells), frame)
    return cells


def _create_cell_based_roi(animal, cells_to_include, time_str, closing_radius=3):
    """Morphologically closed mask of *cells_to_include* at *time_str*."""
    a = animal.attrs
    frame = int(find_frame(a["frame_ref"], a["time_ref"], a["temperature"], time_str, a["dt"])[0])
    masks = animal["TRACKING"]["masks_division_agnostic"][frame]
    final_roi = np.isin(masks, cells_to_include)
    if closing_radius > 0:
        final_roi = morphology.binary_closing(final_roi, morphology.disk(closing_radius))
    logger.info(
        "Created cell-based ROI at %s (frame %d): %d pixels from %d cells",
        time_str, frame, final_roi.sum(), len(cells_to_include),
    )
    return final_roi.astype(bool)


def _roi_to_contour(roi_mask):
    """Largest contour of *roi_mask* as a napari shape ``(1, N, 2)``."""
    contours = measure.find_contours(roi_mask, 0.5)
    if not contours:
        logger.warning("No contours found in ROI mask")
        return np.empty((0, 2))
    return max(contours, key=len).reshape(1, -1, 2)


def _store_roi_in_animal(animal, roi_contour, registration_name):
    """Store *roi_contour* as ``SPACE_REGISTRATION/<registration_name>/rois``."""
    reg = animal.require_group("SPACE_REGISTRATION").require_group(registration_name)
    if "rois" in reg.keys():
        del reg["rois"]
    reg.create_dataset("rois", data=roi_contour)


# ── Grid snapping ──────────────────────────────────────────────────────────


def _snap_roi_to_grid(roi_mask, ref_backup, threshold=0.5):
    """Snap a cell-based binary ROI to Eulerian grid step boundaries.

    Uses the coordinate system of ``map_points_to_movie``
    (via ``get_coordinate_system_from_backup``), but inverted: the mask is
    projected onto the grid, thresholded per cell, then reconstructed at
    pixel resolution.

    Parameters
    ----------
    roi_mask : np.ndarray
        Boolean mask ``(H, W)`` in reference image space.
    ref_backup : zarr.hierarchy.Group
        Time-averaged backup providing the grid coordinate system.
    threshold : float
        Fraction of a grid cell's clipped area that must overlap the ROI
        for the cell to be included (default 0.5 = majority vote).

    Returns
    -------
    np.ndarray
        Boolean mask of same shape, snapped to grid boundaries.
    """
    start_movie, start_grid, boxsizes, overlaps = (
        get_coordinate_system_from_backup(ref_backup)
    )

    # Step sizes [step_x, step_y, step_t]
    step = boxsizes * (1 - overlaps)
    step_x, step_y = step[0], step[1]

    # Pixel offset of grid cell (0, 0) — same formula as map_points_to_movie
    offset = start_movie - start_grid * step - step / 2
    offset_x, offset_y = offset[0], offset[1]

    h, w = roi_mask.shape

    # Cell (j, i) covers rows [offset_y + j*step_y, +step_y)
    #                    cols [offset_x + i*step_x, +step_x)
    j_min = int(np.floor((0 - offset_y) / step_y))
    j_max = int(np.ceil((h - offset_y) / step_y))
    i_min = int(np.floor((0 - offset_x) / step_x))
    i_max = int(np.ceil((w - offset_x) / step_x))

    snapped = np.zeros_like(roi_mask)
    n_included = 0

    for j in range(j_min, j_max):
        for i in range(i_min, i_max):
            r0 = offset_y + j * step_y
            r1 = r0 + step_y
            c0 = offset_x + i * step_x
            c1 = c0 + step_x

            r0c = max(0, int(np.round(r0)))
            r1c = min(h, int(np.round(r1)))
            c0c = max(0, int(np.round(c0)))
            c1c = min(w, int(np.round(c1)))

            if r0c >= r1c or c0c >= c1c:
                continue

            cell_area = (r1c - r0c) * (c1c - c0c)
            roi_pixels = int(roi_mask[r0c:r1c, c0c:c1c].sum())

            if cell_area > 0 and roi_pixels / cell_area > threshold:
                snapped[r0c:r1c, c0c:c1c] = True
                n_included += 1

    logger.info(
        "Grid snap: %d cells included (step %.1f x %.1f px, threshold %.0f%%)",
        n_included, step_x, step_y, threshold * 100,
    )
    return snapped


# ── ROI advection ───────────────────────────────────────────────────────────


def advect_roi_to_timepoints(
    reference,
    roi_name,
    roi_mask,
    definition_frame,
    target_times,
    closing_radius=3,
    ref_backup=None,
):
    """Advect a ROI mask from *definition_frame* to each target timepoint.

    Stores two contour variants per timepoint:
    - ``tollo_gfp_<tp>_<roi_name>/rois`` — cell-based contour
    - ``tollo_gfp_<tp>_<roi_name>_grid/rois`` — grid-snapped contour
      (only when *ref_backup* is provided)
    """
    cells = _identify_cells_in_roi(reference, roi_mask, definition_frame)

    for time_str in target_times:
        tp_short = time_str.split("h")[0] + "h"
        reg_name = f"tollo_gfp_{tp_short}_{roi_name}"

        logger.info("Advecting ROI '%s' to %s...", roi_name, time_str)
        advected_mask = _create_cell_based_roi(
            reference, cells, time_str, closing_radius
        )

        # Store original cell-based contour
        contour = _roi_to_contour(advected_mask)
        _store_roi_in_animal(reference, contour, reg_name)
        logger.info(
            "  Stored cell-based   → %s (%d points)",
            reg_name, contour.shape[1] if contour.ndim == 3 else 0,
        )

        # Store grid-snapped contour
        if ref_backup is not None:
            snapped_mask = _snap_roi_to_grid(advected_mask, ref_backup)
            snapped_contour = _roi_to_contour(snapped_mask)
            _store_roi_in_animal(reference, snapped_contour, f"{reg_name}_grid")
            logger.info(
                "  Stored grid-snapped → %s_grid (%d points)",
                reg_name,
                snapped_contour.shape[1] if snapped_contour.ndim == 3 else 0,
            )


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    config = load_config()
    paths = get_paths(config)
    ref_zarr_path = paths["REF_ZARR_PYAP"]

    reference = zarr.open(ref_zarr_path, mode="a")
    logger.info("Reference animal: %s", reference.attrs.get("name", "unknown"))

    cache_dir = Path(paths["OUTPUT_BASE"]) / "fig_morphodynamics_comparison" / "cache"
    logger.info("Resolving ROI input files from: %s", cache_dir)

    # Grid coordinate system from the movie's time-averaged Eulerian backup
    movie_zarr = zarr.open_group(paths["MOVIE_ANIMAL_PATH"], mode="r")
    ref_backup = movie_zarr["TIME_AVERAGE"][GRID_NAME]["D2"][GRID_BACKUP]
    logger.info("Grid coordinate system loaded from %s/%s", GRID_NAME, GRID_BACKUP)

    for roi_name, cfg in ROI_CONFIGS.items():
        logger.info("=" * 60)
        logger.info("Processing ROI: %s", roi_name)
        logger.info("=" * 60)

        roi_mask = np.load(cache_dir / cfg["roi_image_filename"]).astype(bool)

        advect_roi_to_timepoints(
            reference=reference,
            roi_name=roi_name,
            roi_mask=roi_mask,
            definition_frame=cfg["definition_frame"],
            target_times=TARGET_TIMES,
            closing_radius=cfg["closing_radius"],
            ref_backup=ref_backup,
        )

    logger.info("All ROIs advected and stored.")


if __name__ == "__main__":
    main()
