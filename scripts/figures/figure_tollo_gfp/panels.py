"""Napari scene builder for figure_tollo_gfp.

A single scene displays the reference (vi) tissue and all 4 registered
Tollo:GFP images, with advected ROI contours overlaid.  The temporal
affine transform from ``plot_rescaling`` aligns each TIF's single frame
to the correct timepoint on the reference movie.

YAML exports select the specific viewer frame for each timepoint.
"""

import logging

import numpy as np
from pyanimalprocessing.plotting.property_plotting import plot_rescaling

from ._helpers import CONTRAST_LIMITS, ROI_GRID_STYLES, ROI_STYLES, TIMEPOINTS

logger = logging.getLogger(__name__)


def _find_layer_affine(viewer, zarr_name):
    """Find the affine of a layer placed by plot_rescaling for a given zarr."""
    for layer in viewer.layers:
        if zarr_name in layer.name:
            return layer.affine.affine_matrix
    return None


def build_tollo_gfp_scene(viewer, shared_data):
    """Build a napari scene with all Tollo:GFP timepoints.

    Layers added (bottom → top):
    1. Reference vi image channels (affine-registered, multi-frame)
    2. All TIF image channels (affine-registered, each placed at its timepoint)
    3. ROI contour shapes for each timepoint
    """
    ref_zarr = shared_data["ref_zarr"]
    tif_zarrs = shared_data["tif_zarrs"]
    roi_contours = shared_data["roi_contours"]
    roi_grid_contours = shared_data.get("roi_grid_contours", {})
    grid_type = shared_data["grid_type"]

    # Collect all animals: reference first, then TIF zarrs in timepoint order
    animals = [ref_zarr]
    for tp_key in TIMEPOINTS:
        tif_zarr = tif_zarrs.get(tp_key)
        if tif_zarr is not None:
            animals.append(tif_zarr)

    # Add all images in registered space (spatial + temporal alignment)
    plot_rescaling(
        animals,
        dimension="D2",
        grid_type=grid_type,
        viewer=viewer,
        show_time_legend=True,
        lazy_load=True,
    )

    # Set per-timepoint contrast limits
    for tp_key, clims in CONTRAST_LIMITS.items():
        tif_zarr = tif_zarrs.get(tp_key)
        if tif_zarr is None:
            continue
        zarr_name = tif_zarr.attrs["name"]
        for layer in viewer.layers:
            for channel, (cmin, cmax) in clims.items():
                if channel in layer.name and zarr_name in layer.name:
                    layer.contrast_limits = (cmin, cmax)

    # Add ROI contour shapes for each timepoint.
    # Reuse the exact affine that plot_rescaling assigned to each TIF layer
    # so the ROI lands at the same viewer position as the TIF image.
    for tp_key in TIMEPOINTS:
        time_str, label = TIMEPOINTS[tp_key]
        tp_rois = roi_contours.get(tp_key, {})
        tif_zarr = tif_zarrs.get(tp_key)
        if tif_zarr is None or not tp_rois:
            continue

        # Build a time-only affine: take the TIF layer's time row (row 0)
        # but keep spatial as identity (ROI coords are in reference space).
        zarr_name = tif_zarr.attrs["name"]
        tif_affine = _find_layer_affine(viewer, zarr_name)
        if tif_affine is None:
            logger.warning("No layer found for %s, skipping ROI", zarr_name)
            continue

        roi_affine = np.eye(4)
        roi_affine[0, :] = tif_affine[0, :]  # time scale + offset only

        for roi_name, contour in tp_rois.items():
            style = ROI_STYLES.get(roi_name, {})
            roi_points = contour[0]  # (N, 2) — (row, col)
            timed_points = np.column_stack(
                [
                    np.zeros(roi_points.shape[0]),  # frame 0 of the TIF
                    roi_points,
                ]
            )  # (N, 3) — (t, row, col)

            viewer.add_shapes(
                [timed_points],
                shape_type="polygon",
                name=f"ROI {roi_name} ({label})",
                affine=roi_affine,
                **style,
            )

        # Grid-snapped ROI contours (same affine, different style)
        tp_grid_rois = roi_grid_contours.get(tp_key, {})
        for roi_name, contour in tp_grid_rois.items():
            grid_style = ROI_GRID_STYLES.get(roi_name, {})
            roi_points = contour[0]  # (N, 2) — (row, col)
            timed_points = np.column_stack(
                [
                    np.zeros(roi_points.shape[0]),
                    roi_points,
                ]
            )

            viewer.add_shapes(
                [timed_points],
                shape_type="polygon",
                name=f"ROI {roi_name} grid ({label})",
                affine=roi_affine,
                **grid_style,
            )

    logger.info("Tollo:GFP scene ready with %d animals", len(animals))
