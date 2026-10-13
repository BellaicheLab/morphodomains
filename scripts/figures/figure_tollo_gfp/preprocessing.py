"""Preprocessing for figure_tollo_gfp.

Opens the TIF zarr animals and the reference zarr, and loads advected ROI
contours from the reference's ``SPACE_REGISTRATION`` groups.

The actual image warping is handled by napari via the affine transform
stored in each zarr (``GRID/EULERIAN/GRID_PROPERTIES/matrix_transform``).
"""

import logging

import numpy as np
import zarr

from ._helpers import TIMEPOINTS

logger = logging.getLogger(__name__)


def _load_roi_contour(reference, tp_key, roi_name, suffix=""):
    """Load an advected ROI contour from the reference zarr.

    Parameters
    ----------
    suffix : str
        Appended to the registration name (e.g. ``"_grid"`` for the
        grid-snapped variant).

    Returns ``None`` if the registration group does not exist.
    """
    reg_name = f"tollo_gfp_{tp_key}_{roi_name}{suffix}"
    try:
        contour = np.array(reference.SPACE_REGISTRATION[reg_name].rois)
        logger.info(
            "Loaded ROI '%s%s' for %s: %d points",
            roi_name,
            suffix,
            tp_key,
            contour.shape[1] if contour.ndim == 3 else contour.shape[0],
        )
        return contour
    except (KeyError, AttributeError):
        logger.warning(
            "ROI '%s%s' not found for timepoint %s", roi_name, suffix, tp_key
        )
        return None


def load_tollo_gfp_data(paths, fig_config):
    """Load zarr animals and ROI contours for the Tollo:GFP figure.

    Parameters
    ----------
    paths : dict
        Resolved paths from config (uppercase keys).
    fig_config : dict
        The ``figure_tollo_gfp`` section from ``figure_config.yaml``.

    Returns
    -------
    dict
        Shared data consumed by scene builders:
        - ``ref_zarr``: opened reference zarr (read-only)
        - ``tif_zarrs``: dict mapping tp_key → opened TIF zarr
        - ``roi_contours``: dict mapping tp_key → {roi_name: contour}
        - ``grid_type``: grid type string (e.g. "eulerian")
        - ``gfp_channel``: GFP channel index
    """
    gfp_channel = fig_config.get("gfp_channel", 1)
    grid_type = fig_config.get("grid_type", "eulerian")
    roi_names = fig_config.get("roi_names", ["flow", "extended_compression"])

    # Reference animal (pyanimalprocessing zarr — has segmentation + ROI data)
    ref_zarr_path = paths["REF_ZARR_PYAP"]
    ref_zarr = zarr.open(ref_zarr_path, "r")
    logger.info("Reference: %s", ref_zarr.attrs.get("name", ref_zarr_path))

    # Load TIF zarrs and ROI contours per timepoint
    tif_zarrs = {}
    roi_contours = {}
    roi_grid_contours = {}

    for tp_key in TIMEPOINTS:
        panel_name = f"tollo_gfp_{tp_key}"
        enabled_panels = fig_config.get("enabled_panels", {})
        if not enabled_panels.get(panel_name, True):
            logger.info("Skipping disabled panel: %s", panel_name)
            continue

        zarr_path_key = f"TOLLO_GFP_{tp_key.upper()}_ZARR"
        zarr_path = paths.get(zarr_path_key)
        if zarr_path is None:
            logger.warning("Path key '%s' not found in config", zarr_path_key)
            continue

        try:
            tif_zarr = zarr.open(zarr_path, "r")
            tif_zarrs[tp_key] = tif_zarr
            logger.info(
                "Opened %s: IMAGE.D2.raw shape %s",
                tp_key,
                tif_zarr.IMAGE.D2.raw.shape,
            )
        except Exception as e:
            logger.warning("Cannot open zarr %s: %s", zarr_path, e)
            continue

        # Load advected ROI contours (stored in reference zarr)
        tp_rois = {}
        tp_grid_rois = {}
        for roi_name in roi_names:
            contour = _load_roi_contour(ref_zarr, tp_key, roi_name)
            if contour is not None:
                tp_rois[roi_name] = contour
            grid_contour = _load_roi_contour(
                ref_zarr, tp_key, roi_name, suffix="_grid"
            )
            if grid_contour is not None:
                tp_grid_rois[roi_name] = grid_contour
        roi_contours[tp_key] = tp_rois
        roi_grid_contours[tp_key] = tp_grid_rois

    return {
        "ref_zarr": ref_zarr,
        "tif_zarrs": tif_zarrs,
        "roi_contours": roi_contours,
        "roi_grid_contours": roi_grid_contours,
        "grid_type": grid_type,
        "gfp_channel": gfp_channel,
    }
