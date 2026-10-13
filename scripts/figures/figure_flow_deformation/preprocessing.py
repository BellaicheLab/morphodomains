"""Data loading and preprocessing for figure_flow_deformation.

Loads zarr groups, tensor backups, cleans outlines, and computes
velocity / deformation fields once (shared across all panels).
"""

import logging

import numpy as np
import zarr
from pyanimalprocessing.plotting.property_plotting import read_property
from scripts.figures.config import aoa_backup
from skimage.measure import label

logger = logging.getLogger(__name__)


def _clean_outlines(raw_outline):
    """Remove connected component 1 (background) from outlines.

    Parameters
    ----------
    raw_outline : np.ndarray
        Raw outline array of shape (T, H, W).

    Returns
    -------
    np.ndarray
        Cleaned outline with NaN background.
    """
    clean = np.empty_like(raw_outline, dtype=float)
    clean[:] = np.nan

    for i in range(raw_outline.shape[0]):
        tmp = label(raw_outline[i], connectivity=1)
        mask = tmp != 1
        clean[i][mask] = raw_outline[i][mask]

    # Replace NaN (outside-image) with 1.0 (white) so they blend with
    # the light viewer background.
    np.nan_to_num(clean, copy=False, nan=1.0)

    return clean


def load_flow_deformation_data(paths, fig_config):
    """Load all data needed by figure_flow_deformation panels.

    Parameters
    ----------
    paths : dict
        Resolved paths from config (uppercase keys).
    fig_config : dict
        The figure_flow_deformation config section from figure_config.yaml.

    Returns
    -------
    dict
        Shared data for all figure_flow_deformation panels.
    """
    logger.info("Loading zarr groups...")
    animal = zarr.open_group(paths["MOVIE_ANIMAL_PATH"], "a")
    avgAnimal = zarr.open(paths["AVERAGES_241002_EULERIAN"])

    AOA_vi_4h = aoa_backup(avgAnimal, "AOA_avgAnimals_vi_4h_0.98olap")

    logger.info("Cleaning outlines...")
    raw_outline = np.array(animal.IMAGE.D2.outlines).astype(float)
    clean_outline = _clean_outlines(raw_outline)

    logger.info("Loading raw image...")
    raw_image = np.array(animal.IMAGE.D2.raw)[:, 0].astype(float)  # drop the pyap channel axis

    # The averaged pattern is overlaid onto this movie via its SAP registration
    # (reference_animal supplies the affine; origins are the archetype grid).
    logger.info("Reading velocity fields (UPIV)...")
    velocity_fields = read_property(
        AOA_vi_4h, "UPIV", reference_animal=animal,
        grid_bulk_threshold=0.55, significance=0,
    )

    logger.info("Reading deformation fields (EpsilonPIV)...")
    deformation_fields = read_property(
        AOA_vi_4h, "EpsilonPIV", reference_animal=animal,
        grid_bulk_threshold=0.55, significance=0,
    )

    return {
        "AOA_vi_4h": AOA_vi_4h,
        "clean_outline": clean_outline,
        "raw_image": raw_image,
        "velocity_fields": velocity_fields,
        "deformation_fields": deformation_fields,
        "results_piv_base": paths["RESULTS_PIV_BASE"],
    }
