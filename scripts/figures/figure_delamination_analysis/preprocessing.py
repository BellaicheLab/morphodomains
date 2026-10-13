"""Preprocessing for figure_delamination_analysis.

Opens the pre-generated averaged zarr archives (``average_tolloi_pyap.zarr`` /
``average_vi_pyap.zarr``), treated as fixed inputs: pyanimalprocessing does not
generate Lagrangian grids.
"""

import logging
import os

import zarr

logger = logging.getLogger(__name__)


def load_delamination_data(paths, fig_config):
    """Open the averaged zarr data for the delamination panels.

    Parameters
    ----------
    paths : dict
        Resolved paths from config (uppercase keys).  The keys
        ``AVERAGES_PYAP_TOLLOI`` and ``AVERAGES_PYAP_VI`` are read from the
        top-level ``paths:`` section of ``figure_config.yaml``.
    fig_config : dict
        The ``figure_delamination_analysis`` section from ``figure_config.yaml``.

    Returns
    -------
    dict
        Shared data consumed by panel functions.
    """
    avg_tolloi_path = paths["AVERAGES_PYAP_TOLLOI"]
    avg_vi_path = paths["AVERAGES_PYAP_VI"]

    missing = [p for p in (avg_tolloi_path, avg_vi_path) if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            "Averaged delamination zarr store(s) not found: "
            + ", ".join(missing)
            + ". Restore them from the Zenodo data archive."
        )

    logger.info("Opening averaged zarrs:")
    logger.info("  Tolloi: %s", avg_tolloi_path)
    logger.info("  Vi:     %s", avg_vi_path)
    aoa_tolloi = zarr.open(avg_tolloi_path, mode="r")
    aoa_vi = zarr.open(avg_vi_path, mode="r")

    return {
        "aoa_tolloi": aoa_tolloi,
        "aoa_vi": aoa_vi,
    }
