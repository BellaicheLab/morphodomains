"""Atlas gene transforms panel for figure_spatial_transcriptome.

Panels:
- atlas_gene_transforms: Atlas genes viewed with/without projective transform
  (multi-figure, self-managed)
"""

import logging
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.io
from skimage.transform import ProjectiveTransform, resize, warp

from vgep.plotter.palettes import INFERNO_CROPPED
from vgep.processing import normalize_01

from scripts.figures import save_and_close

from .preprocessing import PROJECTIVE_TRANSFORM_MATRIX

logger = logging.getLogger(__name__)


def atlas_gene_transforms(shared_data, output_base):
    """Atlas genes viewed as original + projective-transformed images.

    Reads atlas .mat files from GEP_RESULTS_BASE, applies a common mask,
    and optionally applies a projective transformation.

    Checks fig_config.preprocessing.run_atlas_generation and run_transform_atlas.
    Self-managed: saves each figure internally and returns None.
    """
    paths = shared_data["_paths"]
    fig_config = shared_data["_fig_config"]
    preproc_cfg = fig_config.get("preprocessing", {})

    if not preproc_cfg.get("run_atlas_generation", False):
        logger.info("Atlas generation disabled (run_atlas_generation=false). Skipping.")
        return None

    run_transform = preproc_cfg.get("run_transform_atlas", False)

    base_dir = Path(paths["GEP_RESULTS_BASE"])
    fields = ["Gene_dotseg", "Gene_bg50", "Gene_bg", "Gene"]

    all_signals = []
    folder_names = []

    # First pass: collect all signals
    for folder in base_dir.glob("**/AOA*"):
        mat_file = folder / "AOA_backup.mat"
        if not mat_file.exists():
            continue

        try:
            data = scipy.io.loadmat(str(mat_file))
            signal = None
            for field in fields:
                if field in data:
                    signal = data[field][:, :, 0, 0, 0]
                    break

            if signal is None:
                logger.warning("No valid field found in %s", folder)
                continue

            signal = resize(
                signal,
                (signal.shape[0] * 15, signal.shape[1] * 15),
                preserve_range=True,
                order=0,
            )
            all_signals.append(signal)
            folder_names.append(folder.name)

        except Exception as e:
            logger.warning("Error processing %s: %s", folder, e)

    if not all_signals:
        logger.warning("No atlas signals found. Skipping.")
        return None

    # Common mask
    masks = [~np.isnan(sig) for sig in all_signals]
    common_mask = np.logical_and.reduce(masks)

    rows = np.any(common_mask, axis=1)
    cols = np.any(common_mask, axis=0)
    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]

    rectangle_mask = np.zeros_like(all_signals[0], dtype=bool)
    rectangle_mask[y_min - 10 : y_max + 10, x_min - 10 : x_max + 10] = True

    rectangle_image = np.empty_like(all_signals[0])
    rectangle_image[:] = np.nan
    rectangle_image[rectangle_mask] = 0

    masked_signals = []
    for sig in all_signals:
        masked_sig = sig.copy()
        masked_sig[~common_mask] = np.nan
        masked_sig[common_mask] = sig[common_mask]
        masked_signals.append(masked_sig)

    # Original (masked) figures
    for masked_signal, folder_name in zip(masked_signals, folder_names):
        fig, ax = plt.subplots(figsize=(8, 6))
        normalized = normalize_01(masked_signal)
        ax.imshow(rectangle_image, cmap="grey", alpha=1.0, vmin=0, vmax=1)
        im = ax.imshow(normalized, cmap=INFERNO_CROPPED, vmin=0, vmax=1)
        fig.colorbar(im, ax=ax, shrink=0.8)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis("off")
        ax.set_title("")

        clean_name = folder_name.replace(" ", "_").replace("/", "_").replace("\\", "_")
        path = os.path.join(output_base, f"atlas_visualizations/original/{clean_name}")
        save_and_close(fig, path)

    # Transformed figures
    if run_transform:
        transform = ProjectiveTransform(PROJECTIVE_TRANSFORM_MATRIX)

        for masked_signal, folder_name in zip(masked_signals, folder_names):
            h, w = masked_signal.shape
            corners = np.array([[0, 0], [0, h], [w, h], [w, 0]])
            transformed_corners = transform(corners)

            min_xy = np.floor(transformed_corners.min(axis=0)).astype(int)
            max_xy = np.ceil(transformed_corners.max(axis=0)).astype(int)
            out_shape = (max_xy[1] - min_xy[1], max_xy[0] - min_xy[0])

            warped_signal = warp(
                masked_signal, transform, output_shape=out_shape,
                cval=np.nan, preserve_range=True,
            )
            warped_rectangle = warp(
                rectangle_image, transform, output_shape=out_shape,
                cval=np.nan, preserve_range=True,
            )

            fig, ax = plt.subplots(figsize=(8, 6))
            normalized_warped = normalize_01(warped_signal)
            ax.imshow(warped_rectangle, cmap="grey", alpha=1.0, vmin=0, vmax=1)
            im = ax.imshow(normalized_warped, cmap=INFERNO_CROPPED, vmin=0, vmax=1)
            fig.colorbar(im, ax=ax, shrink=0.8)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.axis("off")
            ax.set_title("")
            ax.set_xlim(200, 679)
            ax.set_ylim(215, 125)

            clean_name = folder_name.replace(" ", "_").replace("/", "_").replace("\\", "_")
            path = os.path.join(output_base, f"atlas_visualizations/transformed/{clean_name}")
            save_and_close(fig, path)

    return None
