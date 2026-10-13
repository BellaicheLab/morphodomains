"""Spatial transcriptome visualization panel for figure_spatial_transcriptome.

Panels:
- spatial_transcriptome: Additive-blend napari-style visualization of the vGEP
"""

import logging

import matplotlib.pyplot as plt
import napari.utils.colormaps
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from vgep.utils import to_matrix_form

logger = logging.getLogger(__name__)


def spatial_transcriptome(shared_data, output_base):
    """Visualize the spatial transcriptome using additive blending.

    Uses pre-computed transcriptome from preprocessing.
    """
    st_df = shared_data["st_transcriptome_df"]
    geometry = shared_data["geometry"]
    macrochaete_path = shared_data["macrochaete_path"]
    genes_to_show = shared_data["genes_to_show"]
    vbounds = shared_data["vbounds"]
    napari_colors = shared_data["napari_colors"]

    fig, ax = plt.subplots(figsize=(2, 2), dpi=300)

    base_shape = to_matrix_form(st_df[genes_to_show[0]], geometry=geometry).shape
    combined_image = np.zeros((*base_shape, 4), dtype=np.float32)
    valid_mask = np.ones(base_shape, dtype=bool)

    first_matrix = to_matrix_form(st_df[genes_to_show[0]], geometry=geometry)
    valid_mask = ~np.isnan(first_matrix)

    for gene, color_name, (vmin, vmax) in zip(genes_to_show, napari_colors, vbounds):
        if gene not in st_df.columns:
            logger.debug("Gene %s not found; skipping.", gene)
            continue

        matrix_form = to_matrix_form(st_df[gene], geometry=geometry)
        valid_mask &= ~np.isnan(matrix_form)

        normalized = np.clip((matrix_form - vmin) / (vmax - vmin), 0, 1)
        cmap = napari.utils.colormaps.AVAILABLE_COLORMAPS[color_name]
        rgba = cmap.map(normalized)

        combined_image[..., :3] += rgba[..., :3] * rgba[..., 3][..., np.newaxis]
        combined_image[..., 3] = np.maximum(combined_image[..., 3], rgba[..., 3])

    combined_image[~valid_mask] = 0
    np.clip(combined_image, 0, 1, out=combined_image)

    ax.imshow(combined_image)
    ax.axis("off")
    ax.set_ylim(23, 3)
    ax.set_xlim(14, 47)

    # Legend
    legend_elements = []
    for gene, color_name in zip(genes_to_show, napari_colors):
        if gene not in st_df.columns:
            continue
        cmap = napari.utils.colormaps.AVAILABLE_COLORMAPS[color_name]
        color = cmap.map([1.0])[0]
        legend_elements.append(Patch(facecolor=color, label=gene))

    ax.legend(
        handles=legend_elements,
        loc="center left", bbox_to_anchor=(1.02, 0.5),
        fontsize=4, frameon=False, title="Genes", title_fontsize=5,
    )

    # Macrochaete markers
    POINTS_FACE_COLOR = np.array([1.0, 0.0, 0.0, 1.0])
    POINTS_EDGE_COLOR = np.array([0.41176471, 0.41176471, 0.41176471, 1.0])
    macrochaete = pd.read_csv(macrochaete_path, header=None)
    for m in range(macrochaete.shape[1]):
        circ = plt.Circle(
            (macrochaete.iloc[0, m], macrochaete.iloc[1, m]),
            0.3, ec=POINTS_EDGE_COLOR, fc=POINTS_FACE_COLOR,
        )
        ax.add_patch(circ)

    plt.tight_layout()
    return fig
