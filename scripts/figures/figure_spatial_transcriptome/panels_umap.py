"""UMAP panel functions for figure_spatial_transcriptome.

Panels:
- umap_cell_types: UMAP coloured by cell type
- marker_genes: UMAPs of marker gene expression (multi-figure, self-managed)
"""

import logging
import os

import matplotlib.pyplot as plt
import scanpy as sc
from vgep.plotter.palettes import INFERNO_CROPPED
from vgep.processing import normalize_01

from scripts.figures import save_and_close

logger = logging.getLogger(__name__)


def umap_cell_types(shared_data, output_base):
    """UMAP of single cells coloured by cell type annotation."""
    adata = shared_data["adata"]

    colors = list(plt.cm.tab10.colors)
    colors[4] = (0.8, 0.8, 0.8)

    fig, ax = plt.subplots(figsize=(6, 6))

    sc.pl.umap(
        adata,
        color="cell_type",
        frameon=True,
        show=False,
        ax=ax,
        title="Cell Types",
        legend_loc="on data",
        legend_fontsize=8,
        size=2.5,
        palette=colors,
    )

    ax.set_box_aspect(1)
    ax.set_xlabel("UMAP n\u00b01", fontsize=7)
    ax.set_ylabel("UMAP n\u00b02", fontsize=7)
    ax.tick_params(axis="both", which="major", labelsize=5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title("")
    plt.tight_layout()

    return fig


def marker_genes(shared_data, output_base):
    """UMAPs of marker gene expression. Saves one figure per gene."""
    adata = shared_data["adata"]
    marker_categories = shared_data["marker_categories"]

    for category_name, genes in marker_categories.items():
        for gene_name in genes:
            if gene_name not in adata.var_names:
                logger.warning("Gene %s not found. Skipping.", gene_name)
                continue

            fig, ax = plt.subplots(figsize=(2.25, 2.25))

            expression_values = adata[:, gene_name].X.toarray().flatten()
            normalized_values = normalize_01(expression_values)
            umap_coords = adata.obsm["X_umap"]

            scatter = ax.scatter(
                umap_coords[:, 0],
                umap_coords[:, 1],
                c=normalized_values,
                s=4.0,
                cmap=INFERNO_CROPPED,
                vmin=0,
                vmax=1,
                rasterized=True,
                edgecolors="none",
            )

            ax.set_box_aspect(1)

            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label("Normalized Expression", rotation=270, labelpad=15, fontsize=10)
            cbar.ax.tick_params(labelsize=7)

            ax.set_xlabel("UMAP n\u00b01", fontsize=7)
            ax.set_ylabel("UMAP n\u00b02", fontsize=7)
            ax.set_title(f"{gene_name}", fontsize=7, pad=10)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.tick_params(labelsize=5)

            plt.tight_layout()

            path = os.path.join(
                output_base, f"marker_genes/svg/{category_name}_{gene_name}"
            )
            save_and_close(fig, path)

    return None
