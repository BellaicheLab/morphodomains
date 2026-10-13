"""Panels for the common-frame gene atlas figure."""

import numpy as np
import matplotlib.pyplot as plt


def gene_atlas_maps(shared_data, output_base):
    """Grid of per-gene AOA gene-expression maps on the common archetype grid."""
    gene_maps = shared_data["gene_maps"]
    render_genes = shared_data["_fig_config"].get("pipeline", {}).get("render_genes")
    genes = render_genes or sorted(gene_maps)
    genes = [g for g in genes if g in gene_maps]

    ncol = min(6, len(genes))
    nrow = int(np.ceil(len(genes) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.0 * ncol, 2.0 * nrow), squeeze=False)

    for ax in axes.ravel():
        ax.axis("off")
    for ax, gene in zip(axes.ravel(), genes):
        gmap = gene_maps[gene]
        vmax = np.nanpercentile(gmap, 99) if np.isfinite(gmap).any() else 1.0
        ax.imshow(gmap, cmap="magma", vmin=0, vmax=vmax or 1.0, interpolation="nearest")
        ax.set_title(gene, fontsize=7)
        ax.axis("off")

    fig.suptitle("Common-frame gene atlas (per-gene AOA)", fontsize=9)
    fig.tight_layout()
    return fig
