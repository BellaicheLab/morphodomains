"""Panels for the neck-landmark atlas agreement."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .. import figsize
from ..figure_clustering_stability.panels import CMAP
from ..figure_gene_atlas.preprocessing import (
    _CANONICAL_ARCHETYPE_YX,
    _CANONICAL_GRID_FRAME,
)
from .preprocessing import _MIN_SPLIT_ANIMALS, _WELL_COVERED, between_gene_baseline


def agreement_map(shared_data, output_base):
    """Where the atlas changes when an anterior landmark is added.

    Each box holds the correlation of its standardized gene profile between the atlas registered
    on the 8 macrochaetae and the one registered on the 8 macrochaetae plus the neck. The
    macrochaetae and the mean neck are drawn on top, so the landmark-free anterior half of the
    tissue is visible against its own agreement. Anterior is to the right and the midline at the
    top, as in the tissue maps of Figures 1 and 2.
    """
    agreement = shared_data["per_box_r"].T  # (A-P, M-L) -> display A-P horizontally
    y_min, y_max, x_min, x_max = _CANONICAL_GRID_FRAME
    archetype = np.asarray(_CANONICAL_ARCHETYPE_YX)
    neck = shared_data["neck_positions"].mean(0)

    rows, cols = np.where(np.isfinite(agreement))
    box_y = (y_max - y_min) / agreement.shape[0]
    box_x = (x_max - x_min) / agreement.shape[1]
    crop = (
        x_min + (cols.min() - 1) * box_x, x_min + (cols.max() + 2) * box_x,
        y_min + (rows.min() - 1) * box_y, y_min + (rows.max() + 2) * box_y,
    )

    fig, ax = plt.subplots(figsize=figsize("1x1"))
    image = ax.imshow(
        agreement, origin="lower", extent=(x_min, x_max, y_min, y_max), aspect="equal",
        cmap=CMAP, vmin=0.90, vmax=1.0,
    )
    # White on the inferno ramp, which runs from black to orange: yellow would vanish in the
    # tissue interior, where almost every box sits.
    ax.scatter(
        archetype[:, 1], archetype[:, 0], s=9, c="white", edgecolor="0.15",
        linewidth=0.4, zorder=3,
    )
    ax.scatter(
        neck[1], neck[0], s=26, marker="P", c="white", edgecolor="0.15",
        linewidth=0.4, zorder=3,
    )
    ax.set_xlim(crop[0], crop[1])
    ax.set_ylim(crop[3], crop[2])  # midline at the top, as in the tissue maps of Figures 1 and 2
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines[:].set_visible(False)

    bar_y = crop[3] - 0.05 * (crop[3] - crop[2])
    ax.plot([crop[0] + 20, crop[0] + 120], [bar_y] * 2, color="k", lw=1.2,
            solid_capstyle="butt")
    # Offset in points, not data units: the vertical axis is inverted.
    ax.annotate("100 µm", xy=(crop[0] + 70, bar_y), xytext=(0, 3),
                textcoords="offset points", ha="center", va="bottom", fontsize=5)

    bar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.02, extend="min")
    bar.set_label("agreement between atlases ($r$)")
    fig.tight_layout()
    return fig


def agreement_by_gene(shared_data, output_base):
    """Every gene's map, compared between the two atlases and between two halves of its animals.

    One row per gene: the filled point is how much the map moves when the neck is added to the
    registration, the open point how much it moves when the atlas is rebuilt from a subset of its
    images instead. The segment between them is the margin by which the neck matters less.
    """
    per_gene = shared_data["per_gene"]
    genes = sorted(shared_data["genes"], key=lambda g: per_gene[g]["r_neck"])
    r_neck = np.array([per_gene[g]["r_neck"] for g in genes])
    r_split = np.array([per_gene[g]["r_split_half"] for g in genes])
    y = np.arange(len(genes))

    fig, ax = plt.subplots(figsize=figsize("2x1"))
    ax.set_axisbelow(True)
    ax.grid(axis="x", linestyle=":", linewidth=0.5, color="0.85")
    paired = np.isfinite(r_split)
    ax.hlines(y[paired], r_split[paired], r_neck[paired], color="0.75", lw=0.7, zorder=2)
    ax.scatter(
        r_split, y, s=11, facecolor="none", edgecolor="0.45", linewidth=0.7, zorder=3,
        label="subsampled images (8 landmarks)",
    )
    ax.scatter(
        r_neck, y, s=13, c="#4C72B0", linewidth=0, zorder=4, label="8 landmarks vs 8 + neck",
    )

    ax.set_yticks(y)
    ax.set_yticklabels(genes, fontstyle="italic")
    ax.set_ylim(-0.8, len(genes) - 0.2)
    ax.set_xlim(0.4, 1.02)
    ax.set_xticks([0.4, 0.6, 0.8, 1.0])
    ax.set_xlabel("correlation between atlases ($r$)")
    ax.tick_params(axis="y", length=0)
    ax.legend(
        frameon=False, fontsize=5, loc="lower left", bbox_to_anchor=(0, 1.002),
        handletextpad=0.3, labelspacing=0.25, borderpad=0.1,
    )
    ax.spines[["top", "right", "left"]].set_visible(False)
    fig.tight_layout()
    return fig


def agreement_distributions(shared_data, output_base):
    """The two correlations as distributions, over the genes where both are defined.

    Adding the neck to the registration is compared with subsampling the images: the same atlas,
    the same genes, the same boxes, perturbed two different ways.
    """
    per_gene = shared_data["per_gene"]
    paired = [g for g in shared_data["genes"] if np.isfinite(per_gene[g]["r_split_half"])]
    values = [
        np.array([per_gene[g]["r_neck"] for g in paired]),
        np.array([per_gene[g]["r_split_half"] for g in paired]),
    ]

    fig, ax = plt.subplots(figsize=figsize("1x1"))
    parts = ax.violinplot(values, positions=[1, 2], widths=0.7, showextrema=False)
    for body, colour in zip(parts["bodies"], ("#4C72B0", "0.55")):
        body.set_facecolor(colour)
        body.set_alpha(0.30)
        body.set_linewidth(0)
        # A correlation cannot exceed 1; the kernel density would otherwise spill past it.
        vertices = body.get_paths()[0].vertices
        vertices[:, 1] = np.minimum(vertices[:, 1], 1.0)

    jitter = np.linspace(-0.10, 0.10, len(paired))
    for x, series, colour in zip((1, 2), values, ("#4C72B0", "0.45")):
        ax.scatter(x + jitter, series, s=8, c=colour, linewidth=0, alpha=0.85, zorder=3)
        ax.plot([x - 0.24, x + 0.24], [np.median(series)] * 2, color="0.2", lw=1.2, zorder=4)

    ax.set_xlim(0.4, 2.6)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["8 landmarks\nvs 8 + neck", "subsampled images\n(8 landmarks)"])
    ax.set_ylabel("correlation between atlases ($r$)")
    ax.tick_params(axis="x", length=0)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def atlas_statistics(shared_data, output_base):
    """Write every number behind the agreement panels to a text file, and render nothing.

    Self-managed panel (returns None).
    """
    per_gene = shared_data["per_gene"]
    genes = shared_data["genes"]
    well = shared_data["well_covered"]
    paired = [g for g in genes if np.isfinite(per_gene[g]["r_split_half"])]
    baseline = between_gene_baseline(shared_data["maps_macro8"], well)

    def summary(values):
        values = np.asarray(values, float)
        return (f"n={len(values):<3d} mean {values.mean():.4f} ± {values.std(ddof=1):.4f}   "
                f"median {np.median(values):.4f}   min {values.min():.4f}   "
                f"max {values.max():.4f}")

    neck_all = [per_gene[g]["r_neck"] for g in genes]
    neck_well = [per_gene[g]["r_neck"] for g in well]
    neck_paired = [per_gene[g]["r_neck"] for g in paired]
    split_paired = [per_gene[g]["r_split_half"] for g in paired]

    lines = [
        "Agreement between the gene atlas built with and without the neck landmark",
        "=" * 78,
        "",
        "Both atlases are built from the same animals, genes and grid boxes; they differ",
        "only in whether the neck constrained the per-animal rescaling. The 9-landmark",
        "archetype is the reference 8 macrochaetae plus the mean of the held-out necks, so",
        "both builds tile the identical 40 x 67 boxes.",
        "",
        f"Animals            {shared_data['n_atlas_animals']}"
        f"   (atlas stores carrying a neck landmark and a gene channel)",
        f"Genes              {len(genes)}"
        f"   ({len(well)} with >= {_WELL_COVERED} animals, {len(paired)} with >= "
        f"{_MIN_SPLIT_ANIMALS})",
        "",
        "SUMMARY",
        "-" * 78,
        f"8 vs 8+neck, all genes        {summary(neck_all)}",
        f"8 vs 8+neck, >= {_WELL_COVERED} animals     {summary(neck_well)}",
        "",
        f"8 vs 8+neck, paired subset    {summary(neck_paired)}",
        f"split-half floor, same genes  {summary(split_paired)}",
        f"  neck agreement exceeds the split-half floor for "
        f"{int((np.array(neck_paired) > np.array(split_paired)).sum())}/{len(paired)} genes",
        "",
        f"different genes (baseline)    {summary(baseline)}",
        f"  {int((np.abs(baseline) > 0.9).sum())}/{len(baseline)} gene pairs exceed |r| = 0.9",
        "",
        f"per-box gene profile r        median {shared_data['median_per_box_r']:.4f}"
        f"   over {shared_data['n_boxes_evaluated']} boxes",
        f"  restricted to the {shared_data['n_boxes_full']} boxes carrying >= 95% of the genes"
        f"   median {shared_data['median_r_full']:.4f}",
        f"  the {shared_data['n_weak_boxes']} boxes below r = 0.9 carry a median of "
        f"{shared_data['median_genes_weak']:.0f} genes, against "
        f"{shared_data['median_genes_evaluated']:.0f} overall: they are the partially covered"
        " tissue border",
        "",
        "ALONG THE ANTERO-POSTERIOR AXIS",
        "-" * 78,
        f"macrochaetae span {shared_data['macro_x_range'][0]:.0f} to "
        f"{shared_data['macro_x_range'][1]:.0f} um; mean neck at "
        f"{shared_data['neck_x']:.0f} um; "
        f"{shared_data['frac_boxes_anterior'] * 100:.0f}% of the boxes lie anterior of the last"
        " macrochaeta",
        f"{'region':<32}{'boxes':>7}{'median r':>11}{'covered':>9}{'median r':>11}",
        *[
            f"{region['label']:<32}{region['n']:>7}{region['median']:>11.4f}"
            f"{region['n_full']:>9}{region['median_full']:>11.4f}"
            for region in shared_data["ap_regions"]
        ],
        "",
        "PER GENE",
        "-" * 78,
        f"{'gene':<10}{'animals':>8}{'boxes':>8}{'r_8_vs_9':>11}{'r_split_half':>14}",
    ]
    for gene in sorted(genes, key=lambda g: -per_gene[g]["r_neck"]):
        stats = per_gene[gene]
        split = (f"{stats['r_split_half']:.4f}" if np.isfinite(stats["r_split_half"])
                 else "-")
        lines.append(
            f"{gene:<10}{stats['n_animals']:>8}{stats['n_boxes']:>8}"
            f"{stats['r_neck']:>11.4f}{split:>14}"
        )

    path = Path(output_base) / "atlas_statistics.txt"
    path.write_text("\n".join(lines) + "\n")
    return None
