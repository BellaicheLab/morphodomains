"""Shared plotting helpers for figure_spatial_transcriptome.

- Correlation bar plots (validation + training)
- Reconstruction comparison grids
- vGEP vs Atlas comparison figures
- CDF and survival plots
- Protein vs Fish CDF comparison

All panels rely on the framework style (rcParams + figsize presets from
apply_style); they do not hardcode fonts, dpi, or on-figure text annotations.
"""

import logging

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from vgep.utils import to_matrix_form
from vgep.processing import normalize_01

from scripts.figures import figsize

logger = logging.getLogger(__name__)

from .preprocessing import METHOD_SPECS


def create_correlation_bar_plot(corr, our_color, size):
    """Vertical grouped bar plot of per-gene correlations, one bar per method.

    Parameters
    ----------
    corr : dict
        A 'validation'/'training' sub-dict: method key -> sorted
        correlation array, plus 'genes_sorted'.
    our_color : str
        Hex color for the vGEP ("our") bars (lets the training panel recolor them).
    size : str
        ``figsize`` preset name (e.g. "1x3").

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    gene_names_sorted = corr["genes_sorted"]
    n = len(METHOD_SPECS)
    width = 0.9 / (n + 1)
    ind = np.arange(len(gene_names_sorted)) * 0.9
    offsets = (np.arange(n) - (n - 1) / 2) * width

    fig, ax = plt.subplots(figsize=figsize(size))
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, linestyle="--", alpha=0.3, color="gray")

    ymax = 0.0
    for spec, off in zip(METHOD_SPECS, offsets):
        arr = corr[spec["key"]]
        color = our_color if spec["key"] == "our" else spec["color"]
        ax.bar(ind + off, arr, width, label=spec["label"],
               color=color, edgecolor="black", linewidth=0.4)
        ymax = max(ymax, np.nanmax(arr))

    ax.set_xticks(ind)
    ax.set_xticklabels(gene_names_sorted, rotation=45, ha="right", fontstyle="italic")
    ax.set_ylabel("Pearson $r$")
    ax.set_ylim(0, ymax + 0.12)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.35), frameon=False, ncol=3)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()

    return fig


def create_reconstruction_comparison_figure(
    gene_list, method_columns, atlas_df, truth_geometry,
    transformed_macrochaete, white_to_red_cmap,
):
    """Grid of spatial reconstructions: one column per method + Ground Truth.

    Parameters
    ----------
    gene_list : list[str]
        Genes to plot (rows).
    method_columns : list[tuple]
        Ordered ``(column_title, expression_df, geometry, correlations)`` per
        method — each method's expression rasterized with its own geometry (these
        differ because methods reconstruct in different flipped frames).
        ``correlations`` is that method's per-gene Pearson r, in ``gene_list``
        order, annotated on each cell.
    atlas_df : DataFrame
        Ground-truth atlas; rasterized with ``truth_geometry`` as the last column.
    truth_geometry : DataFrame
        Geometry for the Ground Truth column.
    transformed_macrochaete : DataFrame
        Macrochaete marker positions (in the rendered frame).
    white_to_red_cmap : Colormap
        White-to-red colormap.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    # The ground truth is what everything is scored against, so it carries no r.
    columns = list(method_columns) + [("Ground Truth", atlas_df, truth_geometry, None)]
    n_genes = len(gene_list)
    n_cols = len(columns)
    # No preset fits an n-method-wide grid; scale a fixed cell size instead.
    fig, axs = plt.subplots(
        n_genes, n_cols, figsize=(1.1 * n_cols, 1.1 * n_genes), squeeze=False
    )

    for j, (col_title, _, _, _) in enumerate(columns):
        axs[0, j].set_title(col_title)

    for i, gene in enumerate(gene_list):
        axs[i, 0].set_ylabel(gene, rotation=0, ha="right", va="center",
                             labelpad=18, fontstyle="italic")
        for j, (_, expr_df, geom, corr) in enumerate(columns):
            ax = axs[i, j]
            matrix = normalize_01(to_matrix_form(expr_df[gene], geometry=geom))
            im = ax.imshow(matrix, cmap=white_to_red_cmap, vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

            if corr is not None and np.isfinite(corr[i]):
                ax.text(0.98, 0.02, f"{corr[i]:.2f}", transform=ax.transAxes,
                        ha="right", va="bottom", fontsize=6)

            for m in range(transformed_macrochaete.shape[1]):
                circ = plt.Circle(
                    (transformed_macrochaete.iloc[0, m], transformed_macrochaete.iloc[1, m]),
                    0.3, ec="red", fc="red", alpha=0.7,
                )
                ax.add_patch(circ)

    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.88, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax)
    fig.subplots_adjust(top=0.95, right=0.85)

    return fig


def create_gene_vgep_vs_atlas_figure(
    gene_name, vgep_df, atlas_df_indexed, common_idx,
    transformed_macrochaete, output_dir, white_to_red_cmap,
    title=None, has_atlas=True,
):
    """Create a 2-panel figure comparing vGEP vs Atlas for one gene.

    Saves to output_dir in pdf/png/svg formats.

    Parameters
    ----------
    gene_name : str
    vgep_df, atlas_df_indexed : DataFrame
        Indexed by (x, y).
    common_idx : Index
        Common positions.
    transformed_macrochaete : DataFrame
    output_dir : Path
    white_to_red_cmap : Colormap
    title : str, optional
    has_atlas : bool
    """
    if title is None:
        title = gene_name

    fig, axs = plt.subplots(1, 2, figsize=figsize("0.5x1"))

    gene_vgep = vgep_df.loc[common_idx, gene_name]
    gene_geometry = pd.DataFrame(index=common_idx)
    gene_geometry["x"] = [idx[0] for idx in common_idx]
    gene_geometry["y"] = [idx[1] for idx in common_idx]

    vgep_matrix = to_matrix_form(gene_vgep, geometry=gene_geometry)
    vgep_normalized = normalize_01(vgep_matrix)

    axs[0].imshow(vgep_normalized, cmap=white_to_red_cmap, vmin=0, vmax=1)
    axs[0].set_title("vGEP")
    axs[0].axis("off")

    if has_atlas:
        gene_atlas = atlas_df_indexed.loc[common_idx, gene_name]
        atlas_matrix = to_matrix_form(gene_atlas, geometry=gene_geometry)
        atlas_normalized = normalize_01(atlas_matrix)
        mask = ~np.isnan(vgep_normalized) & ~np.isnan(atlas_normalized)
        corr = np.corrcoef(vgep_normalized[mask], atlas_normalized[mask])[0, 1]
        logger.debug("%s: Pearson correlation (vGEP vs atlas) = %.4f", gene_name, corr)
        im = axs[1].imshow(atlas_normalized, cmap=white_to_red_cmap, vmin=0, vmax=1)
        axs[1].set_title("Atlas")
    else:
        blank = np.full_like(vgep_normalized, np.nan)
        im = axs[1].imshow(blank, cmap=white_to_red_cmap, vmin=0, vmax=1)
        axs[1].set_title("Atlas (no data)")
    axs[1].axis("off")

    for ax in axs:
        for m in range(transformed_macrochaete.shape[1]):
            circ = plt.Circle(
                (transformed_macrochaete.iloc[0, m], transformed_macrochaete.iloc[1, m]),
                0.3, ec="red", fc="red", alpha=0.7,
            )
            ax.add_patch(circ)

    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.88, 0.15, 0.03, 0.7])
    fig.colorbar(im, cax=cbar_ax)

    output_dir.mkdir(parents=True, exist_ok=True)
    for fmt in ["pdf", "png", "svg"]:
        fig.savefig(output_dir / f"{gene_name}_vGEP_vs_Atlas.{fmt}",
                    format=fmt, bbox_inches="tight")
    plt.close(fig)


# Headroom reserved above the axes for the median labels, in rows. Cohorts need 3 or 4
# depending on how their medians bunch up; reserving a constant keeps the axes box identical
# across panels, which is what lets them be aligned in assembly.
# Fixed axes rectangle for every CDF/survival panel. tight_layout would size the left
# margin from the y-label text, so panels whose label is shorter came out fractionally
# wider and would not align in assembly.
_CDF_AXES_RECT = {"left": 0.205, "right": 0.98, "bottom": 0.16, "top": 0.80}

_MEDIAN_LABEL_ROWS = 4
_MEDIAN_ROW_HEIGHT = 7


def _annotate_medians(ax, medians, min_gap=0.13):
    """Draw each method's median as a vertical rule labelled with its value above the axes.

    Labels are stacked into rows so medians closer than ``min_gap`` in data units never
    overprint — with five methods several medians sit within a few hundredths of each other.
    A transparent placeholder pins the top of the stack so panels needing fewer rows do not
    get a taller axes from ``tight_layout``.
    """
    rows = []
    for x, color in sorted(medians):
        row = next(
            (i for i, taken in enumerate(rows) if all(abs(x - t) >= min_gap for t in taken)),
            len(rows),
        )
        if row == len(rows):
            rows.append([])
        rows[row].append(x)

        ax.axvline(x, color=color, alpha=0.35, linewidth=1.4, zorder=0)
        ax.annotate(
            f"{x:.2f}", xy=(x, 1.0), xycoords=("data", "axes fraction"),
            xytext=(0, 2 + row * _MEDIAN_ROW_HEIGHT), textcoords="offset points",
            ha="center", va="bottom", color=color, fontsize=5, clip_on=False,
        )

    ax.annotate(
        "0.00", xy=(0.5, 1.0), xycoords="axes fraction",
        xytext=(0, 2 + (_MEDIAN_LABEL_ROWS - 1) * _MEDIAN_ROW_HEIGHT),
        textcoords="offset points", ha="center", va="bottom", fontsize=5,
        alpha=0, clip_on=False,
    )


def create_cdf_and_survival_plots(corr):
    """Create CDF and survival plots comparing all methods.

    Parameters
    ----------
    corr : dict
        Method key -> correlation array (e.g. a 'validation'/'training'
        sub-dict, or a combined dict). Each method is drawn with its registry color.

    Returns
    -------
    fig_cdf, fig_survival : matplotlib.figure.Figure
    """
    markers = ["o", "s", "^", "D", "v", "P"]

    fig_cdf, ax_cdf = plt.subplots(figsize=figsize("1x1"))
    fig_survival, ax_s = plt.subplots(figsize=figsize("1x1"))

    medians = []
    for idx, spec in enumerate(METHOD_SPECS):
        arr = corr[spec["key"]]
        arr_sorted = np.sort(arr[~np.isnan(arr)])
        cdf = np.arange(1, len(arr_sorted) + 1) / len(arr_sorted)
        marker = markers[idx % len(markers)]

        for ax, y in ((ax_cdf, cdf), (ax_s, 1 - cdf + 1 / len(arr_sorted))):
            ax.plot(arr_sorted, y, label=spec["label"], color=spec["color"],
                    linewidth=1.6, marker=marker, markersize=2.6,
                    markeredgewidth=0, solid_joinstyle="round")
        medians.append((float(np.median(arr_sorted)), spec["color"]))

    for ax, ylabel, legend_loc in [
        (ax_cdf, "Cumulative fraction of genes", "upper left"),
        (ax_s, "Fraction of genes above $r$", "lower left"),
    ]:
        _annotate_medians(ax, medians)
        ax.set_xlabel("Pearson correlation (Atlas)")
        ax.set_ylabel(ylabel)
        ax.set_xlim(-0.1, 1.05)
        ax.set_ylim(0, 1.05)
        ax.legend(loc=legend_loc, frameon=False, fontsize=5,
                  handlelength=1.6, labelspacing=0.28, borderpad=0.2)
        ax.spines[["top", "right"]].set_visible(False)

    for fig in (fig_cdf, fig_survival):
        fig.subplots_adjust(**_CDF_AXES_RECT)

    return fig_cdf, fig_survival


def create_protein_vs_fish_cdf(our_corrs, validation_genes_list, protein_genes, fish_genes):
    """Create CDF and survival plots comparing Protein vs smiFISH genes in vGEP.

    Parameters
    ----------
    our_corrs : np.ndarray
        Correlation values for vGEP (ordered by validation_genes_list).
    validation_genes_list : list[str]
    protein_genes, fish_genes : list[str]

    Returns
    -------
    fig_cdf, fig_survival : matplotlib.figure.Figure
    """
    gene_to_corr = dict(zip(validation_genes_list, our_corrs))
    protein_corrs = np.array([gene_to_corr[g] for g in protein_genes if g in gene_to_corr])
    fish_corrs = np.array([gene_to_corr[g] for g in fish_genes if g in gene_to_corr])

    protein_sorted = np.sort(protein_corrs[~np.isnan(protein_corrs)])
    fish_sorted = np.sort(fish_corrs[~np.isnan(fish_corrs)])

    protein_cdf = np.arange(1, len(protein_sorted) + 1) / len(protein_sorted)
    fish_cdf = np.arange(1, len(fish_sorted) + 1) / len(fish_sorted)

    cohorts = [
        ("Protein", protein_sorted, protein_cdf, "#1f77b4", "o"),
        ("smiFISH", fish_sorted, fish_cdf, "#d62728", "s"),
    ]
    medians = [(float(np.median(s)), color) for _, s, _, color, _ in cohorts]

    fig_cdf, ax_cdf = plt.subplots(figsize=figsize("1x1"))
    fig_survival, ax_s = plt.subplots(figsize=figsize("1x1"))
    for name, sorted_vals, cdf, color, marker in cohorts:
        label = f"{name} (n={len(sorted_vals)})"
        for ax, y in ((ax_cdf, cdf), (ax_s, 1 - cdf + 1 / len(sorted_vals))):
            ax.plot(sorted_vals, y, label=label, color=color, linewidth=1.6,
                    marker=marker, markersize=2.6, markeredgewidth=0)

    for ax, ylabel, legend_loc in [
        (ax_cdf, "Cumulative fraction of genes", "upper left"),
        (ax_s, "Fraction of genes above $r$", "lower left"),
    ]:
        _annotate_medians(ax, medians)
        ax.set_xlabel("Pearson correlation (Atlas)")
        ax.set_ylabel(ylabel)
        ax.set_xlim(-0.1, 1.05)
        ax.set_ylim(0, 1.05)
        ax.legend(loc=legend_loc, frameon=False, fontsize=5,
                  handlelength=1.6, labelspacing=0.28, borderpad=0.2)
        ax.spines[["top", "right"]].set_visible(False)

    for fig in (fig_cdf, fig_survival):
        fig.subplots_adjust(**_CDF_AXES_RECT)

    return fig_cdf, fig_survival
