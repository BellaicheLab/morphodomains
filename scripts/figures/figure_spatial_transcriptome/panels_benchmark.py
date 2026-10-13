"""Benchmark panel functions for figure_spatial_transcriptome.

Panels:
- validation_correlation_bars: Bar plot of validation gene correlations
- training_correlation_bars: Bar plot of training gene correlations
- reconstruction_comparison_validation: Grid of spatial reconstructions (validation)
- reconstruction_comparison_training: Grid of spatial reconstructions (training)
- vgep_vs_atlas_comparisons: vGEP vs atlas side-by-side for specific genes
- reconstruction_timing: Horizontal bar chart of compute times per method
"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from scripts.figures import figsize

logger = logging.getLogger(__name__)

from .preprocessing import (
    METHOD_SPECS,
    OUR_COLOR,
    OUR_COLOR_TRAINING,
    VGEP_VS_ATLAS_GENES,
)
from ._helpers import (
    create_correlation_bar_plot,
    create_gene_vgep_vs_atlas_figure,
    create_reconstruction_comparison_figure,
)


def _reconstruction_method_columns(shared_data, cohort):
    """(label, expression, geometry, correlations) per method for the grids.

    All methods render in the perler (double-flipped) frame — they share the same
    bin row order, and to_matrix_form is positional — except PERLER, which carries
    its own restructured geometry.
    """
    gp = shared_data["geometry_perler"]
    expr_geom = {
        "our": (shared_data["our_expression"], gp),
        "novosparc": (shared_data["novosparc_expression"], gp),
        "perler": (shared_data["perler_df"], shared_data["perler_geom"]),
        "tangram": (shared_data["tangram_expression"], gp),
        "cytospace": (shared_data["cytospace_expression"], gp),
    }
    # correlations[cohort][key] is sorted by the same index as genes_sorted, which
    # is the row order of the grid, so it indexes positionally against gene_list.
    corr = shared_data["correlations"][cohort]
    return [
        (spec["label"], *expr_geom[spec["key"]], corr[spec["key"]])
        for spec in METHOD_SPECS
    ]


def validation_correlation_bars(shared_data, output_base):
    """Vertical bar plot comparing validation gene correlations."""
    return create_correlation_bar_plot(
        shared_data["correlations"]["validation"], our_color=OUR_COLOR, size="1x3",
    )


def training_correlation_bars(shared_data, output_base):
    """Vertical bar plot comparing training gene correlations."""
    return create_correlation_bar_plot(
        shared_data["correlations"]["training"], our_color=OUR_COLOR_TRAINING, size="1x3",
    )


def reconstruction_comparison_validation(shared_data, output_base):
    """Grid of spatial reconstructions for validation genes."""
    return create_reconstruction_comparison_figure(
        gene_list=shared_data["correlations"]["validation"]["genes_sorted"],
        method_columns=_reconstruction_method_columns(shared_data, "validation"),
        atlas_df=shared_data["atlas_df"],
        truth_geometry=shared_data["geometry_perler"],
        transformed_macrochaete=shared_data["transformed_macrochaete"],
        white_to_red_cmap=shared_data["white_to_red"],
    )


def reconstruction_comparison_training(shared_data, output_base):
    """Grid of spatial reconstructions for training genes."""
    return create_reconstruction_comparison_figure(
        gene_list=shared_data["correlations"]["training"]["genes_sorted"],
        method_columns=_reconstruction_method_columns(shared_data, "training"),
        atlas_df=shared_data["atlas_df"],
        truth_geometry=shared_data["geometry_perler"],
        transformed_macrochaete=shared_data["transformed_macrochaete"],
        white_to_red_cmap=shared_data["white_to_red"],
    )


def vgep_vs_atlas_comparisons(shared_data, output_base):
    """Side-by-side vGEP vs atlas for specific genes.

    Saves figures internally (multi-format). Returns None.
    """
    vgep_comp = shared_data["vgep_comparison"]
    vgep_df = vgep_comp["vgep_df"]
    atlas_df_indexed = vgep_comp["atlas_df_indexed"]
    common_idx = vgep_comp["common_idx"]
    transformed_macrochaete = shared_data["transformed_macrochaete"]
    white_to_red = shared_data["white_to_red"]

    output_dir = Path(output_base) / "benchmarking" / "reconstructions"

    for gene_name, gene_title in VGEP_VS_ATLAS_GENES:
        if gene_name not in vgep_df.columns:
            logger.debug("%s not found in vGEP. Skipping.", gene_name)
            continue

        if len(common_idx) == 0:
            logger.debug("No common positions for %s. Skipping.", gene_name)
            continue

        has_atlas = gene_name in atlas_df_indexed.columns

        create_gene_vgep_vs_atlas_figure(
            gene_name=gene_name,
            vgep_df=vgep_df,
            atlas_df_indexed=atlas_df_indexed,
            common_idx=common_idx,
            transformed_macrochaete=transformed_macrochaete,
            output_dir=output_dir,
            white_to_red_cmap=white_to_red,
            title=gene_title,
            has_atlas=has_atlas,
        )
        status = "with atlas" if has_atlas else "no atlas data"
        logger.debug("Saved %s vGEP vs Atlas figure (%s).", gene_name, status)

    return None


def reconstruction_timing(shared_data, output_base):
    """Horizontal bar chart comparing single-reconstruction compute times.

    Mean time per method with s.d. error bars over the timed repeats.
    """
    benchmark = shared_data["benchmark_times"]

    means = [benchmark[spec["timing"]]["mean"] for spec in METHOD_SPECS]
    stds = [benchmark[spec["timing"]]["std"] for spec in METHOD_SPECS]

    fig, ax = plt.subplots(figsize=figsize("1x1"))

    y_pos = np.arange(len(METHOD_SPECS))
    # Log scale: the methods span 1.2 s to 820 s, and on a linear axis every method
    # except PERLER collapses into the axis line.
    ax.barh(y_pos, means, xerr=stds, height=0.6,
            color=[spec["color"] for spec in METHOD_SPECS],
            edgecolor="black", linewidth=0.4, capsize=2, error_kw={"linewidth": 0.6})
    ax.set_xscale("log")

    for y, mean, std in zip(y_pos, means, stds):
        ax.text((mean + std) * 1.25, y, f"{mean:.1f} \u00b1 {std:.1f}",
                va="center", ha="left", fontsize=5)

    ax.set_yticks(y_pos)
    ax.set_yticklabels([spec["label"] for spec in METHOD_SPECS])
    ax.set_xlabel("Time per reconstruction (s)")
    ax.set_xlim(0.5, max(means) * 9)
    ax.invert_yaxis()
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()

    return fig
