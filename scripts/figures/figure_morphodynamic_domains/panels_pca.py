"""PCA panel functions for figure_morphodynamic_domains.

Panels:
- pca_explained_variance: Variance barplot + cumulative line
- pca_loadings_combined: Feature importance for PC1 & PC2 (multi-figure, self-managed)
- spatial_pca_maps: Spatial maps of PC0 & PC1 (multi-figure, self-managed)
"""

import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm
from vgep.plotter import geometric

from scripts.figures import save_and_close

from .preprocessing import (
    HEATMAP_TICK_LABEL_SIZE,
    PCA_AXIS_LABEL_SIZE,
    PCA_TICK_LABEL_SIZE,
)


# ── Single-figure panels ─────────────────────────────────────────────────────


def pca_explained_variance(shared_data, output_base):
    """Variance barplot + cumulative line."""
    mypca = shared_data["mypca"]
    n_components = shared_data["n_components"]

    fig, ax = plt.subplots(figsize=(3.5, 2.5), dpi=300)

    x = range(1, len(mypca.explained_variance_) + 1)
    ax.bar(
        x,
        mypca.explained_variance_ratio_,
        color="#2E5A88",
        width=0.7,
        edgecolor="black",
        linewidth=0.5,
    )

    cumulative_variance = np.cumsum(mypca.explained_variance_ratio_)
    ax2 = ax.twinx()
    ax2.plot(x, cumulative_variance, "r.-", color="#CC3311")

    ax.axvline(
        x=n_components,
        color="#000000",
        linestyle="--",
        linewidth=1.5,
        label=f"Used for clustering\n(n={n_components})",
        zorder=3,
    )
    ax.legend(fontsize=6, loc="upper right", framealpha=0.9)

    ax.set_ylabel("Explained variance ratio", fontsize=8)
    ax.set_ylim(0, max(mypca.explained_variance_ratio_) * 1.1)
    ax.tick_params(axis="y", labelsize=7)

    ax2.set_ylabel("Cumulative explained variance", fontsize=8)
    ax2.set_ylim(0, 1.1)
    ax2.tick_params(axis="y", labelsize=7)

    ax.set_xlabel("Principal component", fontsize=8)
    ax.set_xlim(0.5, len(mypca.explained_variance_) + 0.5)
    ax.tick_params(axis="x", labelsize=7)

    ax.yaxis.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()

    return fig


# ── Multi-figure panels ──────────────────────────────────────────────────────


def pca_loadings_combined(shared_data, output_base):
    """2 figs (PC1, PC2) -> pca/."""
    mypca = shared_data["mypca"]
    Mrph_sc = shared_data["Mrph_sc"]

    import pandas as pd

    loadings = pd.DataFrame(
        mypca.components_.T,
        columns=[f"PC{i}" for i in range(mypca.components_.shape[0])],
        index=Mrph_sc.columns,
    )

    cmap = plt.cm.RdBu_r

    for i in range(2):
        fig, ax = plt.subplots(figsize=(7, 8))
        sort_loadings = loadings.iloc[:, i].sort_values(ascending=False)
        x_vals = sort_loadings.values
        y_labels = sort_loadings.index

        norm = TwoSlopeNorm(vmin=min(x_vals), vcenter=0, vmax=max(x_vals))
        colors = cmap(norm(x_vals))

        ax.barh(y_labels, x_vals, color=colors)

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax)
        cbar.set_label(
            "Loading Value",
            rotation=270,
            labelpad=15,
            fontsize=PCA_AXIS_LABEL_SIZE,
        )

        ax.set_xlabel("Loading value", fontsize=PCA_AXIS_LABEL_SIZE)
        ax.axvline(x=0, color="k", linestyle="--", linewidth=0.8)
        ax.tick_params(axis="y", labelsize=HEATMAP_TICK_LABEL_SIZE)
        ax.tick_params(axis="x", labelsize=PCA_TICK_LABEL_SIZE)
        plt.tight_layout()

        path = os.path.join(output_base, f"pca/panel_2_pc{i + 1}_feature_importance")
        save_and_close(fig, path)

    return None


def spatial_pca_maps(shared_data, output_base):
    """2 figs (PC0, PC1) -> spatial_maps/."""
    Mrph_PCA = shared_data["Mrph_PCA"]
    macrochaete = shared_data["macrochaete"]

    geometry = Mrph_PCA.reset_index()[["x", "y"]]
    data = Mrph_PCA.reset_index()

    pc_cols = [c for c in ["PC0", "PC1"] if c in data.columns]
    max_abs_pc = np.nanmax(np.abs(data[pc_cols].values)) if pc_cols else 1.0
    if max_abs_pc == 0 or np.isnan(max_abs_pc):
        max_abs_pc = 1.0
    pc_norm = TwoSlopeNorm(vmin=-max_abs_pc, vcenter=0.0, vmax=max_abs_pc)

    for pc_col, title, fname in [
        ("PC0", "First Principal Component", "panel_3a_pca1"),
        ("PC1", "Second Principal Component", "panel_3b_pca2"),
    ]:
        fig, ax = plt.subplots(figsize=(7.5, 7.5))
        geometric(
            transcriptome=data,
            color=pc_col,
            geometry=geometry,
            ax=ax,
            title=title,
            colorbar=True,
            colorbar_horizontal=False,
            spines=False,
            cropped=True,
            show=False,
            cmap="RdBu_r",
            norm=pc_norm,
            annotation=macrochaete,
        )
        path = os.path.join(output_base, f"spatial_maps/{fname}")
        save_and_close(fig, path)

    return None
