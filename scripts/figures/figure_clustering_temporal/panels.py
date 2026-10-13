"""Panel functions for the temporal-robustness figure.

Panels:
- cluster_map_early / _late / _pooled: per-period 5-cluster spatial maps,
  colours aligned to the reference pooled partition.
- heatmap_early / _late: per-property z-scored cluster-mean heatmaps.
- ari_matrix: pairwise ARI between the three period partitions.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, ListedColormap
from vgep.plotter import geometric

from scripts.figures import figsize
from scripts.figures.figure_morphodynamic_domains._helpers import create_cluster_heatmap
from scripts.figures.figure_morphodynamic_domains.preprocessing import COLORS_5
from scripts.figures.figure_clustering_stability.panels import CMAP

from .preprocessing import NEUTRAL_COLOR, PERIOD_LABELS


def _cluster_map(shared_data, key):
    """Spatial domain map for one period, coloured by its pooled counterpart.

    Uses the continuous branch of ``geometric`` with a ``BoundaryNorm`` so colours stay
    fixed per domain id even when a window lacks some ids.
    """
    period = shared_data["periods"][key]
    palette = COLORS_5 + [NEUTRAL_COLOR]

    fig, ax = plt.subplots(figsize=figsize("1x1"))
    geometric(
        transcriptome=pd.DataFrame({"domain": period["labels"].astype(float)}),
        color=["domain"],
        geometry=period["geometry"],
        ax=ax,
        colorbar=True,
        colorbar_horizontal=False,
        spines=False,
        cropped=True,
        show=False,
        cmap=ListedColormap(palette),
        norm=BoundaryNorm(np.arange(-0.5, len(palette) + 0.5, 1.0), len(palette)),
        annotation=shared_data["macrochaete"],
    )
    return fig


def cluster_map_early(shared_data, output_base):
    """5-cluster map for the early period (18–22 hAPF)."""
    return _cluster_map(shared_data, "early")


def cluster_map_late(shared_data, output_base):
    """5-cluster map for the late period (22–26 hAPF)."""
    return _cluster_map(shared_data, "late")


def cluster_map_pooled(shared_data, output_base):
    """5-cluster map for the reference pooled partition (18–26 hAPF)."""
    return _cluster_map(shared_data, "pooled")


def _property_heatmap(shared_data, key):
    """Per-property z-scored cluster-mean heatmap for one window (figure_morphodynamic_domains style).

    The features are already z-scored across all bins of the three windows
    (done in preprocessing), so ``create_cluster_heatmap`` only groups by cluster
    and averages — the three heatmaps therefore share a common scale.
    """
    period = shared_data["periods"][key]
    n_clusters = period["labels"].nunique()
    return create_cluster_heatmap(
        shared_data["zscored_features"][key],
        period["labels"],
        color_list=COLORS_5,
        column_labels=[f"{n_clusters}-MD{i}" for i in range(1, n_clusters + 1)],
        cell_width_pt=31.35,
        normalize_zscore=False,
    )


def heatmap_early(shared_data, output_base):
    """Per-property z-scored cluster heatmap, early period (18–22 hAPF)."""
    return _property_heatmap(shared_data, "early")


def heatmap_late(shared_data, output_base):
    """Per-property z-scored cluster heatmap, late period (22–26 hAPF)."""
    return _property_heatmap(shared_data, "late")


def ari_matrix(shared_data, output_base):
    """Pairwise ARI between the period partitions."""
    ari = shared_data["ari_matrix"]
    labels = [PERIOD_LABELS[n] for n in shared_data["period_names"]]

    fig, ax = plt.subplots(figsize=figsize("1x1"))
    im = ax.imshow(ari, cmap=CMAP, vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    for i in range(ari.shape[0]):
        for j in range(ari.shape[1]):
            ax.text(
                j, i, f"{ari[i, j]:.2f}", ha="center", va="center",
                color="white" if ari[i, j] < 0.6 else "black", fontsize=5,
            )
    cbar = fig.colorbar(im, ax=ax, label="Adjusted Rand Index")
    cbar.outline.set_linewidth(0.5)
    plt.tight_layout()
    return fig