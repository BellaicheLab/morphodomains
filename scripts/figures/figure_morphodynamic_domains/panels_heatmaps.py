"""Heatmap panel functions for figure_morphodynamic_domains.

Panels:
- heatmap_5: 5-cluster z-score clustermap
- heatmap_2: 2-cluster z-score clustermap
- heatmap_13: 13-cluster z-score clustermap (all clusters coloured)
- heatmap_aggregated: Aggregated from greyed 13-cluster heatmap

All four are emitted at the size and styling of the assembled figure, so they can
replace the placed artwork directly.
"""

import pandas as pd

from ._helpers import create_cluster_heatmap
from .preprocessing import (
    COLORS_2,
    COLORS_5,
    COLORS_13,
    COLORS_13_GREYED,
    get_highlighted_clusters_from_colors,
)

# Cell width per panel, in points, as the panels are placed in the figure. They
# were scaled to a common height from different source aspect ratios, so each
# panel keeps its own column width.
CELL_WIDTH_2 = 43.8
CELL_WIDTH_5 = 31.35
CELL_WIDTH_13 = 30.13

# Left-to-right order of the three highlighted domains in the aggregated panel.
# The palette lists them by cluster id; the figure orders them 13-MD1..13-MD3.
HIGHLIGHT_ORDER = [5, 7, 9]


# ── Single-figure panels ─────────────────────────────────────────────────────


def heatmap_5(shared_data, output_base):
    """5-cluster z-score heatmap."""
    return create_cluster_heatmap(
        shared_data["Mrph"],
        shared_data["cluster_labels_clean"][5],
        color_list=COLORS_5,
        column_labels=[f"5-MD{i}" for i in range(1, 6)],
        cell_width_pt=CELL_WIDTH_5,
    )


def heatmap_2(shared_data, output_base):
    """2-cluster z-score heatmap."""
    return create_cluster_heatmap(
        shared_data["Mrph"],
        shared_data["cluster_labels_clean"][2],
        color_list=COLORS_2,
        column_labels=["2-MD1", "2-MD2"],
        cell_width_pt=CELL_WIDTH_2,
    )


def heatmap_13(shared_data, output_base):
    """13-cluster z-score heatmap, every cluster in its own colour."""
    clusters = shared_data["cluster_labels_clean"][13]

    # 13-MD1 to 13-MD3 are the three anterior domains, and they must be those same
    # three here as in the aggregated panel -- so they lead, in the published order,
    # and the nine pooled there follow by cluster id as 13-MD4 to 13-MD12.
    order = HIGHLIGHT_ORDER + [
        c for c in sorted(clusters.unique()) if c not in HIGHLIGHT_ORDER
    ]

    return create_cluster_heatmap(
        shared_data["Mrph"],
        clusters,
        color_list=[COLORS_13[c] for c in order],
        column_labels=[f"13-MD{i}" for i in range(1, len(order) + 1)],
        cell_width_pt=CELL_WIDTH_13,
        cluster_order=order,
    )


def heatmap_aggregated(shared_data, output_base):
    """Aggregated from greyed 13-cluster heatmap."""
    cluster_labels_13 = shared_data["cluster_labels_clean"][13]
    highlighted_clusters = get_highlighted_clusters_from_colors(COLORS_13_GREYED)

    colors_by_cluster = dict(
        zip(highlighted_clusters["cluster_indices"], highlighted_clusters["meta_colors"][1:])
    )

    meta_clusters = pd.Series(0, index=cluster_labels_13.index, dtype=int)
    for meta_idx, cluster_idx in enumerate(HIGHLIGHT_ORDER, start=1):
        meta_clusters[cluster_labels_13 == cluster_idx] = meta_idx

    return create_cluster_heatmap(
        shared_data["Mrph"],
        meta_clusters,
        color_list=[highlighted_clusters["meta_colors"][0]]
        + [colors_by_cluster[c] for c in HIGHLIGHT_ORDER],
        column_labels=["Aggregated"] + [f"13-MD{i}" for i in range(1, 4)],
        cell_width_pt=CELL_WIDTH_13,
    )
