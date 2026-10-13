"""Figure: temporal robustness of the spatial clustering.

Re-runs the figure_morphodynamic_domains Ward spatial clustering (k=5) independently on each of the
two pooled 4h windows (early 18–22, late 22–26 hAPF) and on the reference pooled
partition, and compares the resulting morphogenetic domains.

Panels:
- cluster_map_early / _late / _pooled: per-period 5-cluster maps
  (colours aligned to the pooled partition)
- heatmap_early / _late: per-property z-scored cluster-mean heatmaps
  (z-norm pooled across the two windows)
- ari_matrix: pairwise ARI between the three partitions
"""

from .preprocessing import load_clustering_temporal_data
from .panels import (
    ari_matrix,
    cluster_map_early,
    cluster_map_late,
    cluster_map_pooled,
    heatmap_early,
    heatmap_late,
)

FIGURE_NAME = "figure_clustering_temporal"

SCENES = {}

PANELS = {
    "cluster_map_early": cluster_map_early,
    "cluster_map_late": cluster_map_late,
    "cluster_map_pooled": cluster_map_pooled,
    "heatmap_early": heatmap_early,
    "heatmap_late": heatmap_late,
    "ari_matrix": ari_matrix,
}


def preprocess(paths, fig_config):
    """Load all data needed by the clustering-temporal panels."""
    return load_clustering_temporal_data(paths, fig_config)