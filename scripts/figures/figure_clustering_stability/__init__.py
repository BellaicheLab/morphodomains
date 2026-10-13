"""Figure: clustering stability vs. number of PCA components.

Re-runs the figure_morphodynamic_domains Ward spatial clustering (k=5) while sweeping the number of
PCA components and summarises the result as consensus matrices, an ARI curve,
and per-bin stability maps.

Panels:
- consensus_matrix_pcs_2_30: consensus matrix over the full PC range
- consensus_matrix_pcs_10_20: consensus matrix over the narrow PC band
- ari_vs_npcs: ARI of each per-#PC partition vs the reference 5-cluster
- stability_map_pcs_2_30 / _10_20: per-bin consensus on the tissue
"""

from .preprocessing import load_clustering_stability_data
from .panels import (
    ari_vs_npcs,
    cluster_maps_per_pc,
    consensus_matrix_pcs_2_30,
    consensus_matrix_pcs_10_20,
    stability_map_pcs_2_30,
    stability_map_pcs_10_20,
)

FIGURE_NAME = "figure_clustering_stability"

SCENES = {}

PANELS = {
    "consensus_matrix_pcs_2_30": consensus_matrix_pcs_2_30,
    "consensus_matrix_pcs_10_20": consensus_matrix_pcs_10_20,
    "ari_vs_npcs": ari_vs_npcs,
    "stability_map_pcs_2_30": stability_map_pcs_2_30,
    "stability_map_pcs_10_20": stability_map_pcs_10_20,
    "cluster_maps_per_pc": cluster_maps_per_pc,
}


def preprocess(paths, fig_config):
    """Load all data needed by the clustering-stability panels."""
    return load_clustering_stability_data(paths, fig_config)