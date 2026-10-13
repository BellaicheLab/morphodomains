"""figure_grn: Gene regulatory network analysis and TF specificity.

Panels:
- tf_specificity_heatmap_5: Clustermap of cluster-specific TF expression (5 morpho clusters)
- tf_specificity_heatmap_13: Clustermap of cluster-specific TF expression (13 clusters, subset 5/7/9)
- top_tfs_by_tau: Bar plot of top 15 TFs ranked by Tau specificity score
- tau_distribution: Tau score distribution with Otsu threshold
- tf_cluster_network: TF-cluster bipartite network
"""

from .preprocessing import load_grn_data
from .panels import (
    tau_distribution,
    tf_cluster_network,
    tf_specificity_heatmap_5,
    tf_specificity_heatmap_13,
    top_tfs_by_tau,
)

FIGURE_NAME = "figure_grn"

SCENES = {}

PANELS = {
    "tf_specificity_heatmap_5": tf_specificity_heatmap_5,
    "tf_specificity_heatmap_13": tf_specificity_heatmap_13,
    "top_tfs_by_tau": top_tfs_by_tau,
    "tau_distribution": tau_distribution,
    "tf_cluster_network": tf_cluster_network,
}


def preprocess(paths, fig_config):
    """Load all data needed by figure_grn panels.

    Parameters
    ----------
    paths : dict
        Resolved paths from config (uppercase keys).
    fig_config : dict
        The figure_grn config section from figure_config.yaml.

    Returns
    -------
    dict
        Shared data for all figure_grn panels.
    """
    return load_grn_data(paths, fig_config)
