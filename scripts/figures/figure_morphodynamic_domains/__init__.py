"""figure_morphodynamic_domains: Morphogenetic PCA and clustering.

Panels:
- pca_explained_variance: Variance barplot + cumulative line
- pca_loadings_combined: Feature importance for PC1 & PC2
- spatial_pca_maps: Spatial maps of PC0 & PC1
- dendrogram_5: 5-cluster dendrogram
- dendrogram_13: 13-cluster dendrogram (greyed)
- cluster_map_5: Spatial map, 5 clusters
- cluster_map_13_greyed: Spatial map, 13 clusters highlighted
- cluster_map_13_full: Spatial map, all 13 clusters coloured
- cluster_maps_all: All k=2..13 spatial maps
- silhouette_scores: Silhouette vs k
- heatmap_5: 5-cluster z-score clustermap
- heatmap_2: 2-cluster z-score clustermap
- heatmap_13: 13-cluster z-score clustermap (all clusters coloured)
- heatmap_aggregated: Aggregated from greyed 13-cluster heatmap
- boxplots_k2: Box plots for 2-cluster scenario
- boxplots_k5: Box plots for 5-cluster scenario
- cluster_property_maps: Spatial maps of cluster-mean z-scored properties
- spatial_*: Napari scene builders for 9 tensor/scalar quantities
"""

from .preprocessing import load_morphodynamic_domains_data
from .panels_pca import (
    pca_explained_variance,
    pca_loadings_combined,
    spatial_pca_maps,
)
from .panels_clustering import (
    cluster_map_5,
    cluster_map_13_full,
    cluster_map_13_greyed,
    cluster_maps_all,
    dendrogram_5,
    dendrogram_13,
    silhouette_scores,
)
from .panels_heatmaps import (
    heatmap_2,
    heatmap_5,
    heatmap_13,
    heatmap_aggregated,
)
from .panels_distributions import (
    boxplots_k2,
    boxplots_k5,
    cluster_property_maps,
)
from .panels_spatial import SPATIAL_SCENES

FIGURE_NAME = "figure_morphodynamic_domains"

SCENES = {**SPATIAL_SCENES}

PANELS = {
    # PCA
    "pca_explained_variance": pca_explained_variance,
    "pca_loadings_combined": pca_loadings_combined,
    "spatial_pca_maps": spatial_pca_maps,
    # Clustering
    "dendrogram_5": dendrogram_5,
    "dendrogram_13": dendrogram_13,
    "cluster_map_5": cluster_map_5,
    "cluster_map_13_greyed": cluster_map_13_greyed,
    "cluster_map_13_full": cluster_map_13_full,
    "cluster_maps_all": cluster_maps_all,
    "silhouette_scores": silhouette_scores,
    # Heatmaps
    "heatmap_5": heatmap_5,
    "heatmap_2": heatmap_2,
    "heatmap_13": heatmap_13,
    "heatmap_aggregated": heatmap_aggregated,
    # Distributions
    "boxplots_k2": boxplots_k2,
    "boxplots_k5": boxplots_k5,
    "cluster_property_maps": cluster_property_maps,
}


def preprocess(paths, fig_config):
    """Load all data needed by figure_morphodynamic_domains panels.

    Parameters
    ----------
    paths : dict
        Resolved paths from config (uppercase keys).
    fig_config : dict
        The figure_morphodynamic_domains config section from figure_config.yaml.

    Returns
    -------
    dict
        Shared data for all figure_morphodynamic_domains panels.
    """
    return load_morphodynamic_domains_data(paths, fig_config)
