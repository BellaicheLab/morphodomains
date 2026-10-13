"""Panel functions for figure_grn.

Each panel function takes (shared_data, output_base) and returns a Figure or None.
The shared_data dict is populated by preprocessing.load_grn_data().
"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from scripts.figures.figure_morphodynamic_domains.panels_heatmaps import HIGHLIGHT_ORDER

logger = logging.getLogger(__name__)


def tf_specificity_heatmap_5(shared_data, output_base):
    """Clustermap of cluster-specific TF expression (5 morphogenetic clusters).

    Parameters
    ----------
    shared_data : dict
        Must contain: specific_genes_5, z_bin_mean_5, tf_list.
    output_base : str
        Output directory path.

    Returns
    -------
    plt.Figure
        The generated heatmap figure.
    """
    specific_genes = shared_data["specific_genes_5"]
    z_scores_df = shared_data["z_bin_mean_5"]
    tf_list = shared_data["tf_list"]
    display_columns = [0, 1, 2, 3, 4]

    # Filter to specific genes that are also in tf_list
    specific_tfs = specific_genes[specific_genes].index.intersection(tf_list)
    specific_tfs = specific_tfs.intersection(z_scores_df.index)

    cluster_means = z_scores_df.loc[specific_tfs]
    available_cols = [c for c in display_columns if c in cluster_means.columns]
    cluster_means = cluster_means[available_cols]

    # Sort genes by their maximum expression cluster
    gene_order = cluster_means.idxmax(axis=1).sort_values().index
    sorted_means = cluster_means.loc[gene_order]
    sorted_means.columns = [f"5-MD{c + 1}" for c in sorted_means.columns]

    g = sns.clustermap(
        sorted_means,
        row_cluster=True,
        col_cluster=False,
        dendrogram_ratio=0.1,
        cmap="coolwarm",
        robust=True,
        figsize=(4, 8),
        cbar_kws={"label": "Z-Score (across clusters + SC)"},
        yticklabels=True,
    )

    g.ax_row_dendrogram.set_visible(False)
    g.ax_col_dendrogram.set_visible(False)
    g.ax_heatmap.set_xlabel("Morphodynamic domain")
    g.ax_heatmap.set_ylabel("Specific Transcription Factors")
    g.ax_heatmap.set_title("Cluster-Specific TF Expression Patterns", pad=20)
    g.ax_heatmap.tick_params(axis="y", labelsize=8)

    plt.tight_layout()
    return g.fig


def tf_specificity_heatmap_13(shared_data, output_base):
    """Clustermap of cluster-specific TF expression (13 clusters, subset 5/7/9).

    Parameters
    ----------
    shared_data : dict
        Must contain: specific_genes_13, z_bin_mean_13, tf_list.
    output_base : str
        Output directory path.

    Returns
    -------
    plt.Figure or None
        The generated heatmap figure, or None if no TFs found.
    """
    specific_genes = shared_data["specific_genes_13"]
    z_scores_df = shared_data["z_bin_mean_13"]
    tf_list = shared_data["tf_list"]
    display_columns = HIGHLIGHT_ORDER  # 13-MD1..13-MD3

    specific_tfs = specific_genes[specific_genes].index.intersection(tf_list)
    specific_tfs = specific_tfs.intersection(z_scores_df.index)

    cluster_means = z_scores_df.loc[specific_tfs]

    # Determine home cluster per TF
    spatial_cols = [
        c for c in cluster_means.columns if isinstance(c, (int, np.integer))
    ]
    tf_home_cluster = cluster_means[spatial_cols].idxmax(axis=1)

    tfs_in_display_clusters = tf_home_cluster[
        tf_home_cluster.isin(display_columns)
    ].index

    if len(tfs_in_display_clusters) == 0:
        logger.warning("No TFs found with max expression in clusters %s", display_columns)
        return None

    available_cols = [c for c in display_columns if c in cluster_means.columns]
    if not available_cols:
        return None

    sorted_means = cluster_means.loc[tfs_in_display_clusters, available_cols]
    gene_order = tf_home_cluster.loc[tfs_in_display_clusters].sort_values().index
    sorted_means = sorted_means.loc[gene_order]
    sorted_means.columns = [f"13-MD{display_columns.index(c) + 1}" for c in sorted_means.columns]

    g = sns.clustermap(
        sorted_means,
        row_cluster=True,
        col_cluster=False,
        dendrogram_ratio=0.08,
        cmap="coolwarm",
        robust=True,
        figsize=(2.2, 5.5),
        cbar_kws={"label": "Mean Z-Score"},
        yticklabels=True,
    )

    g.ax_row_dendrogram.set_visible(False)
    g.ax_col_dendrogram.set_visible(False)
    g.ax_heatmap.set_xlabel("Morphodynamic domain")
    g.ax_heatmap.set_ylabel("Specific Transcription Factors")
    g.ax_heatmap.set_title("13-MD Specific TF Expression", pad=10)
    g.ax_heatmap.tick_params(axis="y", labelsize=6)
    g.ax_heatmap.tick_params(axis="x", labelsize=7)

    plt.tight_layout()
    return g.fig


def top_tfs_by_tau(shared_data, output_base):
    """Bar plot of top 15 TFs ranked by Tau specificity score.

    Parameters
    ----------
    shared_data : dict
        Must contain: tau_scores_5, combinations_df_5.
    output_base : str
        Output directory path.

    Returns
    -------
    plt.Figure
        The generated bar plot figure.
    """
    tau_scores = shared_data["tau_scores_5"]
    combinations_df = shared_data["combinations_df_5"]
    cluster_label = 4  # Default cluster for the main panel

    tf_cluster = combinations_df[combinations_df[cluster_label] == 1].index
    tau_cluster = tau_scores.loc[tau_scores.index.intersection(tf_cluster)]
    top15_cluster = tau_cluster.sort_values(ascending=False).head(15)

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    ax.bar(top15_cluster.index, top15_cluster.values, color="skyblue")
    ax.set_title(f"Top 15 TFs by Tau Score (5-MD{cluster_label + 1})")
    ax.set_xlabel("Transcription Factors")
    ax.set_ylabel("Tau Score")
    ax.tick_params(axis="x", rotation=45)

    return fig


def tau_distribution(shared_data, output_base):
    """Plot Tau score distribution with Otsu threshold.

    Parameters
    ----------
    shared_data : dict
        Must contain: tau_scores_5, otsu_threshold_5.
    output_base : str
        Output directory path.

    Returns
    -------
    plt.Figure
        The generated histogram figure.
    """
    from vgep.plotter import plot_tau_distribution

    tau_scores = shared_data["tau_scores_5"]
    otsu_thresh = shared_data["otsu_threshold_5"]

    fig = plot_tau_distribution(
        tau_scores,
        cutoff=otsu_thresh,
        title="Tau score distribution (5 clusters) - Cluster-restricted genes",
        bins=40,
        fig_size=(3, 2),
    )
    return fig


def tf_cluster_network(shared_data, output_base):
    """Generate TF-cluster bipartite network visualization.

    Parameters
    ----------
    shared_data : dict
        Must contain: combinations_df_5, tau_scores_5.
    output_base : str
        Output directory path.

    Returns
    -------
    None
        Network is saved directly by create_tf_cluster_network.
    """
    from vgep.grn.network import create_tf_cluster_network

    combinations_df = shared_data["combinations_df_5"]
    tau_scores = shared_data["tau_scores_5"]

    create_tf_cluster_network(
        combinations_df,
        tau_scores,
        output_path=str(Path(output_base) / "tf_cluster_network"),
    )
    return None


