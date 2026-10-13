"""Clustering panel functions for figure_morphodynamic_domains.

Panels:
- dendrogram_5: 5-cluster dendrogram
- dendrogram_13: 13-cluster dendrogram (greyed)
- cluster_map_5: Spatial map of 5 clusters
- cluster_map_13_greyed: Spatial map of 13 clusters (highlighted)
- cluster_map_13_full: Spatial map of 13 clusters, each its own colour
- cluster_maps_all: All k=2..13 spatial maps (multi-figure, self-managed)
- silhouette_scores: Silhouette score vs k
"""

import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from sklearn.cluster import AgglomerativeClustering
from vgep.plotter import geometric

from scripts.figures import save_and_close

from ._helpers import plot_dendrogram
from .preprocessing import COLORS_5, COLORS_13, COLORS_13_GREYED


# ── Single-figure panels ─────────────────────────────────────────────────────


def dendrogram_5(shared_data, output_base):
    """5-cluster dendrogram."""
    connectivity_matrix = shared_data["connectivity_matrix"]
    data_pca = shared_data["data_pca"]

    model = AgglomerativeClustering(
        n_clusters=5, linkage="ward", connectivity=connectivity_matrix,
        compute_distances=True,
    )

    fig, ax = plt.subplots(figsize=(1.5, 4.5))
    plot_dendrogram(model.fit(data_pca), n_leaves=5, cluster_colors=COLORS_5, ax=ax)
    # Crop just under the lowest of the 4 drawn merges (32.1).
    ax.set_ylim(30, ax.get_ylim()[1])

    return fig


def dendrogram_13(shared_data, output_base):
    """13-cluster dendrogram (greyed)."""
    connectivity_matrix = shared_data["connectivity_matrix"]
    data_pca = shared_data["data_pca"]

    model = AgglomerativeClustering(
        n_clusters=13, linkage="ward", connectivity=connectivity_matrix,
        compute_distances=True,
    )

    fig, ax = plt.subplots(figsize=(3, 4.5))
    plot_dendrogram(
        model.fit(data_pca), n_leaves=13, cluster_colors=COLORS_13_GREYED, ax=ax,
        base_color="#b3b3b3FF",
    )
    # Crop just under the lowest of the 12 drawn merges (19.9).
    ax.set_ylim(18, ax.get_ylim()[1])

    return fig


def cluster_map_5(shared_data, output_base):
    """Spatial map, 5 clusters."""
    cluster_labels_clean = shared_data["cluster_labels_clean"]
    Mrph_PCA = shared_data["Mrph_PCA"]
    macrochaete = shared_data["macrochaete"]

    data = cluster_labels_clean.copy()
    geometry = Mrph_PCA.reset_index()[["x", "y"]]

    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    cmap = ListedColormap(COLORS_5)

    geometric(
        transcriptome=data,
        color=[5],
        geometry=geometry,
        ax=ax,
        title="Spatial Distribution of Clusters",
        colorbar=True,
        colorbar_horizontal=False,
        spines=False,
        cropped=True,
        show=False,
        cmap=cmap,
        annotation=macrochaete,
    )

    return fig


def cluster_map_13_greyed(shared_data, output_base):
    """Spatial map, 13 clusters highlighted."""
    cluster_labels_clean = shared_data["cluster_labels_clean"]
    Mrph_PCA = shared_data["Mrph_PCA"]
    macrochaete = shared_data["macrochaete"]

    cleaned_clusters_13_series = cluster_labels_clean[13]
    cluster_col_name = "cluster"
    plot_data = cleaned_clusters_13_series.to_frame(name=cluster_col_name)

    geometry = Mrph_PCA.reset_index()[["x", "y"]]
    data = plot_data.reset_index()

    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    n_unique_clusters = cleaned_clusters_13_series.nunique()
    cmap = ListedColormap(COLORS_13_GREYED[:n_unique_clusters])

    geometric(
        transcriptome=data,
        color=cluster_col_name,
        geometry=geometry,
        ax=ax,
        title="Spatial Distribution of 13 Clusters (Highlighted)",
        colorbar=True,
        colorbar_horizontal=False,
        spines=False,
        cropped=True,
        show=False,
        cmap=cmap,
        annotation=macrochaete,
    )

    return fig


def cluster_map_13_full(shared_data, output_base):
    """Spatial map, all 13 clusters in their own colours (no greying)."""
    cluster_labels_clean = shared_data["cluster_labels_clean"]
    Mrph_PCA = shared_data["Mrph_PCA"]
    macrochaete = shared_data["macrochaete"]

    clusters = cluster_labels_clean[13]

    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    geometric(
        transcriptome=clusters.to_frame(name="cluster").reset_index(),
        color="cluster",
        geometry=Mrph_PCA.reset_index()[["x", "y"]],
        ax=ax,
        title="Spatial Distribution of 13 Clusters",
        colorbar=True,
        colorbar_horizontal=False,
        spines=False,
        cropped=True,
        show=False,
        cmap=ListedColormap(COLORS_13[: clusters.nunique()]),
        annotation=macrochaete,
    )

    return fig


def silhouette_scores(shared_data, output_base):
    """Silhouette vs k for the merged partition and each single 4h window."""
    curves = shared_data["silhouette_by_period"]
    n_clusters_range = list(range(2, 14))

    styles = {
        "merged": ("Merged (18–26 hAPF)", "#2E5A88"),
        "early": ("Early (18–22 hAPF)", "#EE7733"),
        "late": ("Late (22–26 hAPF)", "#009988"),
    }

    fig, ax = plt.subplots(figsize=(3, 2.25), dpi=300)

    for key, (label, color) in styles.items():
        scores = curves[key]
        optimal_n = n_clusters_range[int(np.argmax(scores))]
        ax.plot(
            n_clusters_range, scores, marker="o", ms=4, linestyle="-", color=color,
            label=f"{label} — optimal {optimal_n}",
        )
        ax.plot(
            optimal_n, np.max(scores), marker="s", markersize=7, color=color,
            markeredgecolor="black", markeredgewidth=0.5, zorder=5,
        )

    ax.set_xlabel("Number of clusters")
    ax.set_ylabel("Silhouette Score")
    ax.set_title("Silhouette Score vs. Number of Clusters")
    ax.grid(True, linestyle="--", alpha=0.7)
    ax.legend(fontsize=5)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    return fig


# ── Multi-figure panels ──────────────────────────────────────────────────────


def cluster_maps_all(shared_data, output_base):
    """12 figs (k=2..13) -> spatial_maps/."""
    cluster_labels_clean = shared_data["cluster_labels_clean"]
    macrochaete = shared_data["macrochaete"]

    for k in range(2, 14):
        fig = plt.figure(figsize=(8, 7.5))
        ax = fig.add_subplot(111)

        geometric(
            cluster_labels_clean,
            [k],
            ax=ax,
            show=False,
            colorbar=True,
            colorbar_horizontal=False,
            annotation=macrochaete,
        )

        path = os.path.join(output_base, f"spatial_maps/clusters_{k}")
        save_and_close(fig, path)

    return None
