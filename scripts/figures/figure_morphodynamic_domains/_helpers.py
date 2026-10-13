"""Shared private plotting helpers for figure_morphodynamic_domains.

Contains reusable plotting functions called by multiple panel modules:
plot_dendrogram, create_cluster_heatmap, plot_cluster_property_distributions,
and spatial masking utilities.
"""

import itertools
import logging
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import to_rgb
from matplotlib.patches import Rectangle
from scipy.cluster.hierarchy import (
    dendrogram,
    leaves_list,
    linkage,
    optimal_leaf_ordering,
)
from scipy.spatial.distance import pdist
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


# ── Published panel conventions ───────────────────────────────────────────────

# Geometry of the heatmap panels in the assembled figure, in points: 30 feature
# rows of 5.73 pt, a 6.5 pt colour bar, and domain names in boxes two rows tall.
# Emitting the panels at this size lets them replace the placed artwork without
# rescaling. Column width is per panel -- they were scaled to a common height.
CELL_HEIGHT_PT = 5.727
LABEL_SIZE_PT = 5
COLORBAR_WIDTH_PT = 6.5
DOMAIN_BOX_HEIGHT_PT = 11.5

# The figures name the deviatoric component CE and drop "Normalized" from the
# rates; the pipeline keeps its own names upstream of the plot.
_PUBLISHED_NAMES = {
    "elongation": "CE",
    "Normalized division rate": "Division rate",
    "Normalized apoptosis rate": "Apoptosis rate",
}


def _published_feature_name(name):
    for pipeline_name, published in _PUBLISHED_NAMES.items():
        name = name.replace(pipeline_name, published)
    return name


def _emphasise_source(name):
    """Bold the deformation source letter, as the figure sets it."""
    source = re.match(r"([ADGRS]) → (.*)", name)
    return rf"$\mathbf{{{source[1]}}}$ → {source[2]}" if source else name


def _readable_ink(color):
    """Black or white, whichever reads on ``color`` (ITU-R BT.601 luma)."""
    r, g, b = to_rgb(color)
    return "black" if 0.299 * r + 0.587 * g + 0.114 * b > 0.5 else "white"


# ── Config accessor ───────────────────────────────────────────────────────────


# ── Dendrogram ────────────────────────────────────────────────────────────────


def plot_dendrogram(model, n_leaves, cluster_colors, ax, base_color="#000000FF"):
    """Draw the top ``n_leaves`` of a fitted Ward tree, each leaf coloured by its cluster.

    ``cluster_colors`` is indexed by the model's cluster id. The model must be fitted
    with ``compute_distances=True`` so the branch heights are its own merge costs.
    """
    n_samples = len(model.labels_)
    counts = np.zeros(len(model.children_))
    for i, merge in enumerate(model.children_):
        counts[i] = sum(1 if c < n_samples else counts[c - n_samples] for c in merge)
    linkage_matrix = np.column_stack([model.children_, model.distances_, counts]).astype(float)

    R = dendrogram(
        linkage_matrix, truncate_mode="lastp", p=n_leaves, ax=ax,
        leaf_rotation=90, leaf_font_size=10, show_contracted=False,
        count_sort="descendent", distance_sort=True,
        above_threshold_color=base_color, color_threshold=-1,
        # scipy labels a one-sample leaf by its index; label every leaf by its size.
        leaf_label_func=lambda i: f"({1 if i < n_samples else int(counts[i - n_samples])})",
    )

    # Leaf legs start at height 0; scipy places leaf i at x = 5 + 10 i.
    leaf_top = {}
    for xs, ys in zip(R["icoord"], R["dcoord"]):
        if ys[0] == 0:
            leaf_top[xs[0]] = ys[1]
        if ys[3] == 0:
            leaf_top[xs[3]] = ys[2]

    for i, node in enumerate(R["leaves"]):
        while node >= n_samples:
            node = model.children_[node - n_samples][0]
        x = 5 + 10 * i
        ax.vlines(x, 0, leaf_top[x], colors=cluster_colors[model.labels_[node]],
                  linewidth=2, zorder=5)

    ax.tick_params(axis="x", length=0)
    ax.set_yticks([])
    for side in ("bottom", "left", "right", "top"):
        ax.spines[side].set_visible(False)


# ── Cluster heatmap ───────────────────────────────────────────────────────────


def create_cluster_heatmap(
    Mrph_original,
    cluster_labels,
    color_list,
    column_labels,
    cell_width_pt,
    cluster_order=None,
    normalize_zscore=True,
    vmin=-1.5,
    vmax=1.5,
):
    """Cluster-mean z-score heatmap, laid out as in the assembled figure.

    ``cluster_order`` gives the columns left to right; ``column_labels`` and
    ``color_list`` follow that order. Returns the Figure.
    """
    if isinstance(cluster_labels, pd.Series) and cluster_labels.index.equals(
        Mrph_original.index
    ):
        cluster_labels_series = cluster_labels
    else:
        cluster_labels_series = pd.Series(
            cluster_labels, index=Mrph_original.index, name="cluster_id"
        )

    features = Mrph_original.rename(columns=_published_feature_name)
    if normalize_zscore:
        features = pd.DataFrame(
            StandardScaler().fit_transform(features),
            index=features.index,
            columns=features.columns,
        )
    data = features.groupby(cluster_labels_series).mean()
    if cluster_order is not None:
        data = data.loc[cluster_order]

    # One feature order for every cluster heatmap, clustered once on the full
    # bin x feature matrix. Letting each panel cluster its own rows puts the same
    # domain's values at a different height in each k, which reads as a difference
    # in the values themselves. Optimal leaf ordering then rotates the branches to
    # keep similar features adjacent, recovering the block structure.
    distances = pdist(features.T.values)
    tree = optimal_leaf_ordering(linkage(distances, method="average"), distances)
    data = data[features.columns[leaves_list(tree)]]

    n_col, n_row = data.shape
    heatmap_w, heatmap_h = n_col * cell_width_pt, n_row * CELL_HEIGHT_PT
    label_w = max(len(c) for c in data.columns) * LABEL_SIZE_PT * 0.52
    width = 2 + heatmap_w + 2.5 + label_w + 2 + COLORBAR_WIDTH_PT + 22
    height = 2 + heatmap_h + DOMAIN_BOX_HEIGHT_PT + 2

    fig = plt.figure(figsize=(width / 72, height / 72))
    ax = fig.add_axes(
        [2 / width, (2 + DOMAIN_BOX_HEIGHT_PT) / height, heatmap_w / width, heatmap_h / height]
    )
    mesh = ax.imshow(
        data.T.values, aspect="auto", cmap="RdBu_r", vmin=vmin, vmax=vmax,
        interpolation="nearest",
    )

    ax.set_xticks([])
    ax.set_yticks(np.arange(n_row))
    ax.set_yticklabels(
        [_emphasise_source(c) for c in data.columns], fontsize=LABEL_SIZE_PT
    )
    ax.yaxis.tick_right()
    ax.tick_params(axis="y", length=2, width=0.4, pad=1.5)
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Domain names in filled boxes under their column, as in the assembled figure.
    for i, (name, color) in enumerate(zip(column_labels, color_list)):
        ax.add_patch(
            Rectangle(
                (i - 0.5, n_row - 0.5), 1, DOMAIN_BOX_HEIGHT_PT / CELL_HEIGHT_PT,
                facecolor=color, edgecolor="black", linewidth=0.5, clip_on=False,
            )
        )
        ax.text(
            i, n_row - 0.5 + 0.5 * DOMAIN_BOX_HEIGHT_PT / CELL_HEIGHT_PT, name,
            ha="center", va="center", fontsize=LABEL_SIZE_PT, fontweight="bold",
            color=_readable_ink(color), clip_on=False,
        )

    cax = fig.add_axes(
        [
            (2 + heatmap_w + 2.5 + label_w + 2) / width,
            (2 + DOMAIN_BOX_HEIGHT_PT + 0.085 * heatmap_h) / height,
            COLORBAR_WIDTH_PT / width,
            0.83 * heatmap_h / height,
        ]
    )
    cbar = fig.colorbar(mesh, cax=cax, ticks=np.arange(vmin, vmax + 0.01, 0.5))
    cbar.outline.set_linewidth(0.4)
    cax.tick_params(labelsize=LABEL_SIZE_PT, length=2, width=0.4, pad=1.5)
    cax.set_ylabel("z-normalized values", fontsize=LABEL_SIZE_PT, labelpad=3)

    return fig


# ── Box-plot distributions ────────────────────────────────────────────────────


def plot_cluster_property_distributions(
    data_df,
    property_columns,
    cluster_label_column,
    num_clusters_to_plot=None,
    palette="viridis",
    cluster_color_list=None,
):
    """Generate box plots for each property comparing distributions across clusters.

    Returns a list of Figure objects.
    """
    from statannotations.Annotator import Annotator

    unique_clusters_in_data = sorted(data_df[cluster_label_column].unique())

    clusters_to_include = unique_clusters_in_data
    if num_clusters_to_plot is not None:
        if all(isinstance(c, (int, float)) for c in unique_clusters_in_data):
            clusters_to_include = [
                c for c in unique_clusters_in_data if c < num_clusters_to_plot
            ]
        else:
            clusters_to_include = unique_clusters_in_data[:num_clusters_to_plot]

    df_filtered = data_df[data_df[cluster_label_column].isin(clusters_to_include)]

    plot_order_original_type = sorted(df_filtered[cluster_label_column].unique())
    df_filtered_plotting = df_filtered.reset_index()
    plot_order = [str(c) for c in plot_order_original_type]
    df_filtered_plotting[cluster_label_column] = pd.Categorical(
        df_filtered_plotting[cluster_label_column].astype(str),
        categories=plot_order,
        ordered=True,
    )

    current_seaborn_palette = palette
    if cluster_color_list:
        if len(plot_order_original_type) <= len(cluster_color_list):
            current_seaborn_palette = {
                str(label_original): cluster_color_list[i]
                for i, label_original in enumerate(plot_order_original_type)
            }

    figs = []
    for prop_col in property_columns:
        fig, ax = plt.subplots(figsize=(7, 6))

        sns.boxplot(
            x=cluster_label_column,
            y=prop_col,
            data=df_filtered_plotting,
            ax=ax,
            palette=current_seaborn_palette,
            order=plot_order,
            hue=cluster_label_column,
            legend=False,
        )

        ax.set_title(f"Distribution of {prop_col}\nby Cluster", fontsize=10)
        ax.set_xlabel("Cluster Label", fontsize=9)
        ax.set_ylabel(prop_col, fontsize=9)
        ax.tick_params(
            axis="x", rotation=45 if len(plot_order) > 5 else 0, labelsize=8
        )
        ax.tick_params(axis="y", labelsize=8)

        if len(plot_order) >= 2:
            box_pairs = list(itertools.combinations(plot_order, 2))
            if box_pairs:
                try:
                    annotator = Annotator(
                        ax,
                        box_pairs,
                        data=df_filtered_plotting,
                        x=cluster_label_column,
                        y=prop_col,
                        order=plot_order,
                    )
                    annotator.configure(
                        test="Mann-Whitney",
                        text_format="star",
                        loc="inside",
                        verbose=0,
                        line_offset=0.02,
                        text_offset=1,
                        comparisons_correction=None,
                        fontsize=10,
                    )
                    annotator.apply_and_annotate()

                    current_y_lim = ax.get_ylim()
                    y_axis_range = current_y_lim[1] - current_y_lim[0]

                    for i, (c1, c2) in enumerate(box_pairs):
                        data1 = df_filtered_plotting[
                            df_filtered_plotting[cluster_label_column] == c1
                        ][prop_col]
                        data2 = df_filtered_plotting[
                            df_filtered_plotting[cluster_label_column] == c2
                        ][prop_col]
                        mean1 = data1.mean()
                        mean2 = data2.mean()

                        fold_change_text = "N/A"
                        if pd.notna(mean1) and pd.notna(mean2):
                            if mean2 != 0:
                                fold_change = mean1 / mean2
                                fold_change_text = f"FC: {fold_change:.2f}"
                            elif mean1 == 0 and mean2 == 0:
                                fold_change_text = "FC: 1.00"
                            else:
                                fold_change_text = "FC: Inf"

                        idx1 = plot_order.index(c1)
                        idx2 = plot_order.index(c2)
                        x_mid = (idx1 + idx2) / 2.0

                        current_prop_data = df_filtered_plotting[prop_col].dropna()
                        if not current_prop_data.empty:
                            data_y_range_prop = (
                                current_prop_data.max() - current_prop_data.min()
                            )
                            if data_y_range_prop == 0:
                                data_y_range_prop = 1
                        else:
                            data_y_range_prop = y_axis_range

                        pair_data_max = max(
                            data1.max() if pd.notna(data1.max()) else -np.inf,
                            data2.max() if pd.notna(data2.max()) else -np.inf,
                        )
                        if pair_data_max == -np.inf:
                            pair_data_max = current_y_lim[0]

                        fc_text_y = pair_data_max + (data_y_range_prop * 0.04)
                        fc_text_y = min(
                            fc_text_y, current_y_lim[0] + y_axis_range * 0.90
                        )

                        fc_font_size = 7 if len(box_pairs) <= 3 else 6
                        ax.text(
                            x_mid,
                            fc_text_y,
                            fold_change_text,
                            ha="center",
                            va="bottom",
                            fontsize=fc_font_size,
                            color="darkslateblue",
                        )

                except Exception as e:
                    logger.debug(
                        "Could not add statistical or FC annotation for %s: %s",
                        prop_col, e,
                    )

        final_y_min, final_y_max = ax.get_ylim()
        padding = (final_y_max - final_y_min) * 0.15
        ax.set_ylim(final_y_min, final_y_max + padding)
        plt.tight_layout(pad=1.0)
        figs.append(fig)

    return figs
