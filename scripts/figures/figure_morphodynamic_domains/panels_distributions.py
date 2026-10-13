"""Distribution and property-map panel functions for figure_morphodynamic_domains.

Panels:
- boxplots_k2: Box plots for 2-cluster scenario (multi-figure, self-managed)
- boxplots_k5: Box plots for 5-cluster scenario (multi-figure, self-managed)
- cluster_property_maps: Spatial maps of cluster-mean z-scored properties (multi-figure, self-managed)
"""

import os

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from sklearn.preprocessing import StandardScaler
from vgep.plotter import geometric

from scripts.figures import save_and_close

from ._helpers import plot_cluster_property_distributions
from .preprocessing import COLORS_2, COLORS_5, COLUMNS_RENAMING


def _sanitize(name):
    """Sanitize a property name for use in filenames."""
    return name.replace(" ", "_").replace("/", "_").replace("(", "").replace(")", "")


def boxplots_k2(shared_data, output_base):
    """N figs (1/property) -> boxplots/."""
    Mrph = shared_data["Mrph"]
    cluster_labels_clean = shared_data["cluster_labels_clean"]

    data_for_plotting = Mrph.join(cluster_labels_clean[2].rename("cluster_id_k2"))
    properties_to_plot = Mrph.columns.tolist()

    figs = plot_cluster_property_distributions(
        data_df=data_for_plotting,
        property_columns=properties_to_plot,
        cluster_label_column="cluster_id_k2",
        num_clusters_to_plot=2,
        cluster_color_list=COLORS_2,
    )

    for i, fig in enumerate(figs):
        path = os.path.join(
            output_base,
            f"boxplots/box_plot_prop_k2_{_sanitize(properties_to_plot[i])}",
        )
        save_and_close(fig, path)
    return None


def boxplots_k5(shared_data, output_base):
    """N figs (1/property) -> boxplots/."""
    Mrph = shared_data["Mrph"]
    cluster_labels_clean = shared_data["cluster_labels_clean"]

    data_for_plotting = Mrph.join(cluster_labels_clean[5].rename("cluster_id_k5"))
    properties_to_plot = Mrph.columns.tolist()

    figs = plot_cluster_property_distributions(
        data_df=data_for_plotting,
        property_columns=properties_to_plot,
        cluster_label_column="cluster_id_k5",
        num_clusters_to_plot=5,
        cluster_color_list=COLORS_5,
    )

    for i, fig in enumerate(figs):
        path = os.path.join(
            output_base,
            f"boxplots/box_plot_prop_k5_{_sanitize(properties_to_plot[i])}",
        )
        save_and_close(fig, path)
    return None


def cluster_property_maps(shared_data, output_base):
    """k x prop x time figs -> spatial_maps/."""
    cluster_labels_clean = shared_data["cluster_labels_clean"]
    Mrph = shared_data["Mrph"]
    macrochaete = shared_data["macrochaete"]

    all_properties = list(
        set(
            name.replace(" (early)", "").replace(" (late)", "")
            for name in COLUMNS_RENAMING.values()
        )
    )

    properties_by_cluster = {2: all_properties, 5: all_properties, 13: all_properties}

    for n_clusters, props in properties_by_cluster.items():
        for prop_base in props:
            for timepoint in ["early", "late"]:
                prop_name = f"{prop_base} ({timepoint})"
                if prop_name not in Mrph.columns:
                    continue

                clusters = cluster_labels_clean[n_clusters]

                prop_values = Mrph[prop_name].values.reshape(-1, 1)
                scaler = StandardScaler()
                prop_zscored = scaler.fit_transform(prop_values).flatten()
                prop_zscored_series = pd.Series(prop_zscored, index=Mrph.index)

                mean_property = prop_zscored_series.groupby(clusters).mean()
                property_map = pd.DataFrame(index=cluster_labels_clean.index)
                property_map["property_value"] = (
                    clusters.map(mean_property).astype(float)
                )

                geometry = property_map.reset_index()[["x", "y"]]
                plot_data = property_map.reset_index()

                fig, ax = plt.subplots(figsize=(7.5, 7.5))
                norm = TwoSlopeNorm(vmin=-1.5, vcenter=0, vmax=1.5)
                geometric(
                    transcriptome=plot_data,
                    color="property_value",
                    geometry=geometry,
                    ax=ax,
                    colorbar=True,
                    colorbar_horizontal=False,
                    spines=False,
                    cropped=True,
                    show=False,
                    cmap="RdBu_r",
                    norm=norm,
                    annotation=macrochaete,
                )

                sanitized_name = (
                    prop_base.replace(" ", "_").replace("\u2192", "to").lower()
                )
                path = os.path.join(
                    output_base,
                    f"spatial_maps/k{n_clusters}_{sanitized_name}_{timepoint}",
                )
                save_and_close(fig, path)

    return None
