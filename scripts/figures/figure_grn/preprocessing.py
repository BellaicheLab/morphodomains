"""Preprocessing for figure_grn.

Loads raw data, computes tau scores, builds GRN graphs, and assembles
the shared_data dict consumed by panel functions.
"""

import logging
import pickle
from pathlib import Path

import mudata
import numpy as np
import pandas as pd
import scanpy as sc
from skimage.filters import threshold_otsu

from vgep.grn.network import (
    annotate_network_with_leiden_clusters,
    export_network_for_cytoscape,
    fetch_adjacency_file_paths,
    generate_grn_graphs_optimized,
    load_all_adjacencies,
)
logger = logging.getLogger(__name__)

from vgep.grn.specificity import (
    argmax_one_hot,
    compute_pseudobulk_zscores,
    compute_tau_scores,
    filter_tau_by_max_expression,
    find_specific_genes,
)


_CACHE_FILENAME = "_figure_grn_cache.pkl"

# Analysis parameters (not exposed in config — config only controls output)
_N_TOP_VALUES = [10, 30]


def load_grn_data(paths, fig_config):
    """Load and preprocess all data needed by figure_grn panels.

    When ``regenerate_cache`` is *false* (the default) and a cache pickle
    exists, shared_data is loaded directly from disk, skipping all
    computation.  Set ``regenerate_cache: true`` in the YAML to force a
    full recomputation and update the cache.

    Parameters
    ----------
    paths : dict
        Resolved paths from config (uppercase keys).
    fig_config : dict
        The figure_grn config section.

    Returns
    -------
    dict
        Shared data dict consumed by all figure_grn panel functions.
    """
    output_base = Path(paths["OUTPUT_BASE"]) / fig_config["output_dir"]
    cache_path = output_base / _CACHE_FILENAME

    regenerate = fig_config.get("regenerate_cache", False)

    if not regenerate and cache_path.exists():
        import time
        age_hours = (time.time() - cache_path.stat().st_mtime) / 3600
        logger.info("Loading figure_grn cache from %s (age: %.1fh)", cache_path, age_hours)
        if age_hours > 72:
            logger.warning(
                "Cached figure_grn data is %.1fh old. "
                "Run with --regenerate-cache to rebuild.", age_hours,
            )
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    # ------------------------------------------------------------------
    # Full computation
    # ------------------------------------------------------------------
    logger.info("Computing figure_grn data from scratch...")

    # --- Load TF list ---
    tf_list = pd.read_csv(paths["SCENIC_TF_LIST"], header=None)[0].tolist()

    # --- Load cluster labels (produced by figure_morphodynamic_domains) ---
    cluster_labels_5 = pd.read_csv(
        Path(paths["OUTPUT_BASE"]) / "fig_morphodynamic_domains" / "clustering" / "cluster_labels_5.csv",
        index_col=[0, 1],
    ).iloc[:, 0]

    cluster_labels_clean = pd.read_csv(
        Path(paths["OUTPUT_BASE"]) / "fig_morphodynamic_domains" / "clustering" / "cleaned_clusters.csv",
        index_col=[0, 1],
    )
    cluster_labels_13 = cluster_labels_clean.loc[:, "13"]

    cluster_df_5 = pd.DataFrame(
        cluster_labels_5.values, columns=["cluster"], index=cluster_labels_5.index
    )
    cluster_df_13 = pd.DataFrame(
        cluster_labels_13.values, columns=["cluster"], index=cluster_labels_13.index
    )

    # --- Load spatial transcriptome ---
    transcriptome = mudata.read(
        Path(paths["GEP_RESULTS_BASE"]) / "spatial_transcriptome_output" / "best_transcriptome.h5mu"
    )
    transcriptome_df = transcriptome["st"].to_df()
    transcriptome_obs = transcriptome["st"].obs.copy()
    transcriptome_obs[["x", "y"]] = transcriptome_obs[["x", "y"]].astype(int)
    transcriptome_df = pd.concat([transcriptome_df, transcriptome_obs], axis=1)
    transcriptome_df = transcriptome_df.set_index(["x", "y"])
    transcriptome_df.columns = transcriptome_df.columns.str.replace("st:", "")

    # --- Load and process single-cell data ---
    adata = sc.read_h5ad(paths["SC_SINGLE_CELL_H5AD"])
    adata = _process_single_cell(adata, transcriptome)

    # --- Compute SC pseudobulks ---
    sc_group_expression = _compute_sc_pseudobulks(adata, paths, transcriptome_df)

    # --- Extend cluster DataFrames with "outside" positions ---
    outside_coords = transcriptome_df.index.difference(cluster_df_5.index)
    outside_cluster_5 = pd.DataFrame(
        {"cluster": [max(cluster_df_5["cluster"]) + 1] * len(outside_coords)},
        index=outside_coords,
    )
    outside_cluster_13 = pd.DataFrame(
        {"cluster": [max(cluster_df_13["cluster"]) + 1] * len(outside_coords)},
        index=outside_coords,
    )

    extended_cluster_df_5 = pd.concat([cluster_df_5, outside_cluster_5])
    extended_cluster_df_13 = pd.concat([cluster_df_13, outside_cluster_13])

    valid_coords = transcriptome_df.index.intersection(extended_cluster_df_5.index)
    transcriptome_df = transcriptome_df.loc[valid_coords]
    cluster_df_5 = extended_cluster_df_5.loc[valid_coords]
    cluster_df_13 = extended_cluster_df_13.loc[valid_coords]

    # --- Compute tau scores and z-scores ---
    data_5 = _compute_tau_analysis(
        cluster_df_5, transcriptome_df, sc_group_expression, tf_list, n_clusters=5
    )
    data_13 = _compute_tau_analysis(
        cluster_df_13, transcriptome_df, sc_group_expression, tf_list, n_clusters=13
    )

    # Filter 13-cluster tau to genes present in 5-cluster analysis
    data_13["tau_scores"] = data_13["tau_scores"].loc[
        data_13["tau_scores"].index.intersection(data_5["tau_scores"].index)
    ]

    # --- Load adjacency data ---
    adjacency_folder = paths["GRN_ADJACENCY_DIR"]
    adjacency_files = fetch_adjacency_file_paths(folder_path=adjacency_folder)
    combined_adjacencies = load_all_adjacencies(adjacency_files)

    # --- GRN generation (Cytoscape export) ---
    _generate_grn_graphs_all(
        combinations_df_5=data_5["combinations_df"],
        combinations_df_13=data_13["combinations_df"],
        combined_adjacencies=combined_adjacencies,
        tf_list=tf_list,
        z_bin_mean_5=data_5["z_bin_mean"],
        z_bin_mean_13=data_13["z_bin_mean"],
        tau_scores_5=data_5["tau_scores"],
        tau_scores_13=data_13["tau_scores"],
        output_base=output_base,
    )

    # --- Assemble shared_data ---
    shared_data = {
        "tf_list": tf_list,
        "tau_scores_5": data_5["tau_scores"],
        "specific_genes_5": data_5["specific_genes"],
        "specific_genes_13": data_13["specific_genes"],
        "z_bin_mean_5": data_5["z_bin_mean"],
        "z_bin_mean_13": data_13["z_bin_mean"],
        "combinations_df_5": data_5["combinations_df"],
        "otsu_threshold_5": data_5["otsu_threshold"],
    }

    # --- Save cache ---
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(shared_data, f)
    logger.debug("Saved figure_grn cache to %s", cache_path)

    return shared_data


def _process_single_cell(adata, transcriptome):
    """Assign cell types and filter single-cell data."""
    cell_type_to_cluster_ids_map = {
        "Muscle & Trachea": [0, 9],
        "Epithelium Notum": [1, 2, 3, 4, 5, 7],
        "Epithelium Neck": [8],
        "Sensory Organ precursor": [11],
        "Unassigned": [6],
    }

    adata.obs["cell_type"] = "Unassigned"
    for cell_type_name, cluster_id_list in cell_type_to_cluster_ids_map.items():
        adata.obs.loc[
            adata.obs["seurat_clusters"].isin(cluster_id_list), "cell_type"
        ] = cell_type_name
    adata.obs["cell_type"] = adata.obs["cell_type"].astype("category")

    # Split Epithelium Notum based on mapping
    if "meta" in transcriptome.mod and "mapped" in transcriptome["meta"].layers:
        mapped_layer = transcriptome["meta"].layers["mapped"]
        mapped_counts = np.asarray(mapped_layer.sum(axis=0)).ravel()
        meta_cell_names = transcriptome["meta"].var_names
        is_mapped_series = pd.Series(
            mapped_counts > 0, index=meta_cell_names, dtype=bool
        )

        common_cells_map = adata.obs_names.intersection(is_mapped_series.index)
        adata.obs["is_mapped"] = False
        adata.obs.loc[common_cells_map, "is_mapped"] = is_mapped_series.loc[
            common_cells_map
        ].values

        adata.obs["cell_type"] = adata.obs["cell_type"].astype(object)
        mask_ep_notum = adata.obs["cell_type"] == "Epithelium Notum"
        mask_mapped = adata.obs["is_mapped"]

        adata.obs.loc[mask_ep_notum & mask_mapped, "cell_type"] = (
            "Epithelium Notum (Mapped)"
        )
        adata.obs.loc[mask_ep_notum & ~mask_mapped, "cell_type"] = (
            "Epithelium Notum (Unmapped)"
        )
        adata.obs["cell_type"] = adata.obs["cell_type"].astype("category")

        if "is_mapped" in adata.obs.columns:
            adata.obs.drop(columns=["is_mapped"], inplace=True)

    # Remove mapped and unmapped epithelium notum
    for label in ["Epithelium Notum (Mapped)", "Epithelium Notum (Unmapped)"]:
        mask = adata.obs["cell_type"] == label
        if mask.any():
            adata = adata[~mask].copy()
            adata.obs["cell_type"] = adata.obs["cell_type"].astype("category")

    return adata


def _compute_sc_pseudobulks(adata, paths, transcriptome_df):
    """Compute single-cell pseudobulk expression grouped by cell type."""
    raw_counts = pd.read_csv(paths["SC_RAW_COUNTS_CSV_ALL_CELLS"], index_col=0).T

    # Filter to genes matching spatial transcriptome
    common_genes = raw_counts.columns.intersection(transcriptome_df.columns)
    raw_counts = raw_counts.loc[:, common_genes]

    # Normalize to target sum 10,000
    sc_sums = raw_counts.sum(axis=1)
    sc_sums[sc_sums == 0] = 1
    raw_counts = raw_counts.div(sc_sums, axis=0).multiply(10000)

    # Align cells
    common_cells = adata.obs_names.intersection(raw_counts.index)
    raw_counts_filtered = raw_counts.loc[common_cells]
    groups = adata.obs.loc[common_cells, "cell_type"]

    # Group by cell type and mean (genes as rows)
    sc_pseudobulk = raw_counts_filtered.groupby(groups, observed=True).mean().T
    if "Epithelium Notum (Mapped)" in sc_pseudobulk.columns:
        sc_pseudobulk = sc_pseudobulk.drop(columns=["Epithelium Notum (Mapped)"])

    sc_group_column_map = {
        "Epithelium Notum (Unmapped)": "Epithelium Notum (Unmapped)",
        "Sensory Organ precursor": "SOP",
        "Epithelium Neck": "Epithelium Neck",
        "Muscle & Trachea": "Muscle & Trachea",
        "Unassigned": "Unassigned",
    }
    sc_group_cols = [col for col in sc_group_column_map if col in sc_pseudobulk.columns]
    sc_group_expression = sc_pseudobulk.loc[:, sc_group_cols].rename(
        columns={col: sc_group_column_map[col] for col in sc_group_cols}
    )
    return sc_group_expression


def _compute_tau_analysis(cluster_df, transcriptome_df, sc_group_expression, tf_list, n_clusters):
    """Compute tau scores, z-scores, specific genes, and combinations for one clustering."""
    tau_scores_spatial, avg_expr_spatial = compute_tau_scores(
        cluster_df["cluster"], transcriptome_df
    )
    tau_scores = tau_scores_spatial.copy()

    # Build combined expression (spatial + SC)
    sc_groups_aligned = sc_group_expression.reindex(avg_expr_spatial.index).fillna(0)
    combined_expression = pd.concat([avg_expr_spatial, sc_groups_aligned], axis=1)

    # Filter tau to genes whose max is in a primary cluster
    outside_cluster_id = int(cluster_df["cluster"].max())
    primary_clusters = sorted(
        [c for c in avg_expr_spatial.columns if c != outside_cluster_id]
    )
    tau_scores = filter_tau_by_max_expression(tau_scores, combined_expression, primary_clusters)

    # Z-scores
    z_bin_mean = compute_pseudobulk_zscores(combined_expression)

    # Otsu threshold for specificity
    otsu_thresh = threshold_otsu(tau_scores.values)
    specific_genes, _ = find_specific_genes(tau_scores, threshold=otsu_thresh)

    # Combinations (argmax one-hot)
    avg_expression = avg_expr_spatial.loc[tau_scores.index]
    specific_idx = specific_genes[specific_genes == True].index
    combinations = argmax_one_hot(avg_expression.loc[specific_idx])
    combinations_df = pd.DataFrame(
        combinations, columns=avg_expression.columns, index=specific_idx
    )

    # Remove outside cluster column
    if outside_cluster_id in combinations_df.columns:
        combinations_df = combinations_df.drop(columns=[outside_cluster_id])

    # Filter to TFs only
    combinations_df = combinations_df.loc[combinations_df.index.isin(tf_list)]

    return {
        "tau_scores": tau_scores,
        "average_expression": avg_expression,
        "z_bin_mean": z_bin_mean,
        "specific_genes": specific_genes,
        "combinations_df": combinations_df,
        "otsu_threshold": otsu_thresh,
    }


def _generate_grn_graphs_all(
    combinations_df_5,
    combinations_df_13,
    combined_adjacencies,
    tf_list,
    z_bin_mean_5,
    z_bin_mean_13,
    tau_scores_5,
    tau_scores_13,
    output_base,
):
    """Generate GRN graphs and export for Cytoscape.

    Parameters
    ----------
    combinations_df_5, combinations_df_13 : pd.DataFrame
        TF-cluster assignment matrices.
    combined_adjacencies : pd.DataFrame
        Combined adjacency data from all SCENIC runs.
    tf_list : list
        List of transcription factor names.
    z_bin_mean_5, z_bin_mean_13 : pd.DataFrame
        Z-score matrices for cluster expression.
    tau_scores_5, tau_scores_13 : pd.Series
        Tau specificity scores.
    output_base : Path
        Output directory for figure_grn.
    """
    output_base = Path(output_base)

    logger.info("Generating Gene Regulatory Networks")

    for n_top in _N_TOP_VALUES:
        logger.info("Processing n_top = %s", n_top)

        cytoscape_dir = output_base / f"cytoscape_exports_top_{n_top}"
        cytoscape_dir.mkdir(exist_ok=True, parents=True)

        _generate_grn_graphs(
            combinations_df_5,
            combinations_df_13,
            combined_adjacencies,
            tf_list,
            z_bin_mean_5,
            z_bin_mean_13,
            tau_scores_5,
            tau_scores_13,
            n_top,
            cytoscape_dir,
        )


def _generate_grn_graphs(
    combinations_df_5,
    combinations_df_13,
    combined_adjacencies,
    tf_list,
    z_bin_mean_5,
    z_bin_mean_13,
    tau_scores_5,
    tau_scores_13,
    n_top,
    cytoscape_dir,
):
    """Generate GRN graphs for 5 and 13 cluster analyses."""
    # 5-cluster GRNs
    logger.info("Generating GRNs for 5-cluster analysis (n_top=%s)...", n_top)
    grn_graphs_5 = generate_grn_graphs_optimized(
        combinations_df=combinations_df_5,
        combined_adjacencies=combined_adjacencies,
        tf_list=tf_list,
        min_occurrence_fraction=0.8,
        n_top=n_top,
        z_cluster_means=z_bin_mean_5,
    )

    logger.info("Applying Leiden community detection to networks (5 clusters)...")
    for cluster, G in grn_graphs_5.items():
        if G.number_of_nodes() >= 3:
            annotate_network_with_leiden_clusters(G, resolution=1.0)

    for cluster, G in grn_graphs_5.items():
        export_network_for_cytoscape(
            G,
            cytoscape_dir / "5_clusters",
            f"grn_cluster_{cluster}",
            tau_scores=tau_scores_5,
        )

    # 13-cluster GRNs
    logger.info("Generating GRNs for 13-cluster analysis (n_top=%s)...", n_top)
    grn_graphs_13 = generate_grn_graphs_optimized(
        combinations_df=combinations_df_13,
        combined_adjacencies=combined_adjacencies,
        tf_list=tf_list,
        min_occurrence_fraction=0.8,
        n_top=n_top,
        z_cluster_means=z_bin_mean_13,
    )

    logger.info("Applying Leiden community detection to networks (13 clusters)...")
    for cluster, G in grn_graphs_13.items():
        if G.number_of_nodes() >= 3:
            annotate_network_with_leiden_clusters(G, resolution=1.0)

    for cluster, G in grn_graphs_13.items():
        export_network_for_cytoscape(
            G,
            cytoscape_dir / "13_clusters",
            f"grn_cluster_{cluster}",
            tau_scores=tau_scores_13,
        )

    return grn_graphs_5, grn_graphs_13
