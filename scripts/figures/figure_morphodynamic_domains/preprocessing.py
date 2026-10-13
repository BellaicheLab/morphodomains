"""Preprocessing for figure_morphodynamic_domains.

Loads morphological CSV data, runs PCA, performs agglomerative clustering
(k=2..13), cleans cluster labels, and assembles the shared_data dict
consumed by panel functions.
"""

import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.neighbors import radius_neighbors_graph
from sklearn.preprocessing import StandardScaler

from pyanimalprocessing.spatial_statistics import clean_clusters
from scripts.build_morpho_clustering import build as build_morpho_clustering
from scripts.figures.config import aoa_backup

logger = logging.getLogger(__name__)

_CACHE_FILENAME = "_figure_morphodynamic_domains_cache.pkl"

# ── Column renaming ──────────────────────────────────────────────────────────

COLUMNS_RENAMING = {
    # Early timepoint (frame 73)
    "tr_EG_frame73": "G → dilation (early)",
    "tr_ED_frame73": "D → dilation (early)",
    "tr_EA_frame73": "A → dilation (early)",
    "tr_ER_frame73": "R → dilation (early)",
    "tr_ES_frame73": "S → dilation (early)",
    "DiffEigEG_frame73": "G → elongation (early)",
    "DiffEigED_frame73": "D → elongation (early)",
    "DiffEigEA_frame73": "A → elongation (early)",
    "DiffEigER_frame73": "R → elongation (early)",
    "DiffEigES_frame73": "S → elongation (early)",
    "u_frame73": "Velocity (early)",
    "ianiso_frame73": "Anisotropy (early)",
    "dnD_nrmlz_frame73": "Normalized division rate (early)",
    "dnA_nrmlz_frame73": "Normalized apoptosis rate (early)",
    "CellArea_frame73": "Cellular area (early)",
    # Late timepoint (frame 121)
    "tr_EG_frame121": "G → dilation (late)",
    "tr_ED_frame121": "D → dilation (late)",
    "tr_EA_frame121": "A → dilation (late)",
    "tr_ER_frame121": "R → dilation (late)",
    "tr_ES_frame121": "S → dilation (late)",
    "DiffEigEG_frame121": "G → elongation (late)",
    "DiffEigED_frame121": "D → elongation (late)",
    "DiffEigEA_frame121": "A → elongation (late)",
    "DiffEigER_frame121": "R → elongation (late)",
    "DiffEigES_frame121": "S → elongation (late)",
    "u_frame121": "Velocity (late)",
    "ianiso_frame121": "Anisotropy (late)",
    "dnD_nrmlz_frame121": "Normalized division rate (late)",
    "dnA_nrmlz_frame121": "Normalized apoptosis rate (late)",
    "CellArea_frame121": "Cellular area (late)",
}

# ── Color palettes ────────────────────────────────────────────────────────────

COLORS_5 = ["#892A81", "#84CA69", "#FF5720", "#C1BBB1", "#004A20"]
COLORS_2 = [COLORS_5[0], COLORS_5[-1]]

# Clusters 5, 7 and 9 carry the same hues here as in COLORS_13_GREYED, so a reader
# can carry a domain from the greyed panel to the full one by colour. The other
# entries were re-picked around those three for maximum perceptual separation
# (min dE 30 pairwise, and none near the #b3b3b3 the greyed panel uses for
# "not selected").
COLORS_13 = [
    "#1565c7",
    "#E46C6C",
    "#F5C400",
    "#68312A",
    "#0092A1",
    "#A6D3E6",
    "#C25192",
    "#F32F01",
    "#938F28",
    "#18551C",
    "#332288",
    "#66C2A5",
    "#A65628",
]

# Clusters 5, 7 and 9 are the three anterior subdomains that 5-MD5 splits into at
# k=13 (28 + 141 + 45 = 214 bins); every other cluster is greyed so the panel reads
# as that decomposition.
COLORS_13_GREYED = [
    "#b3b3b3FF",
    "#b3b3b3FF",
    "#b3b3b3FF",
    "#b3b3b3FF",
    "#b3b3b3FF",
    "#A6D3E6",
    "#b3b3b3FF",
    "#F32F01",
    "#b3b3b3FF",
    "#18551C",
    "#b3b3b3FF",
    "#b3b3b3FF",
    "#b3b3b3FF",
]

# ── Layout / font-size constants ──────────────────────────────────────────────

SCALE_POSITION = [1900, 2930]
BBOX_X = [3250, 3350]
BBOX_Y_OFFSET = 50

HEATMAP_TICK_LABEL_SIZE = 16
PCA_TICK_LABEL_SIZE = 16
PCA_AXIS_LABEL_SIZE = 18

# ── Quantity config for spatial tensor plots ──────────────────────────────────

QUANTITY_CONFIG = {
    "ER": {
        "quantity_name": "ER",
        "display_name": "R",
        "type": "tensor",
        "scale": 1500,
        "color": "#99288c",
        "vector_style": "line",
        "unit": "h\u207b\u00b9",
        "scale_value": 0.1,
    },
    "CellIaniso": {
        "quantity_name": "CellIaniso",
        "display_name": "Cell anisotropy",
        "type": "scalar",
        "scale": 200,
        "color": "#2e8ac3",
        "vector_style": "line",
        "unit": "",
        "scale_value": 0.5,
    },
    "dnA": {
        "quantity_name": "dnA",
        "display_name": "Apoptosis rate",
        "type": "scalar",
        "scale": 20,
        "color": "#E94F37",
        "vector_style": "line",
        "unit": "h\u207b\u00b9",
        "scale_value": 5,
    },
    "dnD": {
        "quantity_name": "dnD",
        "display_name": "Division rate",
        "type": "scalar",
        "scale": 5,
        "color": "#1d650a",
        "vector_style": "line",
        "unit": "h\u207b\u00b9",
        "scale_value": 20,
    },
    "EG": {
        "quantity_name": "EG",
        "display_name": "G",
        "type": "tensor",
        "scale": 3000,
        "color": "#17357b",
        "vector_style": "line",
        "unit": "h\u207b\u00b9",
        "scale_value": 0.05,
    },
    "EA": {
        "quantity_name": "EA",
        "display_name": "A",
        "type": "tensor",
        "scale": 3000,
        "color": "#474545",
        "vector_style": "line",
        "unit": "h\u207b\u00b9",
        "scale_value": 0.05,
    },
    "ED": {
        "quantity_name": "ED",
        "display_name": "D",
        "type": "tensor",
        "scale": 750,
        "color": "#14812a",
        "vector_style": "line",
        "unit": "h\u207b\u00b9",
        "scale_value": 0.2,
    },
    "ES": {
        "quantity_name": "ES",
        "display_name": "S",
        "type": "tensor",
        "scale": 750,
        "color": "#12d0fb",
        "vector_style": "line",
        "unit": "h\u207b\u00b9",
        "scale_value": 0.2,
    },
    "CellArea": {
        "quantity_name": "CellArea",
        "display_name": "Cell area",
        "type": "scalar",
        "scale": 2.5,
        "color": "#2E86AB",
        "vector_style": "line",
        "unit": "\u00b5m\u00b2",
        "scale_value": 20,
    },
    "U": {
        "quantity_name": "U",
        "display_name": "Velocity",
        "type": "tensor",
        "scale": 20,
        "color": "#FF732E",
        "vector_style": "triangle",
        "unit": "\u00b5m.h\u207b\u00b9",
        "scale_value": 5,
        "linewidth": 5,
        "vector_scale_factor": 1.0,
    },
    "Epsilon": {
        "quantity_name": "Epsilon",
        "display_name": "Deformation rate",
        "type": "tensor",
        "scale": 3000,
        "color": "blue",
        "vector_style": "line",
        "unit": "h\u207b\u00b9",
        "scale_value": 0.05,
        "vector_scale_factor": 1.0,
    },
}


# ── Private helpers ───────────────────────────────────────────────────────────


def _run_pca(Mrph):
    """Scale data and run PCA.

    Returns
    -------
    mypca : PCA
        Fitted PCA object.
    Mrph_PCA : pd.DataFrame
        PCA-transformed data with PC columns.
    Mrph_sc : pd.DataFrame
        StandardScaler-transformed morphological data.
    """
    scY = StandardScaler()
    Y_filtered_sc = scY.fit_transform(Mrph)
    Mrph_sc = pd.DataFrame(Y_filtered_sc, columns=Mrph.columns, index=Mrph.index)

    mypca = PCA()
    Mrph_PCA = mypca.fit_transform(Mrph_sc)
    Mrph_PCA = pd.DataFrame(
        Mrph_PCA,
        index=Mrph_sc.index,
        columns=["PC" + str(i) for i in range(Mrph_PCA.shape[1])],
    )

    return mypca, Mrph_PCA, Mrph_sc


def _clustering_prep(Mrph_sc):
    """Prepare spatial connectivity and PCA for agglomerative clustering.

    Returns
    -------
    coords : np.ndarray
        (N, 2) coordinate array.
    connectivity_matrix : sparse matrix
        Radius-based spatial connectivity.
    data_pca : np.ndarray
        PCA-transformed data, reduced to components explaining 90% variance.
    n_components : int
        Number of PCA components used.
    """
    coords = Mrph_sc.reset_index()[["x", "y"]].values
    coords_transformed = np.column_stack([-coords[:, 0], coords[:, 1]])

    connectivity_matrix = radius_neighbors_graph(
        coords_transformed, radius=np.sqrt(2), mode="connectivity", include_self=False
    )

    pca = PCA()
    data_pca = pca.fit_transform(Mrph_sc)

    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    n_components = np.argmax(cumulative_variance >= 0.9) + 1
    data_pca = data_pca[:, :n_components]

    return coords, connectivity_matrix, data_pca, n_components


def _run_clustering(coords, connectivity_matrix, data_pca, index):
    """Run agglomerative clustering for k=2..13.

    Returns
    -------
    all_clusters : pd.DataFrame
        Columns 2..13, each holding cluster labels.
    silhouette_scores : list of float
        Silhouette score for each k.
    """
    all_clusters = pd.DataFrame(index=index)
    silhouette_scores = []

    for k in range(2, 14):
        model = AgglomerativeClustering(
            n_clusters=k, linkage="ward", connectivity=connectivity_matrix
        )
        labels = model.fit_predict(data_pca)
        all_clusters[k] = labels
        silhouette_scores.append(silhouette_score(data_pca, labels))

    return all_clusters, silhouette_scores


def get_highlighted_clusters_from_colors(colors_13_greyed):
    """Extract non-grey clusters from colors_13_greyed.

    Returns
    -------
    dict
        Keys: cluster_indices, colors, labels, meta_colors.
    """
    grey_color = "#b3b3b3FF"
    aggregated_grey = "#A9A9A9"

    highlighted = {"cluster_indices": [], "colors": []}

    for idx, color in enumerate(colors_13_greyed):
        if color.upper() != grey_color.upper():
            highlighted["cluster_indices"].append(idx)
            highlighted["colors"].append(color)

    highlighted["labels"] = ["Aggregated"] + [
        f"Cluster {idx}" for idx in highlighted["cluster_indices"]
    ]
    highlighted["meta_colors"] = [aggregated_grey] + highlighted["colors"]

    return highlighted


def _any_spatial_tensor_panel_enabled(fig_config):
    """Check if any spatial_* napari panel is enabled."""
    from scripts.figures import is_panel_enabled

    panels = fig_config.get("panels", {})
    return any(
        k.startswith("spatial_")
        and v.get("kind") == "napari"
        and is_panel_enabled(fig_config, k)
        for k, v in panels.items()
    )


def _load_spatial_tensor_data(paths, shared_data):
    """Load zarr refs and clean outlines for spatial scenes.

    Zarr groups are not picklable, so this must run after the cache block.
    Adds keys directly to *shared_data*: ``AOA_vi_4h``, ``animal``,
    ``clean_outline_spatial``, ``available_quantities``,
    ``tracking_mask``.
    """
    import zarr
    from skimage.measure import label as skimage_label

    animal = zarr.open_group(paths["MOVIE_ANIMAL_PATH"], "a")
    avgAnimal = zarr.open(paths["AVERAGES_240920_LAGRANGIAN"])

    AOA_vi_4h = aoa_backup(avgAnimal, "AOA_avgAnimals_vi_4h_0.98olap")

    # Clean tissue outlines
    logger.info("Processing tissue outlines for spatial tensors...")
    raw_outline = np.array(animal.IMAGE.D2.outlines).astype(float)
    clean_outline = np.empty_like(raw_outline)
    clean_outline[:] = np.nan
    for i in range(raw_outline.shape[0]):
        tmp = skimage_label(raw_outline[i], connectivity=1)
        tmp = tmp != 1
        clean_outline[i][tmp] = raw_outline[i][tmp]
    # Fill outside-tissue NaN with white (1.0) for clean look on light theme
    np.nan_to_num(clean_outline, copy=False, nan=1.0)

    # Which quantities are available in the zarr?
    available_in_zarr = set(AOA_vi_4h.keys()) - {"Macrocaetes"}
    available_quantities = [q for q in QUANTITY_CONFIG if q in available_in_zarr]

    # Tracking mask for Lagrangian origins
    tracking_mask = np.array(animal["TRACKING"]["masks_division_agnostic"])
    logger.info("Loaded tracking mask: %s", tracking_mask.shape)

    shared_data["AOA_vi_4h"] = AOA_vi_4h
    shared_data["animal"] = animal
    shared_data["clean_outline_spatial"] = clean_outline
    shared_data["available_quantities"] = available_quantities
    shared_data["tracking_mask"] = tracking_mask


def _save_outputs(shared_data, output_base, fig_config):
    """Ensure CSV outputs exist (idempotent, skips already-written files)."""
    outputs_cfg = fig_config.get("outputs", {})

    if outputs_cfg.get("cleaned_clusters", {}).get("enabled", True):
        csv_path = output_base / outputs_cfg.get("cleaned_clusters", {}).get(
            "path", "clustering/cleaned_clusters.csv"
        )
        if not csv_path.exists():
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            shared_data["cluster_labels_clean"].to_csv(csv_path)
            logger.debug("Saved cleaned_clusters.csv to %s", csv_path)

    if outputs_cfg.get("cluster_labels_5", {}).get("enabled", True):
        csv_path = output_base / outputs_cfg.get("cluster_labels_5", {}).get(
            "path", "clustering/cluster_labels_5.csv"
        )
        if not csv_path.exists():
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            shared_data["all_clusters"][5].to_csv(csv_path)
            logger.debug("Saved cluster_labels_5.csv to %s", csv_path)

    if outputs_cfg.get("pca_scores", {}).get("enabled", True):
        csv_path = output_base / outputs_cfg.get("pca_scores", {}).get(
            "path", "pca/pca_scores.csv"
        )
        if not csv_path.exists():
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            shared_data["Mrph_PCA"].iloc[:, :13].to_csv(csv_path)
            logger.debug("Saved pca_scores.csv to %s", csv_path)


# ── Main loader ───────────────────────────────────────────────────────────────


def load_morphodynamic_domains_data(paths, fig_config):
    """Load and preprocess all data needed by figure_morphodynamic_domains panels.

    When ``regenerate_cache`` is *false* (the default) and a cache pickle
    exists, shared_data is loaded directly from disk, skipping all
    computation.  Set ``regenerate_cache: true`` in the YAML to force a
    full recomputation and update the cache.

    Parameters
    ----------
    paths : dict
        Resolved paths from config (uppercase keys).
    fig_config : dict
        The figure_morphodynamic_domains config section.

    Returns
    -------
    dict
        Shared data dict consumed by all figure_morphodynamic_domains panel functions.
    """
    output_base = Path(paths["OUTPUT_BASE"]) / fig_config["output_dir"]
    cache_path = output_base / _CACHE_FILENAME

    regenerate = fig_config.get("regenerate_cache", False)

    if not regenerate and cache_path.exists():
        import time

        age_hours = (time.time() - cache_path.stat().st_mtime) / 3600
        logger.info(
            "Loading figure_morphodynamic_domains cache from %s (age: %.1fh)", cache_path, age_hours
        )
        if age_hours > 72:
            logger.warning(
                "Cached figure_morphodynamic_domains data is %.1fh old. "
                "Run with --regenerate-cache to rebuild.",
                age_hours,
            )
        with open(cache_path, "rb") as f:
            shared_data = pickle.load(f)

        # Ensure CSV outputs exist
        _save_outputs(shared_data, output_base, fig_config)

        # Load zarr-based spatial tensor data (not picklable → after cache)
        if _any_spatial_tensor_panel_enabled(fig_config):
            _load_spatial_tensor_data(paths, shared_data)

        return shared_data

    # ------------------------------------------------------------------
    # Full computation
    # ------------------------------------------------------------------
    logger.info("Computing figure_morphodynamic_domains data from scratch...")

    # --- Load raw data ---
    macrochaete = pd.read_csv(paths["MACROCHAETE_CSV"], header=None)
    Mrph = build_morpho_clustering(paths["AVERAGES_240920_LAGRANGIAN"])
    Mrph = Mrph.rename(columns=COLUMNS_RENAMING)

    # --- PCA ---
    mypca, Mrph_PCA, Mrph_sc = _run_pca(Mrph)

    # --- Clustering prep ---
    coords, connectivity_matrix, data_pca, n_components = _clustering_prep(Mrph_sc)

    # --- Run clustering k=2..13 ---
    all_clusters, silhouette_scores = _run_clustering(
        coords, connectivity_matrix, data_pca, Mrph_sc.index
    )

    # --- Per-window silhouette curves (same pipeline on each single 4h window) ---
    # Lets the silhouette panel compare the optimal k of the early/late windows
    # against the pooled (merged) matrix.
    silhouette_by_period = {"merged": silhouette_scores}
    for name, frame_indices in {"early": {"frame73": 72}, "late": {"frame121": 120}}.items():
        df_window = build_morpho_clustering(paths["AVERAGES_240920_LAGRANGIAN"], frame_indices)
        _, _, sc_window = _run_pca(df_window)
        _, conn_window, pca_window, _ = _clustering_prep(sc_window)
        _, silhouette_by_period[name] = _run_clustering(
            None, conn_window, pca_window, sc_window.index
        )

    # --- Clean clusters ---
    cluster_labels_clean = clean_clusters(all_clusters.copy())

    # --- Save outputs (for figure_grn dependency) ---
    outputs_cfg = fig_config.get("outputs", {})

    clustering_dir = output_base / "clustering"
    clustering_dir.mkdir(parents=True, exist_ok=True)

    if outputs_cfg.get("cleaned_clusters", {}).get("enabled", True):
        csv_path = output_base / outputs_cfg.get("cleaned_clusters", {}).get(
            "path", "clustering/cleaned_clusters.csv"
        )
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        cluster_labels_clean.to_csv(csv_path)
        logger.debug("Saved cleaned_clusters.csv to %s", csv_path)

    if outputs_cfg.get("cluster_labels_5", {}).get("enabled", True):
        csv_path = output_base / outputs_cfg.get("cluster_labels_5", {}).get(
            "path", "clustering/cluster_labels_5.csv"
        )
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        # 5-cluster dendrogram labels
        model_5 = AgglomerativeClustering(
            n_clusters=5, linkage="ward", connectivity=connectivity_matrix
        )
        cluster_labels_5 = model_5.fit_predict(data_pca)
        pd.Series(cluster_labels_5, index=Mrph_sc.index).to_csv(csv_path)
        logger.debug("Saved cluster_labels_5.csv to %s", csv_path)

    if outputs_cfg.get("pca_scores", {}).get("enabled", True):
        csv_path = output_base / outputs_cfg.get("pca_scores", {}).get(
            "path", "pca/pca_scores.csv"
        )
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        Mrph_PCA.iloc[:, :13].to_csv(csv_path)
        logger.debug("Saved pca_scores.csv to %s", csv_path)

    # --- Assemble shared_data ---
    shared_data = {
        "Mrph": Mrph,
        "macrochaete": macrochaete,
        "mypca": mypca,
        "Mrph_PCA": Mrph_PCA,
        "Mrph_sc": Mrph_sc,
        "coords": coords,
        "connectivity_matrix": connectivity_matrix,
        "data_pca": data_pca,
        "n_components": n_components,
        "all_clusters": all_clusters,
        "cluster_labels_clean": cluster_labels_clean,
        "silhouette_scores": silhouette_scores,
        "silhouette_by_period": silhouette_by_period,
    }

    # --- Save cache ---
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(shared_data, f)
    logger.debug("Saved figure_morphodynamic_domains cache to %s", cache_path)

    # Load zarr-based spatial tensor data (not picklable → after cache)
    if _any_spatial_tensor_panel_enabled(fig_config):
        _load_spatial_tensor_data(paths, shared_data)

    return shared_data
