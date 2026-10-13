"""Preprocessing for the clustering-stability figure.

Re-runs the figure_morphodynamic_domains morphogenetic clustering (Ward + spatial connectivity,
k=5, with spatial cleaning) while sweeping the number of PCA components, and
builds consensus matrices that quantify how stable the 5-cluster partition is
to that choice. The clustering is deterministic, so the number of PCs is the
sole perturbation axis.
"""

import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score

from pyanimalprocessing.spatial_statistics import clean_clusters
from scripts.build_morpho_clustering import build as build_morpho_clustering
from scripts.figures.figure_morphodynamic_domains.preprocessing import (
    COLUMNS_RENAMING,
    _clustering_prep,
    _run_pca,
)

logger = logging.getLogger(__name__)

_CACHE_FILENAME = "_figure_clustering_stability_cache.pkl"

N_CLUSTERS = 5
PC_RANGE_FULL = range(2, 31)      # broad: all 30 PCs
PC_RANGE_NARROW = range(10, 21)   # narrow band around the reference 14-PC choice


def _cluster_at(data_pca, connectivity_matrix, index):
    """Ward clustering at k=5 followed by the spatial cleaning step."""
    labels = AgglomerativeClustering(
        n_clusters=N_CLUSTERS, linkage="ward", connectivity=connectivity_matrix
    ).fit_predict(data_pca)
    cleaned = clean_clusters(pd.DataFrame({0: labels}, index=index))
    return cleaned[0].to_numpy()


def _consensus(labels_by_p, pc_values):
    """Mean co-clustering matrix over the given PC counts (bin x bin, in [0, 1])."""
    partitions = [labels_by_p[p] for p in pc_values]
    n = partitions[0].shape[0]
    co = np.zeros((n, n))
    for labels in partitions:
        co += labels[:, None] == labels[None, :]
    return co / len(partitions)


def _per_bin_stability(consensus, reference_labels):
    """Mean consensus of each bin with the other bins of its reference cluster."""
    ref = reference_labels.to_numpy()
    stability = np.empty(consensus.shape[0])
    for i in range(consensus.shape[0]):
        same = ref == ref[i]
        same[i] = False
        stability[i] = consensus[i, same].mean() if same.any() else np.nan
    return pd.Series(stability, index=reference_labels.index, name="stability")


def load_clustering_stability_data(paths, fig_config):
    """Sweep #PCs, build consensus matrices, and assemble the shared_data dict.

    When ``regenerate_cache`` is false (default) and the cache exists, the
    pickle is loaded directly.  Set ``regenerate_cache: true`` to rebuild.

    Parameters
    ----------
    paths : dict
        Resolved paths from config (uppercase keys).
    fig_config : dict
        The figure_clustering_stability config section.

    Returns
    -------
    dict
        Shared data consumed by the panel functions.
    """
    output_base = Path(paths["OUTPUT_BASE"]) / fig_config.get(
        "output_dir", "fig_clustering_stability"
    )
    cache_path = output_base / _CACHE_FILENAME

    if not fig_config.get("regenerate_cache", False) and cache_path.exists():
        logger.info("Loading clustering-stability cache from %s", cache_path)
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    logger.info("Computing clustering-stability data from scratch...")

    # --- Reuse the figure_morphodynamic_domains inputs: features, PCA scores, spatial connectivity ---
    macrochaete = pd.read_csv(paths["MACROCHAETE_CSV"], header=None)
    Mrph = build_morpho_clustering(paths["AVERAGES_240920_LAGRANGIAN"])
    Mrph = Mrph.rename(columns=COLUMNS_RENAMING)
    _, Mrph_PCA, Mrph_sc = _run_pca(Mrph)
    coords, connectivity_matrix, data_pca_90, n_components = _clustering_prep(Mrph_sc)
    logger.info("Reference clustering uses %d PCs (90%% variance)", n_components)

    index = Mrph_sc.index
    geometry = Mrph_PCA.reset_index()[["x", "y"]]

    # --- Reference 5-cluster partition: ordering / annotation / ARI reference ---
    reference_labels_5 = pd.Series(
        _cluster_at(data_pca_90, connectivity_matrix, index), index=index
    )

    # --- One sweep over the full PC range (the narrow range is a sub-slice) ---
    scores = Mrph_PCA.to_numpy()
    labels_by_p = {}
    ari = {}
    for p in PC_RANGE_FULL:
        labels = _cluster_at(scores[:, :p], connectivity_matrix, index)
        labels_by_p[p] = labels
        ari[p] = adjusted_rand_score(reference_labels_5.to_numpy(), labels)
        logger.info("PCs=%2d  ARI vs reference=%.3f", p, ari[p])

    ranges = {"2_30": list(PC_RANGE_FULL), "10_20": list(PC_RANGE_NARROW)}
    consensus = {key: _consensus(labels_by_p, pcs) for key, pcs in ranges.items()}
    stability_per_bin = {
        key: _per_bin_stability(mat, reference_labels_5)
        for key, mat in consensus.items()
    }

    # Shared bin order: group by reference cluster so both heatmaps are comparable
    order = np.argsort(reference_labels_5.to_numpy(), kind="stable")

    shared_data = {
        "index": index,
        "coords": coords,
        "geometry": geometry,
        "macrochaete": macrochaete,
        "n_components": n_components,
        "reference_labels_5": reference_labels_5,
        "labels_by_p": labels_by_p,
        "consensus": consensus,
        "ari": ari,
        "stability_per_bin": stability_per_bin,
        "order": order,
    }

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(shared_data, f)
    logger.debug("Saved clustering-stability cache to %s", cache_path)

    return shared_data