"""Preprocessing for the temporal-robustness figure.

The reference clustering pools two 4h windows (early frame73 ≈ 18–22 hAPF +
late frame121 ≈ 22–26 hAPF) into one 30-feature matrix. Here the full pipeline
(build → StandardScaler → PCA ≥90% variance → Ward + spatial connectivity →
``clean_clusters``) is run on each window separately, alongside the pooled one.

Every partition has five domains. Ward splits off a 2-bin outlier domain in the
early window at every k from 4 to 8, so early is cut at k=6 and that domain is
dropped. Each window's silhouette-optimal k (k=2..13) is also recorded.
"""

import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from pyanimalprocessing.spatial_statistics import clean_clusters
from scripts.build_morpho_clustering import build as build_morpho_clustering
from scripts.figures.figure_morphodynamic_domains.preprocessing import (
    COLORS_5,
    COLUMNS_RENAMING,
    _clustering_prep,
    _run_pca,
)

logger = logging.getLogger(__name__)

_CACHE_FILENAME = "_figure_clustering_temporal_cache.pkl"

# Ward cut per window: 5 everywhere except early, which needs 6 to yield 5
# well-populated domains once its outlier pair is dropped (see MIN_DOMAIN_BINS).
PERIOD_K = {"early": 6, "late": 5, "pooled": 5}
MIN_DOMAIN_BINS = 10
# Id and colour for a domain with no counterpart in the pooled partition. The colour is
# deliberately not a grey — COLORS_5 already spends #c0c0c0 on 5-MD4.
UNMATCHED_ID = len(COLORS_5)
NEUTRAL_COLOR = "#efe4d4"
# 4h-window frame indices per developmental period (see zarr TimeArray).
PERIOD_FRAMES = {
    "early": {"frame73": 72},
    "late": {"frame121": 120},
    "pooled": {"frame73": 72, "frame121": 120},
}
PERIOD_LABELS = {
    "early": "18–22 hAPF",
    "late": "22–26 hAPF",
    "pooled": "Pooled (18–26 hAPF)",
}

# Single-window periods that get a per-property cluster heatmap (pooled excluded).
HEATMAP_TIMEPOINTS = ["early", "late"]
# Base feature → display name (figure_morphodynamic_domains names with the "(early)" timepoint suffix stripped).
BASE_FEATURE_NAMES = {
    col.replace("_frame73", ""): name.replace(" (early)", "")
    for col, name in COLUMNS_RENAMING.items()
    if col.endswith("_frame73")
}


def _cluster_period(df, n_clusters):
    """Full pipeline on one window: PCA ≥90% → Ward+connectivity → clean.

    Domains below ``MIN_DOMAIN_BINS`` are dropped — a handful of outlier bins is
    not a morphodynamic domain, and keeping one would saturate the shared z-score
    scale of the property heatmaps. The silhouette-optimal k is also swept
    (k=2..13) and returned for reporting, but never used for the cut.
    """
    _, _, Mrph_sc = _run_pca(df)
    _, connectivity_matrix, data_pca, n_pcs = _clustering_prep(Mrph_sc)

    silhouettes = []
    for k in range(2, 14):
        km = AgglomerativeClustering(
            n_clusters=k, linkage="ward", connectivity=connectivity_matrix
        ).fit_predict(data_pca)
        silhouettes.append(silhouette_score(data_pca, km))
    k_optimal = int(np.argmax(silhouettes)) + 2

    labels = AgglomerativeClustering(
        n_clusters=n_clusters, linkage="ward", connectivity=connectivity_matrix
    ).fit_predict(data_pca)
    labels = clean_clusters(pd.DataFrame({0: labels}, index=Mrph_sc.index))[0]

    sizes = labels.value_counts()
    labels = labels[labels.isin(sizes[sizes >= MIN_DOMAIN_BINS].index)]
    return labels, n_pcs, k_optimal


def _match_to_reference(labels, reference):
    """Give a window domain a pooled colour only when the two are each other's best match.

    A Hungarian bijection forces a one-to-one correspondence that does not exist here:
    the late window has a domain whose bins lie entirely inside the pooled posterior
    domain, and pairing it with the pooled domain left over implies a migration that
    never happened. Requiring mutual plurality instead leaves such a domain unmatched,
    and it is drawn in a neutral colour.
    """
    common = labels.index.intersection(reference.index)
    overlap = pd.crosstab(labels.loc[common], reference.loc[common])
    best_reference = overlap.idxmax(axis=1)
    best_window = overlap.idxmax(axis=0)

    mapping = {}
    for domain in np.unique(labels.to_numpy()):
        if domain in overlap.index:
            candidate = best_reference[domain]
            if best_window[candidate] == domain:
                mapping[domain] = candidate
                continue
        mapping[domain] = UNMATCHED_ID
    return labels.map(mapping)


def load_clustering_temporal_data(paths, fig_config):
    """Cluster each period independently, align colours, and score pairwise ARI.

    When ``regenerate_cache`` is false (default) and the cache exists, the
    pickle is loaded directly.  Set ``regenerate_cache: true`` to rebuild.

    Parameters
    ----------
    paths : dict
        Resolved paths from config (uppercase keys).
    fig_config : dict
        The figure_clustering_temporal config section.

    Returns
    -------
    dict
        Shared data consumed by the panel functions.
    """
    output_base = Path(paths["OUTPUT_BASE"]) / fig_config.get(
        "output_dir", "fig_clustering_temporal"
    )
    cache_path = output_base / _CACHE_FILENAME

    if not fig_config.get("regenerate_cache", False) and cache_path.exists():
        logger.info("Loading clustering-temporal cache from %s", cache_path)
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    logger.info("Computing clustering-temporal data from scratch...")

    macrochaete = pd.read_csv(paths["MACROCHAETE_CSV"], header=None)
    zarr_path = paths["AVERAGES_240920_LAGRANGIAN"]

    # --- Cluster each period (full pipeline, own ≥90% PCA space) ---
    raw_labels = {}
    n_pcs = {}
    k_optimal = {}
    geometry = {}
    features = {}  # raw 15-property matrix per single window (base-named columns)
    for name, frame_indices in PERIOD_FRAMES.items():
        df = build_morpho_clustering(zarr_path, frame_indices)
        raw_labels[name], n_pcs[name], k_optimal[name] = _cluster_period(
            df, PERIOD_K[name]
        )
        # Bins in a dropped sub-threshold domain leave the window entirely.
        n_dropped = len(df) - len(raw_labels[name])
        df = df.loc[raw_labels[name].index]
        geometry[name] = df.reset_index()[["x", "y"]]
        if name in HEATMAP_TIMEPOINTS:
            base = df.rename(columns=lambda c: c.rsplit("_frame", 1)[0])
            features[name] = base.rename(columns=BASE_FEATURE_NAMES)
        logger.info(
            "%s: cut at k=%d → %d domains over %d bins (%d dropped below %d), "
            "%d PCs (≥90%% var), own silhouette optimum k=%d",
            name, PERIOD_K[name], raw_labels[name].nunique(), len(df),
            n_dropped, MIN_DOMAIN_BINS, n_pcs[name], k_optimal[name],
        )

    # --- Per-property z-scores, normalized across all bins of the 3 windows ---
    pooled_bins = pd.concat([features[t] for t in HEATMAP_TIMEPOINTS], axis=0)
    scaler = StandardScaler().fit(pooled_bins.to_numpy())
    zscored_features = {
        t: pd.DataFrame(
            scaler.transform(features[t].to_numpy()),
            index=features[t].index,
            columns=features[t].columns,
        )
        for t in HEATMAP_TIMEPOINTS
    }

    # --- Colour each window by the pooled domain it actually corresponds to ---
    reference = raw_labels["pooled"]
    aligned = {
        name: _match_to_reference(labels, reference)
        for name, labels in raw_labels.items()
    }
    for name in aligned:
        unmatched = int((aligned[name] == UNMATCHED_ID).sum())
        if unmatched:
            logger.info(
                "%s: %d bins in domains with no counterpart in the pooled partition",
                name, unmatched,
            )

    # --- Pairwise ARI on each pair's common bins ---
    period_names = ["early", "late", "pooled"]
    ari_matrix = np.ones((len(period_names), len(period_names)))
    for i, a in enumerate(period_names):
        for j, b in enumerate(period_names):
            if i < j:
                common = raw_labels[a].index.intersection(raw_labels[b].index)
                score = adjusted_rand_score(
                    raw_labels[a].loc[common], raw_labels[b].loc[common]
                )
                ari_matrix[i, j] = ari_matrix[j, i] = score
                logger.info("ARI %s vs %s = %.3f", a, b, score)

    periods = {
        name: {
            "labels": aligned[name],
            "geometry": geometry[name],
            "n_pcs": n_pcs[name],
            "n_domains": int(aligned[name].nunique()),
            "k_cut": PERIOD_K[name],
            "k_optimal": k_optimal[name],
            "n_bins": len(aligned[name]),
        }
        for name in period_names
    }

    shared_data = {
        "period_names": period_names,
        "periods": periods,
        "ari_matrix": ari_matrix,
        "zscored_features": zscored_features,
        "macrochaete": macrochaete,
    }

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(shared_data, f)
    logger.debug("Saved clustering-temporal cache to %s", cache_path)

    return shared_data