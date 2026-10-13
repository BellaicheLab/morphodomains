"""Preprocessing for the spatio-temporal clustering figure.

The reference partition clusters one row per spatial bin (two pooled 4h windows) under a
spatial contiguity constraint. Here each observation is a (bin, frame) pair and Ward runs
under a graph linking spatial neighbours within a frame and the same bin across adjacent
frames, so domains are contiguous in space and time.

Same pipeline as the reference (``_run_pca`` then Ward under connectivity, PCs up to >=90%
variance), except:

1. the connectivity graph carries temporal edges (``_space_time_graph``);
2. features are z-scored across space and time (15 columns), not per window (30 columns);
3. ``_clean_frames`` replaces ``clean_clusters``: within a frame, pieces below
   ``MIN_PIECE_BINS`` are dissolved into their surroundings; a domain's largest piece
   always survives.

Frames are the 45 unique 4h sliding windows with centres 19h58 to 24h03 hAPF. They overlap
by 98%, so part of the temporal smoothness comes from the windowing itself.
"""

import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import zarr
from scipy.sparse.csgraph import connected_components
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.neighbors import radius_neighbors_graph

from pyanimalprocessing.rescaling.timings import median_time
from pyanimalprocessing.spatial_statistics import clean_clusters
from scripts.build_morpho_clustering import build as build_morpho_clustering
from scripts.figures.config import aoa_backup
from scripts.figures.figure_morphodynamic_domains.preprocessing import (
    _clustering_prep,
    _run_pca,
)
from scripts.figures.figure_clustering_stability.preprocessing import (
    _consensus,
    _per_bin_stability,
)
from scripts.figures._clustering_helpers import align_to_reference

logger = logging.getLogger(__name__)

_CACHE_FILENAME = "_figure_clustering_spatiotemporal_cache_k{}.pkl"

# The reference partition always has 5 domains — it is what the space-time
# result is compared against, and is independent of the k used for the space-time cut.
N_CLUSTERS_REFERENCE = 5
AOA_GROUP_4H = "AOA_avgAnimals_vi_4h_0.98olap"
# 4h-window slots whose window lies inside 18-26 hAPF (see the zarr TimeArray): window
# centres 19h58 to 24h03, delimited by the early and late windows of the reference partition.
SLOT_RANGE = range(72, 122)
# Within a frame, 4-connected pieces below this many bins are dissolved.
MIN_PIECE_BINS = 3
# Silhouette sweep. Undefined at k=1, and beyond 8 the extra domains have no reference
# counterpart to compare against.
K_RANGE = range(2, 9)
N_STILLS = 5


def _retained_slots(zarr_path):
    """Slots covering 18-26 hAPF, with the bit-identical duplicates removed.

    The 4h TimeArray has 217 slots but only 196 distinct windows; a consecutive
    duplicate carries the same data and would enter the clustering twice.
    """
    g4 = aoa_backup(zarr.open(str(zarr_path), mode="r"), AOA_GROUP_4H)
    time_array = [tuple(t) for t in g4.attrs["TimeArray"]]

    slots, windows = [], []
    for i in SLOT_RANGE:
        if slots and time_array[i] == time_array[slots[-1]]:
            continue
        slots.append(i)
        windows.append(time_array[i])
    return slots, windows


def _stack_frames(zarr_path, slots):
    """One 15-column feature frame per slot, each with its own (x, y) index.

    ``build`` is called one slot at a time on purpose: a multi-slot call pools the
    features column-wise and intersects the surviving bins across every slot, which
    is the opposite of the row-wise stacking we need.
    """
    frames = []
    for slot in slots:
        df = build_morpho_clustering(zarr_path, {f"frame{slot + 1}": slot})
        frames.append(df.rename(columns=lambda c: c.rsplit("_frame", 1)[0]))
    return frames


def _space_time_graph(frames):
    """8-connected within each frame, plus an edge joining a bin to itself at t±1.

    The spatial blocks use the same ``radius=sqrt(2)`` neighbourhood as the reference
    2D pipeline (``_clustering_prep``); that function cannot be reused here because it
    would run the radius search over the stacked frames, where every (x, y) recurs
    once per frame as a coincident point.

    Returns the stacked graph and the per-frame spatial blocks, which
    ``_clean_frames`` needs to widen a doomed bin's vote to its 8-neighbourhood.
    """
    blocks, coords, offsets = [], [], []
    total = 0
    for df in frames:
        c = df.reset_index()[["x", "y"]].to_numpy()
        blocks.append(
            radius_neighbors_graph(
                c, radius=np.sqrt(2), mode="connectivity", include_self=False
            )
        )
        coords.append(c)
        offsets.append(total)
        total += len(c)

    graph = sp.block_diag(blocks, format="csr")

    rows, cols = [], []
    for j in range(len(coords) - 1):
        position = {tuple(v): i for i, v in enumerate(coords[j])}
        for i, v in enumerate(coords[j + 1]):
            previous = position.get(tuple(v))
            if previous is not None:
                rows.append(offsets[j] + previous)
                cols.append(offsets[j + 1] + i)

    temporal = sp.coo_matrix(
        (np.ones(len(rows)), (rows, cols)), shape=graph.shape
    ).tocsr()
    graph = graph + temporal
    graph = graph + graph.T
    graph.data[:] = 1.0
    return graph, blocks


def _small_pieces(frame_labels, block4):
    """Mask of bins in a piece below ``MIN_PIECE_BINS``, excluding each domain's largest."""
    small = np.zeros(len(frame_labels), dtype=bool)
    for domain in np.unique(frame_labels):
        member = np.flatnonzero(frame_labels == domain)
        _, piece = connected_components(block4[member][:, member], directed=False)
        sizes = np.bincount(piece)
        doomed = (sizes < MIN_PIECE_BINS) & (np.arange(len(sizes)) != sizes.argmax())
        small[member[doomed[piece]]] = True
    return small


def _clean_frames(labels, frames, blocks):
    """Dissolve every 4-connected piece below ``MIN_PIECE_BINS``, frame by frame.

    Pieces are 4-connected on purpose: a bin touching its domain only at a corner is its
    own piece, and that is the case that reads as a speck on screen even though Ward's
    sqrt(2) graph considers it attached.

    A doomed bin takes the majority label of its 4-neighbours that are not themselves
    doomed, widening to the 8-neighbourhood when the cardinal ones are all doomed — the
    same widening ``clean_clusters`` uses to break a tie. Excluding doomed bins from the
    vote means a reassigned bin always joins a piece that survives, so the doomed set
    shrinks and the loop ends; a bin that never gains a surviving neighbour is kept.
    """
    cleaned, n_moved, offset = [], 0, 0
    for df, block8 in zip(frames, blocks):
        coords = df.reset_index()[["x", "y"]].to_numpy()
        block4 = radius_neighbors_graph(
            coords, radius=1.0, mode="connectivity", include_self=False
        )
        frame = labels[offset : offset + len(df)]
        small = _small_pieces(frame, block4)
        while small.any():
            updated = frame.copy()
            for i in np.flatnonzero(small):
                for block in (block4, block8):
                    neighbours = block.indices[block.indptr[i] : block.indptr[i + 1]]
                    donors = frame[neighbours[~small[neighbours]]]
                    if len(donors):
                        updated[i] = np.bincount(donors).argmax()
                        break
            if np.array_equal(updated, frame):
                break
            n_moved += int((updated != frame).sum())
            frame = updated
            small = _small_pieces(frame, block4)
        cleaned.append(frame)
        offset += len(df)
    return np.concatenate(cleaned), n_moved


def _pooled_reference(zarr_path, n_clusters=N_CLUSTERS_REFERENCE):
    """The reference partition, recomputed from the zarr."""
    Mrph = build_morpho_clustering(zarr_path)
    _, _, Mrph_sc = _run_pca(Mrph)
    _, connectivity_matrix, data_pca, _ = _clustering_prep(Mrph_sc)
    labels = AgglomerativeClustering(
        n_clusters=n_clusters,
        linkage="ward",
        connectivity=connectivity_matrix,
    ).fit_predict(data_pca)
    return clean_clusters(pd.DataFrame({0: labels}, index=Mrph_sc.index))[0]


def _summarise_over_time(labels, modal, frames, reference, centres, n_stills=N_STILLS):
    """Per-bin, per-frame and consensus statistics of a partition over the windows.

    ``labels`` is indexed by the (frame, x, y) stack and its ids are already aligned to
    ``reference``.
    """
    counts = labels.groupby(level=["x", "y"]).value_counts().unstack(fill_value=0)
    n_obs = counts.sum(axis=1)
    # A peripheral bin can be seen only a handful of times, which would make its
    # persistence 1.0 by construction. Half the interval is the cutoff.
    min_bin_frames = int(np.ceil(len(frames) / 2))
    persistence = (counts.max(axis=1) / n_obs).where(n_obs >= min_bin_frames)
    logger.info(
        "bins: %d seen in all %d frames, %d masked from the persistence map "
        "(< %d frames)",
        int((n_obs == len(frames)).sum()), len(frames),
        int((n_obs < min_bin_frames).sum()), min_bin_frames,
    )

    common = modal.index.intersection(reference.index)
    ari = adjusted_rand_score(modal.loc[common], reference.loc[common])
    # A domain present in only part of the interval means Ward cut along time
    # rather than space — the partition would then not be a domain map at all.
    span = labels.groupby(labels).agg(
        n_obs="size",
        n_frames=lambda s: s.index.get_level_values("frame").nunique(),
        n_bins=lambda s: s.droplevel("frame").index.nunique(),
    )
    logger.info("domains over space-time (of %d frames):\n%s", len(frames), span)
    logger.info(
        "ARI(modal domain, reference 5-MD) = %.3f on %d common bins; "
        "mean persistence = %.3f",
        ari, len(common), persistence.mean(),
    )

    labels_by_frame = [labels.xs(j, level="frame") for j in range(len(frames))]
    geometry_by_frame = [df.reset_index()[["x", "y"]] for df in frames]

    # --- Temporal consensus: the clustering-stability statistic with windows in place of PC counts ---
    # Co-clustering is label-free, so unlike persistence it does not depend on how domains
    # are aligned across windows — a bin scores high when it keeps the same company, even
    # if the domain it belongs to is renamed or moves. Restricted to bins present in every
    # window so the matrix is complete.
    temporal_core = labels_by_frame[0].index
    for series in labels_by_frame[1:]:
        temporal_core = temporal_core.intersection(series.index)
    partitions = {
        j: series.loc[temporal_core].to_numpy()
        for j, series in enumerate(labels_by_frame)
    }
    temporal_consensus = _consensus(partitions, range(len(labels_by_frame)))
    temporal_stability = _per_bin_stability(temporal_consensus, modal.loc[temporal_core])
    logger.info(
        "temporal consensus over %d bins present in every window: mean per-bin "
        "stability %.3f (min %.3f)",
        len(temporal_core), temporal_stability.mean(), temporal_stability.min(),
    )

    # --- Agreement with the pooled partition, bin by bin and frame by frame ---
    # Restricted to bins present in every frame and in the pooled partition, so the
    # matrix is rectangular and every row is comparable.
    core = reference.index
    for series in labels_by_frame:
        core = core.intersection(series.index)
    agreement = pd.DataFrame(
        {j: labels_by_frame[j].loc[core] for j in range(len(frames))}, index=core
    )
    matches = agreement.eq(reference.loc[core], axis=0)
    ari_by_frame = [
        adjusted_rand_score(agreement[j], reference.loc[core])
        for j in range(len(frames))
    ]
    # Rows are ordered by when a bin first leaves its pooled domain; bins that never
    # leave sort last. Ordering spatially instead shows the transition is diffuse
    # (Spearman rho = 0.28 vs the A-P axis for 5-MD5), so this is a switch-time
    # reading, not a travelling front.
    never = ~(~matches).any(axis=1)
    switch_frame = (~matches).idxmax(axis=1).where(~never, len(frames))
    agreement_order = np.lexsort(
        (switch_frame.to_numpy(), reference.loc[core].to_numpy())
    )
    logger.info(
        "agreement with pooled over %d core bins: %.1f%% at %s -> %.1f%% at %s; "
        "ARI %.3f -> %.3f (min %.3f at %s)",
        len(core), 100 * matches[0].mean(), centres[0],
        100 * matches[len(frames) - 1].mean(), centres[-1],
        ari_by_frame[0], ari_by_frame[-1],
        min(ari_by_frame), centres[int(np.argmin(ari_by_frame))],
    )

    still_indices = np.linspace(0, len(frames) - 1, n_stills).round().astype(int)

    return {
        "labels_by_frame": labels_by_frame,
        "geometry_by_frame": geometry_by_frame,
        "still_indices": still_indices.tolist(),
        "modal": modal,
        "persistence": persistence,
        "temporal_stability": temporal_stability,
        "agreement": agreement,
        "agreement_order": agreement_order,
        "ari_by_frame": ari_by_frame,
        "reference": reference,
        "ari": ari,
    }


def load_clustering_spatiotemporal_data(paths, fig_config):
    """Cluster (bin, frame) observations under a space-time contiguity constraint.

    Parameters
    ----------
    paths : dict
        Resolved paths from config (uppercase keys).
    fig_config : dict
        The figure_clustering_spatiotemporal config section.
    Returns
    -------
    dict
        Shared data consumed by the panel functions.
    """
    output_base = Path(paths["OUTPUT_BASE"]) / fig_config["output_dir"]
    n_clusters = int(fig_config.get("n_clusters", N_CLUSTERS_REFERENCE))
    cache_path = output_base / _CACHE_FILENAME.format(n_clusters)

    if not fig_config.get("regenerate_cache", False) and cache_path.exists():
        logger.info("Loading clustering-spatiotemporal cache from %s", cache_path)
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    logger.info(
        "Computing clustering-spatiotemporal data from scratch (k=%d)...", n_clusters
    )

    macrochaete = pd.read_csv(paths["MACROCHAETE_CSV"], header=None)
    zarr_path = paths["AVERAGES_240920_LAGRANGIAN"]

    n_stills = int(fig_config.get("n_stills", N_STILLS))
    slots, windows = _retained_slots(zarr_path)
    centres = [median_time(*w) for w in windows]
    logger.info(
        "%d unique 4h windows retained from slots %d-%d (%s -> %s, centres %s -> %s)",
        len(slots), SLOT_RANGE.start, SLOT_RANGE.stop - 1,
        windows[0][0], windows[-1][1], centres[0], centres[-1],
    )

    frames = _stack_frames(zarr_path, slots)
    stacked = pd.concat(frames, keys=range(len(frames)), names=["frame"])

    # Same scaler + PCA as the reference 2D pipeline, truncated by the same rule.
    pca, scores, _ = _run_pca(stacked)
    n_pcs = int(np.argmax(np.cumsum(pca.explained_variance_ratio_) >= 0.9) + 1)
    scores = scores.to_numpy()[:, :n_pcs]

    graph, blocks = _space_time_graph(frames)
    n_graph_components, _ = connected_components(graph, directed=False)
    logger.info(
        "space-time graph: %d observations, %d undirected edges, %d connected "
        "component(s); %d PCs (>=90%% var)",
        graph.shape[0], graph.nnz // 2, n_graph_components, n_pcs,
    )

    # --- Cut at every resolution, scoring the silhouette in the reference space ---
    # Both curves are kept so the peak can be read as a property of the cut rather than
    # of the cleaning: they differ by under 0.01 at every k.
    k_values = sorted(set(K_RANGE) | {n_clusters})
    silhouette_by_k, silhouette_by_k_raw = {}, {}
    labels = None
    for k in k_values:
        raw = AgglomerativeClustering(
            n_clusters=k, linkage="ward", connectivity=graph
        ).fit_predict(scores)
        cleaned, n_moved = _clean_frames(raw, frames, blocks)
        silhouette_by_k_raw[k] = float(silhouette_score(scores, raw))
        silhouette_by_k[k] = float(silhouette_score(scores, cleaned))
        logger.info(
            "k=%d: %d bins reassigned (%.1f%%); silhouette %.4f raw -> %.4f cleaned",
            k, n_moved, 100 * n_moved / len(raw),
            silhouette_by_k_raw[k], silhouette_by_k[k],
        )
        if k == n_clusters:
            labels = pd.Series(cleaned, index=stacked.index)

    best_k = max(silhouette_by_k, key=silhouette_by_k.get)
    logger.info(
        "silhouette optimum over k=%d-%d: k=%d (%.4f); rendering at k=%d",
        k_values[0], k_values[-1], best_k, silhouette_by_k[best_k], n_clusters,
    )

    # --- Align domain ids to the reference partition, via the modal domain ---
    reference = _pooled_reference(zarr_path)
    modal = (
        labels.groupby(level=["x", "y"]).value_counts().unstack(fill_value=0).idxmax(axis=1)
    )
    aligned_modal = align_to_reference(modal, reference)
    # The alignment is a pure relabel, so pairing before/after recovers the mapping —
    # but only for domains that are some bin's modal label. Any other keeps a distinct
    # id of its own, exactly as align_to_reference does for unmatched labels.
    mapping = dict(zip(modal.to_numpy(), aligned_modal.to_numpy()))
    next_id = max(mapping.values()) + 1
    for domain in np.unique(labels.to_numpy()):
        if domain not in mapping:
            mapping[domain] = next_id
            next_id += 1
    labels = labels.map(mapping)

    shared_data = {
        "slots": slots,
        "windows": windows,
        "centres": centres,
        **_summarise_over_time(
            labels, aligned_modal, frames, reference, centres, n_stills
        ),
        "k_values": k_values,
        "silhouette_by_k": silhouette_by_k,
        "silhouette_by_k_raw": silhouette_by_k_raw,
        "n_pcs": n_pcs,
        "n_observations": int(graph.shape[0]),
        "macrochaete": macrochaete,
        "n_clusters": n_clusters,
    }

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(shared_data, f)
    logger.debug("Saved clustering-spatiotemporal cache to %s", cache_path)

    return shared_data
