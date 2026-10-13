"""Preprocessing for the atlas-size sensitivity figure.

The spatial mapping (vGEP) builds the cell->position correlation from only 24
atlas genes. This figure measures how reconstruction quality depends on the
number of atlas genes, via a saturation curve.

The 20 validation genes are held fixed as the evaluation set and never enter the
atlas. Reconstruction quality is the mean per-gene Pearson r over those 20 genes
at the reference cutoff. Two nested gene pools are swept independently:

- curated:  the 24 curated training genes alone
- extended: + stan, yrt and Tollo (27)

Each size reports mean +/- s.d. over random draws (no greedy gene ordering).

A leave-one-out pass over the 44 training + validation genes (atlas = the other 43) gives the
per-gene reconstruction quality, exposing which genes the atlas basis represents
poorly.

The reconstruction reuses figure_spatial_transcriptome's loaders and the vGEP
``transform``/``assess`` path; subsetting the single-cell matrix to the
pooled genes makes each reconstruction ~5x faster (only the pool genes are
projected, which is all the scoring needs).
"""

import gc
import logging
import os
import pickle
from itertools import combinations
from math import comb
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.figures.figure_spatial_transcriptome.preprocessing import (
    NAME_CORRECTIONS,
    TRAINING_GENES,
    VALIDATION_GENES,
    _load_and_filter_raw_counts,
    _load_atlas,
    _load_single_cell,
)
from vgep.processing import scale_data
from vgep.spatial_transcriptome import assess, sc_data_handler, transform

logger = logging.getLogger(__name__)

_CACHE_FILENAME = "_figure_atlas_size_cache_v6.pkl"

# Genes added to the training atlas (atlas-CSV symbol -> single-cell symbol; all
# identical), selected on the qualitative quality of their expression pattern and
# its reproducibility across animals. Tollo never entered the 44-gene
# training/validation split, so it is atlas-eligible like the others.
ADDED_ATLAS_GENES = {
    "stan": "stan",
    "yrt": "yrt",
    "Tollo": "Tollo",
}

# The two nested atlas pools (symbols already renamed to single-cell symbols in
# gene_patterns).
POOLS = {
    "curated": [],
    "extended": list(ADDED_ATLAS_GENES.values()),
}

N_DRAWS = 500
SEED = 42

# Enumerate every k-subset when there are no more than this many. C(n, k) is
# symmetric, so this makes both tails of each sweep exact and leaves only the
# combinatorial middle to sampling.
EXHAUSTIVE_MAX = 50_000


def _sizes_for(n_pool):
    """Every atlas size from 2 to the pool size (one point per added gene)."""
    return list(range(2, n_pool + 1))


def _reference_cutoff(paths):
    """Read the reference vGEP cutoff from results.csv (same scoring as optimize)."""
    csv = (
        Path(paths["GEP_RESULTS_BASE"]) / "spatial_transcriptome_output" / "results.csv"
    )
    recap = pd.read_csv(csv, index_col=0, sep=";")
    scores = recap[["mean", "median"]].round(2).values.astype(float).sum(axis=1)
    best = len(scores) - np.argmax(scores[::-1]) - 1
    return float(recap.iloc[best, 1])


def _reconstruct_and_score(
    sc_scaled, gene_patterns, geometry, atlas_genes, validation_genes, cutoff
):
    """vGEP reconstruct from ``atlas_genes`` and score the full validation set.

    Returns (mean_r, median_r) of the per-gene Pearson correlations over
    ``validation_genes`` (the fixed held-out evaluation set, disjoint from the
    atlas by construction).
    """
    st = transform(
        sc_scaled, gene_patterns[atlas_genes], geometry, cutoff, verbose=False
    )
    st = assess(st, gene_patterns[atlas_genes], gene_patterns[validation_genes])
    v = st["st"].uns["results_recap"].loc["Validation"]
    return float(v["mean"]), float(v["median"])


_WORKER = {}


def _init_worker(sc_scaled, gene_patterns, geometry, cutoff):
    """Hand each worker the read-only reconstruction inputs, once per process."""
    _WORKER.update(
        sc_scaled=sc_scaled, gene_patterns=gene_patterns, geometry=geometry, cutoff=cutoff
    )


def _score_atlas(atlas_genes):
    return _reconstruct_and_score(
        _WORKER["sc_scaled"],
        _WORKER["gene_patterns"],
        _WORKER["geometry"],
        atlas_genes,
        VALIDATION_GENES,
        _WORKER["cutoff"],
    )


def load_atlas_size_data(paths, fig_config):
    """Sweep atlas size via pooled 44-gene cross-validation; cache the result.

    When ``regenerate_cache`` is false (default) and the cache exists, the
    pickle is loaded directly. Set ``regenerate_cache: true`` to rebuild.
    """
    output_base = Path(paths["OUTPUT_BASE"]) / fig_config.get(
        "output_dir", "fig_atlas_size"
    )
    cache_path = output_base / _CACHE_FILENAME

    if not fig_config.get("regenerate_cache", False) and cache_path.exists():
        logger.info("Loading atlas-size cache from %s", cache_path)
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    logger.info("Computing atlas-size sensitivity from scratch...")

    cutoff = _reference_cutoff(paths)
    logger.info("Using reference cutoff %.4f", cutoff)

    # --- Loaders shared with figure_spatial_transcriptome ---
    adata = _load_single_cell(paths)
    raw_counts = _load_and_filter_raw_counts(paths, adata)
    atlas_df, geometry = _load_atlas(paths)

    # _load_atlas keeps only the 44 training + validation genes, so re-read the atlas CSV to
    # pull the extension genes at the same filtered positions, renamed to their
    # single-cell symbols. gene_patterns holds 24 training + 20 validation + 3.
    raw_atlas = pd.read_csv(paths["ATLAS_VALIDATION_TXT"]).rename(
        columns=NAME_CORRECTIONS
    )
    new_patterns = raw_atlas.loc[atlas_df.index, list(ADDED_ATLAS_GENES)].rename(
        columns=ADDED_ATLAS_GENES
    )
    gene_patterns = pd.concat(
        [atlas_df[TRAINING_GENES + VALIDATION_GENES], new_patterns], axis=1
    )

    # Scale once on the full single-cell matrix, then restrict the projection to
    # the 47-gene pool (all that the scoring needs). copy() matters: slicing AnnData
    # returns a view that still references the full 4312-gene matrix, so without it
    # every worker is sent 376 MB instead of 4 MB.
    sc_scaled = scale_data(sc_data_handler(raw_counts))[:, list(gene_patterns.columns)].copy()

    del adata, raw_counts, raw_atlas, new_patterns, atlas_df
    gc.collect()

    # --- Saturation curves: for each pool, random k-gene atlases scored against
    #     the full 20-gene validation set. When the k-subsets number fewer than
    #     N_DRAWS we enumerate them all; at k = pool size the only atlas is the
    #     full panel (a single point). ---
    n_workers = fig_config.get("n_workers") or os.cpu_count()
    logger.info("Sweeping on %d workers", n_workers)
    # A reconstruction leaks ~0.4 MB -- invisible over hundreds of calls, several GB per
    # worker over the tens of thousands an exhaustive size needs. maxtasksperchild counts
    # chunks, so chunksize stays 1 for the cap to mean 200 reconstructions.
    pool = Pool(
        processes=n_workers,
        initializer=_init_worker,
        initargs=(sc_scaled, gene_patterns, geometry, cutoff),
        maxtasksperchild=200,
    )

    pools = {}
    for pool_name, extension_genes in POOLS.items():
        atlas_pool = list(TRAINING_GENES) + extension_genes
        sizes = _sizes_for(len(atlas_pool))
        rng = np.random.default_rng(SEED)
        curve_mean, curve_median, is_exhaustive = {}, {}, {}
        for k in sizes:
            exhaustive = comb(len(atlas_pool), k) <= EXHAUSTIVE_MAX
            if exhaustive:
                atlases = [list(c) for c in combinations(atlas_pool, k)]
            else:
                atlases = [
                    list(rng.choice(atlas_pool, size=k, replace=False))
                    for _ in range(N_DRAWS)
                ]
            scored = pool.map(_score_atlas, atlases, chunksize=1)
            curve_mean[k] = np.array([m for m, _ in scored])
            curve_median[k] = np.array([md for _, md in scored])
            is_exhaustive[k] = exhaustive
            logger.info(
                "%s k=%2d (%d atlases, %s): validation mean r = %.3f +/- %.3f",
                pool_name,
                k,
                len(atlases),
                "exhaustive" if exhaustive else "sampled",
                curve_mean[k].mean(),
                curve_mean[k].std(),
            )
        pools[pool_name] = {
            "sizes": np.array(sizes),
            "curve_mean": curve_mean,
            "curve_median": curve_median,
            "is_exhaustive": is_exhaustive,
            "pool_size": len(atlas_pool),
        }

    pool.close()
    pool.join()

    # --- Reference operating point: the curated 24-gene training atlas. It is the
    #     only 24-subset of the curated pool, so it must equal that curve's last
    #     point -- a check that the sweep reproduces the reference run. ---
    pub_k = len(TRAINING_GENES)
    pub_mean, _ = _reconstruct_and_score(
        sc_scaled,
        gene_patterns,
        geometry,
        list(TRAINING_GENES),
        VALIDATION_GENES,
        cutoff,
    )
    logger.info("Reference curated-%d atlas: validation mean r = %.3f", pub_k, pub_mean)

    # --- Leave-one-out over the 44 genes: per-gene reconstruction with
    #     the other 43 as atlas ---
    loo_genes = list(TRAINING_GENES) + list(VALIDATION_GENES)
    loo_r = {}
    for gene in loo_genes:
        atlas_genes = [g for g in loo_genes if g != gene]
        st = transform(
            sc_scaled, gene_patterns[atlas_genes], geometry, cutoff, verbose=False
        )
        st = assess(st, gene_patterns[atlas_genes], gene_patterns[[gene]])
        loo_r[gene] = float(st["st"].uns["correlations_genes_validation"].iloc[0, 0])
    loo = pd.DataFrame(
        {
            "r": pd.Series(loo_r),
            "panel": [
                "training" if g in TRAINING_GENES else "validation" for g in loo_r
            ],
        }
    ).sort_values("r")

    shared_data = {
        "pools": pools,
        "loo": loo,
        "cutoff": cutoff,
    }

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(shared_data, f)
    logger.debug("Saved atlas-size cache to %s", cache_path)

    return shared_data
