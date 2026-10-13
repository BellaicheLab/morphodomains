"""Preprocessing for figure_spatial_transcriptome.

Loads single-cell data, atlas, runs the five spatial-transcriptome
optimizations (vGEP, Novosparc, PERLER, Tangram, CytoSPACE), computes correlations, and
assembles the shared_data dict consumed by panel functions.

Results are cached to a pickle file for fast re-use.
"""

import gc
import logging
import os
import pickle
import time
from pathlib import Path

logger = logging.getLogger(__name__)

import numpy as np
import pandas as pd
import scanpy as sc

from scripts.build_morpho_clustering import build as build_morpho_clustering
from vgep.optimization import (
    generate_novosparc,
    generate_perler,
    optimize_novosparc,
    optimize_perler,
    optimize_transcriptome,
    generate_tangram,
    optimize_tangram,
    generate_cytospace,
    optimize_cytospace,
)
from vgep.plotter.palettes import WHITE_TO_RED
from vgep.utils import to_matrix_form
from vgep.spatial_transcriptome import generate_vgep
from vgep.utils import remove_index_prefix
from vgep.evaluation.correlations import per_gene_pearson_r

_CACHE_FILENAME = "_figure_spatial_transcriptome_cache.pkl"
_BENCHMARK_CACHE_FILENAME = "_benchmark_times_cache.pkl"

# ── Constants ────────────────────────────────────────────────────────────────

TRAINING_GENES = [
    "al",
    "ara",
    "B-H1",
    "bi",
    "Diap1",
    "esg",
    "exd",
    "eyg",
    "h",
    "hh",
    "lgs",
    "mirr",
    "ptc",
    "salm",
    "sd",
    "tsh",
    "ush",
    "sr",
    "usp",
    "wg",
    "en",
    "rpr",
    "hid",
    "Pax",
]

VALIDATION_GENES = [
    "C15",
    "caup",
    "crp",
    "Dl",
    "dpp",
    "fj",
    "hbs",
    "hth",
    "nw",
    "pdm3",
    "Pvr",
    "Qsox2",
    "robo1",
    "sli",
    "slow",
    "toe",
    "Wnt6",
    "svp",
    "vvl",
    "18w",
]

PROTEIN_VALIDATION_GENES = [
    "C15",
    "caup",
    "Dl",
    "Wnt6",
    "svp",
    "vvl",
    "18w",
]

FISH_VALIDATION_GENES = [
    "slow",
    "sli",
    "pdm3",
    "toe",
    "crp",
    "nw",
    "hbs",
    "hth",
    "robo1",
    "Qsox2",
    "fj",
    "dpp",
    "Pvr",
]

NAME_CORRECTIONS = {
    "BH1": "B-H1",
    "Delta": "Dl",
    "Hid": "hid",
    "Rpr": "rpr",
    "Sr": "sr",
    "diap": "Diap1",
    "Ush": "ush",
    "Fj": "fj",
    "robo": "robo1",
    "slit": "sli",
    "Dpp": "dpp",
    "gene18w": "18w",
    "CG17843": "Qsox2",
    "nw :": "nw",
    "Pdm3": "pdm3",
}

CELL_TYPE_TO_CLUSTER_IDS = {
    "Muscle & Trachea": [0, 9],
    "Epithelium Notum": [1, 2, 3, 4, 5, 7],
    "Epithelium Neck": [8],
    "Sensory Organ precursor": [11],
    "Unassigned": [6],
}

MARKER_CATEGORIES = {
    "Epithelial": ["hh", "pnr", "caup", "eyg", "ush", "mirr"],
    "Muscle": ["twi", "sns", "kon", "lmd", "Him"],
    "SOP": ["sens", "ase", "mira"],
    "Neck": ["Dfd", "oc"],
}

GENES_TO_SHOW = ["B-H1", "sr", "eyg", "esg", "wg", "tsh"]
VBOUNDS = [
    [0, 2.12],
    [0.1, 5.77],
    [0.01, 3.81],
    [0, 1.12],
    [0, 3.94],
    [0.25, 5.82],
]
NAPARI_COLORS = ["bop blue", "bop orange", "blue", "yellow", "green", "red"]

VGEP_VS_ATLAS_GENES = [
    ("Tollo", "Tollo (Toll-8)"),
    ("ea", "ea (easter)"),
    ("spz3", "spz3 (spätzle 3)"),
    ("sr", "sr (stripe)"),
]

# ── Morphodynamics column renaming ────────────────────────────────────────


PROJECTIVE_TRANSFORM_MATRIX = np.array(
    [
        [1.4700, -0.7451, -53.8882],
        [0.0000, 3.2354, -375.5405],
        [0.0000, 0.0028, 0.5346],
    ]
)

# Method colors: Okabe-Ito, which keeps every adjacent pair separable under protanopia
# and tritanopia (worst adjacent dE 9.6 CVD / 18.7 normal vision). vGEP stays black.
OUR_COLOR = "#000000"
OUR_COLOR_TRAINING = "#ca0505"
NOVOSPARC_COLOR = "#D55E00"  # vermillion
PERLER_COLOR = "#009E73"  # bluish green
TANGRAM_COLOR = "#0072B2"  # blue
CYTOSPACE_COLOR = "#CC79A7"  # reddish purple

# Ordered benchmark registry. ``key`` is the dict key used in correlations[...]
# and the per-method expression entries in shared_data; ``timing`` is the key in
# benchmark_times. Plotting helpers and correlation/CSV code loop over this list,
# so adding a method here threads it through every benchmark panel.
METHOD_SPECS = [
    {"key": "our", "label": "vGEP", "color": OUR_COLOR, "timing": "vGEP"},
    {"key": "novosparc", "label": "NovoSpaRc", "color": NOVOSPARC_COLOR, "timing": "Novosparc"},
    {"key": "perler", "label": "Perler", "color": PERLER_COLOR, "timing": "PERLER"},
    {"key": "tangram", "label": "Tangram", "color": TANGRAM_COLOR, "timing": "Tangram"},
    {"key": "cytospace", "label": "CytoSPACE", "color": CYTOSPACE_COLOR, "timing": "CytoSPACE"},
]

MIN_CELLS = 50
MIN_MAX_UMI = 10


# ── Optimization grid presets ────────────────────────────────────────────────


def _get_optimization_grids(mode, output_base):
    """Return grid params and output dirs for vGEP/novosparc/perler/benchmark.

    ``mode == "test"`` uses tiny grids and writes to test-specific
    subdirectories so it doesn't collide with full-grid cached results
    from production runs (each optimizer resumes from its CSV on disk).
    ``mode == "full"`` (default) uses library defaults.
    """
    if mode == "test":
        return {
            "vgep_cutoffs": np.arange(0.97, 0.999, 0.005),  # ~6 cutoffs
            "novosparc_kwargs": {
                "alpha_values": [0.5],
                "epsilon_values": [0.03],
                "s_values": [10],
                "t_values": [10],
                "output_dir": str(output_base / "optimisation_novosparc_test"),
            },
            "perler_kwargs": {
                "k_range": [10],
                "output_dir": str(output_base / "optimisation_perler_test"),
            },
            # Tiny grids for a quick end-to-end run.
            "tangram_kwargs": {
                "n_trials": 2,
                "device": "cpu",
                "search_space": {
                    "density_prior": ["uniform"],
                    "num_epochs": [100],
                    "learning_rate": (0.1, 0.1),
                    "lambda_g2": [0.0],
                    "lambda_r": [0.0],
                },
                "output_dir": str(output_base / "optimisation_tangram_test"),
            },
            "cytospace_kwargs": {
                "search_space": {
                    "distance_metric": ["pearson"],
                    "cells_per_bin": ["single"],
                },
                "output_dir": str(output_base / "optimisation_cytospace_test"),
            },
            "vgep_output_path": str(output_base / "spatial_transcriptome_test_output"),
            "benchmark_repeats": 1,
        }
    return {
        "vgep_cutoffs": np.arange(0.97, 0.999, 0.0005),
        # The vgep default grids are the published search spaces.
        "novosparc_kwargs": {},
        "perler_kwargs": {},
        # Tangram gets a bigger Optuna budget for a fair search vs novosparc's
        # 1,386-config grid, starting from the published optimum; CytoSPACE is
        # exhaustive (10 combos) so needs no budget.
        # Runners default output_dir to GEP_RESULTS_BASE/<method>_output; GPU auto-selected.
        "tangram_kwargs": {
            "n_trials": 150,
            "initial_params": [{
                "density_prior": "uniform", "num_epochs": 500,
                "learning_rate": 0.0092912136952415, "lambda_g2": 1.0, "lambda_r": 0.0,
            }],
        },
        "cytospace_kwargs": {},
        "vgep_output_path": None,  # use paths["GEP_RESULTS_BASE"]
        "benchmark_repeats": None,  # use YAML benchmark_repeats
    }


# ── Private loaders ──────────────────────────────────────────────────────────


def _load_single_cell(paths):
    """Load h5ad, assign cell_type column, return adata."""
    adata = sc.read_h5ad(paths["SC_SINGLE_CELL_H5AD"])

    adata.obs["cell_type"] = "Unassigned"
    for cell_type_name, cluster_ids in CELL_TYPE_TO_CLUSTER_IDS.items():
        adata.obs.loc[adata.obs["seurat_clusters"].isin(cluster_ids), "cell_type"] = (
            cell_type_name
        )
    adata.obs["cell_type"] = adata.obs["cell_type"].astype("category")

    logger.debug("Cell type distribution:")
    for ct, count in adata.obs["cell_type"].value_counts().items():
        logger.debug("  %s: %s cells", ct, count)
    logger.debug("  Total: %s cells", len(adata.obs))

    return adata


def _load_and_filter_raw_counts(paths, adata):
    """Load CSV, filter Epithelium Notum, apply gene thresholds."""
    raw_counts = pd.read_csv(paths["SC_RAW_COUNTS_CSV"], index_col=0).T

    # Filter to Epithelium Notum cells
    epi_cells = adata.obs[adata.obs["cell_type"] == "Epithelium Notum"].index
    raw_counts = raw_counts.loc[raw_counts.index.intersection(epi_cells), :]
    logger.debug("Filtered to %s Epithelium Notum cells for spatial mapping", len(raw_counts))

    # Filter genes: min cells expressing, min max UMI
    cells_per_gene = (raw_counts > 0).sum(axis=0)
    max_umi_per_gene = raw_counts.max(axis=0)
    gene_mask = (cells_per_gene >= MIN_CELLS) & (max_umi_per_gene >= MIN_MAX_UMI)
    raw_counts = raw_counts.loc[:, gene_mask]

    return raw_counts


def _load_atlas(paths):
    """Load atlas CSV, apply name corrections, extract geometry, filter."""
    atlas = pd.read_csv(paths["ATLAS_VALIDATION_TXT"])
    atlas.rename(columns=NAME_CORRECTIONS, inplace=True)

    geometry = atlas.iloc[:, -2:].rename({"coordX": "x", "coordY": "y"}, axis=1)

    published = atlas[TRAINING_GENES + VALIDATION_GENES + ["n_nota_gene"]]
    tmp = published.copy()
    tmp[tmp.isna()] = 0
    positions = published[(tmp.sum(axis=1) > 0) & (published["n_nota_gene"] > 0)].index

    atlas = published.loc[positions, :]
    geometry = geometry.loc[positions, :]

    return atlas, geometry


def _run_vgep_optimization(
    raw_counts,
    atlas_df,
    geometry,
    macrochaete_path,
    paths,
    cutoffs=None,
    output_path=None,
):
    """Run our vGEP transcriptome optimization."""
    gene_expression_patterns = atlas_df.iloc[:, :-1]
    macrochaete = pd.read_csv(macrochaete_path, header=None)

    atlas = gene_expression_patterns.loc[:, TRAINING_GENES]
    validation = gene_expression_patterns.loc[:, VALIDATION_GENES]

    if cutoffs is None:
        cutoffs = np.arange(0.97, 0.999, 0.0005)
    if output_path is None:
        output_path = str(
            Path(paths["GEP_RESULTS_BASE"]) / "spatial_transcriptome_output"
        )
    os.makedirs(output_path, exist_ok=True)

    transcriptome = optimize_transcriptome(
        single_cells=raw_counts,
        atlas=atlas,
        geometry=geometry,
        validation=validation,
        list_cutoffs=cutoffs,
        output_path=output_path,
        macrochaetae=macrochaete,
        plot_recap=False,
    )

    st_df = transcriptome["st"].to_df()
    st_df.columns = remove_index_prefix(st_df.columns)

    return transcriptome, st_df


def _run_novosparc_optimization(raw_counts, atlas_df, geometry, paths, novosparc_kwargs=None):
    """Run Novosparc optimization.

    Applies the first geometry transform (flip x and y) before running.
    ``novosparc_kwargs`` is forwarded to ``optimize_novosparc`` (used to
    inject smaller grids for test runs); ``output_dir`` defaults to the
    canonical ``GEP_RESULTS_BASE/novosparc_output`` when not overridden.
    Returns (results_novosparc, tissue, novosparc_expression, single_cell,
    geometry_flipped, cyc_genes).
    """
    novosparc_kwargs = dict(novosparc_kwargs or {})
    output_dir = novosparc_kwargs.pop("output_dir", None) or str(
        Path(paths["GEP_RESULTS_BASE"]) / "novosparc_output"
    )
    geometry = geometry.copy()
    max_x, max_y = geometry[["x", "y"]].max(axis=0)
    geometry["x"] = max_x - geometry["x"]
    geometry["y"] = max_y - geometry["y"]
    geometry["zcoord"] = 0

    # Cell cycle genes
    cyc_genes = [g for g in raw_counts.columns if g.startswith("Cyc")]

    # Prepare single cell data
    single_cell = sc.AnnData(raw_counts)
    sc.pp.normalize_total(single_cell, target_sum=1e4)
    sc.pp.log1p(single_cell)
    sc.pp.highly_variable_genes(single_cell)
    sc.tl.pca(single_cell)

    variable_genes = single_cell.var.highly_variable[
        single_cell.var.highly_variable
    ].index.tolist()
    single_cell = single_cell[
        :, list(set(TRAINING_GENES + VALIDATION_GENES + cyc_genes + variable_genes))
    ]

    results_novosparc = optimize_novosparc(
        single_cell=single_cell,
        geometry=geometry,
        atlas_df=atlas_df,
        training_genes=TRAINING_GENES,
        validation_genes=VALIDATION_GENES,
        output_dir=output_dir,
        **novosparc_kwargs,
    )
    logger.debug("Novosparc optimization results: %s", results_novosparc)

    if (
        "best_novosparc_obj" in results_novosparc
        and results_novosparc["best_novosparc_obj"] is not None
    ):
        tissue = results_novosparc["best_novosparc_obj"]
        # sdge covers the reconstructed genes (tissue.gene_names, the marker subset),
        # NOT the full dataset var_names — label with the former so shapes match.
        novosparc_expression = pd.DataFrame(
            tissue.sdge.T, index=geometry.index, columns=tissue.gene_names
        )
    else:
        best = results_novosparc["best_params"]
        rep = best.get("rep", "variable")
        dge_rep = single_cell.obsm["X_pca"] if rep == "pca" else single_cell.to_df()
        gen_result = generate_novosparc(
            single_cell=single_cell,
            geometry=geometry,
            atlas_df=atlas_df,
            training_genes=TRAINING_GENES,
            alpha=best["alpha"],
            epsilon=best["epsilon"],
            num_neighbors_s=best["s"],
            num_neighbors_t=best["t"],
            dge_rep=dge_rep,
        )
        tissue = gen_result["tissue"]
        novosparc_expression = gen_result["expression"]

    return (
        results_novosparc,
        tissue,
        novosparc_expression,
        single_cell,
        geometry,
        cyc_genes,
    )


def _run_perler_optimization(
    raw_counts,
    atlas_df,
    geometry,
    macrochaete_path,
    original_min_x,
    original_min_y,
    paths,
    perler_kwargs=None,
):
    """Run PERLER optimization.

    Applies the Perler-specific second geometry flip (bug workaround).
    ``perler_kwargs`` is forwarded to ``optimize_perler`` (used to inject
    a smaller k_range for test runs); ``output_dir`` defaults to the
    canonical ``GEP_RESULTS_BASE/perler_output`` when not overridden.
    Returns (results_perler, perler_expression, geometry_perler,
    transformed_macrochaete).
    """
    perler_kwargs = dict(perler_kwargs or {})
    output_dir = perler_kwargs.pop("output_dir", None) or str(
        Path(paths["GEP_RESULTS_BASE"]) / "perler_output"
    )
    geometry = geometry.copy()

    # Perler-specific second flip
    max_x, max_y = geometry[["x", "y"]].max(axis=0)
    geometry["x"] = max_x - geometry["x"]
    geometry["y"] = max_y - geometry["y"]
    geometry["zcoord"] = 0

    # Transform macrochaete to match the double-flipped geometry frame.
    # Double-flipped geometry = x_orig - min_x_orig, so macrochaete
    # (in the original tissue frame) need to be shifted by the original mins.
    macrochaete = pd.read_csv(macrochaete_path, header=None)
    transformed_macrochaete = macrochaete.copy()
    transformed_macrochaete.iloc[0, :] -= original_min_y
    transformed_macrochaete.iloc[1, :] -= original_min_x

    # Prepare single cell data for Perler (fit on the 24+20 benchmark genes, as published).
    single_cell_perler = sc.AnnData(raw_counts[TRAINING_GENES + VALIDATION_GENES])
    sc.pp.normalize_total(single_cell_perler, target_sum=1e4)
    sc.pp.log1p(single_cell_perler)

    results_perler = optimize_perler(
        single_cell_perler.to_df(),
        atlas_df,
        TRAINING_GENES,
        VALIDATION_GENES,
        geometry,
        output_dir=output_dir,
        **perler_kwargs,
    )
    logger.debug("Perler optimization results: %s", results_perler)

    if (
        "best_perler_obj" in results_perler
        and results_perler["best_perler_obj"] is not None
    ):
        plr = results_perler["best_perler_obj"]
    else:
        best_k = results_perler["best_params"]["k_metagenes"]
        plr = generate_perler(
            single_cell_df=single_cell_perler.to_df(),
            atlas_df=atlas_df,
            training_genes=TRAINING_GENES,
            k_metagenes=best_k,
            geometry=geometry,
        )["plr"]

    plr.spatial_reconstruction(location=plr._PERLER__location)
    perler_expression = plr.result_with_location

    return results_perler, perler_expression, geometry, transformed_macrochaete


# ── Tangram / CytoSPACE benchmark methods ───────────────────────────────────
#
# These optimizers learn a row-stochastic bin←cell projection rather than an
# expression DataFrame. We project the log-normalized single-cell matrix through
# it (``M.T @ S`` for tangram, ``P @ S`` for cytospace) to get a per-bin
# expression DataFrame indexed by the ORIGINAL (un-flipped) ``geometry.index`` —
# the same shape vGEP/novosparc/perler already produce, so it plugs straight into
# the correlation + reconstruction code. They apply no geometry flip.


def _prepare_single_cell_lognorm(raw_counts):
    """Log-normalized AnnData over the full transcriptome (shared by the new methods)."""
    single_cell = sc.AnnData(raw_counts.copy())
    sc.pp.normalize_total(single_cell, target_sum=1e4)
    sc.pp.log1p(single_cell)
    return single_cell


def _dense(single_cell, genes):
    """``single_cell[:, genes].X`` as a dense (n_cells, len(genes)) float64 array."""
    x = single_cell[:, list(genes)].X
    x = x.toarray() if hasattr(x, "toarray") else np.asarray(x)
    return x.astype(np.float64)


# Every gene each method reconstructs and is scored on: 24 training + 20 validation.
_BENCHMARK_GENES = TRAINING_GENES + VALIDATION_GENES


def _per_gene_corr(expr_df, expr_geom, atlas_df, truth_geom, genes):
    """Per-gene Pearson r between a method's rasterized expression and the atlas.

    ``expr_geom`` rasterizes ``expr_df``; ``truth_geom`` rasterizes the atlas
    truth (these differ only for PERLER). NaN bins are masked per gene; a gene
    with no overlapping finite bins yields NaN. Returns an array aligned 1:1 to
    ``genes``.
    """
    return np.array([
        float(per_gene_pearson_r(
            to_matrix_form(expr_df[gene], geometry=expr_geom)[None],
            to_matrix_form(atlas_df[gene], geometry=truth_geom)[None],
        )[0])
        for gene in genes
    ])


def _run_tangram_optimization(raw_counts, atlas_df, geometry, paths, tangram_kwargs=None):
    """Optuna-tuned Tangram mapping. Returns (expression_df, validation_corr)."""
    tangram_kwargs = dict(tangram_kwargs or {})
    output_dir = tangram_kwargs.pop("output_dir", None) or str(
        Path(paths["GEP_RESULTS_BASE"]) / "tangram_output"
    )
    single_cell = _prepare_single_cell_lognorm(raw_counts)
    results = optimize_tangram(
        single_cell=single_cell,
        atlas_df=atlas_df,
        training_genes=TRAINING_GENES,
        validation_genes=VALIDATION_GENES,
        output_dir=output_dir,
        **tangram_kwargs,
    )
    M = np.asarray(results["best_tangram_obj"].X, dtype=np.float64)  # (n_cells, n_spots)
    expression = pd.DataFrame(
        M.T @ _dense(single_cell, _BENCHMARK_GENES),
        index=geometry.index,
        columns=_BENCHMARK_GENES,
    )
    val_corr = _per_gene_corr(expression, geometry, atlas_df, geometry, VALIDATION_GENES)
    return expression, val_corr


def _run_cytospace_optimization(raw_counts, atlas_df, geometry, paths, cytospace_kwargs=None):
    """CytoSPACE balanced-LAP placement. Returns (expression_df, validation_corr)."""
    cytospace_kwargs = dict(cytospace_kwargs or {})
    output_dir = cytospace_kwargs.pop("output_dir", None) or str(
        Path(paths["GEP_RESULTS_BASE"]) / "cytospace_output"
    )
    single_cell = _prepare_single_cell_lognorm(raw_counts)
    results = optimize_cytospace(
        single_cell=single_cell,
        atlas_df=atlas_df,
        training_genes=TRAINING_GENES,
        validation_genes=VALIDATION_GENES,
        output_dir=output_dir,
        **cytospace_kwargs,
    )
    P = np.asarray(results["best_cytospace_obj"], dtype=np.float64)  # (n_bins, n_cells)
    expression = pd.DataFrame(
        P @ _dense(single_cell, _BENCHMARK_GENES),
        index=geometry.index,
        columns=_BENCHMARK_GENES,
    )
    val_corr = _per_gene_corr(expression, geometry, atlas_df, geometry, VALIDATION_GENES)
    return expression, val_corr


def _fix_perler_dataframe(perler_expression, genes):
    """Fix PERLER's unusual top/bottom half structure.

    Returns (perler_df, perler_geom).
    """
    n_rows = len(perler_expression)
    half_point = n_rows // 2
    top_half_idx = perler_expression.index[:half_point]
    bottom_half_idx = perler_expression.index[half_point:]

    fixed_df = pd.DataFrame(index=top_half_idx)
    fixed_df[["x", "y", "zcoord"]] = perler_expression.loc[
        top_half_idx, ["x", "y", "zcoord"]
    ]
    for gene in genes:
        fixed_df[gene] = perler_expression.loc[bottom_half_idx, gene].values

    perler_geom = fixed_df[["x", "y", "zcoord"]]
    return fixed_df, perler_geom


def _compute_correlations(methods, atlas_df):
    """Assemble per-gene validation + training Pearson correlations, per method.

    ``methods`` is an ordered dict ``key -> (expression, expr_geom, truth_geom,
    val_corr)`` — one uniform entry per method. Training correlations are computed
    from each method's expression (``_per_gene_corr`` against the atlas);
    validation correlations are taken directly from the method's ``val_corr``. All
    methods are sorted with the SAME order derived from vGEP so the gene axis
    matches across all bars.

    Returns a dict with 'validation' and 'training' sub-dicts keyed by method
    (plus 'genes_sorted').
    """
    val, train = {}, {}
    for key, (expr, expr_geom, truth_geom, val_corr) in methods.items():
        train[key] = _per_gene_corr(expr, expr_geom, atlas_df, truth_geom, TRAINING_GENES)
        val[key] = np.squeeze(np.asarray(val_corr))

    sort_idx_val = np.argsort(val["our"])
    sort_idx_train = np.argsort(train["our"])
    val = {k: v[sort_idx_val] for k, v in val.items()}
    train = {k: v[sort_idx_train] for k, v in train.items()}
    val["genes_sorted"] = [VALIDATION_GENES[i] for i in sort_idx_val]
    train["genes_sorted"] = [TRAINING_GENES[i] for i in sort_idx_train]

    return {"validation": val, "training": train}


def _prepare_vgep_comparison_data(transcriptome, paths):
    """Re-index vGEP/atlas to (x,y) multi-index, intersect positions.

    Returns dict with vgep_df, atlas_df_indexed, common_idx.
    """
    original_atlas = pd.read_csv(paths["ATLAS_VALIDATION_TXT"])

    vgep_df = transcriptome["st"].to_df()
    vgep_obs = transcriptome["st"].obs.copy()
    vgep_obs[["x", "y"]] = vgep_obs[["x", "y"]].astype(int)
    vgep_df = pd.concat([vgep_df, vgep_obs[["x", "y"]]], axis=1)
    vgep_df = vgep_df.set_index(["x", "y"])
    vgep_df.columns = vgep_df.columns.str.replace("st:", "")

    morpho_data = build_morpho_clustering(paths["AVERAGES_240920_LAGRANGIAN"])

    atlas_df_indexed = original_atlas.copy()
    atlas_df_indexed["x"] = atlas_df_indexed["coordX"].astype(int)
    atlas_df_indexed["y"] = atlas_df_indexed["coordY"].astype(int)
    atlas_df_indexed = atlas_df_indexed.set_index(["x", "y"])

    common_idx = vgep_df.index.intersection(atlas_df_indexed.index).intersection(
        morpho_data.index
    )
    logger.debug("vGEP vs Atlas: %s common positions", len(common_idx))

    return {
        "vgep_df": vgep_df,
        "atlas_df_indexed": atlas_df_indexed,
        "common_idx": common_idx,
    }


def _read_best_params_from_csvs(paths):
    """Read each method's best parameters from its optimization CSV.

    vGEP uses its own results.csv and (mean+median) selection objective. Every
    grid-search method (novosparc/perler/tangram/cytospace) is read
    identically — the max-``score`` row of its ``optimization_results.csv`` —
    and returned raw; the timing call site coerces fields to the right types.

    Returns (best_cutoff, best_rows) where ``best_rows`` maps method name to its
    best-row dict.
    """
    gep_base = Path(paths["GEP_RESULTS_BASE"])

    # vGEP — bespoke results.csv + (mean+median) selection objective.
    recap_df = pd.read_csv(
        gep_base / "spatial_transcriptome_output" / "results.csv", index_col=0, sep=";"
    )
    scores = recap_df[["mean", "median"]].round(2).values.astype(float).sum(axis=1)
    best_index = len(scores) - np.argmax(scores[::-1]) - 1
    best_cutoff = float(recap_df.iloc[best_index, 1])

    # Grid-search methods — uniform: max-score row from optimization_results.csv.
    output_subdirs = {
        "novosparc": "novosparc_output",
        "perler": "perler_output",
        "tangram": "tangram_output",
        "cytospace": "cytospace_output",
    }
    best_rows = {}
    for method, subdir in output_subdirs.items():
        df = pd.read_csv(gep_base / subdir / "optimization_results.csv")
        best_rows[method] = df.loc[df["score"].idxmax()].to_dict()

    return best_cutoff, best_rows


def _time_reconstruction(label, reconstruct, n_repeats):
    """Time a no-arg ``reconstruct`` thunk ``n_repeats`` times → {times, mean, std}."""
    logger.info("  Benchmarking %s (%s repeats)...", label, n_repeats)
    times = []
    for i in range(n_repeats):
        t0 = time.perf_counter()
        reconstruct()
        times.append(time.perf_counter() - t0)
        logger.debug("    %s repeat %s/%s: %.2fs", label, i + 1, n_repeats, times[-1])
    return {"times": times, "mean": float(np.mean(times)), "std": float(np.std(times))}


def _benchmark_reconstruction_times(
    raw_counts,
    atlas_df,
    geometry,
    macrochaete_path,
    output_base,
    paths,
    n_repeats=10,
    regenerate=False,
):
    """Time single reconstructions for each method with best parameters.

    Uses a separate cache file so it can be regenerated independently.

    Parameters
    ----------
    raw_counts : DataFrame
        Filtered raw count matrix (Epithelium Notum cells).
    atlas_df : DataFrame
        Atlas expression with training + validation genes.
    geometry : DataFrame
        Spatial coordinates (original, pre-flip).
    macrochaete_path : str
        Path to macrochaete CSV.
    output_base : Path
        Output directory for cache file.
    paths : dict
        Resolved paths (uppercase keys).
    n_repeats : int
        Number of timing repeats per method.
    regenerate : bool
        If True, ignore existing cache.

    Returns
    -------
    dict
        ``{"vGEP": {"times": [...], "mean": X, "std": Y}, ...}``
    """
    cache_path = Path(output_base) / _BENCHMARK_CACHE_FILENAME

    if cache_path.exists() and not regenerate:
        import time
        age_hours = (time.time() - cache_path.stat().st_mtime) / 3600
        logger.info("Loading benchmark times from %s (age: %.1fh)", cache_path, age_hours)
        if age_hours > 72:
            logger.warning(
                "Cached benchmark times are %.1fh old. "
                "Run with --regenerate-cache to rebuild.", age_hours,
            )
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    logger.info("Benchmarking reconstruction times...")
    best_cutoff, best_rows = _read_best_params_from_csvs(paths)
    macrochaete = pd.read_csv(macrochaete_path, header=None)
    results = {}

    # Each block preps its inputs (not timed), then times one reconstruction thunk.

    # ── vGEP ──────────────────────────────────────────────────────────────
    atlas_train = atlas_df.loc[:, TRAINING_GENES]
    atlas_valid = atlas_df.loc[:, VALIDATION_GENES]
    results["vGEP"] = _time_reconstruction(
        "vGEP",
        lambda: generate_vgep(
            raw_counts, atlas_train, geometry, best_cutoff, atlas_valid,
            morphological_marks=macrochaete,
        ),
        n_repeats,
    )

    # ── Novosparc ── once-flipped geometry + PCA/variable-gene AnnData
    geometry_novo = geometry.copy()
    max_x, max_y = geometry_novo[["x", "y"]].max(axis=0)
    geometry_novo["x"] = max_x - geometry_novo["x"]
    geometry_novo["y"] = max_y - geometry_novo["y"]
    geometry_novo["zcoord"] = 0
    cyc_genes = [g for g in raw_counts.columns if g.startswith("Cyc")]
    single_cell_novo = _prepare_single_cell_lognorm(raw_counts)
    sc.pp.highly_variable_genes(single_cell_novo)
    sc.tl.pca(single_cell_novo)
    variable_genes = single_cell_novo.var.highly_variable[
        single_cell_novo.var.highly_variable
    ].index.tolist()
    single_cell_novo = single_cell_novo[
        :, list(set(TRAINING_GENES + VALIDATION_GENES + cyc_genes + variable_genes))
    ]
    nb = best_rows["novosparc"]
    dge_rep = single_cell_novo.obsm["X_pca"] if nb["rep"] == "pca" else single_cell_novo.to_df()
    results["Novosparc"] = _time_reconstruction(
        "Novosparc",
        lambda: generate_novosparc(
            single_cell=single_cell_novo, geometry=geometry_novo, atlas_df=atlas_df,
            training_genes=TRAINING_GENES,
            alpha=float(nb["alpha"]), epsilon=float(nb["epsilon"]),
            num_neighbors_s=int(nb["s"]), num_neighbors_t=int(nb["t"]), dge_rep=dge_rep,
        ),
        n_repeats,
    )

    # ── PERLER ── double-flipped geometry + log-norm AnnData
    geometry_perler = geometry_novo.copy()
    max_x, max_y = geometry_perler[["x", "y"]].max(axis=0)
    geometry_perler["x"] = max_x - geometry_perler["x"]
    geometry_perler["y"] = max_y - geometry_perler["y"]
    geometry_perler["zcoord"] = 0
    perler_sc_df = _prepare_single_cell_lognorm(
        raw_counts[TRAINING_GENES + VALIDATION_GENES]
    ).to_df()
    pb = best_rows["perler"]
    results["PERLER"] = _time_reconstruction(
        "PERLER",
        lambda: generate_perler(
            single_cell_df=perler_sc_df, atlas_df=atlas_df,
            training_genes=TRAINING_GENES, k_metagenes=int(pb["k_metagenes"]),
            geometry=geometry_perler,
        ),
        n_repeats,
    )

    # ── Tangram / CytoSPACE ── log-norm full-transcriptome AnnData (shared)
    single_cell = _prepare_single_cell_lognorm(raw_counts)

    tb = best_rows["tangram"]
    results["Tangram"] = _time_reconstruction(
        "Tangram",
        lambda: generate_tangram(
            single_cell=single_cell, atlas_df=atlas_df, training_genes=TRAINING_GENES,
            mode=str(tb.get("mode", "cells")), density_prior=str(tb["density_prior"]),
            num_epochs=int(tb["num_epochs"]), learning_rate=float(tb["learning_rate"]),
            lambda_g2=float(tb["lambda_g2"]), lambda_r=float(tb["lambda_r"]),
        ),
        n_repeats,
    )

    cb = best_rows["cytospace"]
    cells_per_bin = (
        int(cb["cells_per_bin"])
        if str(cb["cells_per_bin"]).isdigit()
        else str(cb["cells_per_bin"])
    )
    results["CytoSPACE"] = _time_reconstruction(
        "CytoSPACE",
        lambda: generate_cytospace(
            single_cell=single_cell, atlas_df=atlas_df, training_genes=TRAINING_GENES,
            distance_metric=str(cb["distance_metric"]), cells_per_bin=cells_per_bin,
        ),
        n_repeats,
    )

    # Save cache
    Path(output_base).mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(results, f)

    logger.info("Benchmark results:")
    for method, data in results.items():
        logger.info("  %s: %.2f +/- %.2f s", method, data["mean"], data["std"])

    return results


def _assemble_shared_data(cacheable_data, adata):
    """Combine cached DataFrames with non-cacheable objects.

    Parameters
    ----------
    cacheable_data : dict
        Pickle-safe data (DataFrames, arrays, dicts, strings).
    adata : AnnData
        Single-cell data (not cacheable due to WeakSet internals).

    Returns
    -------
    dict
        Full shared_data dict for panels.
    """
    shared_data = dict(cacheable_data)
    shared_data.update(
        {
            "adata": adata,
            "training_genes": TRAINING_GENES,
            "validation_genes": VALIDATION_GENES,
            "white_to_red": WHITE_TO_RED,
            "protein_validation_genes": PROTEIN_VALIDATION_GENES,
            "fish_validation_genes": FISH_VALIDATION_GENES,
            "marker_categories": MARKER_CATEGORIES,
            "genes_to_show": GENES_TO_SHOW,
            "vbounds": VBOUNDS,
            "napari_colors": NAPARI_COLORS,
        }
    )
    return shared_data


def _save_outputs(shared_data, output_base, fig_config):
    """Ensure per-gene correlation CSVs exist (idempotent)."""
    outputs_cfg = fig_config.get("outputs", {})
    correlations = shared_data.get("correlations")
    if correlations is None:
        return

    for subset_key, output_key in [
        ("validation", "correlation_validation"),
        ("training", "correlation_training"),
    ]:
        cfg = outputs_cfg.get(output_key, {})
        if not cfg.get("enabled", True):
            continue
        csv_path = output_base / cfg.get(
            "path", f"benchmarking/correlation_{subset_key}.csv"
        )
        corr = correlations[subset_key]
        df = pd.DataFrame({"gene": corr["genes_sorted"]})
        for spec in METHOD_SPECS:
            col = "vgep" if spec["key"] == "our" else spec["key"]
            df[col] = corr[spec["key"]]
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv_path, index=False)
        logger.debug("  Saved %s CSV to %s", output_key, csv_path)


# ── Main entry point ─────────────────────────────────────────────────────────


def load_spatial_transcriptome_data(paths, fig_config):
    """Load and precompute all data needed by figure_spatial_transcriptome panels.

    Parameters
    ----------
    paths : dict
        Resolved paths (uppercase keys).
    fig_config : dict
        The figure_spatial_transcriptome section from figure_config.yaml.

    Returns
    -------
    dict
        Shared data for all figure_spatial_transcriptome panels.
    """
    output_base = Path(paths["OUTPUT_BASE"]) / fig_config["output_dir"]
    cache_path = output_base / _CACHE_FILENAME
    regenerate = fig_config.get("regenerate_cache", False)
    macrochaete_path = paths["MACROCHAETE_CSV"]

    opt_mode = fig_config.get("optimization_grid", "full")
    grids = _get_optimization_grids(opt_mode, output_base)
    if opt_mode == "test":
        logger.info("figure_spatial_transcriptome: optimization_grid=test (tiny grids, separate output dirs)")
    benchmark_repeats = (
        grids["benchmark_repeats"]
        if grids["benchmark_repeats"] is not None
        else fig_config.get("benchmark_repeats", 5)
    )

    if cache_path.exists() and not regenerate:
        import time
        age_hours = (time.time() - cache_path.stat().st_mtime) / 3600
        logger.info("Loading cached figure_spatial_transcriptome data from %s (age: %.1fh)", cache_path, age_hours)
        if age_hours > 72:
            logger.warning(
                "Cached figure_spatial_transcriptome data is %.1fh old. "
                "Run with --regenerate-cache to rebuild.", age_hours,
            )
        with open(cache_path, "rb") as f:
            cacheable_data = pickle.load(f)
        # Run benchmark with its own cache (independent of main cache)
        benchmark_times = _benchmark_reconstruction_times(
            cacheable_data["raw_counts"],
            cacheable_data["atlas_df"],
            cacheable_data["geometry"],
            macrochaete_path,
            output_base,
            paths,
            n_repeats=benchmark_repeats,
            regenerate=fig_config.get("regenerate_benchmark", False),
        )
        cacheable_data["benchmark_times"] = benchmark_times
        # Reload non-cacheable objects (adata for UMAP panels)
        adata = _load_single_cell(paths)
        shared_data = _assemble_shared_data(cacheable_data, adata)
        shared_data["macrochaete_path"] = macrochaete_path
        _save_outputs(shared_data, output_base, fig_config)
        return shared_data

    logger.info("Computing figure_spatial_transcriptome data from scratch (this may take a while)...")

    np.random.seed(42)

    # 1. Load basic data
    adata = _load_single_cell(paths)
    raw_counts = _load_and_filter_raw_counts(paths, adata)
    atlas_df, geometry = _load_atlas(paths)

    # 2. Run vGEP optimization
    logger.info("Running vGEP optimization...")
    transcriptome, st_transcriptome_df = _run_vgep_optimization(
        raw_counts,
        atlas_df,
        geometry,
        macrochaete_path,
        paths,
        cutoffs=grids["vgep_cutoffs"],
        output_path=grids["vgep_output_path"],
    )

    # 3. Run Novosparc optimization (applies first geometry flip)
    logger.info("Running Novosparc optimization...")
    (
        results_novosparc,
        tissue,
        novosparc_expression,
        single_cell,
        geometry_flipped,
        cyc_genes,
    ) = _run_novosparc_optimization(
        raw_counts, atlas_df, geometry, paths, novosparc_kwargs=grids["novosparc_kwargs"]
    )

    # 5. Run PERLER optimization (applies second geometry flip)
    logger.info("Running PERLER optimization...")
    original_min_x = geometry["x"].min()
    original_min_y = geometry["y"].min()
    (
        results_perler,
        perler_expression,
        geometry_perler,
        transformed_macrochaete,
    ) = _run_perler_optimization(
        raw_counts,
        atlas_df,
        geometry_flipped,
        macrochaete_path,
        original_min_x,
        original_min_y,
        paths,
        perler_kwargs=grids["perler_kwargs"],
    )

    # 5b. Benchmark single-reconstruction times (has its own cache)
    benchmark_times = _benchmark_reconstruction_times(
        raw_counts,
        atlas_df,
        geometry,
        macrochaete_path,
        output_base,
        paths,
        n_repeats=benchmark_repeats,
        regenerate=regenerate,
    )

    # 6. Fix PERLER dataframe
    perler_df, perler_geom = _fix_perler_dataframe(perler_expression, _BENCHMARK_GENES)

    # Free the heavy novosparc tissue (~830MB) and PERLER model (~813MB) before the
    # multiprocessing benchmark methods: CytoSPACE forks workers that would otherwise
    # copy these resident models and OOM a 16GB machine. Only the per-method validation
    # correlation arrays are needed for the correlation assembly below.
    novosparc_best_corr = results_novosparc["best_correlations"]
    perler_best_corr = results_perler["best_correlations"]
    del tissue, results_novosparc, results_perler
    gc.collect()

    # 6b. Run Tangram and CytoSPACE. They apply no geometry flip, so they
    # reconstruct on the original geometry.
    logger.info("Running Tangram optimization...")
    tangram_expression, tangram_val = _run_tangram_optimization(
        raw_counts, atlas_df, geometry, paths, tangram_kwargs=grids["tangram_kwargs"]
    )
    logger.info("Running CytoSPACE optimization...")
    cytospace_expression, cytospace_val = _run_cytospace_optimization(
        raw_counts, atlas_df, geometry, paths, cytospace_kwargs=grids["cytospace_kwargs"]
    )

    # vGEP expression + validation r, used for the correlation assembly and grids.
    our_expression = transcriptome["st"].to_df()
    our_expression.columns = remove_index_prefix(our_expression.columns)
    # _calculate_gene_correlations returns its result sorted ascending by r, carrying gene
    # identity only in the index — so reindex onto VALIDATION_GENES rather than dropping it.
    our_val = (
        transcriptome["st"]
        .uns["correlations_genes_validation"]
        .reindex(VALIDATION_GENES)
        .iloc[:, 0]
        .to_numpy()
    )

    # 7. Compute correlations. One uniform entry per method:
    # (expression, geometry to rasterize it, geometry of its atlas truth, validation r).
    logger.info("Computing correlations...")
    methods = {
        "our": (our_expression, geometry_flipped, geometry_flipped, our_val),
        "novosparc": (novosparc_expression, geometry_flipped, geometry_flipped,
                      novosparc_best_corr),
        "perler": (perler_df, perler_geom, geometry_perler,
                   perler_best_corr),
        "tangram": (tangram_expression, geometry, geometry, tangram_val),
        "cytospace": (cytospace_expression, geometry, geometry, cytospace_val),
    }
    correlations = _compute_correlations(methods, atlas_df)

    # 8. Prepare vGEP comparison data
    logger.info("Preparing vGEP comparison data...")
    vgep_comparison = _prepare_vgep_comparison_data(transcriptome, paths)

    # Separate cacheable data (plain DataFrames/arrays/dicts) from objects
    # that contain unpicklable internals (AnnData, colormaps).
    cacheable_data = {
        "raw_counts": raw_counts,
        "atlas_df": atlas_df,
        "geometry": geometry,
        "geometry_perler": geometry_perler,
        "st_transcriptome_df": st_transcriptome_df,
        "our_expression": our_expression,
        "novosparc_expression": novosparc_expression,
        "perler_df": perler_df,
        "perler_geom": perler_geom,
        "tangram_expression": tangram_expression,
        "cytospace_expression": cytospace_expression,
        "transformed_macrochaete": transformed_macrochaete,
        "correlations": correlations,
        "vgep_comparison": vgep_comparison,
        "benchmark_times": benchmark_times,
    }

    # Cache
    output_base.mkdir(parents=True, exist_ok=True)
    logger.info("Saving cache to %s", cache_path)
    with open(cache_path, "wb") as f:
        pickle.dump(cacheable_data, f)

    # Add non-cacheable items and return full shared_data
    shared_data = _assemble_shared_data(cacheable_data, adata)
    shared_data["macrochaete_path"] = macrochaete_path
    _save_outputs(shared_data, output_base, fig_config)
    return shared_data
