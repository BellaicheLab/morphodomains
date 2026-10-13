"""Preprocessing for the neck-landmark atlas agreement.

Every atlas animal is registered onto the reference archetype from its 8 macrochaetae with
pyanimalprocessing's ``Grid`` (``xyscale_translation``: independent x/y scaling + translation,
no rotation), and its thorax/neck boundary is projected through that transform. The mean
projected neck extends the archetype to 9 landmarks, and the gene atlas is built on both.
"""

import glob
import itertools
import logging
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
import zarr
from pyanimalprocessing.rescaling.registration import (
    Grid,
    _apply_matrix_transform_to_points,
)

from ..figure_gene_atlas.preprocessing import (
    _CANONICAL_ARCHETYPE_YX,
    _CANONICAL_GRID_FRAME,
    _average_gene_maps,
    _has_gene_channel,
    _run_atlas_pipeline,
)

logger = logging.getLogger(__name__)

_REGISTRATION = "atlas_sr"
_N_MACROS = 8
_MIN_MACROS = 4  # matches the pipeline's nMacroMin
_GRID_MACRO8 = "EULERIAN_MACRO8"
_GRID_MACRO9 = "EULERIAN_MACRO9"
_MIN_SPLIT_ANIMALS = 4  # below this a half-cohort map is a single animal
_MIN_GENES_PER_BOX = 5  # boxes carrying fewer genes give a meaningless profile correlation
_WELL_COVERED = 5  # animals per gene, above which r is not sampling-noise-limited
_FULL_COVERAGE = 0.95  # fraction of genes a box must carry to count as fully covered


def _project(points_yx, matrix_transform):
    """Map (n, 2) ``(y, x)`` points through a ``Grid`` matrix transform."""
    homogeneous = np.hstack((points_yx, np.ones((len(points_yx), 1))))
    return _apply_matrix_transform_to_points(homogeneous, matrix_transform)[:, :2]


def load_neck_positions(paths, fig_config):
    """Project the neck of every neck-clicked atlas animal into the reference archetype frame.

    Cached to a pickle, gated by ``pipeline.overwrite``.
    """
    overwrite = fig_config.get("pipeline", {}).get("overwrite", False)
    cache_path = Path(paths["GENE_ATLAS_BASE"]) / "_neck_registration_spread.pkl"
    if cache_path.exists() and not overwrite:
        logger.info("Loading cached neck positions from %s", cache_path)
        with open(cache_path, "rb") as fh:
            return {"neck_positions": pickle.load(fh)["neck_positions"]}

    archetype = np.asarray(_CANONICAL_ARCHETYPE_YX)
    necks = []
    for path in sorted(glob.glob(paths["ZARR_ATLASES"])):
        animal = zarr.open(path, mode="r")
        scaling = animal.attrs["scaling"]
        landmarks = animal.SPACE_REGISTRATION[_REGISTRATION].landmarks[:]
        macros = landmarks[:_N_MACROS, 1:] * scaling
        if (~np.isnan(macros).any(1)).sum() < _MIN_MACROS:
            continue
        if landmarks.shape[0] <= _N_MACROS or np.isnan(landmarks[_N_MACROS]).any():
            continue
        transform = Grid(
            image_shape=animal.IMAGE.D2.raw.shape[-2:],
            reference_landmarks=macros,
            target_landmarks=archetype,
            transform_type="xyscale_translation",
        ).matrix_transform
        necks.append(_project(landmarks[_N_MACROS:_N_MACROS + 1, 1:] * scaling, transform)[0])

    shared = {"neck_positions": np.asarray(necks)}
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as fh:
        pickle.dump(shared, fh)
    logger.info("Projected the neck of %d atlas animals", len(necks))
    return shared


def _correlate(map_a, map_b):
    """Pearson r over the boxes both maps cover (average_over_animals writes NaN elsewhere)."""
    covered = np.isfinite(map_a) & np.isfinite(map_b)
    if covered.sum() < _MIN_GENES_PER_BOX:
        return np.nan, int(covered.sum())
    return float(np.corrcoef(map_a[covered], map_b[covered])[0, 1]), int(covered.sum())


def _standardize(gene_map):
    """Z-score a gene map over the boxes it covers, leaving the uncovered ones NaN."""
    covered = np.isfinite(gene_map)
    out = np.full_like(gene_map, np.nan)
    out[covered] = (gene_map[covered] - gene_map[covered].mean()) / gene_map[covered].std()
    return out


def _anterior_profile(shared, neck_x):
    """Where along the antero-posterior axis the two atlases disagree.

    The anterior notum lies beyond every macrochaeta. Boxes are binned by their A-P coordinate
    in the archetype frame and reported over every evaluated box, and over the boxes carrying
    nearly all the genes (the profile correlation is fragile on the partially covered border).
    """
    per_box_r = shared["per_box_r"]
    _, _, x_min, x_max = _CANONICAL_GRID_FRAME
    n_boxes_ap = per_box_r.shape[0]
    ap = x_min + (np.arange(n_boxes_ap) + 0.5) * (x_max - x_min) / n_boxes_ap
    genes = shared["genes"]
    n_genes = np.isfinite(np.stack([shared["maps_macro8"][g] for g in genes])).sum(0)

    macro_x = np.asarray(_CANONICAL_ARCHETYPE_YX)[:, 1]
    mid_anterior = (macro_x.max() + neck_x) / 2
    bounds = [
        ("posterior of the macrochaetae", -np.inf, macro_x.min()),
        ("within the macrochaeta span", macro_x.min(), macro_x.max()),
        ("anterior, proximal", macro_x.max(), mid_anterior),
        ("anterior, towards the neck", mid_anterior, np.inf),
    ]

    evaluated = np.isfinite(per_box_r)
    full = evaluated & (n_genes >= _FULL_COVERAGE * len(genes))
    regions = []
    for label, low, high in bounds:
        in_band = ((ap > low) & (ap <= high))[:, None] & np.ones_like(per_box_r, bool)
        values = per_box_r[in_band & evaluated]
        covered = per_box_r[in_band & full]
        regions.append({
            "label": label,
            "x_from": low,
            "x_to": high,
            "n": int(values.size),
            "median": float(np.median(values)) if values.size else np.nan,
            "n_full": int(covered.size),
            "median_full": float(np.median(covered)) if covered.size else np.nan,
        })

    weak = evaluated & (per_box_r < 0.9)
    return {
        "ap_regions": regions,
        "neck_x": neck_x,
        "macro_x_range": (float(macro_x.min()), float(macro_x.max())),
        "n_boxes_evaluated": int(evaluated.sum()),
        "n_boxes_full": int(full.sum()),
        "median_r_full": float(np.median(per_box_r[full])),
        "frac_boxes_anterior": float(
            (evaluated & (ap > macro_x.max())[:, None]).sum() / evaluated.sum()
        ),
        "n_weak_boxes": int(weak.sum()),
        "median_genes_weak": float(np.median(n_genes[weak])) if weak.any() else np.nan,
        "median_genes_evaluated": float(np.median(n_genes[evaluated])),
    }


def _neck_clicked_animals(atlas_glob):
    """Open the atlas animals carrying a neck landmark (writable), grouped by gene."""
    animals_by_gene = defaultdict(list)
    for path in sorted(glob.glob(atlas_glob)):
        animal = zarr.open(path, mode="a")
        landmarks = animal.SPACE_REGISTRATION[_REGISTRATION].landmarks[:]
        if landmarks.shape[0] <= _N_MACROS or np.isnan(landmarks[_N_MACROS]).any():
            continue
        if (~np.isnan(landmarks[:_N_MACROS, 1:]).any(1)).sum() < _MIN_MACROS:
            continue
        if not _has_gene_channel(animal):
            logger.info("Skipping %s: no gene channel", path)
            continue
        animals_by_gene[animal.attrs["gene"]].append(animal)
    return dict(animals_by_gene)


def load_neck_atlas_agreement_data(paths, fig_config):
    """Build the atlas twice from the neck-clicked animals and correlate the two.

    Both builds use the same animals, the same genes and the same grid; they differ only in
    what the per-animal rescaling was fitted on -- the 8 macrochaetae, or the 8 macrochaetae
    plus the neck.

    The 9-landmark archetype adds the mean projected neck to the reference one, keeping the
    macrochaetae coordinates and the grid frame, so both builds tile the same boxes.

    Reference scale: each gene's animals are also split in two (alternating the sorted animals)
    and averaged separately on the 8-macrochaetae grid.

    Returns per-gene correlations, both sets of maps, and the per-box correlation of the
    standardized gene profile. Cached to a pickle, gated by ``pipeline.rebuild_atlases``.
    """
    pipeline_cfg = fig_config.get("pipeline", {})
    base = Path(paths["GENE_ATLAS_BASE"])
    neck_centre = load_neck_positions(paths, fig_config)["neck_positions"].mean(0)

    cache_path = base / "_neck_atlas_agreement.pkl"
    if cache_path.exists() and not pipeline_cfg.get("rebuild_atlases", False):
        logger.info("Loading cached neck/no-neck atlas comparison from %s", cache_path)
        with open(cache_path, "rb") as fh:
            shared = pickle.load(fh)
        return {**shared, **_anterior_profile(shared, float(neck_centre[1]))}

    boxsize = pipeline_cfg.get("boxsize", [40, 40])
    overlap = pipeline_cfg.get("overlap", 0.5)
    n_jobs = pipeline_cfg.get("n_jobs", 4)

    archetype_9 = np.vstack([_CANONICAL_ARCHETYPE_YX, neck_centre])

    animals_by_gene = _neck_clicked_animals(paths["ZARR_ATLASES"])
    n_animals = sum(len(group) for group in animals_by_gene.values())
    logger.info(
        "Neck-clicked cohort: %d animals over %d genes", n_animals, len(animals_by_gene)
    )

    maps_macro8, _ = _run_atlas_pipeline(
        animals_by_gene, str(base / "_gene_archetype_macro8.zarr"),
        str(base / "aoa_macro8"), boxsize, overlap,
        landmarks=_CANONICAL_ARCHETYPE_YX, n_macros=_N_MACROS,
        grid_name=_GRID_MACRO8, n_jobs=n_jobs,
    )
    maps_macro9, _ = _run_atlas_pipeline(
        animals_by_gene, str(base / "_gene_archetype_macro9.zarr"),
        str(base / "aoa_macro9"), boxsize, overlap,
        landmarks=archetype_9, n_macros=_N_MACROS + 1,
        grid_name=_GRID_MACRO9, n_jobs=n_jobs,
    )

    # Same grid_frame on both archetypes must give the same meshgrid, or the boxes of the two
    # atlases are not the same boxes and every correlation below is meaningless.
    probe = next(iter(animals_by_gene.values()))[0]
    if not np.array_equal(
        probe.GRID[_GRID_MACRO8].GRID_PROPERTIES.X[:],
        probe.GRID[_GRID_MACRO9].GRID_PROPERTIES.X[:],
    ):
        raise ValueError("the two archetypes do not tile the same frame")

    splittable = {
        gene: group for gene, group in animals_by_gene.items()
        if len(group) >= _MIN_SPLIT_ANIMALS
    }
    maps_half_a, _ = _average_gene_maps(
        {gene: group[::2] for gene, group in splittable.items()},
        str(base / "aoa_macro8_half_a"), _GRID_MACRO8,
    )
    maps_half_b, _ = _average_gene_maps(
        {gene: group[1::2] for gene, group in splittable.items()},
        str(base / "aoa_macro8_half_b"), _GRID_MACRO8,
    )

    genes = sorted(maps_macro8)
    per_gene = {}
    for gene in genes:
        r_neck, n_boxes = _correlate(maps_macro8[gene], maps_macro9[gene])
        r_split = (
            _correlate(maps_half_a[gene], maps_half_b[gene])[0] if gene in maps_half_a
            else np.nan
        )
        per_gene[gene] = {
            "n_animals": len(animals_by_gene[gene]),
            "n_boxes": n_boxes,
            "r_neck": r_neck,
            "r_split_half": r_split,
        }

    # Per box, does the *profile* over genes change? Each gene map is standardized first, so
    # this measures the shape of the signature and not the genes' different intensity scales.
    stack8 = np.stack([_standardize(maps_macro8[gene]) for gene in genes])
    stack9 = np.stack([_standardize(maps_macro9[gene]) for gene in genes])
    covered = np.isfinite(stack8) & np.isfinite(stack9)
    per_box_r = np.full(stack8.shape[1:], np.nan)
    for iy, ix in zip(*np.where(covered.sum(0) >= _MIN_GENES_PER_BOX)):
        present = covered[:, iy, ix]
        per_box_r[iy, ix] = np.corrcoef(stack8[present, iy, ix], stack9[present, iy, ix])[0, 1]

    well_covered = [g for g in genes if per_gene[g]["n_animals"] >= _WELL_COVERED]
    shared = {
        "per_gene": per_gene,
        "maps_macro8": maps_macro8,
        "maps_macro9": maps_macro9,
        "per_box_r": per_box_r,
        "genes": genes,
        "well_covered": well_covered,
        "n_atlas_animals": n_animals,
        "median_r_neck": float(np.nanmedian([per_gene[g]["r_neck"] for g in well_covered])),
        "median_r_split": float(
            np.nanmedian([per_gene[g]["r_split_half"] for g in well_covered])
        ),
        "median_per_box_r": float(np.nanmedian(per_box_r)),
    }
    with open(cache_path, "wb") as fh:
        pickle.dump(shared, fh)
    logger.info(
        "Neck vs no-neck atlas over %d animals / %d genes: median r %.4f "
        "(split-half floor %.4f) on the %d well-covered genes; per-box profile r %.4f",
        n_animals, len(genes), shared["median_r_neck"], shared["median_r_split"],
        len(well_covered), shared["median_per_box_r"],
    )
    return {**shared, **_anterior_profile(shared, float(neck_centre[1]))}


def between_gene_baseline(gene_maps, genes):
    """Correlations between maps of *different* genes -- the bottom of the scale.

    Anchors the agreement figures: without it, a high correlation between two maps of the same
    tissue could just be shared anatomy rather than a preserved expression pattern.
    """
    return np.array([
        _correlate(gene_maps[a], gene_maps[b])[0]
        for a, b in itertools.combinations(genes, 2)
    ])
