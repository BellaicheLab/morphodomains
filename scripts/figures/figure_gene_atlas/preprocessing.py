"""Preprocessing for the common-frame gene atlas.

Reproduces the MATLAB atlas quantification in Python as one linear pipeline, all
on existing pyanimalprocessing machinery except two additions
(``generate_archetype`` and the ``xyscale_translation`` transform):

  1. write the canonical Gene archetype of the 17hAPF atlas (its 8 macrochaetae
     in microns + its fixed 40x67 grid frame) -> ``generate_archetype`` (explicit landmarks)
  2. rescale every animal onto that one fixed frame with the MATLAB pipeline's diagonal
     transform (independent x/y scaling + translation, NO rotation) -> ``generate_common_grid``
     with ``transform_type="xyscale_translation"`` (the archetype's stored frame gives every
     gene the identical grid, as MATLAB does)
  3. per-box gene expression (GEP) = per-box ``intensity_mean`` of the raw Gene
     channel -> ``analyze_labels(analysis_type="eulerian")`` (same regionprops path
     as per-cell intensity; runs on grid compartments)
  4. time-average the single fixed frame (17h APF as a 1-frame timelapse) ->
     ``average_over_time``
  5. average per gene across animals on the common grid -> ``average_over_animals``

The pipeline augments each atlas zarr in place (additive ``GRID`` / ``TIME_AVERAGE``
groups; ``IMAGE`` / ``SPACE_REGISTRATION`` untouched) and writes the archetype and
the per-gene AOA stores under ``data/gene_atlas/``. The assembled per-gene maps +
grid geometry are cached to a pickle, gated by ``pipeline.overwrite``.
"""

import glob
import logging
import os
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
import zarr
from joblib import Parallel, delayed

logger = logging.getLogger(__name__)

# ── Optional pyanimalprocessing imports (heavy dependency; guarded so --list works) ──
try:
    from pyanimalprocessing.analyze_labels import analyze_labels
    from pyanimalprocessing.average_over_animals import average_over_animals
    from pyanimalprocessing.average_over_time import average_over_time
    from pyanimalprocessing.rescaling.registration import (
        generate_archetype,
        generate_common_grid,
    )

    _HAS_PYANIMALPROCESSING = True
except ImportError:
    _HAS_PYANIMALPROCESSING = False


def _require_pyanimalprocessing():
    if not _HAS_PYANIMALPROCESSING:
        raise ImportError("pyanimalprocessing is required to build the gene atlas.")


# The atlas gene readout is always channel 1 ([Cad, Gene]).
_GENE_CHANNEL = 1
_REGISTRATION = "atlas_sr"

# Canonical Gene archetype of the 17hAPF atlas (the 2019 protein-scale archetype, as
# embedded in its 40x67 common grid). 8 macrochaetae in microns (y, x); the neck landmark is
# excluded (n_macros=8). Every animal -- fish and protein -- is rescaled onto this one frame, so
# all genes share the identical grid, as in the MATLAB pipeline.
_CANONICAL_ARCHETYPE_YX = [
    [-37.8125145, -109.1786175],
    [10.9751895, -87.3504135],
    [9.0800595, 62.7462045],
    [17.7572655, 133.7828265],
    [103.2276555, 35.0425305],
    [173.6478135, 55.1225925],
    [199.5614415, 153.7899345],
    [200.6938215, 80.0648145],
]
# Fixed grid extent (y_min, y_max, x_min, x_max) microns -> 40 x 67 boxes at 40um / 0.5 overlap.
_CANONICAL_GRID_FRAME = (-190.0, 610.0, -470.0, 870.0)


def _has_gene_channel(animal):
    """The atlas readout is channel 1, so a store left with only Cad carries no gene to measure."""
    return animal.IMAGE.D2.raw.shape[1] > _GENE_CHANNEL


def _load_atlas_animals(atlas_glob, gene_filter, max_per_gene):
    """Open atlas zarrs (writable) grouped by gene, keeping only registered ones."""
    groups = defaultdict(list)
    for path in sorted(glob.glob(atlas_glob)):
        animal = zarr.open(path, mode="a")
        gene = animal.attrs.get("gene")
        if gene_filter and gene not in gene_filter:
            continue
        landmarks = animal.SPACE_REGISTRATION[_REGISTRATION].landmarks[:]
        if (~np.isnan(landmarks).any(axis=1)).sum() < 2:
            logger.info("Skipping %s: no usable space registration", path)
            continue
        if not _has_gene_channel(animal):
            logger.info("Skipping %s: no gene channel", path)
            continue
        groups[gene].append(animal)
    if max_per_gene:
        groups = {g: v[:max_per_gene] for g, v in groups.items()}
    return dict(groups)


def _grid_and_measure(store_paths, archetype_path, boxsize, overlap, grid_name):
    """Rescale animals onto the archetype and measure their per-box expression, one at a time.

    Takes store paths rather than open groups so the chunks can run in separate processes.
    Each animal is stamped with the build it was measured under once it is done, so an
    interrupted build resumes where it stopped while a changed archetype still rebuilds.
    """
    archetype = zarr.open(archetype_path, mode="r")
    build = {
        "landmarks": archetype.SPACE_REGISTRATION[_REGISTRATION].landmarks[:].tolist(),
        "boxsize": list(boxsize),
        "overlap": overlap,
    }
    for path in store_paths:
        animal = zarr.open(path, mode="a")
        if "GRID" in animal and grid_name in animal.GRID:
            if animal.GRID[grid_name].attrs.get("measured_from") == build:
                continue
        generate_common_grid(
            [animal], boxsize=boxsize, overlap=overlap, registration_name=_REGISTRATION,
            reference_animal=archetype, transform_type="xyscale_translation",
            grid_type="eulerian", grid_name=grid_name, verbose=True,
        )
        analyze_labels(
            [animal], analysis_type="eulerian", grid_name=grid_name, intensity_image="raw",
            properties=["intensity_mean"], dimension="D2", n_jobs=1, verbose=True,
        )
        animal.GRID[grid_name].attrs["measured_from"] = build


def _average_gene_maps(animals_by_gene, aoa_dir, grid_name="EULERIAN"):
    """Average each gene's animals on the common grid. Returns ({gene: (Y,X) map}, centroids)."""
    os.makedirs(aoa_dir, exist_ok=True)
    gene_maps = {}
    centroids = None
    for gene, group in animals_by_gene.items():
        out = os.path.join(aoa_dir, f"{gene}.zarr")
        average_over_animals(
            group, output_path=out, animal_reference=group[0], rescale="Full",
            grid_name=grid_name,
        )
        avg = zarr.open(out, mode="r")
        # intensity_mean: (T, C, Y, X, D, 1+N); last-axis index 0 is the average.
        gene_maps[gene] = np.asarray(
            avg[f"{grid_name}/D2/default/intensity_mean"][0, _GENE_CHANNEL, :, :, 0, 0], float
        )
        centroids = avg[f"{grid_name}/D2/default/grid_centroids"][:]
    return gene_maps, centroids


def _run_atlas_pipeline(
    animals_by_gene, archetype_path, aoa_dir, boxsize, overlap,
    landmarks=_CANONICAL_ARCHETYPE_YX, n_macros=8, grid_name="EULERIAN", n_jobs=1,
):
    """archetype -> rescale -> GEP -> AOT -> per-gene AOA. Returns {gene: (Y,X) map}.

    ``landmarks`` / ``n_macros`` set which archetype the cohort is registered onto, and
    ``grid_name`` which ``GRID/<name>`` the whole build lives under, so several
    registrations can coexist on the same animals (see figure_neck_archetype_spread).

    ``n_jobs`` splits the per-animal rescale + measurement across processes. The archetype
    stipulates the grid frame, so each animal is gridded independently of its chunk-mates
    and the result does not depend on how the cohort is split.
    """
    all_animals = [a for group in animals_by_gene.values() for a in group]
    logger.info(
        "Atlas pipeline (%s, %d landmarks) over %d animals, %d genes",
        grid_name, n_macros, len(all_animals), len(animals_by_gene),
    )

    logger.info("1/5 writing canonical archetype (fixed 40x67 frame) -> %s", archetype_path)
    generate_archetype(
        output_path=archetype_path, landmarks=landmarks,
        grid_frame=_CANONICAL_GRID_FRAME, n_macros=n_macros, registration_name=_REGISTRATION,
        name="gene_archetype",
    )

    logger.info("2-3/5 rescaling onto archetype + per-box gene expression (%d jobs)", n_jobs)
    store_paths = [animal.store.dir_path() for animal in all_animals]
    # Round-robin so each chunk gets a comparable mix of the small protein and large FISH images.
    chunks = [store_paths[i::n_jobs] for i in range(n_jobs)]
    Parallel(n_jobs=n_jobs)(
        delayed(_grid_and_measure)(chunk, archetype_path, boxsize, overlap, grid_name)
        for chunk in chunks
    )

    logger.info("4/5 time-averaging the fixed frame (17h APF)")
    for animal in all_animals:
        animal.attrs["time_ref"] = "17h00"
        animal.attrs["frame_ref"] = 0
    average_over_time(
        all_animals, properties=["intensity_mean"], grid_type="eulerian",
        grid_name=grid_name, averaged_type=None, time_start="17h00", time_stop="17h05",
        time_window="00h01", time_step=5, verbose=True,
    )

    logger.info("5/5 averaging over animals per gene -> %s", aoa_dir)
    return _average_gene_maps(animals_by_gene, aoa_dir, grid_name)


def load_gene_atlas_data(paths, fig_config):
    """Build (or load the cache of) the common-frame per-gene gene atlas.

    Parameters
    ----------
    paths : dict
        Resolved uppercase paths; reads ``ZARR_ATLASES`` (glob) and ``GENE_ATLAS_BASE``.
    fig_config : dict
        The ``figure_gene_atlas`` config section (its ``pipeline`` block gates the run).

    Returns
    -------
    dict
        ``gene_maps`` ({gene: (Y,X) AOA map}), ``centroids`` ((Y,X,2) box centroids),
        ``archetype_path``.
    """
    pipeline_cfg = fig_config.get("pipeline", {})
    overwrite = pipeline_cfg.get("overwrite", False)
    gene_filter = pipeline_cfg.get("genes")
    max_per_gene = pipeline_cfg.get("max_animals_per_gene")
    boxsize = pipeline_cfg.get("boxsize", [40, 40])
    overlap = pipeline_cfg.get("overlap", 0.5)
    n_jobs = pipeline_cfg.get("n_jobs", 1)

    base = Path(paths["GENE_ATLAS_BASE"])
    archetype_path = str(base / "_gene_archetype.zarr")
    aoa_dir = str(base / "aoa")
    cache_path = base / "_gene_atlas_maps.pkl"

    if cache_path.exists() and not overwrite:
        logger.info("Loading cached gene atlas from %s", cache_path)
        with open(cache_path, "rb") as fh:
            return pickle.load(fh)

    _require_pyanimalprocessing()
    base.mkdir(parents=True, exist_ok=True)
    animals_by_gene = _load_atlas_animals(paths["ZARR_ATLASES"], gene_filter, max_per_gene)
    gene_maps, centroids = _run_atlas_pipeline(
        animals_by_gene, archetype_path, aoa_dir, boxsize, overlap, n_jobs=n_jobs
    )

    shared = {"gene_maps": gene_maps}
    with open(cache_path, "wb") as fh:
        pickle.dump(shared, fh)
    logger.info("Cached gene atlas (%d genes) -> %s", len(gene_maps), cache_path)
    return shared
