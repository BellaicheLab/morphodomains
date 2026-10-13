"""Build the morphological-clustering feature matrix from the Lagrangian zarr.

1. Per-frame morphometric feature extraction for the AOA mean (animal idx 0)
2. AreaRatio cutoff filter at 0.15 (per frame, per pixel)
3. Iterative dead-pixel removal (4 passes, 8-conn ≤ 4 valid neighbors)
4. Apoptosis-rate outlier removal (drop rows where dnA_nrmlz > 0.4 in any frame)

``build`` is called by the morphodynamic-domain, spatial-transcriptome and
clustering figures. The CLI writes the matrix to a CSV::

    python scripts/build_morpho_clustering.py --output path.csv
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import zarr

# === MORPHODOMAINS_DATA bootstrap (mirrors scripts/figures/generate.py) ===
_REPO_ROOT = Path(__file__).resolve().parents[1]                    # morphodomains/
os.environ.setdefault("MORPHODOMAINS_DATA", str(_REPO_ROOT.parents[1]))
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.figures.config import aoa_backup, get_paths, load_config  # noqa: E402
from pyanimalprocessing.rescaling.timings import time_str_to_dec

logger = logging.getLogger(__name__)

# MATLAB frame73 / frame121 → Python 0-based 72 / 120 (4h windows centered on
# 18-22 hAPF "early" and 22-26 hAPF "late"). The output column suffix uses the
# 1-based MATLAB names that downstream code expects.
FRAME_INDICES = {"frame73": 72, "frame121": 120}

ANIMAL_AOA = 0  # animal axis index of the AOA mean (slot 0 in zarr)
AR_CUTOFF = 0.15
DEAD_NEIGHBOR_THRESHOLD = 4  # ≤ this many valid 3×3 cells (incl. center) → dead
DEAD_PIXEL_ITERATIONS = 4
OUTLIER_DNA_NRMLZ_THRESHOLD = 0.4
TIME_MATCH_TOLERANCE_MIN = 3  # ±3 min for 4h↔0h frame matching

TENSOR_NAMES = ["EG", "ED", "EA", "ER", "ES"]
# Final feature order (the 15 morphometric features per time window).
OUTPUT_FEATURES = [
    "tr_EG", "tr_ED", "tr_EA", "tr_ER", "tr_ES",
    "DiffEigEG", "DiffEigED", "DiffEigEA", "DiffEigER", "DiffEigES",
    "u", "ianiso", "dnD_nrmlz", "dnA_nrmlz", "CellArea",
]


def find_matched_0h_frame(ta_4h, ta_0h, idx_4h: int) -> int:
    """Return the largest 0h frame index whose start time is within ±3 min of the 4h frame start."""
    target = round(time_str_to_dec(ta_4h[idx_4h][0]) * 60)
    last = None
    for i, t in enumerate(ta_0h):
        v = round(time_str_to_dec(t[0]) * 60)
        if target - TIME_MATCH_TOLERANCE_MIN <= v < target + TIME_MATCH_TOLERANCE_MIN:
            last = i
    if last is None:
        raise ValueError(f"No 0h frame within ±3 min of 4h frame {idx_4h} (start {target} min)")
    return last


def extract_features_for_frame(g4, g0, idx_4h: int, idx_0h: int):
    """Compute the 15 morphometric features and AR mask for one (frame, animal)."""
    out = {}
    for tname in TENSOR_NAMES:
        # layout: (T, channel=1, Ny, Nx, comp, animals)
        T = g4[tname][idx_4h, 0, :, :, :, ANIMAL_AOA].astype(np.float64)
        a, b, c, d = T[..., 0], T[..., 1], T[..., 2], T[..., 3]
        # Eigenvalues in ascending order (as MATLAB eig), so (λ₁ − λ₂)/2 is negative.
        # Complex eigenvalues (discriminant < 0) keep their real part, i.e. 0.
        out[f"tr_{tname}"] = a + d
        disc = (a - d) ** 2 + 4 * b * c
        with np.errstate(invalid="ignore"):
            out[f"DiffEig{tname}"] = -np.sqrt(np.maximum(disc, 0)) / 2

    U = g4["U"][idx_4h, 0, :, :, :, ANIMAL_AOA].astype(np.float64)
    out["u"] = np.sqrt(U[..., 0] ** 2 + U[..., 1] ** 2)
    out["ianiso"] = g4["CellIaniso"][idx_4h, 0, :, :, 0, ANIMAL_AOA].astype(np.float64)
    out["CellArea"] = g4["CellArea"][idx_4h, 0, :, :, 0, ANIMAL_AOA].astype(np.float64)

    dnD = g4["dnD"][idx_4h, 0, :, :, 0, ANIMAL_AOA].astype(np.float64)
    dnA = g4["dnA"][idx_4h, 0, :, :, 0, ANIMAL_AOA].astype(np.float64)
    nC0 = g0["nCoreRNs"][idx_0h, 0, :, :, 0, ANIMAL_AOA].astype(np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        out["dnD_nrmlz"] = np.where(nC0 > 0, dnD / nC0, np.nan)
        out["dnA_nrmlz"] = np.where(nC0 > 0, dnA / nC0, np.nan)

    AR = g4["AreaRatios"][idx_4h, 0, :, :, 0, ANIMAL_AOA].astype(np.float64)
    return out, AR


def features_to_dataframe(per_frame, ny: int, nx: int) -> pd.DataFrame:
    """Stack per-frame feature dicts into a wide (x, y) → 30-col DataFrame.

    Uses MATLAB's 1-based indices (x ∈ 1..ny, y ∈ 1..nx) and column-major
    flattening, so x is the fast index.
    """
    x = np.tile(np.arange(1, ny + 1), nx)
    y = np.repeat(np.arange(1, nx + 1), ny)
    index = pd.MultiIndex.from_arrays([x, y], names=["x", "y"])

    data = {}
    columns = []
    for frame_label, (feats, _AR) in per_frame.items():
        for fname in OUTPUT_FEATURES:
            col = f"{fname}_{frame_label}"
            columns.append(col)
            data[col] = feats[fname].flatten(order="F")
    return pd.DataFrame(data, index=index, columns=columns)


def apply_ar_cutoff(df: pd.DataFrame, per_frame, cutoff: float) -> pd.DataFrame:
    """Set features to NaN at pixels where the per-frame AreaRatio < cutoff."""
    for frame_label, (_feats, AR) in per_frame.items():
        bad_mask = AR.flatten(order="F") < cutoff
        cols = [c for c in df.columns if c.endswith(f"_{frame_label}")]
        df.loc[bad_mask, cols] = np.nan
    return df


def remove_dead_pixels(df: pd.DataFrame) -> pd.DataFrame:
    """Iterative 8-connectivity dead-pixel removal.

    For ``DEAD_PIXEL_ITERATIONS`` passes, drop any row whose 3×3 spatial
    neighborhood (centre included) contains ≤ DEAD_NEIGHBOR_THRESHOLD valid
    pixels in the current (already-NaN-dropped) frame.
    """
    df = df.dropna(how="any").copy()
    df = df.sort_index()
    for _ in range(DEAD_PIXEL_ITERATIONS):
        rows_to_drop = []
        for ix, iy in df.index:
            neigh = df.loc[(slice(ix - 1, ix + 1), slice(iy - 1, iy + 1)), :]
            if neigh.shape[0] <= DEAD_NEIGHBOR_THRESHOLD:
                rows_to_drop.append((ix, iy))
        if not rows_to_drop:
            break
        df.loc[rows_to_drop, :] = np.nan
        df = df.dropna(how="any")
    return df


def remove_apoptosis_outliers(df: pd.DataFrame, frame_indices=FRAME_INDICES) -> pd.DataFrame:
    """Drop rows where dnA_nrmlz exceeds the threshold in any frame."""
    bad = pd.Series(False, index=df.index)
    for frame_label in frame_indices:
        col = f"dnA_nrmlz_{frame_label}"
        if col in df.columns:
            bad |= df[col] > OUTLIER_DNA_NRMLZ_THRESHOLD
    return df.loc[~bad]


def build(zarr_path: str | Path, frame_indices=FRAME_INDICES) -> pd.DataFrame:
    """End-to-end build: zarr → final feature DataFrame (15 columns per frame).

    ``frame_indices`` maps an output column suffix to a 4h-window frame index;
    the default builds the pooled matrix used in the paper (frame73 + frame121). A
    single-entry dict builds one developmental window (15 columns).
    """
    z = zarr.open(str(zarr_path))
    g4 = aoa_backup(z, "AOA_avgAnimals_vi_4h_0.98olap")
    g0 = aoa_backup(z, "AOA_avgAnimals_vi_0h_0olap")

    ta_4h = g4.attrs["TimeArray"]
    ta_0h = g0.attrs["TimeArray"]
    _, _, ny, nx, _, _ = g4["EG"].shape

    per_frame = {}
    for label, idx_4h in frame_indices.items():
        idx_0h = find_matched_0h_frame(ta_4h, ta_0h, idx_4h)
        logger.info(
            "Frame %s: 4h idx=%d (start %s) ↔ 0h idx=%d (start %s)",
            label, idx_4h, ta_4h[idx_4h][0], idx_0h, ta_0h[idx_0h][0],
        )
        per_frame[label] = extract_features_for_frame(g4, g0, idx_4h, idx_0h)

    df = features_to_dataframe(per_frame, ny=ny, nx=nx)
    logger.info("Initial grid: %d rows × %d cols", *df.shape)

    df = apply_ar_cutoff(df, per_frame, cutoff=AR_CUTOFF)
    n_after_ar = (~df.isna().any(axis=1)).sum()
    logger.info("After AR ≥ %.2f cutoff: %d rows still complete", AR_CUTOFF, n_after_ar)

    df = remove_dead_pixels(df)
    logger.info("After dead-pixel removal (%d iterations): %d rows", DEAD_PIXEL_ITERATIONS, len(df))

    df = remove_apoptosis_outliers(df, frame_indices)
    logger.info("After dnA_nrmlz > %.1f outlier removal: %d rows", OUTLIER_DNA_NRMLZ_THRESHOLD, len(df))

    expected_cols = [f"{f}_{lab}" for lab in frame_indices for f in OUTPUT_FEATURES]
    df = df[expected_cols].sort_index()
    logger.info("Final: %d rows × %d cols", *df.shape)
    return df


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zarr", default=None, help="Override Lagrangian zarr path")
    parser.add_argument("--output", default=None, help="Write the feature matrix to this CSV")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    cfg = load_config(base_dir=os.environ["MORPHODOMAINS_DATA"])
    paths = get_paths(cfg)
    zarr_path = args.zarr or paths["AVERAGES_240920_LAGRANGIAN"]

    logger.info("Source zarr: %s", zarr_path)
    df = build(zarr_path)

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.output)
        logger.info("Wrote %s", args.output)


if __name__ == "__main__":
    main()
