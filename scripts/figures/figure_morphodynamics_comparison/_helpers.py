"""Figure-5-specific helper functions.

ROI computation, Moran data orchestration, cache I/O, colorbars,
and constants that are specific to the vi/Tolloi comparison analysis.
"""

import logging
import os

import numpy as np
import pandas as pd

from pyanimalprocessing.gep import advect_grid_to_movie
from pyanimalprocessing.spatial_statistics import (
    compute_local_spatial_association,
    compute_moran,
    create_diverging_moran_colorbar,
)
from pyanimalprocessing.rescaling.timings import find_closest_time_index, median_time
from pyanimalprocessing.plotting.property_plotting import read_property
from scripts.figures.config import aoa_backup
from vgep.plotter.palettes import WHITE_TO_RED, mpl_to_napari
from scripts.figures.figure_morphodynamic_domains.preprocessing import COLORS_5
from scripts.figures.figure_flow_deformation._helpers import IMAGE_SIZE

logger = logging.getLogger(__name__)


# ── Napari marker positions ──────────────────────────────────────────────────

POINTS_DATA = np.array(
    [
        [68, 345.59017449, 692.12181548],
        [68, 650.49377292, 780.64221502],
        [68, 684.91837274, 1604.37371076],
        [68, 758.68537236, 1953.53750895],
        [68, 1154.56827031, 1405.20281179],
        [68, 1535.69776834, 1506.01771127],
        [68, 1734.86866731, 2005.17440869],
        [68, 1705.36186746, 1611.75041072],
    ]
)

POINTS_DATA_2 = np.array(
    [
        [84.0, 331.89913007, 693.65264471],
        [84.0, 672.6087448, 791.24283586],
        [84.0, 718.83567745, 1601.07021157],
        [84.0, 771.91104457, 1958.90091247],
        [84.0, 1138.30228854, 1409.3140465],
        [84.0, 1475.58768604, 1505.19212903],
        [84.0, 1682.75282866, 1991.43097618],
        [84.0, 1631.38957016, 1618.19129774],
    ]
)

POINTS_FACE_COLOR = np.array([[1.0, 0.0, 0.0, 1.0]] * 8)
POINTS_EDGE_COLOR = np.array([[0.41176471, 0.41176471, 0.41176471, 1.0]] * 8)

# ── Scale bar settings (matching figure_flow_deformation convention) ────────────────────────

VELOCITY_SCALE_SETTINGS = {
    "length": 5,  # um.h^-1
    "scale_position": [1900, 2930],
    "bbox_x": [3250, 3350],
    "bbox_y_offset": 50,
    "color": "#FF732E",
    "vector_style": "triangle",
    "scale": 40,
    "text_string": "{length} \u00b5m.h\u207b\u00b9",
}

DEFORMATION_SCALE_SETTINGS = {
    "length": 0.05,  # h^-1
    "scale_position": [1900, 2930],
    "bbox_x": [3250, 3350],
    "bbox_y_offset": 50,
    "scale": 3000,
    "color": "blue",
    "vector_style": "line",
    "text_string": "{length} h\u207b\u00b9",
}


# ── ROI helpers ──────────────────────────────────────────────────────────────


def _largest_component(labeled_array, num_features):
    """Return the label of the largest connected component."""
    sizes = np.bincount(labeled_array.ravel())
    # Ignore background (label 0)
    sizes[0] = 0
    return int(np.argmax(sizes))


def filter_roi_by_component(roi_mask, component_label):
    """Filter a boolean ROI mask to keep only one connected component.

    Parameters
    ----------
    roi_mask : np.ndarray
        Boolean mask (2-D or 3-D with time as first axis).
    component_label : int, ``"largest"``, or None
        Which component to keep.  ``"largest"`` selects the component
        with the most pixels (robust to label renumbering).
        ``None`` keeps all components.

    Returns
    -------
    np.ndarray
        Filtered boolean mask.
    """
    from scipy.ndimage import label as ndimage_label

    if component_label is None:
        return roi_mask

    pick_largest = component_label == "largest"
    filtered_mask = np.zeros_like(roi_mask, dtype=bool)

    if roi_mask.ndim == 2:
        labeled_array, num_features = ndimage_label(roi_mask)
        if num_features == 0:
            return filtered_mask
        target = (
            _largest_component(labeled_array, num_features)
            if pick_largest
            else component_label
        )
        if target <= num_features:
            filtered_mask = labeled_array == target
        else:
            logger.warning(
                f"Component label {component_label} not found. "
                f"Only {num_features} components exist."
            )
    elif roi_mask.ndim == 3:
        for t in range(roi_mask.shape[0]):
            if np.any(roi_mask[t]):
                labeled_array, num_features = ndimage_label(roi_mask[t])
                target = (
                    _largest_component(labeled_array, num_features)
                    if pick_largest
                    else component_label
                )
                if target <= num_features:
                    filtered_mask[t] = labeled_array == target

    return filtered_mask


def save_connected_components_debug(roi_2d, cache_dir, name, selected_component=None):
    """Save a PNG showing all connected components with numbered labels.

    Parameters
    ----------
    roi_2d : np.ndarray
        2-D boolean mask (single time slice, before component filtering).
    cache_dir : str
        Directory to save the debug PNG.
    name : str
        Descriptive name used in filename and title (e.g. "contraction_LL",
        "flow_HH").
    selected_component : int or None
        Currently selected component index (highlighted in title).
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from scipy.ndimage import center_of_mass, label as ndimage_label

    labeled, n_components = ndimage_label(roi_2d)
    if n_components == 0:
        logger.warning("No connected components found for %s", name)
        return

    # Distinct colours per component (skip 0 = background)
    cmap_colors = ["#ffffff"] + [plt.cm.tab10(i % 10) for i in range(n_components)]
    cmap = ListedColormap(cmap_colors, N=n_components + 1)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(labeled, cmap=cmap, origin="lower", vmin=0, vmax=n_components)

    # Label each component at its centroid
    for comp_idx in range(1, n_components + 1):
        cy, cx = center_of_mass(labeled == comp_idx)
        marker = f"  << selected" if comp_idx == selected_component else ""
        ax.text(
            cx,
            cy,
            str(comp_idx),
            fontsize=14,
            fontweight="bold",
            ha="center",
            va="center",
            color="black",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8),
        )
        ax.text(
            cx + 1.5,
            cy + 1.2,
            marker,
            fontsize=8,
            color="red",
            ha="left",
            va="center",
        )

    sel_str = (
        f", selected={selected_component}" if selected_component is not None else ""
    )
    ax.set_title(f"{name}: {n_components} components{sel_str}")
    ax.set_xlabel("grid x")
    ax.set_ylabel("grid y")
    fig.tight_layout()

    os.makedirs(cache_dir, exist_ok=True)
    out_path = os.path.join(cache_dir, f"{name}_components.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Saved connected-components debug plot: %s", out_path)


def compute_contraction_roi(
    p_values,
    ll_map,
    max_idx,
    pval_threshold,
    roi_component,
    cache_dir=None,
):
    """Compute ROI for contraction analysis using Low-Low clusters."""
    roi = np.zeros_like(p_values, dtype=bool)
    roi[max_idx] = p_values[max_idx] < pval_threshold * ll_map[max_idx]
    if cache_dir is not None:
        save_connected_components_debug(
            roi[max_idx],
            cache_dir,
            "contraction_LL",
            selected_component=roi_component,
        )
    roi = filter_roi_by_component(roi, roi_component)
    return roi


def compute_flow_roi(
    p_values,
    hh_map,
    max_idx,
    pval_threshold,
    roi_component,
    cache_dir=None,
):
    """Compute ROI for flow analysis using High-High clusters."""
    roi = np.zeros_like(p_values, dtype=bool)
    roi[max_idx] = p_values[max_idx] < pval_threshold * hh_map[max_idx]
    if cache_dir is not None:
        save_connected_components_debug(
            roi[max_idx],
            cache_dir,
            "flow_HH",
            selected_component=roi_component,
        )
    roi = filter_roi_by_component(roi, roi_component)
    return roi


# ── Advection helpers ────────────────────────────────────────────────────────


def generate_gene_advection_pattern(
    wt15h_zarr,
    gene_name,
    xy_starts_grid,
    xy_start,
    overlap,
    boxsize,
    tracking_mask,
    frame_ref,
):
    """Generate an advected gene-expression pattern from the spatial transcriptome.

    Parameters
    ----------
    wt15h_zarr : zarr.Group
        Spatial transcriptome zarr group (``animal.GEP.SPATIAL_TRANSCRIPTOME_15APF``).
    gene_name : str
        Gene name to extract (e.g. ``"Tollo"``).
    xy_starts_grid, xy_start, overlap, boxsize, tracking_mask, frame_ref
        See :func:`advect_grid_to_movie`.

    Returns
    -------
    np.ndarray
        Advected expression pattern ``(n_frames, H, W)``.
    """
    from vgep.utils import to_matrix_form

    expression_matrix = to_matrix_form(wt15h_zarr, color=gene_name)
    return advect_grid_to_movie(
        expression_matrix,
        xy_starts_grid,
        xy_start,
        overlap,
        boxsize,
        tracking_mask,
        frame_ref,
    )


def generate_atlas_advection_pattern(
    atlas_path,
    gene_name,
    xy_starts_grid,
    xy_start,
    overlap,
    boxsize,
    tracking_mask,
    frame_ref,
):
    """Generate an advected pattern from an atlas CSV file.

    Parameters
    ----------
    atlas_path : str or Path
        Path to the atlas validation CSV (columns include ``coordX``,
        ``coordY``, and gene expression columns).
    gene_name : str
        Gene column name in the atlas CSV.
    xy_starts_grid, xy_start, overlap, boxsize, tracking_mask, frame_ref
        See :func:`advect_grid_to_movie`.

    Returns
    -------
    np.ndarray
        Advected expression pattern ``(n_frames, H, W)``.
    """
    atlas = pd.read_csv(atlas_path)

    if gene_name not in atlas.columns:
        available = [
            c for c in atlas.columns if c not in ["coordX", "coordY", "n_nota_gene"]
        ]
        raise ValueError(
            f"Gene '{gene_name}' not found in atlas. "
            f"Available genes: {available[:10]}..."
        )

    x_coords = atlas["coordX"].astype(int)
    y_coords = atlas["coordY"].astype(int)
    expression_matrix = np.full((x_coords.max() + 1, y_coords.max() + 1), np.nan)

    for _, row in atlas.iterrows():
        x, y = int(row["coordX"]), int(row["coordY"])
        value = row[gene_name]
        if pd.notna(value):
            expression_matrix[x, y] = value

    return advect_grid_to_movie(
        expression_matrix,
        xy_starts_grid,
        xy_start,
        overlap,
        boxsize,
        tracking_mask,
        frame_ref,
    )


def generate_cluster_advection_pattern(
    cluster_labels_path,
    xy_starts_grid,
    xy_start,
    overlap,
    boxsize,
    tracking_mask,
    frame_ref,
):
    """Generate an advected pattern from cluster-label CSV.

    Parameters
    ----------
    cluster_labels_path : str or Path
        Path to CSV with a ``(x, y)`` MultiIndex and a single cluster-ID
        column.
    xy_starts_grid, xy_start, overlap, boxsize, tracking_mask, frame_ref
        See :func:`advect_grid_to_movie`.

    Returns
    -------
    np.ndarray
        Advected cluster pattern ``(n_frames, H, W)``.  Cluster IDs are
        offset by +1 so that cluster 0 is distinguishable from background.
    """
    cluster_labels = pd.read_csv(cluster_labels_path, index_col=[0, 1])
    cluster_col = cluster_labels.columns[0]

    x_coords = cluster_labels.index.get_level_values(0).astype(int)
    y_coords = cluster_labels.index.get_level_values(1).astype(int)
    cluster_matrix = np.full((x_coords.max() + 1, y_coords.max() + 1), np.nan)

    for (x, y), row in cluster_labels.iterrows():
        cluster_matrix[int(x), int(y)] = row[cluster_col] + 1

    return advect_grid_to_movie(
        cluster_matrix,
        xy_starts_grid,
        xy_start,
        overlap,
        boxsize,
        tracking_mask,
        frame_ref,
    )


# ── Moran cache I/O ─────────────────────────────────────────────────────────


def save_moran_cache(moran_data, cache_path):
    """Save Moran computation results to disk cache."""
    cache_dir = os.path.dirname(cache_path)
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)

    cache_data = {
        "max_idx_epsilon": moran_data["max_idx_epsilon"],
        "max_idx_u": moran_data["max_idx_u"],
        "moran_indices_epsilon": np.array(moran_data["moran_indices_epsilon"]),
        "moran_indices_u": np.array(moran_data["moran_indices_u"]),
        "p_values_epsilon": moran_data["p_values_epsilon"],
        "p_values_u": moran_data["p_values_u"],
        "hh_map_epsilon": moran_data["hh_map_epsilon"],
        "ll_map_epsilon": moran_data["ll_map_epsilon"],
        "hl_map_epsilon": moran_data["hl_map_epsilon"],
        "lh_map_epsilon": moran_data["lh_map_epsilon"],
        "hh_map_u": moran_data["hh_map_u"],
        "ll_map_u": moran_data["ll_map_u"],
        "hl_map_u": moran_data["hl_map_u"],
        "lh_map_u": moran_data["lh_map_u"],
        "median_times": np.array(moran_data["median_times"], dtype=object),
        "data_epsilon_dba_iso": moran_data["data_epsilon_dba_iso"],
        "data_u_dba": moran_data["data_u_dba"],
        "time_array": moran_data["time_array"],
    }

    np.savez_compressed(cache_path, **cache_data)
    logger.info(f"Moran cache saved to {cache_path}")


def load_moran_cache(cache_path):
    """Load Moran computation results from disk cache."""
    if not os.path.exists(cache_path):
        logger.info(f"No Moran cache found at {cache_path}")
        return None

    import time

    age_hours = (time.time() - os.path.getmtime(cache_path)) / 3600
    logger.info("Loading Moran cache from %s (age: %.1fh)", cache_path, age_hours)
    if age_hours > 72:
        logger.warning(
            "Cached Moran data is %.1fh old. "
            "Run with --regenerate-cache to rebuild.",
            age_hours,
        )
    cached = np.load(cache_path, allow_pickle=True)

    moran_data = {
        "max_idx_epsilon": int(cached["max_idx_epsilon"]),
        "max_idx_u": int(cached["max_idx_u"]),
        "moran_indices_epsilon": list(cached["moran_indices_epsilon"]),
        "moran_indices_u": list(cached["moran_indices_u"]),
        "p_values_epsilon": cached["p_values_epsilon"].copy(),
        "p_values_u": cached["p_values_u"].copy(),
        "hh_map_epsilon": cached["hh_map_epsilon"].astype(bool).copy(),
        "ll_map_epsilon": cached["ll_map_epsilon"].astype(bool).copy(),
        "hl_map_epsilon": cached["hl_map_epsilon"].astype(bool).copy(),
        "lh_map_epsilon": cached["lh_map_epsilon"].astype(bool).copy(),
        "hh_map_u": cached["hh_map_u"].astype(bool).copy(),
        "ll_map_u": cached["ll_map_u"].astype(bool).copy(),
        "hl_map_u": cached["hl_map_u"].astype(bool).copy(),
        "lh_map_u": cached["lh_map_u"].astype(bool).copy(),
        "median_times": list(cached["median_times"]),
        "data_epsilon_dba_iso": cached["data_epsilon_dba_iso"].copy(),
        "data_u_dba": cached["data_u_dba"].copy(),
        "time_array": cached["time_array"].copy(),
    }

    logger.info(
        f"Cache loaded: max_idx_epsilon={moran_data['max_idx_epsilon']}, "
        f"max_idx_u={moran_data['max_idx_u']}"
    )
    return moran_data


# ── Moran computation orchestrator ──────────────────────────────────────────


def compute_moran_data(avgAnimal, reference_animal, seed=42, permutations=100_000):
    """Compute Moran's I indices and LISA maps.

    Parameters
    ----------
    avgAnimal : zarr.Group
        Average animal data containing tensors.
    reference_animal : zarr.Group
        Registered movie store the average grid is placed onto (its SAP registration).
    seed : int, optional
        Seed forwarded to the LISA permutation test.
    permutations : int, optional
        Number of permutations used to derive each cell's LISA p-value.

    Returns
    -------
    dict
        Moran data dictionary.
    """
    logger.info("Computing Moran data (this may take a while)...")

    DBA_Tolloi_vi = aoa_backup(avgAnimal, "DBA_avgAnimals_Tolloi-avgAnimals_vi_2h_0.96olap")
    time_frame = DBA_Tolloi_vi.attrs.asdict()["TimeArray"]

    logger.info("Reading deformation and velocity fields...")
    b_eps = read_property(
        DBA_Tolloi_vi, "EpsilonPIV",
        reference_animal=reference_animal, grid_bulk_threshold=0.6, significance=0,
    )
    b_u = read_property(
        DBA_Tolloi_vi, "UPIV",
        reference_animal=reference_animal, grid_bulk_threshold=0.6, significance=0,
    )

    # DBA is a 5-D difference backup: read_property returns no area-ratio mask, so
    # rebuild the grid_bulk_threshold=0.6 keep-mask from its own AreaRatios.
    # Both grids are (T, Ny, Nx).
    keep = np.asarray(DBA_Tolloi_vi["AreaRatios"])[:, 0, :, :, 0] > 0.6
    eps = b_eps["property_data"]  # (T, 1, Ny, Nx, 4)
    data_epsilon_dba_iso = np.where(keep, (eps[..., 0] + eps[..., 3])[:, 0] / 2, np.nan)
    data_u_dba = np.where(
        keep, np.linalg.norm(b_u["property_data"][:, 0], axis=-1), np.nan
    )

    logger.info("Computing Moran's I for deformation rate...")
    moran_indices_epsilon = []
    for i in range(data_epsilon_dba_iso.shape[0]):
        m = compute_moran(data_epsilon_dba_iso[i, :, :])
        moran_indices_epsilon.append(m.I)

    logger.info("Computing Moran's I for velocity...")
    moran_indices_u = []
    for i in range(data_u_dba.shape[0]):
        m = compute_moran(data_u_dba[i, :, :])
        moran_indices_u.append(m.I)

    time_array = np.arange(len(moran_indices_epsilon))
    median_times = [median_time(i[0], i[1]) for i in time_frame]
    max_idx_epsilon = int(np.nanargmax(moran_indices_epsilon))
    max_idx_u = int(np.nanargmax(moran_indices_u))

    logger.info(
        "Computing LISA maps for deformation rate (seed=%d, perms=%d)...",
        seed, permutations,
    )
    (
        p_values_epsilon,
        hh_map_epsilon,
        ll_map_epsilon,
        hl_map_epsilon,
        lh_map_epsilon,
    ) = compute_local_spatial_association(
        data_epsilon_dba_iso,
        pval_threshold=0.01,
        seed=seed,
        permutations=permutations,
    )

    logger.info(
        "Computing LISA maps for velocity (seed=%d, perms=%d)...",
        seed, permutations,
    )
    (p_values_u, hh_map_u, ll_map_u, hl_map_u, lh_map_u) = (
        compute_local_spatial_association(
            data_u_dba,
            pval_threshold=0.01,
            seed=seed,
            permutations=permutations,
        )
    )

    logger.info("Moran data computation completed")

    return {
        "max_idx_epsilon": max_idx_epsilon,
        "max_idx_u": max_idx_u,
        "moran_indices_epsilon": moran_indices_epsilon,
        "moran_indices_u": moran_indices_u,
        "p_values_epsilon": p_values_epsilon,
        "p_values_u": p_values_u,
        "hh_map_epsilon": hh_map_epsilon,
        "ll_map_epsilon": ll_map_epsilon,
        "hl_map_epsilon": hl_map_epsilon,
        "lh_map_epsilon": lh_map_epsilon,
        "hh_map_u": hh_map_u,
        "ll_map_u": ll_map_u,
        "hl_map_u": hl_map_u,
        "lh_map_u": lh_map_u,
        "median_times": median_times,
        "data_epsilon_dba_iso": data_epsilon_dba_iso,
        "data_u_dba": data_u_dba,
        "time_array": time_array,
    }


# ── Moran time-series plots ─────────────────────────────────────────────────


def plot_moran_figures(moran_data, start_time="16h47", end_time="26h47"):
    """Create Moran's I time-series plots from computed data.

    Returns
    -------
    tuple
        (fig_U, fig_epsilon) matplotlib Figure objects.
    """
    import matplotlib.pyplot as plt

    logger.info("Creating Moran's I plots...")

    moran_indices_epsilon = np.array(moran_data["moran_indices_epsilon"])
    moran_indices_u = np.array(moran_data["moran_indices_u"])
    median_times = list(moran_data["median_times"])
    max_idx_epsilon = moran_data["max_idx_epsilon"]
    max_idx_u = moran_data["max_idx_u"]

    start_idx = find_closest_time_index(start_time, median_times)
    end_idx = find_closest_time_index(end_time, median_times)

    moran_indices_epsilon = moran_indices_epsilon[start_idx : end_idx + 1]
    moran_indices_u = moran_indices_u[start_idx : end_idx + 1]
    median_times = median_times[start_idx : end_idx + 1]
    time_array = np.arange(len(moran_indices_epsilon))

    max_idx_epsilon = max(
        0, min(max_idx_epsilon - start_idx, len(moran_indices_epsilon) - 1)
    )
    max_idx_u = max(0, min(max_idx_u - start_idx, len(moran_indices_u) - 1))

    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.linewidth": 0.7,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "figure.dpi": 300,
        }
    )

    fig_U, ax_U = plt.subplots(1, 1, figsize=(4.5, 2.5), num="Moran's I - Velocity")
    fig_epsilon, ax_epsilon = plt.subplots(
        1, 1, figsize=(4.5, 2.5), num="Moran's I - Deformation Rate"
    )

    max_val_epsilon = np.nanmax(moran_indices_epsilon)
    max_time_epsilon = median_times[max_idx_epsilon]
    max_val_u = np.nanmax(moran_indices_u)
    max_time_u = median_times[max_idx_u]

    init_val_epsilon = moran_indices_epsilon[0]
    init_val_u = moran_indices_u[0]

    ax_epsilon.plot(
        time_array,
        moran_indices_epsilon,
        color="#1f77b4",
        linewidth=1.5,
        label="Deformation rate",
    )
    ax_epsilon.axvline(
        time_array[max_idx_epsilon],
        color="#ff5555",
        linestyle="--",
        linewidth=1.0,
        label=f"Peak: {max_val_epsilon:.3f} at {max_time_epsilon}",
    )
    ax_epsilon.axhline(
        init_val_epsilon,
        color="#1f77b4",
        linestyle=":",
        linewidth=0.8,
        alpha=0.6,
        label=f"Initial I = {init_val_epsilon:.3f} ({start_time})",
    )

    ax_U.plot(
        time_array, moran_indices_u, color="#ff7f0e", linewidth=1.5, label="Velocity"
    )
    ax_U.axvline(
        time_array[max_idx_u],
        color="#459c12",
        linestyle="--",
        linewidth=1.0,
        label=f"Peak: {max_val_u:.3f} at {max_time_u}",
    )
    ax_U.axhline(
        init_val_u,
        color="#ff7f0e",
        linestyle=":",
        linewidth=0.8,
        alpha=0.6,
        label=f"Initial I = {init_val_u:.3f} ({start_time})",
    )

    for ax, data, title_suffix in [
        (ax_epsilon, moran_indices_epsilon, "deformation rate"),
        (ax_U, moran_indices_u, "velocity"),
    ]:
        ax.set_ylim(0, 1)
        ax.set_xlim(time_array[0] - 0.5, time_array[-1] + 0.5)
        ax.set_title(
            r"Moran's $I$ in $\Delta$ Tollo$^{RNAi}$-v$^{RNAi}$ " + title_suffix,
            pad=8,
        )
        ax.set_ylabel("Moran's $I$", labelpad=5)
        ax.set_xlabel("Time (hAPF)", labelpad=5)
        ax.grid(True, linestyle=":", alpha=0.3, linewidth=0.4)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    subsample_rate = 20
    median_times_sub = median_times[::subsample_rate]
    x_ticks = np.arange(0, len(moran_indices_epsilon), subsample_rate)

    for ax in [ax_epsilon, ax_U]:
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(median_times_sub, rotation=90)
        ax.legend()

    fig_U.tight_layout()
    fig_epsilon.tight_layout()

    return fig_U, fig_epsilon


# ── Moran colorbar generation ───────────────────────────────────────────────


def generate_moran_colorbars(output_base):
    """Generate and save diverging colorbar legends for Local Moran statistics.

    Saves figures directly to ``{output_base}/moran_plots/`` and returns None.
    """
    import os

    from scripts.figures import save_and_close

    colorbars = [
        (
            "#1f77b4",
            "#d62728",
            "Local Moran\nDeformation Rate",
            "LL (Contraction)",
            "HH (Dilation)",
            "moran_deformation_colorbar",
        ),
        (
            "#9467bd",
            "#2ca02c",
            "Local Moran\nVelocity",
            "LL (Low velocity)",
            "HH (High velocity)",
            "moran_velocity_colorbar",
        ),
    ]

    for color_low, color_high, title, ll_label, hh_label, filename in colorbars:
        fig = create_diverging_moran_colorbar(
            color_low, color_high, title, ll_label, hh_label
        )
        save_and_close(fig, os.path.join(output_base, f"moran_plots/{filename}"))

    return None


WHITE_TO_RED_NAPARI = mpl_to_napari(WHITE_TO_RED, "white_to_red")


def add_cluster_overlay(viewer, cluster_pattern):
    """Add the advected 5-domain pattern as an RGBA layer coloured with COLORS_5."""
    rgba = np.zeros(cluster_pattern.shape + (4,), dtype=np.uint8)
    for i, hex_color in enumerate(COLORS_5):
        rgba[cluster_pattern == i + 1] = [int(hex_color[k:k + 2], 16) for k in (1, 3, 5)] + [255]
    viewer.add_image(rgba, name="Morphogenetic Clusters", rgb=True, opacity=0.9)
