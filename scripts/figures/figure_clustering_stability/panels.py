"""Panel functions for the clustering-stability figure.

Panels:
- consensus_matrix_pcs_2_30: consensus heatmap over the full PC range
- consensus_matrix_pcs_10_20: consensus heatmap over the narrow PC band
- ari_vs_npcs: ARI of each per-#PC partition vs the reference 5-cluster
- stability_map_pcs_2_30 / _10_20: per-bin consensus on the tissue
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, to_rgb
from vgep.plotter import geometric

from scripts.figures import figsize, save_and_close
from scripts.figures.figure_morphodynamic_domains.preprocessing import COLORS_5
from scripts.figures._clustering_helpers import align_to_reference

# Matches the cluster-swatch edges of figure_morphodynamic_domains's heatmaps (_helpers.py).
_EDGE_LW = 0.5

# Inferno stopped before its yellow end so the bright-yellow macrochaetae stay legible
# on the tissue maps, which sit mostly at high stability.
CMAP = LinearSegmentedColormap.from_list(
    "inferno_no_yellow", plt.get_cmap("inferno")(np.linspace(0, 0.85, 256))
)


def _plot_consensus(shared_data, key):
    """Consensus heatmap reordered by the reference clusters, with a colour sidebar."""
    order = shared_data["order"]
    consensus = shared_data["consensus"][key][np.ix_(order, order)]
    ref = shared_data["reference_labels_5"].to_numpy()[order]
    n = consensus.shape[0]

    fig = plt.figure(figsize=figsize("1x1"))
    gs = fig.add_gridspec(1, 3, width_ratios=[0.4, 12, 0.5], wspace=0.04)
    ax_side = fig.add_subplot(gs[0])
    ax = fig.add_subplot(gs[1])
    cax = fig.add_subplot(gs[2])

    im = ax.imshow(
        consensus, cmap=CMAP, vmin=0, vmax=1, interpolation="nearest", aspect="auto"
    )
    boundaries = np.where(np.diff(ref) != 0)[0] + 1
    for b in boundaries:
        ax.axhline(b - 0.5, color="white", lw=0.6)
        ax.axvline(b - 0.5, color="white", lw=0.6)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("bins (ordered by selected morphodynamic domain)")

    side = np.array([to_rgb(COLORS_5[c]) for c in ref]).reshape(n, 1, 3)
    ax_side.imshow(side, interpolation="nearest", aspect="auto")
    ax_side.set_xticks([])
    ax_side.set_yticks([])
    ax_side.set_ylabel("selected morphodynamic domain")
    for spine in ax_side.spines.values():
        spine.set_linewidth(_EDGE_LW)

    cbar = fig.colorbar(im, cax=cax, label="co-clustering frequency")
    cbar.outline.set_linewidth(_EDGE_LW)
    return fig


def consensus_matrix_pcs_2_30(shared_data, output_base):
    """Consensus matrix over PCs 2–30."""
    return _plot_consensus(shared_data, "2_30")


def consensus_matrix_pcs_10_20(shared_data, output_base):
    """Consensus matrix over PCs 10–20."""
    return _plot_consensus(shared_data, "10_20")


def ari_vs_npcs(shared_data, output_base):
    """ARI of each per-#PC partition against the reference 5-cluster partition."""
    ari = shared_data["ari"]
    n_components = shared_data["n_components"]
    ps = sorted(ari)
    vals = [ari[p] for p in ps]

    fig, ax = plt.subplots(figsize=figsize("1x1"))
    ax.axvspan(10, 20, color="#cfe8d4", alpha=0.6, label="PCs 10–20")
    ax.axvline(
        n_components,
        color="#CC3311",
        ls="--",
        lw=1,
        label=f"reference ({n_components} PCs)",
    )
    ax.plot(ps, vals, marker="o", ms=3, color="#2E5A88")
    ax.set_xlabel("Number of PCA components")
    ax.set_ylabel("ARI vs reference 5-cluster")
    ax.set_ylim(0, 1.02)
    ax.grid(True, ls="--", alpha=0.6)
    ax.legend(fontsize=7, loc="lower right")
    plt.tight_layout()
    return fig


def _stability_map(shared_data, key):
    """Spatial map of per-bin consensus with same-cluster peers."""
    data = shared_data["stability_per_bin"][key].to_frame(name="stability")

    fig, ax = plt.subplots(figsize=figsize("1x1"))
    geometric(
        transcriptome=data,
        color=["stability"],
        geometry=shared_data["geometry"],
        ax=ax,
        colorbar=True,
        colorbar_horizontal=False,
        spines=False,
        cropped=True,
        show=False,
        cmap=CMAP,
        annotation=shared_data["macrochaete"],
        vmin=0,
        vmax=1,
    )
    return fig


def cluster_maps_per_pc(shared_data, output_base):
    """One spatial 5-cluster map per #PCs (colours aligned to the reference partition)."""
    labels_by_p = shared_data["labels_by_p"]
    reference = shared_data["reference_labels_5"].to_numpy()
    ari = shared_data["ari"]
    cmap = ListedColormap(COLORS_5)

    for p in sorted(labels_by_p):
        idx = shared_data["index"]
        aligned = align_to_reference(pd.Series(labels_by_p[p], index=idx), pd.Series(reference, index=idx)).to_numpy()
        data = pd.DataFrame({"cluster": aligned}, index=shared_data["index"])

        fig, ax = plt.subplots(figsize=figsize("1x1"))
        geometric(
            transcriptome=data,
            color=["cluster"],
            geometry=shared_data["geometry"],
            ax=ax,
            title=f"PCs = {p}  (ARI = {ari[p]:.2f})",
            colorbar=True,
            colorbar_horizontal=False,
            spines=False,
            cropped=True,
            show=False,
            cmap=cmap,
            annotation=shared_data["macrochaete"],
        )
        save_and_close(
            fig, os.path.join(output_base, f"cluster_maps_per_pc/clusters_pc_{p:02d}")
        )

    return None


def stability_map_pcs_2_30(shared_data, output_base):
    """Per-bin stability map over PCs 2–30."""
    return _stability_map(shared_data, "2_30")


def stability_map_pcs_10_20(shared_data, output_base):
    """Per-bin stability map over PCs 10–20."""
    return _stability_map(shared_data, "10_20")