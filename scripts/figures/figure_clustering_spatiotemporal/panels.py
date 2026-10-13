"""Panel functions for the spatio-temporal clustering figure.

Panels:
- domain_movie: mp4 of the domains across the configured 4h sliding windows
- domain_tiff: the same frames as a multi-page TIFF stack
- domain_stills: standalone SVGs at evenly spaced timepoints, for assembly
- domain_stills_column: first/middle/last timepoint stacked as one panel
- persistence_map: per-bin fraction of frames spent in its modal domain
- modal_domain_map: per-bin modal domain, to compare against the reference partition
- silhouette_spacetime: silhouette vs number of domains, over the whole k sweep
- agreement_vs_pooled / agreement_identity: bins x time identity matrix, with and
  without the per-frame ARI curve

Categorical maps use the continuous branch of ``geometric`` with a fixed
``BoundaryNorm``, so domain *d* keeps colour *d* even in frames missing some domains.
"""

import os

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile
from matplotlib.colors import BoundaryNorm, ListedColormap, to_rgb
from vgep.plotter import geometric
from vgep.plotter._utils import get_cropping_boundaries
from vgep.utils._data_handling import to_matrix_form

from scripts.figures import figsize, save_and_close
from scripts.figures.figure_morphodynamic_domains.preprocessing import COLORS_5, COLORS_13
from scripts.figures.figure_clustering_stability.panels import CMAP

from .preprocessing_spatiotemporal_clustering import MIN_PIECE_BINS
from pyanimalprocessing.rescaling.timings import time_str_to_dec

MOVIE_FPS = 5
MOVIE_DPI = 200


def _domain_style(colors):
    """Colormap and norm pinning domain ``d`` to colour ``d``."""
    return ListedColormap(colors), BoundaryNorm(
        np.arange(-0.5, len(colors) + 0.5, 1.0), len(colors)
    )


def _view_box(shared_data):
    """Axis limits covering every bin of every frame.

    ``geometric(cropped=True)`` crops to the bins present in the frame it is handed, and
    the surviving bin set changes from frame to frame — so a per-frame crop would make
    the tissue breathe across the movie and leave the static panels unregistered with
    each other. Deriving the box through the same helper keeps the axis convention
    (display-x is bin-y, display-y is bin-x inverted) automatically consistent.
    """
    every_bin = pd.concat(shared_data["labels_by_frame"])
    every_bin = every_bin[~every_bin.index.duplicated()]
    geometry = every_bin.reset_index()[["x", "y"]]
    matrix = to_matrix_form(
        pd.DataFrame({"cluster": every_bin.astype(float)}),
        geometry=geometry,
        color="cluster",
    )
    x_min, x_max, y_min, y_max = get_cropping_boundaries(matrix)
    return (x_min, x_max), (y_max, y_min)


def _palette(n_clusters):
    """The reference 5-MD colours, extended from the 13-MD palette beyond 5 domains."""
    return COLORS_5 + COLORS_13[: max(0, n_clusters - len(COLORS_5))]


def _draw_domains(labels, geometry, shared_data, ax, xlim, ylim, colorbar=False):
    """Draw one domain map onto ``ax`` with a fixed field of view."""
    cmap, norm = _domain_style(_palette(shared_data["n_clusters"]))
    geometric(
        transcriptome=pd.DataFrame({"domain": labels.astype(float)}),
        color=["domain"],
        geometry=geometry,
        ax=ax,
        colorbar=colorbar,
        colorbar_horizontal=False,
        spines=False,
        cropped=False,
        show=False,
        cmap=cmap,
        norm=norm,
        annotation=shared_data["macrochaete"],
    )
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)


def _span_label(centres):
    """Window-centre span in whole hours, e.g. "20_24hAPF".

    Derived from the rendered windows so the filename cannot drift from them.
    """
    return f"{round(time_str_to_dec(centres[0]))}_{round(time_str_to_dec(centres[-1]))}hAPF"


def _movie_frames(shared_data):
    """One RGB frame per retained window, shared by the mp4 and the TIFF stack.

    Drawn straight off the canvas with no ``bbox_inches``, so every frame is the same
    pixel size — libx264 chokes on a size change, and a TIFF stack must be rectangular.
    """
    xlim, ylim = _view_box(shared_data)
    for index, centre in enumerate(shared_data["centres"]):
        fig, ax = plt.subplots(figsize=figsize("1x1"), dpi=MOVIE_DPI)
        # The global style sets savefig.transparent; an exported frame must be opaque.
        fig.patch.set_facecolor("white")
        _draw_domains(
            shared_data["labels_by_frame"][index],
            shared_data["geometry_by_frame"][index],
            shared_data, ax, xlim, ylim,
        )
        ax.set_title(f"{centre} APF")
        fig.canvas.draw()
        # Copy before closing: the buffer is a view onto the canvas.
        frame = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
        plt.close(fig)
        yield frame


def domain_movie(shared_data, output_base):
    """mp4 of the space-time domains across every retained 4h window."""
    k = shared_data["n_clusters"]
    span = _span_label(shared_data["centres"])
    movie_path = os.path.join(output_base, f"movie/domains_{span}_k{k}.mp4")
    os.makedirs(os.path.dirname(movie_path), exist_ok=True)

    writer = imageio.get_writer(
        movie_path, fps=MOVIE_FPS, codec="libx264", macro_block_size=1,
        ffmpeg_params=["-pix_fmt", "yuv420p", "-crf", "18"],
    )
    try:
        for frame in _movie_frames(shared_data):
            writer.append_data(frame)
    finally:
        writer.close()

    return None


def domain_tiff(shared_data, output_base):
    """The movie's frames as one multi-page TIFF stack, for ImageJ and figure assembly.

    Frame-for-frame identical to the mp4 — same renderer, same field of view, same
    titles — but lossless, so a still can be pulled out at full quality.
    """
    k = shared_data["n_clusters"]
    span = _span_label(shared_data["centres"])
    stack_path = os.path.join(output_base, f"movie/domains_{span}_k{k}.tif")
    os.makedirs(os.path.dirname(stack_path), exist_ok=True)

    stack = np.stack(list(_movie_frames(shared_data)))
    tifffile.imwrite(stack_path, stack, photometric="rgb", compression="deflate")

    return None


def domain_stills(shared_data, output_base):
    """Standalone SVGs at evenly spaced timepoints, for the Illustrator filmstrip."""
    xlim, ylim = _view_box(shared_data)
    k = shared_data["n_clusters"]

    for index in shared_data["still_indices"]:
        centre = shared_data["centres"][index]
        fig, ax = plt.subplots(figsize=figsize("1x1"))
        _draw_domains(
            shared_data["labels_by_frame"][index],
            shared_data["geometry_by_frame"][index],
            shared_data, ax, xlim, ylim,
        )
        save_and_close(
            fig,
            os.path.join(output_base, f"stills/k{k}/domains_{centre}APF"),
        )

    return None


def domain_stills_column(shared_data, output_base):
    """First, middle and last still stacked in one column as a single assembled panel.

    Same maps and same field of view as ``domain_stills``, drawn top to bottom in
    chronological order so the three timepoints ship as one figure element rather than
    three files to align by hand.
    """
    xlim, ylim = _view_box(shared_data)
    k = shared_data["n_clusters"]
    stills = shared_data["still_indices"]
    indices = [stills[0], stills[len(stills) // 2], stills[-1]]

    fig, axes = plt.subplots(len(indices), 1, figsize=figsize("3x1"))
    for ax, index in zip(axes, indices):
        _draw_domains(
            shared_data["labels_by_frame"][index],
            shared_data["geometry_by_frame"][index],
            shared_data, ax, xlim, ylim,
        )
    fig.subplots_adjust(hspace=0.02)

    span = _span_label(shared_data["centres"])
    save_and_close(
        fig, os.path.join(output_base, f"stills/domains_column_{span}_k{k}")
    )
    return None


def modal_domain_map(shared_data, output_base):
    """Per-bin modal domain over the interval, to compare with the reference map."""
    modal = shared_data["modal"]
    xlim, ylim = _view_box(shared_data)

    fig, ax = plt.subplots(figsize=figsize("1x1"))
    _draw_domains(
        modal, modal.reset_index()[["x", "y"]], shared_data, ax, xlim, ylim,
        colorbar=True,
    )
    k = shared_data["n_clusters"]
    save_and_close(fig, os.path.join(output_base, f"comparison/modal_domain_map_k{k}"))
    return None


def _agreement(shared_data, output_base, show_ari):
    """Bins x time matrix of domain identity, with the per-frame ARI curve optional.

    Rows are spatial bins grouped by their pooled domain and then ordered by the frame
    at which they first leave it; columns are the 4h windows. A solid block is a bin
    that keeps its pooled domain throughout.
    """
    agreement = shared_data["agreement"]
    reference = shared_data["reference"].loc[agreement.index]
    order = shared_data["agreement_order"]
    palette = _palette(shared_data["n_clusters"])
    hours = [time_str_to_dec(c) for c in shared_data["centres"]]

    ordered = agreement.to_numpy()[order]
    grouped = reference.to_numpy()[order]

    fig = plt.figure(figsize=figsize("2x2"))
    gs = fig.add_gridspec(
        2 if show_ari else 1, 2, width_ratios=[0.35, 12],
        height_ratios=[4, 1.15] if show_ari else [1], wspace=0.03, hspace=0.10,
    )
    ax_side = fig.add_subplot(gs[0, 0])
    ax = fig.add_subplot(gs[0, 1])

    cmap, norm = _domain_style(palette)
    ax.imshow(
        ordered, aspect="auto", interpolation="nearest", cmap=cmap, norm=norm,
        extent=[hours[0], hours[-1], len(ordered), 0],
    )
    for boundary in np.where(np.diff(grouped) != 0)[0] + 1:
        ax.axhline(boundary, color="white", lw=0.8)
    ax.set_yticks([])
    ax.set_ylabel("spatial bins")
    if show_ari:
        ax.set_xticks([])
    else:
        ax.set_xlabel("developmental time (hAPF, 4 h window centre)")

    ax_side.imshow(
        np.array([to_rgb(palette[c]) for c in grouped]).reshape(-1, 1, 3),
        aspect="auto", interpolation="nearest", extent=[0, 1, len(grouped), 0],
    )
    ax_side.set_xticks([])
    ax_side.set_yticks([])

    if show_ari:
        ax_ari = fig.add_subplot(gs[1, 1])
        ax_ari.plot(hours, shared_data["ari_by_frame"], color="#2E5A88", lw=1.2)
        ax_ari.set_xlim(hours[0], hours[-1])
        ax_ari.set_ylim(0, 1)
        ax_ari.set_xlabel("developmental time (hAPF, 4 h window centre)")
        ax_ari.set_ylabel("ARI vs pooled")
        ax_ari.grid(True, ls="--", alpha=0.4)
        for side in ("top", "right"):
            ax_ari.spines[side].set_visible(False)

    k = shared_data["n_clusters"]
    name = "agreement_vs_pooled" if show_ari else "agreement_identity"
    save_and_close(fig, os.path.join(output_base, f"agreement/{name}_k{k}"))
    return None


def agreement_vs_pooled(shared_data, output_base):
    """Bins x time identity matrix with the per-frame ARI curve underneath."""
    return _agreement(shared_data, output_base, show_ari=True)


def agreement_identity(shared_data, output_base):
    """Bins x time identity matrix alone, without the ARI curve."""
    return _agreement(shared_data, output_base, show_ari=False)


def temporal_stability_map(shared_data, output_base):
    """Per-bin co-clustering with its own domain's bins, across the 45 windows.

    The clustering-stability statistic with time windows in place of PC counts. Unlike the
    persistence map this is label-free: a bin scores high when it keeps the same company
    over the interval, whether or not the domain containing it is renamed or moves.
    """
    data = shared_data["temporal_stability"].to_frame(name="stability")
    xlim, ylim = _view_box(shared_data)

    fig, ax = plt.subplots(figsize=figsize("1x1"))
    geometric(
        transcriptome=data,
        color=["stability"],
        geometry=data.reset_index()[["x", "y"]],
        ax=ax,
        colorbar=True,
        colorbar_horizontal=False,
        spines=False,
        cropped=False,
        show=False,
        cmap=CMAP,
        annotation=shared_data["macrochaete"],
        vmin=0,
        vmax=1,
    )
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    k = shared_data["n_clusters"]
    save_and_close(
        fig, os.path.join(output_base, f"stability/temporal_stability_map_k{k}")
    )
    return None


def persistence_map(shared_data, output_base):
    """Per-bin fraction of frames spent in its modal domain."""
    data = shared_data["persistence"].to_frame(name="persistence")
    xlim, ylim = _view_box(shared_data)

    fig, ax = plt.subplots(figsize=figsize("1x1"))
    geometric(
        transcriptome=data,
        color=["persistence"],
        geometry=data.reset_index()[["x", "y"]],
        ax=ax,
        colorbar=True,
        colorbar_horizontal=False,
        spines=False,
        cropped=False,
        show=False,
        cmap=CMAP,
        annotation=shared_data["macrochaete"],
        vmin=0,
        vmax=1,
    )
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    k = shared_data["n_clusters"]
    save_and_close(fig, os.path.join(output_base, f"persistence/persistence_map_k{k}"))
    return None


def silhouette_spacetime(shared_data, output_base):
    """Silhouette vs k for the space-time cut, before and after removing small pieces."""
    k_values = shared_data["k_values"]
    curves = {
        f"Pieces < {MIN_PIECE_BINS} bins removed": (
            shared_data["silhouette_by_k"], "#2E5A88", 1.0,
        ),
        "Raw Ward cut": (shared_data["silhouette_by_k_raw"], "#2E5A88", 0.35),
    }

    fig, ax = plt.subplots(figsize=figsize("1x1"))
    for label, (scores, color, alpha) in curves.items():
        values = [scores[k] for k in k_values]
        optimal = k_values[int(np.argmax(values))]
        ax.plot(
            k_values, values, marker="o", ms=4, linestyle="-", color=color, alpha=alpha,
            label=f"{label} — optimal {optimal}",
        )
        ax.plot(
            optimal, max(values), marker="s", markersize=7, color=color, alpha=alpha,
            markeredgecolor="black", markeredgewidth=0.5, zorder=5,
        )

    ax.set_xlabel("Number of domains")
    ax.set_ylabel("Silhouette score")
    ax.set_xticks(k_values)
    ax.grid(True, linestyle="--", alpha=0.7)
    ax.legend(fontsize=5)
    ax.set_ylim(bottom=0)

    save_and_close(fig, os.path.join(output_base, "silhouette/silhouette_spacetime"))
    return None
