"""Comparison panels for figure_morphodynamics_comparison.

Panels:
- contraction_analysis: Contraction rate comparison (vi vs Tolloi)
- velocity_analysis: Velocity vector polar plot
- velocity_amplitude: Velocity amplitude comparison (barplot)
- rearrangement_anisotropy: Rearrangement anisotropy (ER eigenvalue difference) barplot
- cell_shape_trace: Cell shape isotropic change (ES trace) barplot
- cell_shape_anisotropy: Cell shape anisotropic change (ES eigenvalue difference) barplot
- apoptosis_rate: Apoptosis rate (dnA/nCoreRNs) barplot
"""

import logging

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

from pyanimalprocessing.plotting.comparison_plots import compare_genotypes_barplot
from pyanimalprocessing.rescaling.timings import find_closest_time_index, median_time
from pyanimalprocessing.spatial_statistics import (
    per_animal_mean_over_roi,
    per_animal_weighted_mean_in_roi,
)

from scripts.figures.config import aoa_backup

from ._helpers import compute_contraction_roi, compute_flow_roi

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Small wiring helpers shared by the panel functions
# ---------------------------------------------------------------------------


def _flow_roi_from_shared(shared_data):
    md = shared_data["moran_data"]
    return compute_flow_roi(
        md["p_values_u"],
        md["hh_map_u"],
        md["max_idx_u"],
        shared_data["pval_threshold_roi"],
        shared_data["roi_component_velocity"],
    )


def _peak_t_idx_in_lagrangian(shared_data, backup):
    """Map the Moran U peak time to the closest Lagrangian time index."""
    md = shared_data["moran_data"]
    peak_time = md["median_times"][md["max_idx_u"]]
    lag_medians = [median_time(f[0], f[1]) for f in backup.attrs["TimeArray"]]
    return find_closest_time_index(peak_time, lag_medians)


def _ncorerns_normalizer(avgAnimal_lagrangian, geno):
    """Per-animal nCoreRNs map (Ny, Nx, animals) at 18h00, used for the apoptosis rate."""
    inst = aoa_backup(avgAnimal_lagrangian, f"AOA_avgAnimals_{geno}_0h_0olap")
    inst_medians = [median_time(f[0], f[1]) for f in inst.attrs["TimeArray"]]
    ref_idx = find_closest_time_index("18h00", inst_medians)
    # (T, 1, Ny, Nx, 1, animals): drop the weighted-mean animal (index 0)
    return inst["nCoreRNs"][ref_idx, 0, :, :, 0, 1:]


def _lagrangian_barplot(
    shared_data,
    tensor_name,
    matrix_op,
    ylabel,
    *,
    normalize_by_ncorerns=False,
    negate=False,
):
    """Run the vi vs Tolloi comparison for a Lagrangian tensor over the flow ROI."""
    roi_flow = _flow_roi_from_shared(shared_data)
    md = shared_data["moran_data"]
    roi_2d = roi_flow[md["max_idx_u"]]

    values = {}
    for geno in ("vi", "tolloi"):
        backup = aoa_backup(
            shared_data["avgAnimal_lagrangian"], f"AOA_avgAnimals_{geno}_2h_0.96olap"
        )
        t_idx = _peak_t_idx_in_lagrangian(shared_data, backup)

        normalizer = None
        if normalize_by_ncorerns:
            normalizer = _ncorerns_normalizer(shared_data["avgAnimal_lagrangian"], geno)

        values[geno] = per_animal_weighted_mean_in_roi(
            backup,
            tensor_name,
            roi_2d,
            t_idx,
            matrix_op=matrix_op,
            normalizer=normalizer,
        )

    clean_vi = values["vi"][~np.isnan(values["vi"])]
    clean_tolloi = values["tolloi"][~np.isnan(values["tolloi"])]
    if negate:
        clean_vi = -clean_vi
        clean_tolloi = -clean_tolloi

    fig, p_value = compare_genotypes_barplot(clean_vi, clean_tolloi, ylabel=ylabel)
    logger.info(
        "Mann-Whitney %s %s: p=%.4f (n_vi=%d, n_tolloi=%d)",
        tensor_name,
        matrix_op or "",
        p_value,
        len(clean_vi),
        len(clean_tolloi),
    )
    return fig


# ---------------------------------------------------------------------------
# Lagrangian barplots (velocity ROI)
# ---------------------------------------------------------------------------


def rearrangement_anisotropy(shared_data, output_base):
    r"""R :math:`\to` elongation (ER anisotropy) in velocity ROI."""
    return _lagrangian_barplot(
        shared_data, "ER", "anisotropy", r"R $\to$ elongation (h$^{-1}$)"
    )


def cell_shape_trace(shared_data, output_base):
    r"""S :math:`\to` contraction (ES trace, negated) in velocity ROI."""
    return _lagrangian_barplot(
        shared_data,
        "ES",
        "trace",
        r"S $\to$ contraction (h$^{-1}$)",
        negate=True,
    )


def cell_shape_anisotropy(shared_data, output_base):
    r"""S :math:`\to` elongation (ES anisotropy) in velocity ROI."""
    return _lagrangian_barplot(
        shared_data, "ES", "anisotropy", r"S $\to$ elongation (h$^{-1}$)"
    )


def apoptosis_rate(shared_data, output_base):
    """Normalized apoptosis rate (dnA / nCoreRNs) in velocity ROI."""
    return _lagrangian_barplot(
        shared_data,
        "dnA",
        None,
        "Normalized apoptosis rate",
        normalize_by_ncorerns=True,
    )


# ---------------------------------------------------------------------------
# Eulerian panels (contraction + velocity)
# ---------------------------------------------------------------------------


def contraction_analysis(shared_data, output_base):
    """Contraction rate comparison between vi and Tolloi genotypes."""
    md = shared_data["moran_data"]
    roi_contraction = compute_contraction_roi(
        md["p_values_epsilon"],
        md["ll_map_epsilon"],
        md["max_idx_epsilon"],
        shared_data["pval_threshold_roi"],
        shared_data["roi_component_deformation"],
    )

    iso_vi = per_animal_mean_over_roi(
        shared_data["AOA_vi"],
        "EpsilonPIV",
        roi_contraction,
        component="iso",
    )
    iso_tolloi = per_animal_mean_over_roi(
        shared_data["AOA_Tolloi"],
        "EpsilonPIV",
        roi_contraction,
        component="iso",
    )

    fig, p_value = compare_genotypes_barplot(
        -iso_vi, -iso_tolloi,
        ylabel=r"Contraction Rate (h$^{-1}$)",
    )
    logger.info("Mann-Whitney contraction test: p-value = %s", p_value)
    return fig


def velocity_analysis(shared_data, output_base):
    """Polar plot of mean velocity vector in the flow ROI (vi vs Tolloi)."""
    roi_flow = _flow_roi_from_shared(shared_data)

    vecs_vi = per_animal_mean_over_roi(
        shared_data["AOA_vi"], "UPIV", roi_flow,
        component="dev",
    )
    vecs_tolloi = per_animal_mean_over_roi(
        shared_data["AOA_Tolloi"], "UPIV", roi_flow,
        component="dev",
    )

    # Reverse component order and flip the vertical sign so the polar plot's 0°
    # points in tissue "forward".
    avg_vi = np.mean(vecs_vi[:, ::-1], axis=0)
    avg_tolloi = np.mean(vecs_tolloi[:, ::-1], axis=0)
    avg_vi[1] = -avg_vi[1]
    avg_tolloi[1] = -avg_tolloi[1]

    mag_vi = np.linalg.norm(avg_vi)
    angle_vi = np.arctan2(avg_vi[1], avg_vi[0])
    mag_tolloi = np.linalg.norm(avg_tolloi)
    angle_tolloi = np.arctan2(avg_tolloi[1], avg_tolloi[0])

    import matplotlib.patches as mpatches

    fig = plt.figure(figsize=(2.5, 2.5))
    ax = plt.subplot(111, projection="polar")
    ax.set_facecolor("white")

    ax.arrow(
        angle_tolloi, 0, 0, mag_tolloi,
        head_width=0.05, head_length=0.05,
        fc="#F8AE8B", ec="#F8AE8B", linewidth=1.5, zorder=10,
    )
    ax.arrow(
        angle_vi, 0, 0, mag_vi,
        head_width=0.05, head_length=0.05,
        fc="#98C3E4", ec="#98C3E4", linewidth=1.5, zorder=10,
    )

    r_max = max(mag_vi, mag_tolloi) + 0.2
    ax.set_ylim(0, r_max)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=4))
    ax.grid(True, linestyle=":", alpha=0.9, color="gray", linewidth=0.7)
    ax.tick_params(axis="y", labelsize=8, colors="gray")
    ax.set_rlabel_position(90)

    legend_elements = [
        mpatches.Patch(facecolor="skyblue", label=f"V$^{{RNAi}}$ ({mag_vi:.2f} µm/h)"),
        mpatches.Patch(facecolor="salmon", label=f"Tollo$^{{RNAi}}$ ({mag_tolloi:.2f} µm/h)"),
    ]
    ax.legend(
        handles=legend_elements, loc="upper left", bbox_to_anchor=(0.05, 0.95),
        framealpha=0.7, fontsize="small", handlelength=1.0, handletextpad=0.5,
    )

    plt.tight_layout()
    return fig


def velocity_amplitude(shared_data, output_base):
    """Velocity amplitude comparison between vi and Tolloi genotypes."""
    roi_flow = _flow_roi_from_shared(shared_data)

    vecs_vi = per_animal_mean_over_roi(
        shared_data["AOA_vi"], "UPIV", roi_flow,
        component="dev",
    )
    vecs_tolloi = per_animal_mean_over_roi(
        shared_data["AOA_Tolloi"], "UPIV", roi_flow,
        component="dev",
    )

    # Reverse vector component order before computing magnitude
    amplitudes_vi = np.linalg.norm(vecs_vi[:, ::-1], axis=1)
    amplitudes_tolloi = np.linalg.norm(vecs_tolloi[:, ::-1], axis=1)

    fig, p_value = compare_genotypes_barplot(
        amplitudes_vi, amplitudes_tolloi,
        ylabel="Velocity Amplitude (µm/h)",
    )
    logger.info("Mann-Whitney velocity amplitude: p-value = %s", p_value)
    return fig