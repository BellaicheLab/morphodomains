"""CDF and survival panel functions for figure_spatial_transcriptome.

Thin wrappers around _helpers functions:
- cdf_validation, survival_validation
- cdf_training, survival_training
- cdf_combined, survival_combined
- cdf_protein_vs_fish, survival_protein_vs_fish
"""

import numpy as np

from .preprocessing import METHOD_SPECS
from ._helpers import create_cdf_and_survival_plots, create_protein_vs_fish_cdf


def _combined(shared_data):
    """Per-method validation+training correlations concatenated, keyed by method."""
    val = shared_data["correlations"]["validation"]
    train = shared_data["correlations"]["training"]
    return {s["key"]: np.concatenate([val[s["key"]], train[s["key"]]]) for s in METHOD_SPECS}


def _modality_inputs(shared_data):
    """vGEP correlations + gene lists for the protein-vs-FISH split."""
    val = shared_data["correlations"]["validation"]
    return (val["our"], list(val["genes_sorted"]),
            shared_data["protein_validation_genes"], shared_data["fish_validation_genes"])


def cdf_validation(shared_data, output_base):
    """CDF plot for validation gene correlations."""
    fig_cdf, _ = create_cdf_and_survival_plots(shared_data["correlations"]["validation"])
    return fig_cdf


def survival_validation(shared_data, output_base):
    """Survival plot for validation gene correlations."""
    _, fig_survival = create_cdf_and_survival_plots(shared_data["correlations"]["validation"])
    return fig_survival


def cdf_training(shared_data, output_base):
    """CDF plot for training gene correlations."""
    fig_cdf, _ = create_cdf_and_survival_plots(shared_data["correlations"]["training"])
    return fig_cdf


def survival_training(shared_data, output_base):
    """Survival plot for training gene correlations."""
    _, fig_survival = create_cdf_and_survival_plots(shared_data["correlations"]["training"])
    return fig_survival


def cdf_combined(shared_data, output_base):
    """CDF plot combining validation + training gene correlations."""
    fig_cdf, _ = create_cdf_and_survival_plots(_combined(shared_data))
    return fig_cdf


def survival_combined(shared_data, output_base):
    """Survival plot combining validation + training gene correlations."""
    _, fig_survival = create_cdf_and_survival_plots(_combined(shared_data))
    return fig_survival


def cdf_protein_vs_fish(shared_data, output_base):
    """CDF comparing Protein vs smiFISH genes in vGEP."""
    fig_cdf, _ = create_protein_vs_fish_cdf(*_modality_inputs(shared_data))
    return fig_cdf


def survival_protein_vs_fish(shared_data, output_base):
    """Survival comparing Protein vs smiFISH genes in vGEP."""
    _, fig_survival = create_protein_vs_fish_cdf(*_modality_inputs(shared_data))
    return fig_survival
