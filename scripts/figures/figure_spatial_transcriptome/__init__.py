"""figure_spatial_transcriptome: Spatial transcriptome generation and benchmarking.

Panels:
- umap_cell_types: UMAP coloured by cell type
- marker_genes: UMAPs of marker gene expression
- atlas_gene_transforms: Atlas genes with/without projective transform
- spatial_transcriptome: Additive-blend napari-style spatial transcriptome
- validation_correlation_bars: Bar plot of validation gene correlations
- training_correlation_bars: Bar plot of training gene correlations
- reconstruction_comparison_validation: Reconstruction grid (validation)
- reconstruction_comparison_training: Reconstruction grid (training)
- vgep_vs_atlas_comparisons: vGEP vs atlas side-by-side
- reconstruction_timing: Bar chart of compute times per method
- cdf_validation: CDF for validation gene correlations
- survival_validation: Survival plot for validation gene correlations
- cdf_training: CDF for training gene correlations
- survival_training: Survival plot for training gene correlations
- cdf_combined: CDF combining validation + training
- survival_combined: Survival combining validation + training
- cdf_protein_vs_fish: CDF comparing Protein vs smiFISH
- survival_protein_vs_fish: Survival comparing Protein vs smiFISH
"""

from .preprocessing import load_spatial_transcriptome_data
from .panels_umap import umap_cell_types, marker_genes
from .panels_atlas import atlas_gene_transforms
from .panels_transcriptome import spatial_transcriptome
from .panels_benchmark import (
    validation_correlation_bars,
    training_correlation_bars,
    reconstruction_comparison_validation,
    reconstruction_comparison_training,
    vgep_vs_atlas_comparisons,
    reconstruction_timing,
)
from .panels_cdf import (
    cdf_validation,
    survival_validation,
    cdf_training,
    survival_training,
    cdf_combined,
    survival_combined,
    cdf_protein_vs_fish,
    survival_protein_vs_fish,
)

FIGURE_NAME = "figure_spatial_transcriptome"

SCENES = {}

PANELS = {
    # UMAP
    "umap_cell_types": umap_cell_types,
    "marker_genes": marker_genes,
    # Atlas
    "atlas_gene_transforms": atlas_gene_transforms,
    # Spatial transcriptome
    "spatial_transcriptome": spatial_transcriptome,
    # Benchmark
    "validation_correlation_bars": validation_correlation_bars,
    "training_correlation_bars": training_correlation_bars,
    "reconstruction_comparison_validation": reconstruction_comparison_validation,
    "reconstruction_comparison_training": reconstruction_comparison_training,
    "vgep_vs_atlas_comparisons": vgep_vs_atlas_comparisons,
    "reconstruction_timing": reconstruction_timing,
    # CDF / Survival
    "cdf_validation": cdf_validation,
    "survival_validation": survival_validation,
    "cdf_training": cdf_training,
    "survival_training": survival_training,
    "cdf_combined": cdf_combined,
    "survival_combined": survival_combined,
    "cdf_protein_vs_fish": cdf_protein_vs_fish,
    "survival_protein_vs_fish": survival_protein_vs_fish,
}


def preprocess(paths, fig_config):
    """Load all data needed by figure_spatial_transcriptome panels.

    Parameters
    ----------
    paths : dict
        Resolved paths from config (uppercase keys).
    fig_config : dict
        The figure_spatial_transcriptome config section from figure_config.yaml.

    Returns
    -------
    dict
        Shared data for all figure_spatial_transcriptome panels.
    """
    return load_spatial_transcriptome_data(paths, fig_config)
