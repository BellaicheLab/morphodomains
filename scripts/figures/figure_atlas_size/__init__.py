"""Figure: sensitivity of the vGEP reconstruction to atlas size.

For each atlas size k the full vGEP pipeline is run on a random k-gene subset of
the atlas-eligible pool (cutoff fixed at the reference optimum) and scored against
the full, fixed 20-gene validation set. Two nested pools are swept: the 24
curated training genes alone, and the same pool extended with stan, yrt and
Tollo (27).

Panels:
- saturation_curves: quality vs. atlas size, both pools on one axis.
- per_gene_loo: leave-one-out reconstruction quality per gene.
"""

from .panels import per_gene_loo, sampling_check, saturation_curves
from .preprocessing import load_atlas_size_data

FIGURE_NAME = "figure_atlas_size"

SCENES = {}

PANELS = {
    "saturation_curves": saturation_curves,
    "sampling_check": sampling_check,
    "per_gene_loo": per_gene_loo,
}


def preprocess(paths, fig_config):
    """Load all data needed by the atlas-size panels."""
    return load_atlas_size_data(paths, fig_config)
