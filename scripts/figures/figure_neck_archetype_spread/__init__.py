"""Figure: what the neck landmark is worth to the gene atlas.

The gene atlas is built twice from the neck-clicked animals -- registered on the 8 macrochaetae,
and on the 8 macrochaetae plus the neck -- and the two are correlated, against the floor set by
rebuilding it from half the animals.

Panels (matplotlib):
- agreement_by_gene: each gene's neck effect beside the floor set by resampling its animals.
- agreement_distributions: the same two correlations as distributions.
- agreement_map: per-box agreement of the two atlases, over the tissue.
- atlas_statistics: writes every underlying number to atlas_statistics.txt (no figure).
"""

from .panels import (
    agreement_by_gene,
    agreement_distributions,
    agreement_map,
    atlas_statistics,
)
from .preprocessing import load_neck_atlas_agreement_data, load_neck_positions

FIGURE_NAME = "figure_neck_archetype_spread"

SCENES = {}

PANELS = {
    "agreement_by_gene": agreement_by_gene,
    "agreement_distributions": agreement_distributions,
    "agreement_map": agreement_map,
    "atlas_statistics": atlas_statistics,
}


def preprocess(paths, fig_config):
    """Compute (or load) the neck positions and the neck/no-neck atlas comparison."""
    return {
        **load_neck_positions(paths, fig_config),
        **load_neck_atlas_agreement_data(paths, fig_config),
    }
