"""Figure: common-frame gene atlas (archetype -> rescale -> GEP -> AOA).

Reproduces the MATLAB atlas quantification in Python on pyanimalprocessing: builds a gene
archetype from the atlas macrochaetae, rescales every animal onto it with the
MATLAB pipeline's diagonal transform (independent x/y scaling + translation, no
rotation), measures per-box gene expression via analyze_labels on grid compartments,
and averages per gene across animals (AOT -> AOA).

Panels (matplotlib):
- gene_atlas_maps: per-gene AOA gene-expression maps on the common archetype grid.
"""

from .panels import gene_atlas_maps
from .preprocessing import load_gene_atlas_data

FIGURE_NAME = "figure_gene_atlas"

SCENES = {}

PANELS = {
    "gene_atlas_maps": gene_atlas_maps,
}


def preprocess(paths, fig_config):
    """Build (or load) the common-frame per-gene gene atlas."""
    return load_gene_atlas_data(paths, fig_config)
