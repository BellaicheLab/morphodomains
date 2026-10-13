"""Figure: spatio-temporal clustering of the morphodynamic domains.

Runs the figure_morphodynamic_domains Ward clustering (k=5) over (spatial bin, time frame) observations
under a connectivity graph carrying both spatial and temporal edges, so the domains are
contiguous in space *and* time and a bin may change domain over the interval. Frames are
the 45 unique 4h sliding windows with centres 19h58 to 24h03 hAPF. Within a frame, pieces
of a domain below 3 bins are dissolved into their surroundings.

Panels:
- domain_movie: mp4 across every retained window
- domain_tiff: the same frames as a multi-page TIFF stack
- domain_stills: standalone SVGs at evenly spaced timepoints
- domain_stills_column: first/middle/last timepoint stacked as one panel
- agreement_vs_pooled: bins x time identity matrix with the per-frame ARI curve
- agreement_identity: the identity matrix alone
- persistence_map: per-bin fraction of frames spent in its modal domain
- temporal_stability_map: per-bin co-clustering with its own domain across the windows
- modal_domain_map: per-bin modal domain, compared against the reference partition
- silhouette_spacetime: silhouette vs number of domains, over the whole k sweep
"""

from .preprocessing_spatiotemporal_clustering import (
    load_clustering_spatiotemporal_data,
)
from .panels import (
    agreement_identity,
    agreement_vs_pooled,
    domain_movie,
    domain_stills,
    domain_stills_column,
    domain_tiff,
    modal_domain_map,
    persistence_map,
    silhouette_spacetime,
    temporal_stability_map,
)

FIGURE_NAME = "figure_clustering_spatiotemporal"

SCENES = {}

PANELS = {
    "domain_movie": domain_movie,
    "domain_tiff": domain_tiff,
    "agreement_vs_pooled": agreement_vs_pooled,
    "agreement_identity": agreement_identity,
    "domain_stills": domain_stills,
    "domain_stills_column": domain_stills_column,
    "persistence_map": persistence_map,
    "temporal_stability_map": temporal_stability_map,
    "modal_domain_map": modal_domain_map,
    "silhouette_spacetime": silhouette_spacetime,
}


def preprocess(paths, fig_config):
    """Load all data needed by the clustering-spatiotemporal panels."""
    return load_clustering_spatiotemporal_data(paths, fig_config)
