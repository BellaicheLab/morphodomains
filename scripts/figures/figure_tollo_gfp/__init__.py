"""Figure Tollo:GFP: Fluorescence images at 4 timepoints with ROI overlays.

All images (reference vi + 4 Tollo:GFP TIFs) are displayed in registered
coordinate space via napari affine transforms.  Advected flow/compression
ROI contours are overlaid at each timepoint.

Scenes (napari):
- build_tollo_gfp_scene: All images + ROIs in a single scene

Panels (matplotlib): none

YAML exports select the specific viewer frame for each timepoint.
"""

from .panels import build_tollo_gfp_scene
from .preprocessing import load_tollo_gfp_data

FIGURE_NAME = "figure_tollo_gfp"

PANELS = {}

SCENES = {
    "build_tollo_gfp_scene": build_tollo_gfp_scene,
}


def preprocess(paths, fig_config):
    """Load zarr animals and advected ROI contours."""
    return load_tollo_gfp_data(paths, fig_config)
