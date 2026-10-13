"""Velocity tensor map scene builder for figure_flow_deformation.

Scene builder:
- build_velocity_scene: Configures viewer with velocity fields and scale bar.
  Exports are defined in YAML (frames + output paths).
"""

import logging

import numpy as np
from pyanimalprocessing.plotting.napari_plotting import (
    add_legend,
    add_tensor_legends,
    add_tensor_scale_bar,
    create_bbox_for_scale,
    setup_mask_picker,
)
from pyanimalprocessing.plotting.property_plotting import plot_property

from scripts.figures._napari_helpers import reformat_time_legends

from ._helpers import (
    IMAGE_SIZE,
    VELOCITY_SCALE_SETTINGS,
    add_marker_points,
    apply_spatial_masks,
)

logger = logging.getLogger(__name__)


def build_velocity_scene(viewer, shared_data):
    """Configure the viewer with velocity fields and scale bar.

    Parameters
    ----------
    viewer : napari.Viewer
        The napari viewer (created by dispatcher).
    shared_data : dict
        Must contain ``"clean_outline"``, ``"velocity_fields"`` (a ``read_property``
        bundle), ``"raw_image"``, ``"AOA_vi_4h"``, and ``"_fig_config"``.
    """
    logger.info("Building velocity scene...")

    clean_outline = shared_data["clean_outline"]
    bundle = shared_data["velocity_fields"]
    AOA_vi_4h = shared_data["AOA_vi_4h"]
    fig_config = shared_data.get("_fig_config", {})

    masks = fig_config.get("masks", [])
    if masks:
        bundle = apply_spatial_masks(bundle, masks)

    settings = VELOCITY_SCALE_SETTINGS
    n_frames = bundle["property_data"].shape[0]
    start_frame = int(round(np.asarray(bundle["affine_transform"])[0, -1]))

    viewer.add_image(clean_outline, name="outline")
    viewer.add_image(
        shared_data["raw_image"], name="raw",
        colormap="gray_r", contrast_limits=(97, 160),
    )

    # Velocity arrows (display name becomes the layer name).
    plot_property(
        **{**bundle, "property_name": settings["quantity"]},
        viewer=viewer,
        vector_scale=settings["scale"],
        rescale_property="Rotation",
        vectors_kwargs={
            "edge_color": settings["color"],
            "vector_style": settings["vector_style"],
            "edge_width": 15,
            "opacity": 1,
        },
    )

    add_tensor_scale_bar(
        viewer,
        position=settings["scale_position"],
        length=settings["length"],
        scale=settings["scale"],
        n_frames=n_frames,
        color=settings["color"],
        vector_style=settings["vector_style"],
        edge_width=15,
        start_frame=start_frame,
    )

    add_tensor_legends(
        viewer,
        time_array=AOA_vi_4h.attrs["TimeArray"],
        image_size=IMAGE_SIZE,
        quantity=settings["quantity"],
        start_frame=start_frame,
        quantity_color=settings["color"],
        quantity_anchor="lower_left",
        quantity_translation=[-250, 50],
        time_color="black",
        time_anchor="lower_left",
        time_translation=[-130, 50],
        legend_size=30,
    )

    # Scale bar text label.
    bbox = create_bbox_for_scale(
        settings["scale_position"], settings["bbox_x"], settings["bbox_y_offset"]
    )
    add_legend(
        settings["text_string"].format(length=settings["length"]),
        viewer=viewer, color=settings["color"], position="center",
        translation=[0, -20], size=30, bbox=bbox, name="scale legend",
    )

    add_marker_points(viewer)

    if fig_config.get("pick_masks", False):
        setup_mask_picker(viewer)

    reformat_time_legends(viewer)
    logger.info("Velocity scene ready")