"""Velocity (U) scene builders for figure_morphodynamics_comparison.

Scene builders add all layers to the viewer.  The dispatcher handles
exports via YAML-defined ``exports:`` entries.

Scenes:
- build_velocity_vi_scene: Vi control velocity field
- build_velocity_tolloi_scene: Tolloi velocity field
- build_velocity_difference_scene: Velocity difference with LISA + overlays
"""

import logging
import os

import numpy as np
from pyanimalprocessing.plotting.napari_export import convert_points_to_shapes
from pyanimalprocessing.plotting.napari_plotting import (
    add_legend,
    add_tensor_legends,
    add_tensor_scale_bar,
    create_bbox_for_scale,
    get_viewer_offset,
    setup_mask_picker,
    display_spatial_clusters,
)
from pyanimalprocessing.plotting.property_plotting import plot_property, read_property
from pyanimalprocessing.rescaling.geometry import (
    export_roi_to_image_space,
    find_affine_transform,
)
from pyanimalprocessing.spatial_statistics import (
    create_high_res_roi_outline,
    mask_and_crop_moran,
)

from vgep.processing import normalize_01

from scripts.figures._napari_helpers import reformat_time_legends
from scripts.figures.figure_flow_deformation._helpers import apply_spatial_masks

from ._helpers import (
    WHITE_TO_RED_NAPARI,
    add_cluster_overlay,
    IMAGE_SIZE,
    POINTS_DATA,
    POINTS_EDGE_COLOR,
    POINTS_FACE_COLOR,
    VELOCITY_SCALE_SETTINGS,
    compute_flow_roi,
)

logger = logging.getLogger(__name__)


def build_velocity_vi_scene(viewer, shared_data):
    """Build velocity field scene for vi control."""
    logger.info("Building velocity vi scene...")

    AOA_vi = shared_data["AOA_vi"]
    animal = shared_data["animal"]
    max_idx_u = shared_data["moran_data"]["max_idx_u"]
    outlines = shared_data["outlines"]

    bundle = read_property(
        AOA_vi, "UPIV", reference_animal=animal,
        grid_bulk_threshold=0.6, significance=0.1,
    )

    fig_config = shared_data.get("_fig_config", {})
    masks = fig_config.get("masks", [])
    if masks:
        bundle = apply_spatial_masks(bundle, masks)

    settings = VELOCITY_SCALE_SETTINGS
    quantity_name = "Velocity (Control)"
    n_frames = bundle["property_data"].shape[0]
    start_frame = int(round(np.asarray(bundle["affine_transform"])[0, -1]))

    viewer.add_image(
        outlines, name="Tissue Outlines", contrast_limits=[0, 1], colormap="gray"
    )

    plot_property(
        **{**bundle, "property_name": quantity_name},
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
        time_array=AOA_vi.attrs["TimeArray"],
        image_size=IMAGE_SIZE,
        quantity=quantity_name,
        start_frame=start_frame,
        quantity_color=settings["color"],
        quantity_anchor="lower_left",
        quantity_translation=[-250, 50],
        time_color="black",
        time_anchor="lower_left",
        time_translation=[-130, 50],
        legend_size=30,
    )

    # Scale bar text label
    bbox = create_bbox_for_scale(
        settings["scale_position"], settings["bbox_x"], settings["bbox_y_offset"]
    )
    legend_text = settings["text_string"].format(length=settings["length"])
    add_legend(
        legend_text, viewer=viewer, color=settings["color"], position="center",
        translation=[0, -20], size=30, bbox=bbox, name="scale legend",
    )

    offset = get_viewer_offset(viewer, len(AOA_vi.attrs["TimeArray"]))
    viewer_frame = max_idx_u + offset

    # Add marker points and convert to shapes
    points_data = POINTS_DATA.copy()
    points_data[:, 0] = viewer_frame
    viewer.add_points(
        points_data,
        face_color=POINTS_FACE_COLOR,
        border_color=POINTS_EDGE_COLOR,
        size=50,
    )
    convert_points_to_shapes(viewer)

    viewer.dims.set_point(0, viewer_frame)

    if fig_config.get("pick_masks", False):
        setup_mask_picker(viewer, config_section="figure_morphodynamics_comparison")

    reformat_time_legends(viewer)
    logger.info("Velocity vi scene ready")


def build_velocity_tolloi_scene(viewer, shared_data):
    """Build velocity field scene for Tolloi."""
    logger.info("Building velocity tolloi scene...")

    AOA_Tolloi = shared_data["AOA_Tolloi"]
    AOA_vi = shared_data["AOA_vi"]
    animal = shared_data["animal"]
    max_idx_u = shared_data["moran_data"]["max_idx_u"]
    outlines = shared_data["outlines"]

    bundle = read_property(
        AOA_Tolloi, "UPIV", reference_animal=animal,
        grid_bulk_threshold=0.6, significance=0.1,
    )

    fig_config = shared_data.get("_fig_config", {})
    masks = fig_config.get("masks", [])
    if masks:
        bundle = apply_spatial_masks(bundle, masks)

    settings = VELOCITY_SCALE_SETTINGS
    quantity_name = "Velocity (TolloRNAi)"
    n_frames = bundle["property_data"].shape[0]
    start_frame = int(round(np.asarray(bundle["affine_transform"])[0, -1]))

    viewer.add_image(
        outlines, name="Tissue Outlines", contrast_limits=[0, 1], colormap="gray"
    )

    plot_property(
        **{**bundle, "property_name": quantity_name},
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
        time_array=AOA_vi.attrs["TimeArray"],
        image_size=IMAGE_SIZE,
        quantity=quantity_name,
        start_frame=start_frame,
        quantity_color=settings["color"],
        quantity_anchor="lower_left",
        quantity_translation=[-250, 50],
        time_color="black",
        time_anchor="lower_left",
        time_translation=[-130, 50],
        legend_size=30,
    )

    # Scale bar text label
    bbox = create_bbox_for_scale(
        settings["scale_position"], settings["bbox_x"], settings["bbox_y_offset"]
    )
    legend_text = settings["text_string"].format(length=settings["length"])
    add_legend(
        legend_text, viewer=viewer, color=settings["color"], position="center",
        translation=[0, -20], size=30, bbox=bbox, name="scale legend",
    )

    offset = get_viewer_offset(viewer, len(AOA_vi.attrs["TimeArray"]))
    viewer_frame = max_idx_u + offset

    # Add marker points and convert to shapes
    points_data = POINTS_DATA.copy()
    points_data[:, 0] = viewer_frame
    viewer.add_points(
        points_data,
        face_color=POINTS_FACE_COLOR,
        border_color=POINTS_EDGE_COLOR,
        size=50,
    )
    convert_points_to_shapes(viewer)

    viewer.dims.set_point(0, viewer_frame)

    if fig_config.get("pick_masks", False):
        setup_mask_picker(viewer, config_section="figure_morphodynamics_comparison")

    reformat_time_legends(viewer)
    logger.info("Velocity tolloi scene ready")


def build_velocity_difference_scene(viewer, shared_data):
    """Build velocity difference scene with LISA overlays and ROIs.

    Adds ALL layers (including optional Tollo / Atlas Tollo / Cluster
    overlays).  The YAML ``exports:`` entries select which layers appear
    in each exported file via the ``layers:`` key.
    """
    logger.info("Building velocity difference scene...")

    DBA_Tolloi_vi = shared_data["DBA_Tolloi_vi"]
    AOA_vi = shared_data["AOA_vi"]
    animal = shared_data["animal"]
    moran_data = shared_data["moran_data"]
    max_idx_u = moran_data["max_idx_u"]
    p_values_u = moran_data["p_values_u"]
    hh_map_u = moran_data["hh_map_u"]
    ll_map_u = moran_data["ll_map_u"]
    outlines = shared_data["outlines"]
    xy_starts_grid = shared_data["xy_starts_grid"]
    xy_starts = shared_data["xy_starts"]
    overlap = shared_data["overlap"]
    boxsizes = shared_data["boxsizes"]
    pval_threshold = shared_data["pval_threshold_roi"]
    roi_component = shared_data["roi_component_velocity"]

    # Generate U difference fields (DBA is 5-D; read_property auto-detects)
    bundle = read_property(
        DBA_Tolloi_vi, "UPIV", reference_animal=animal,
        grid_bulk_threshold=0.6, significance=0.3,
    )

    fig_config = shared_data.get("_fig_config", {})
    masks = fig_config.get("masks", [])
    frame_array = AOA_vi.attrs["FrameArray"]
    frame_offset = int(np.floor(np.median(frame_array[0]))) - 1
    if masks:
        bundle = apply_spatial_masks(bundle, masks)

    settings = VELOCITY_SCALE_SETTINGS
    quantity_name = "Velocity difference"
    n_frames = bundle["property_data"].shape[0]
    start_frame = int(round(np.asarray(bundle["affine_transform"])[0, -1]))

    viewer.add_image(
        outlines, name="Tissue Outlines", contrast_limits=[0, 1], colormap="gray"
    )

    plot_property(
        **{**bundle, "property_name": quantity_name},
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
        time_array=AOA_vi.attrs["TimeArray"],
        image_size=IMAGE_SIZE,
        quantity=quantity_name,
        start_frame=start_frame,
        quantity_color=settings["color"],
        quantity_anchor="lower_left",
        quantity_translation=[-250, 50],
        time_color="black",
        time_anchor="lower_left",
        time_translation=[-130, 50],
        legend_size=30,
    )

    # Scale bar text label
    bbox = create_bbox_for_scale(
        settings["scale_position"], settings["bbox_x"], settings["bbox_y_offset"]
    )
    legend_text = settings["text_string"].format(length=settings["length"])
    add_legend(
        legend_text, viewer=viewer, color=settings["color"], position="center",
        translation=[0, -20], size=30, bbox=bbox, name="scale legend",
    )

    time_offset = get_viewer_offset(viewer, len(AOA_vi.attrs["TimeArray"]))

    affine_transform = find_affine_transform(
        np.array(list(xy_starts_grid)),
        np.array(list(xy_starts[0])),
        overlap[:2],
        np.array(list(boxsizes[0][:2])),
    )

    # Mask and crop Moran data for visualization
    p_vals_viz, crop_y, crop_x, affine_viz = mask_and_crop_moran(
        p_values_u,
        affine_transform,
        masks,
        IMAGE_SIZE,
        frame_offset=frame_offset,
    )
    hh_viz = hh_map_u[:, crop_y, crop_x]
    ll_viz = ll_map_u[:, crop_y, crop_x]

    # LISA overlays
    display_spatial_clusters(
        viewer,
        p_vals_viz,
        hh_viz,
        "u_higher_iso_value",
        "I Forest",
        affine_viz,
        time_offset=time_offset,
    )
    display_spatial_clusters(
        viewer,
        p_vals_viz,
        ll_viz,
        "u_lower_iso_value",
        "I Purple",
        affine_viz,
        time_offset=time_offset,
    )

    # ROI outline
    roi_upscale_factor = 20
    roi_outline_width = 5

    output_base = shared_data["_output_base"]
    cache_dir_early = os.path.join(output_base, "cache")
    roi_flow = compute_flow_roi(
        p_values_u,
        hh_map_u,
        max_idx_u,
        pval_threshold,
        roi_component,
        cache_dir=cache_dir_early,
    )
    roi_slice = roi_flow[max_idx_u]

    # Crop ROI to tissue extent before upscaling (uses same crop as Moran)
    roi_slice_viz = roi_slice[crop_y, crop_x]
    high_res_outline = create_high_res_roi_outline(
        roi_slice_viz,
        upscale_factor=roi_upscale_factor,
        outline_width=roi_outline_width,
    )

    high_res_labels = np.zeros(
        (time_offset + p_values_u.shape[0],) + high_res_outline.shape,
        dtype=np.float16,
    )
    high_res_labels[:] = high_res_outline

    scale_matrix = np.eye(len(affine_viz))
    scale_matrix[0, 0] = 1 / roi_upscale_factor
    scale_matrix[1, 1] = 1 / roi_upscale_factor

    shift = (roi_upscale_factor - 1) / 2.0
    translate_matrix = np.eye(len(affine_viz))
    translate_matrix[0, -1] = -shift
    translate_matrix[1, -1] = -shift

    new_affine = affine_viz @ scale_matrix @ translate_matrix
    high_res_labels[high_res_labels == 0] = np.nan
    viewer.add_image(
        high_res_labels, name="Flow ROI", affine=new_affine, colormap="blue"
    )

    # ── Save Flow ROI cache (side effect) ────────────────────────────────
    output_base = shared_data["_output_base"]
    cache_dir = os.path.join(output_base, "cache")
    os.makedirs(cache_dir, exist_ok=True)
    np.save(os.path.join(cache_dir, "flow_roi.npy"), roi_slice)
    np.save(os.path.join(cache_dir, "max_idx_u.npy"), max_idx_u)

    # Debug plot: grid-level ROI mask
    import matplotlib.pyplot as plt

    fig_dbg, ax = plt.subplots(figsize=(4, 3))
    ax.imshow(roi_slice.astype(float), cmap="Blues", origin="lower", vmin=0, vmax=1)
    ax.set_title(f"Flow ROI  (max_idx_u={max_idx_u}, n={roi_slice.sum()})")
    ax.set_xlabel("grid x")
    ax.set_ylabel("grid y")
    fig_dbg.tight_layout()
    fig_dbg.savefig(os.path.join(cache_dir, "flow_roi_grid.png"), dpi=100)
    plt.close(fig_dbg)

    # Image-space ROI (mirrors deformation panel pattern)
    image_shape = outlines.shape[-2:]
    flow_roi_image = export_roi_to_image_space(
        roi_slice,
        xy_starts_grid,
        xy_starts,
        overlap,
        boxsizes,
        image_shape,
    )
    np.save(os.path.join(cache_dir, "flow_roi_image.npy"), flow_roi_image)

    from skimage.io import imsave

    imsave(
        os.path.join(cache_dir, "flow_roi_image.png"),
        (flow_roi_image * 255).astype(np.uint8),
    )

    logger.info(
        "Flow ROI cache saved: %d grid positions, max_idx_u=%d",
        roi_slice.sum(),
        max_idx_u,
    )

    # ── Add overlay layers (always, if data available) ───────────────────
    tollo_pattern = shared_data.get("tollo_pattern")
    atlas_tollo_pattern = shared_data.get("atlas_tollo_pattern")
    cluster_pattern = shared_data.get("cluster_pattern")

    if tollo_pattern is not None:
        viewer.add_image(
            normalize_01(tollo_pattern),
            name="Tollo Expression",
            colormap=WHITE_TO_RED_NAPARI,
            contrast_limits=(0, 1),
            opacity=0.8,
        )

    if atlas_tollo_pattern is not None:
        viewer.add_image(
            normalize_01(atlas_tollo_pattern),
            name="Atlas Tollo Expression",
            colormap=WHITE_TO_RED_NAPARI,
            contrast_limits=(0, 1),
            opacity=0.8,
        )

    if cluster_pattern is not None:
        add_cluster_overlay(viewer, cluster_pattern)

    # ── Set viewer frame and add marker points ───────────────────────────
    viewer_frame = max_idx_u + time_offset

    points_data = POINTS_DATA.copy()
    points_data[:, 0] = viewer_frame
    viewer.add_points(
        points_data,
        face_color=POINTS_FACE_COLOR,
        border_color=POINTS_EDGE_COLOR,
        size=50,
    )
    convert_points_to_shapes(viewer)

    viewer.dims.set_point(0, viewer_frame)

    if fig_config.get("pick_masks", False):
        setup_mask_picker(viewer, config_section="figure_morphodynamics_comparison")

    reformat_time_legends(viewer)
    logger.info("Velocity difference scene ready")
