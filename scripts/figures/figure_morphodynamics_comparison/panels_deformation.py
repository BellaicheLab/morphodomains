"""Deformation (epsilon) scene builders for figure_morphodynamics_comparison.

Scene builders add all layers to the viewer.  The dispatcher handles
exports via YAML-defined ``exports:`` entries.

Scenes:
- build_deformation_vi_scene: Vi control deformation field
- build_deformation_tolloi_scene: Tolloi deformation field
- build_deformation_difference_scene: Deformation difference with LISA + overlays
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
    DEFORMATION_SCALE_SETTINGS,
    IMAGE_SIZE,
    POINTS_DATA_2,
    POINTS_EDGE_COLOR,
    POINTS_FACE_COLOR,
    compute_contraction_roi,
)

logger = logging.getLogger(__name__)


def build_deformation_vi_scene(viewer, shared_data):
    """Build deformation field scene for vi control."""
    logger.info("Building deformation vi scene...")

    AOA_vi = shared_data["AOA_vi"]
    animal = shared_data["animal"]
    max_idx_epsilon = shared_data["moran_data"]["max_idx_epsilon"]
    outlines = shared_data["outlines"]

    bundle = read_property(
        AOA_vi, "EpsilonPIV", reference_animal=animal,
        grid_bulk_threshold=0.6, significance=0,
    )

    fig_config = shared_data.get("_fig_config", {})
    masks = fig_config.get("masks", [])
    if masks:
        bundle = apply_spatial_masks(bundle, masks)

    settings = DEFORMATION_SCALE_SETTINGS
    quantity_name = "Deformation rate (Control)"
    n_frames = bundle["property_data"].shape[0]
    start_frame = int(round(np.asarray(bundle["affine_transform"])[0, -1]))

    viewer.add_image(
        outlines, name="Tissue Outlines", contrast_limits=[0, 1], colormap="gray"
    )

    plot_property(
        **{**bundle, "property_name": quantity_name},
        viewer=viewer,
        vector_scale=settings["scale"],
        scalar_scale=settings["scale"],
        rescale_property="Rotation",
        vectors_kwargs={
            "edge_color": settings["color"],
            "vector_style": settings["vector_style"],
            "edge_width": 8,
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
        edge_width=8,
        start_frame=start_frame,
        scalar=settings["length"],
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
    viewer_frame = max_idx_epsilon + offset

    # Hide deviatoric-bar layers (isotropic-only display); keep iso points visible.
    for layer in viewer.layers:
        if layer.name in (f"{quantity_name}_positive", f"{quantity_name}_negative"):
            layer.visible = False

    # Add marker points and convert to shapes
    points_data = POINTS_DATA_2.copy()
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
    logger.info("Deformation vi scene ready")


def build_deformation_tolloi_scene(viewer, shared_data):
    """Build deformation field scene for Tolloi."""
    logger.info("Building deformation tolloi scene...")

    AOA_Tolloi = shared_data["AOA_Tolloi"]
    AOA_vi = shared_data["AOA_vi"]
    animal = shared_data["animal"]
    max_idx_epsilon = shared_data["moran_data"]["max_idx_epsilon"]
    outlines = shared_data["outlines"]

    bundle = read_property(
        AOA_Tolloi, "EpsilonPIV", reference_animal=animal,
        grid_bulk_threshold=0.6, significance=0,
    )

    fig_config = shared_data.get("_fig_config", {})
    masks = fig_config.get("masks", [])
    if masks:
        bundle = apply_spatial_masks(bundle, masks)

    settings = DEFORMATION_SCALE_SETTINGS
    quantity_name = "Deformation rate (TolloRNAi)"
    n_frames = bundle["property_data"].shape[0]
    start_frame = int(round(np.asarray(bundle["affine_transform"])[0, -1]))

    viewer.add_image(
        outlines, name="Tissue Outlines", contrast_limits=[0, 1], colormap="gray"
    )

    plot_property(
        **{**bundle, "property_name": quantity_name},
        viewer=viewer,
        vector_scale=settings["scale"],
        scalar_scale=settings["scale"],
        rescale_property="Rotation",
        vectors_kwargs={
            "edge_color": settings["color"],
            "vector_style": settings["vector_style"],
            "edge_width": 8,
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
        edge_width=8,
        start_frame=start_frame,
        scalar=settings["length"],
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
    viewer_frame = max_idx_epsilon + offset

    # Hide deviatoric-bar layers (isotropic-only display); keep iso points visible.
    for layer in viewer.layers:
        if layer.name in (f"{quantity_name}_positive", f"{quantity_name}_negative"):
            layer.visible = False

    # Add marker points and convert to shapes
    points_data = POINTS_DATA_2.copy()
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
    logger.info("Deformation tolloi scene ready")


def build_deformation_difference_scene(viewer, shared_data):
    """Build deformation difference scene with LISA overlays and ROIs.

    Adds ALL layers (including optional Tollo / Atlas Tollo / Cluster
    overlays).  The YAML ``exports:`` entries select which layers appear
    in each exported file via the ``layers:`` key.
    """
    logger.info("Building deformation difference scene...")

    DBA_Tolloi_vi = shared_data["DBA_Tolloi_vi"]
    AOA_vi = shared_data["AOA_vi"]
    animal = shared_data["animal"]
    moran_data = shared_data["moran_data"]
    max_idx_epsilon = moran_data["max_idx_epsilon"]
    p_values_epsilon = moran_data["p_values_epsilon"]
    ll_map_epsilon = moran_data["ll_map_epsilon"]
    hh_map_epsilon = moran_data["hh_map_epsilon"]
    outlines = shared_data["outlines"]
    xy_starts_grid = shared_data["xy_starts_grid"]
    xy_starts = shared_data["xy_starts"]
    overlap = shared_data["overlap"]
    boxsizes = shared_data["boxsizes"]
    pval_threshold = shared_data["pval_threshold_roi"]
    contraction_threshold = shared_data["contraction_threshold"]
    roi_component = shared_data["roi_component_deformation"]
    output_base = shared_data["_output_base"]

    # Generate epsilon difference fields (DBA is 5-D; read_property auto-detects)
    bundle = read_property(
        DBA_Tolloi_vi, "EpsilonPIV", reference_animal=animal,
        grid_bulk_threshold=0.6, significance=0,
    )

    fig_config = shared_data.get("_fig_config", {})
    masks = fig_config.get("masks", [])
    frame_array = AOA_vi.attrs["FrameArray"]
    frame_offset = int(np.floor(np.median(frame_array[0]))) - 1
    if masks:
        bundle = apply_spatial_masks(bundle, masks)

    settings = DEFORMATION_SCALE_SETTINGS
    quantity_name = "Deformation rate difference"
    n_frames = bundle["property_data"].shape[0]
    start_frame = int(round(np.asarray(bundle["affine_transform"])[0, -1]))

    # Isotropic deformation grid (T, Ny, Nx) for the restrictive contraction ROI,
    # NaN outside the grid_bulk_threshold=0.6 area-ratio keep-mask.
    eps = bundle["property_data"]  # (T, 1, Ny, Nx, 4)
    keep = np.asarray(DBA_Tolloi_vi["AreaRatios"])[:, 0, :, :, 0] > 0.6
    iso_vals = np.where(keep, (eps[..., 0] + eps[..., 3])[:, 0] / 2, np.nan)

    viewer.add_image(
        outlines, name="Tissue Outlines", contrast_limits=[0, 1], colormap="gray"
    )

    plot_property(
        **{**bundle, "property_name": quantity_name},
        viewer=viewer,
        vector_scale=settings["scale"],
        scalar_scale=settings["scale"],
        rescale_property="Rotation",
        vectors_kwargs={
            "edge_color": settings["color"],
            "vector_style": settings["vector_style"],
            "edge_width": 8,
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
        edge_width=8,
        start_frame=start_frame,
        scalar=settings["length"],
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
        p_values_epsilon,
        affine_transform,
        masks,
        IMAGE_SIZE,
        frame_offset=frame_offset,
    )
    ll_viz = ll_map_epsilon[:, crop_y, crop_x]
    hh_viz = hh_map_epsilon[:, crop_y, crop_x]

    # LISA overlays
    display_spatial_clusters(
        viewer,
        p_vals_viz,
        ll_viz,
        "epsilon_lower_iso_value",
        "I Blue",
        affine_viz,
        time_offset=time_offset,
    )
    display_spatial_clusters(
        viewer,
        p_vals_viz,
        hh_viz,
        "epsilon_higher_iso_value",
        "I Bordeaux",
        affine_viz,
        time_offset=time_offset,
    )

    # ROI outline
    roi_upscale_factor = 20
    roi_outline_width = 5

    cache_dir = os.path.join(output_base, "cache")
    roi_contraction = compute_contraction_roi(
        p_values_epsilon,
        ll_map_epsilon,
        max_idx_epsilon,
        pval_threshold,
        roi_component,
        cache_dir=cache_dir,
    )
    roi_slice = roi_contraction[max_idx_epsilon]

    # Crop ROI to tissue extent before upscaling (uses same crop as Moran)
    roi_slice_viz = roi_slice[crop_y, crop_x]
    high_res_outline = create_high_res_roi_outline(
        roi_slice_viz,
        upscale_factor=roi_upscale_factor,
        outline_width=roi_outline_width,
    )

    high_res_labels = np.zeros(
        (time_offset + p_values_epsilon.shape[0],) + high_res_outline.shape,
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
    additional_scaling = 1.02
    vertical_scale_matrix = np.eye(len(affine_viz))
    vertical_scale_matrix[0, 0] *= additional_scaling
    new_affine = new_affine @ vertical_scale_matrix

    high_res_labels[high_res_labels == 0] = np.nan
    viewer.add_image(
        high_res_labels, name="Contraction ROI", affine=new_affine, colormap="green"
    )

    # Restrictive ROI
    contraction_values_at_max = iso_vals[max_idx_epsilon]
    restrictive_roi_slice = roi_slice & (
        contraction_values_at_max <= contraction_threshold
    )
    restrictive_roi_slice_viz = restrictive_roi_slice[crop_y, crop_x]

    restrictive_outline = create_high_res_roi_outline(
        restrictive_roi_slice_viz,
        upscale_factor=roi_upscale_factor,
        outline_width=roi_outline_width,
    )
    restrictive_labels = np.zeros(
        (time_offset + p_values_epsilon.shape[0],) + restrictive_outline.shape,
        dtype=np.float16,
    )
    restrictive_labels[:] = restrictive_outline
    restrictive_labels[restrictive_labels == 0] = np.nan
    viewer.add_image(
        restrictive_labels,
        name="Restrictive Contraction ROI",
        affine=new_affine,
        colormap="magenta",
    )

    logger.info(
        f"Restrictive ROI: {restrictive_roi_slice.sum()} pixels "
        f"(vs {roi_slice.sum()} in Moran ROI)"
    )

    # ── Save ROI cache (side effect) ─────────────────────────────────────
    cache_dir = os.path.join(output_base, "cache")
    os.makedirs(cache_dir, exist_ok=True)
    np.save(os.path.join(cache_dir, "moran_roi.npy"), roi_slice)
    np.save(os.path.join(cache_dir, "restrictive_roi.npy"), restrictive_roi_slice)
    np.save(os.path.join(cache_dir, "max_idx_epsilon.npy"), max_idx_epsilon)

    image_shape = outlines.shape[-2:]
    restrictive_roi_image = export_roi_to_image_space(
        restrictive_roi_slice,
        xy_starts_grid,
        xy_starts,
        overlap,
        boxsizes,
        image_shape,
    )
    moran_roi_image = export_roi_to_image_space(
        roi_slice,
        xy_starts_grid,
        xy_starts,
        overlap,
        boxsizes,
        image_shape,
    )
    np.save(os.path.join(cache_dir, "restrictive_roi_image.npy"), restrictive_roi_image)
    np.save(os.path.join(cache_dir, "moran_roi_image.npy"), moran_roi_image)

    from skimage.io import imsave

    imsave(
        os.path.join(cache_dir, "restrictive_roi_image.png"),
        (restrictive_roi_image * 255).astype(np.uint8),
    )
    imsave(
        os.path.join(cache_dir, "moran_roi_image.png"),
        (moran_roi_image * 255).astype(np.uint8),
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
    viewer_frame = max_idx_epsilon + time_offset

    points_data = POINTS_DATA_2.copy()
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
    logger.info("Deformation difference scene ready")
