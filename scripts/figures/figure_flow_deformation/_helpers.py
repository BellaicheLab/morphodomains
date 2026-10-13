"""Constants and utilities shared across figure_flow_deformation panels."""

import logging

import numpy as np

logger = logging.getLogger(__name__)

# Image dimensions (pixels)
IMAGE_SIZE = (2079, 3763)

# ── Timepoints and marker positions ────────────────────────────────────────

TIMEPOINTS = [
    {"frame": 46, "label": "a"},
    {"frame": 94, "label": "b"},
    {"frame": 142, "label": "c"},
]

POINTS_DATA_BY_FRAME = {
    46: np.array(
        [
            [46.0, 338.3431875, 694.47452728],
            [46.0, 627.31173114, 772.22681362],
            [46.0, 669.53530456, 1594.17904289],
            [46.0, 745.53773672, 1957.30177432],
            [46.0, 1173.69858154, 1394.56470198],
            [46.0, 1586.59015738, 1497.78759594],
            [46.0, 1750.23620878, 1991.2433817],
            [46.0, 1796.71208137, 1591.60001826],
        ]
    ),
    94: np.array(
        [
            [94.0, 341.48412323, 684.45504855],
            [94.0, 701.83660988, 794.5627528],
            [94.0, 731.86598376, 1605.35584777],
            [94.0, 1137.26253124, 1415.16981314],
            [94.0, 789.90928407, 1979.0794818],
            [94.0, 1452.82252682, 1504.19139095],
            [94.0, 1651.02998059, 1988.05076339],
            [94.0, 1581.07440867, 1629.52845731],
        ]
    ),
    142: np.array(
        [
            [142.0, 356.12878453, 673.1756363],
            [142.0, 722.15899849, 844.75229909],
            [142.0, 722.15899849, 1710.26124251],
            [142.0, 790.7896636, 2102.98115957],
            [142.0, 1084.37639771, 1473.86672933],
            [142.0, 1365.81485921, 1552.61097625],
            [142.0, 1518.26915145, 1984.94404378],
            [142.0, 1450.00603552, 1643.62846415],
        ]
    ),
}

POINTS_FACE_COLOR = np.array([[1.0, 1.0, 0.0, 1.0]] * 8)  # Yellow
POINTS_EDGE_COLOR = np.array([[0.0, 0.0, 0.0, 1.0]] * 8)  # Black

# ── Velocity (panel 2) scale bar settings ──────────────────────────────────

VELOCITY_SCALE_SETTINGS = {
    "length": 5,  # um.h^-1
    "scale_position": [1900, 2930],
    "bbox_x": [3250, 3350],
    "bbox_y_offset": 50,
    "color": "#FF732E",
    "quantity": "Velocity",
    "vector_style": "triangle",
    "scale": 20,
    "text_string": "{length} \u00b5m.h\u207b\u00b9",
}

# ── Deformation (panel 3) scale bar settings ───────────────────────────────

DEFORMATION_SCALE_SETTINGS = {
    "length": 0.05,  # h^-1
    "scale_position": [1900, 2930],
    "bbox_x": [3250, 3350],
    "bbox_y_offset": 50,
    "scale": 3000,
    "color": "blue",
    "quantity": "Deformation rate",
    "vector_style": "line",
    "text_string": "{length} h\u207b\u00b9",
}


def apply_spatial_masks(bundle, masks):
    """Clear grid boxes inside the config's circular movie-pixel ``masks`` from a bundle.

    The bundle origins are in the archetype frame, so reposition them to movie pixels with
    the bundle affine before the
    radius test, then drop the hit boxes from ``property_origin_mask`` — all frames, or
    the mask's ``frames`` window (viewer coords → array index = frame − affine offset).
    """
    from scipy.spatial.distance import cdist

    origin_mask = bundle["property_origin_mask"]
    if not masks or origin_mask is None:
        return bundle

    affine = np.asarray(bundle["affine_transform"])
    offset = int(round(affine[0, -1]))
    yx = bundle["property_origin"].reshape(-1, 2)
    movie_yx = (affine[1:3, 1:3] @ yx.T).T + affine[1:3, -1]  # archetype → movie pixels
    ny, nx = bundle["property_origin"].shape[:2]

    origin_mask = origin_mask.copy()
    for m in masks:
        hit = cdist(movie_yx, np.array(m["center"], float).reshape(1, 2)).ravel() <= float(m["radius"])
        if not hit.any():
            continue
        iy, ix = np.unravel_index(np.flatnonzero(hit), (ny, nx))
        frames = m.get("frames")
        t = slice(None) if not frames else slice(max(0, frames[0] - offset), frames[1] - offset + 1)
        origin_mask[t, :, iy, ix, :] = False
    return {**bundle, "property_origin_mask": origin_mask}


def add_marker_points(viewer):
    """Add all timepoint marker points and convert to shapes.

    Adds markers for all frames at once (time-aware 3D coordinates),
    then converts to Shapes so they export correctly in SVG.

    Parameters
    ----------
    viewer : napari.Viewer
        The napari viewer.
    """
    from pyanimalprocessing.plotting.napari_export import convert_points_to_shapes

    all_points = np.concatenate(
        [POINTS_DATA_BY_FRAME[tp["frame"]] for tp in TIMEPOINTS]
    )
    n_points = len(all_points)

    viewer.add_points(
        all_points,
        face_color=np.tile(POINTS_FACE_COLOR[0], (n_points, 1)),
        border_color=np.tile(POINTS_EDGE_COLOR[0], (n_points, 1)),
        size=50,
        name="points_data",
    )
    convert_points_to_shapes(viewer)


