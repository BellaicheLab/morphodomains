"""Constants and utilities for figure_tollo_gfp."""

# ROI contour rendering styles (napari shapes layer kwargs)
# Cell-based (original) contours — thin, dashed-look
ROI_STYLES = {
    "flow": {
        "edge_color": "#FF0000",
        "edge_width": 2,
        "face_color": "transparent",
    },
    "extended_compression": {
        "edge_color": "#33CC33",
        "edge_width": 2,
        "face_color": "transparent",
    },
    "restrictive_compression": {
        "edge_color": "#FFA500",
        "edge_width": 2,
        "face_color": "transparent",
    },
}

# Grid-snapped contours — thick, prominent
ROI_GRID_STYLES = {
    "flow": {
        "edge_color": "#FF0000",
        "edge_width": 5,
        "face_color": "transparent",
    },
    "extended_compression": {
        "edge_color": "#33CC33",
        "edge_width": 5,
        "face_color": "transparent",
    },
    "restrictive_compression": {
        "edge_color": "#FFA500",
        "edge_width": 5,
        "face_color": "transparent",
    },
}

# Timepoint definitions: key -> (time string, display label)
TIMEPOINTS = {
    "15h": ("15h00", "15h APF"),
    "20h": ("20h00", "20h APF"),
    "25h": ("25h00", "25h APF"),
    "28h": ("28h00", "28h APF"),
}

# Per-timepoint contrast limits: {tp_key: {"TolloGFP": (min, max), "ECad": (min, max)}}
# 15h uses napari defaults (0-255).
CONTRAST_LIMITS = {
    "20h": {"TolloGFP": (0, 255), "ECad": (0, 255)},
    "25h": {"TolloGFP": (0, 255), "ECad": (0, 255)},
    "28h": {"TolloGFP": (0, 255), "ECad": (0, 255)},
}
