#!/usr/bin/env python
"""Figure generation dispatcher.

Reads figure_config.yaml and runs enabled figures/panels via a registry.

Usage:
    python scripts/figures/generate.py                          # Run all enabled
    python scripts/figures/generate.py --figure grn             # Run figure_grn only
    python scripts/figures/generate.py --figure grn --panel tf_specificity_heatmap_5
    python scripts/figures/generate.py --list                   # List all panels
    python scripts/figures/generate.py --config path/to/config.yaml
    python scripts/figures/generate.py --figure flow_deformation --display-all  # Keep viewers open

Warning: ``--display-all`` forces every panel to ``display: true``, so
napari viewers stay open after export and matplotlib figures are not
closed. On a full run this opens one napari window per napari panel
(~20+ windows, memory-heavy) and keeps all matplotlib figures in
memory. Intended for interactive inspection of a single figure or
panel, not for batch runs.
"""

import argparse
import gc
import logging
import os
import subprocess
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from tqdm import tqdm

# _REPO_ROOT = morphodomains/ (parents[2] of scripts/figures/generate.py)
# _DATA_ROOT = directory containing data/ (MORPHODOMAINS_DATA env var; defaults to the
#              Zenodo archive root, two levels above the repo).
_REPO_ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MORPHODOMAINS_DATA", str(_REPO_ROOT.parents[1]))
_DATA_ROOT = Path(os.environ["MORPHODOMAINS_DATA"])
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.figures import (
    apply_style,
    create_viewer,
    get_figure_module,
    is_panel_enabled,
    save_and_close,
)
from scripts.figures.config import get_paths, load_config

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate paper figures")
    parser.add_argument(
        "--figure",
        type=str,
        default=None,
        help="Run a specific figure (e.g. 'grn' or 'figure_grn')",
    )
    parser.add_argument(
        "--panel",
        type=str,
        action="append",
        default=None,
        help="Run specific panel(s) within the selected figure (repeatable)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all registered figures and their panels",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to figure_config.yaml (default: scripts/figures/figure_config.yaml)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load config, import modules, run preprocess() only (no rendering)",
    )
    parser.add_argument(
        "--save-all",
        action="store_true",
        help="Force save=true and enabled=true on all panels and exports",
    )
    parser.add_argument(
        "--no-movies",
        action="store_true",
        help="Skip multi-frame exports (movies); only render single-frame SVGs/PNGs",
    )
    parser.add_argument(
        "--regenerate-cache",
        action="store_true",
        help="Force cache regeneration for all figures in this run (overrides YAML)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Redirect output_base to a separate directory (e.g. figures_raw_test)",
    )
    parser.add_argument(
        "--display-all",
        action="store_true",
        help="Force display=true on all panels (napari viewers stay open, "
             "matplotlib figures not closed). Heavy — intended for "
             "interactive inspection of a single figure, not full runs.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Show detailed logging and library output (default: quiet with progress bars only)",
    )
    return parser.parse_args()


def normalize_figure_name(name):
    """Normalize 'grn' -> 'figure_grn'; 'figure_grn' is kept."""
    if name is None:
        return None
    if not name.startswith("figure_"):
        return f"figure_{name}"
    return name


# ---------------------------------------------------------------------------
# Config overrides for testing
# ---------------------------------------------------------------------------


def _apply_save_all(config):
    """Force save=true and enabled=true on all panels and exports."""
    # Enable all figures
    for fig_name in config.get("enabled_figures", {}):
        config["enabled_figures"][fig_name] = True

    for fig_name, fig_cfg in config.get("figures", {}).items():
        # Enable all panels
        if "enabled_panels" in fig_cfg:
            for panel_name in fig_cfg["enabled_panels"]:
                fig_cfg["enabled_panels"][panel_name] = True
        for panel_name, panel_cfg in fig_cfg.get("panels", {}).items():
            panel_cfg["save"] = True
            panel_cfg["display"] = False
            # Enable all exports
            for export_name, export_cfg in panel_cfg.get("exports", {}).items():
                if isinstance(export_cfg, dict):
                    export_cfg["enabled"] = True


def _apply_no_movies(config):
    """Disable multi-frame exports (movies) — keep only single-frame SVGs/PNGs."""
    for fig_cfg in config.get("figures", {}).values():
        for panel_cfg in fig_cfg.get("panels", {}).values():
            for export_name, export_cfg in list(panel_cfg.get("exports", {}).items()):
                if not isinstance(export_cfg, dict):
                    continue
                frames = export_cfg.get("frames")
                # Skip exports with frame ranges (movies)
                if isinstance(frames, list) and len(frames) > 1:
                    export_cfg["enabled"] = False
                elif isinstance(frames, str) and ":" in frames:
                    export_cfg["enabled"] = False


def _apply_output_dir(config, output_dir):
    """Redirect output_base to a separate directory."""
    base = config["paths"].get("base_workspace", "")
    config["paths"]["output_base"] = str(Path(base) / output_dir)


def _apply_regenerate_cache(config):
    """Force cache regeneration for all figures (top-level and nested)."""
    for fig_cfg in config.get("figures", {}).values():
        fig_cfg["regenerate_cache"] = True
        # figure_morphodynamics_comparison uses a nested `cache:` section instead of the top-level key
        cache_section = fig_cfg.get("cache")
        if isinstance(cache_section, dict):
            cache_section["regenerate_cache"] = True


def _apply_display_all(config):
    """Force display=true on every panel.

    Napari viewers are not closed after export (GUI windows remain on
    screen) and matplotlib figures are not closed after saving. Heavy
    on a full run (one viewer per napari panel); intended for
    interactive inspection of a single figure or panel.
    """
    for fig_cfg in config.get("figures", {}).values():
        for panel_cfg in fig_cfg.get("panels", {}).values():
            panel_cfg["display"] = True


# ---------------------------------------------------------------------------
# Frame specification parsing
# ---------------------------------------------------------------------------


def _parse_frames(value):
    """Parse YAML frames specification.

    Supports: None, int, list[int], "all", "start:stop", "start:stop:step".
    """
    if value is None or value == "all":
        return None  # save_viewer_state interprets None as all frames
    if isinstance(value, int):
        return value
    if isinstance(value, list):
        return value
    if isinstance(value, str) and ":" in value:
        parts = value.split(":")
        return list(range(*[int(p) for p in parts]))
    return value


# ---------------------------------------------------------------------------
# Panel listing
# ---------------------------------------------------------------------------


def list_all_panels(config):
    """Print a summary of all figures and panels."""
    figures_cfg = config.get("figures", {})
    enabled_figures = config.get("enabled_figures", {})

    for fig_name, fig_cfg in figures_cfg.items():
        status = ["enabled" if enabled_figures.get(fig_name, False) else "disabled"]

        desc = fig_cfg.get("description", "")
        print(f"\n{'=' * 60}")
        print(f"  {fig_name} [{', '.join(status)}]")
        if desc:
            print(f"  {desc}")
        print(f"{'=' * 60}")

        panels = fig_cfg.get("panels", {})
        for panel_name, panel_cfg in panels.items():
            p_enabled = is_panel_enabled(fig_cfg, panel_name)
            kind = panel_cfg.get("kind", "matplotlib")
            p_desc = panel_cfg.get("description", "")
            marker = "+" if p_enabled else "-"

            # Show scene and export count for napari panels
            if kind == "napari":
                scene_name = panel_cfg.get("scene", panel_name)
                exports = panel_cfg.get("exports", {})
                n_exports = len(exports)
                extra = f" scene={scene_name} ({n_exports} exports)"
            else:
                exports = panel_cfg.get("exports", {})
                extra = f" ({len(exports)} exports)" if exports else ""

            print(f"  {marker} {panel_name:<30s} [{kind}]{extra} {p_desc}")


# ---------------------------------------------------------------------------
# Dispatchers
# ---------------------------------------------------------------------------


def _dispatch_napari_panel(
    panel_name, panel_cfg, scenes, shared_data, output_base, config
):
    """Dispatch a single napari panel: build scene, then run YAML exports."""
    scene_name = panel_cfg.get("scene", panel_name)
    scene_func = scenes.get(scene_name)
    if not scene_func:
        logger.warning("  Scene '%s' not found, skipping", scene_name)
        return

    from pyanimalprocessing.plotting.napari_export import save_viewer_state

    viewer = create_viewer(config)
    try:
        logger.info("  Building scene: %s", scene_name)
        scene_func(viewer, shared_data)

        save = panel_cfg.get("save", True)
        # Remember the frame the scene builder intended so we can
        # restore it after multi-frame exports (movies) which leave
        # the viewer at their last frame.
        scene_frame = viewer.dims.current_step[0]
        for export_name, export_cfg in panel_cfg.get("exports", {}).items():
            if not save:
                break
            if not export_cfg.get("enabled", True):
                continue
            path = os.path.join(str(output_base), export_cfg["path"])
            frames = _parse_frames(export_cfg.get("frames"))
            layers = export_cfg.get("layers")
            # Restore the scene frame before each export so that
            # single-frame exports always use the scene builder's
            # intended position, not a stale position from a
            # previous multi-frame export.
            if frames is None or isinstance(frames, int):
                viewer.dims.set_point(0, scene_frame)
            logger.info(
                "    Exporting %s → %s (viewer frame=%d)",
                export_name,
                export_cfg["path"],
                viewer.dims.current_step[0],
            )
            save_viewer_state(
                viewer,
                path,
                frames=frames,
                layers=layers,
                fps=export_cfg.get("fps", 10),
                crf=export_cfg.get("crf", 18),
                scale_factor=export_cfg.get("scale_factor", 1),
                interleave_tensors=export_cfg.get(
                    "interleave_tensors",
                    panel_cfg.get("interleave_tensors", True),
                ),
            )
    finally:
        if not panel_cfg.get("display", False):
            viewer.close()
            gc.collect()


def _dispatch_matplotlib_panel(panel_name, panel_cfg, panels, shared_data, output_base):
    """Dispatch a single matplotlib panel: run scene, save exports."""
    scene_name = panel_cfg.get("scene", panel_name)
    scene_func = panels.get(scene_name)
    if not scene_func:
        logger.warning("  Scene '%s' not found, skipping", scene_name)
        return

    display = panel_cfg.get("display", False)
    save = panel_cfg.get("save", True)

    logger.info("  Running %s...", scene_name)
    result = scene_func(shared_data, str(output_base))

    if result is None:
        return  # Self-managed panel

    exports = panel_cfg.get("exports", {})
    if not exports:
        logger.warning(
            "  Panel '%s' returned a figure but has no exports defined in YAML",
            scene_name,
        )
        return

    # Normalize result to dict[str, Figure]
    if isinstance(result, Figure):
        export_name = next(iter(exports))
        figures = {export_name: result}
    elif isinstance(result, dict):
        figures = result
    else:
        return

    for export_name, export_cfg in exports.items():
        fig = figures.get(export_name)
        if fig is None:
            logger.warning("    Export '%s' has no matching figure", export_name)
            continue
        path = os.path.join(str(output_base), export_cfg["path"])
        if save:
            save_and_close(fig, path, close=not display)
        elif not display:
            plt.close(fig)


def dispatch_figure(fig_name, fig_cfg, config, dry_run=False):
    """Run all enabled panels for a single figure.

    Parameters
    ----------
    fig_name : str
        e.g. "figure_grn"
    fig_cfg : dict
        The figure's config section.
    config : dict
        Full config (for paths, style, etc.).
    dry_run : bool
        If True, run preprocessing only and log shared_data keys.
    """
    fig_module = get_figure_module(fig_name)
    paths = get_paths(config)
    output_base = Path(paths["OUTPUT_BASE"]) / fig_cfg.get("output_dir", fig_name)
    output_base.mkdir(parents=True, exist_ok=True)

    # Build unified scene registry from module
    panels_cfg = fig_cfg.get("panels", {})
    all_scenes = {
        **getattr(fig_module, "PANELS", {}),
        **getattr(fig_module, "SCENES", {}),
    }

    # Collect enabled panels
    enabled_panels = [
        (name, cfg)
        for name, cfg in panels_cfg.items()
        if is_panel_enabled(fig_cfg, name)
    ]

    # Progress bar: preprocessing (1 step) + each enabled panel
    n_steps = 1 + (len(enabled_panels) if not dry_run else 0)
    pbar = tqdm(
        total=n_steps,
        desc=f"  {fig_name}",
        unit="step",
        leave=False,
    )

    # Run preprocessing
    pbar.set_postfix_str("preprocessing")
    logger.info("Running preprocessing for %s...", fig_name)
    shared_data = fig_module.preprocess(paths, fig_cfg)
    shared_data["_output_base"] = str(output_base)
    shared_data["_paths"] = paths
    shared_data["_fig_config"] = fig_cfg
    pbar.update(1)

    if dry_run:
        logger.info(
            "  [dry-run] shared_data keys (%d): %s",
            len(shared_data),
            ", ".join(sorted(k for k in shared_data if not k.startswith("_"))),
        )
        shared_data.clear()
        pbar.close()
        return

    for panel_name, panel_cfg in enabled_panels:
        pbar.set_postfix_str(panel_name)

        kind = panel_cfg.get("kind", "matplotlib")

        if kind == "napari":
            _dispatch_napari_panel(
                panel_name,
                panel_cfg,
                all_scenes,
                shared_data,
                output_base,
                config,
            )
        else:
            _dispatch_matplotlib_panel(
                panel_name,
                panel_cfg,
                all_scenes,
                shared_data,
                output_base,
            )
        pbar.update(1)

    pbar.close()

    # Release shared_data after all panels for this figure are done
    shared_data.clear()


def main():
    args = parse_args()

    # ── Logging & noise suppression ──────────────────────────────────────
    log_level = logging.DEBUG if args.verbose else logging.WARNING
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    # Always suppress noisy third-party loggers
    logging.getLogger("numexpr").setLevel(logging.WARNING)
    logging.getLogger("joblib").setLevel(logging.WARNING)
    plt.set_loglevel("WARNING")
    if not args.verbose:
        warnings.filterwarnings("ignore")

    config = load_config(args.config, base_dir=_DATA_ROOT)

    # Apply config overrides for testing
    if args.save_all:
        _apply_save_all(config)
    if args.no_movies:
        _apply_no_movies(config)
    if args.regenerate_cache:
        _apply_regenerate_cache(config)
    if args.output_dir:
        _apply_output_dir(config, args.output_dir)
    # Apply --display-all last so it overrides the display=false that
    # --save-all forces.
    if args.display_all:
        _apply_display_all(config)

    # Apply style
    style_cfg = config.get("style", {})
    apply_style(style_cfg)

    # List mode
    if args.list:
        list_all_panels(config)
        return

    # Determine which figures to run
    target_figure = normalize_figure_name(args.figure)
    target_panel = args.panel

    figures_cfg = config.get("figures", {})
    enabled_overrides = config.get("enabled_figures", {})
    any_mpl_display = False
    any_napari_display = False

    # Collect figures to run
    figures_to_run = []
    for fig_name, fig_cfg in figures_cfg.items():
        # Filter by --figure
        if target_figure and fig_name != target_figure:
            continue

        # Check enabled
        if target_figure != fig_name and not enabled_overrides.get(fig_name, False):
            continue

        # Filter panels if --panel specified
        if target_panel:
            panels = fig_cfg.get("panels", {})
            target_set = set(target_panel)
            missing = target_set - set(panels)
            if missing:
                tqdm.write(f"ERROR: Panel(s) {missing} not found in {fig_name}")
            found = target_set & set(panels)
            if not found:
                continue
            fig_cfg["enabled_panels"] = {pname: (pname in found) for pname in panels}

        figures_to_run.append((fig_name, fig_cfg))

    # ── Run figures ──────────────────────────────────────────────────────
    if target_figure and figures_to_run:
        # Single figure: run in-process (fast, interactive)
        fig_name, fig_cfg = figures_to_run[0]
        dispatch_figure(fig_name, fig_cfg, config, dry_run=args.dry_run)
    elif figures_to_run:
        # Multiple figures: subprocess per figure for memory isolation
        fig_pbar = tqdm(
            figures_to_run,
            desc="Figures",
            unit="fig",
            leave=True,
        )
        for fig_name, fig_cfg in fig_pbar:
            fig_pbar.set_postfix_str(fig_name)
            cmd = [sys.executable, __file__, "--figure", fig_name]
            for flag in (
                "--dry-run",
                "--save-all",
                "--no-movies",
                "--regenerate-cache",
                "--verbose",
            ):
                if getattr(args, flag.lstrip("-").replace("-", "_")):
                    cmd.append(flag)
            if args.output_dir:
                cmd += ["--output-dir", args.output_dir]
            if args.config:
                cmd += ["--config", args.config]
            result = subprocess.run(cmd, cwd=str(_REPO_ROOT))
            if result.returncode != 0:
                tqdm.write(f"ERROR: {fig_name} failed (exit {result.returncode})")
        fig_pbar.close()

    # Check if any panels have display=true
    for fig_name, fig_cfg in figures_to_run:
        for pname, panel_cfg in fig_cfg.get("panels", {}).items():
            if is_panel_enabled(fig_cfg, pname) and panel_cfg.get("display", False):
                kind_str = panel_cfg.get("kind", "matplotlib")
                if kind_str == "napari":
                    any_napari_display = True
                else:
                    any_mpl_display = True

    if not args.dry_run:
        if any_mpl_display:
            plt.show()
        if any_napari_display:
            import napari

            napari.run()


if __name__ == "__main__":
    main()
