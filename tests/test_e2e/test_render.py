"""Tier 2: Full rendering tests for the figure generation framework.

Dispatches each figure through the framework and verifies output files.
Takes minutes to hours depending on caches and number of figures.

Run:
    pytest tests/test_e2e/test_render.py -v --tb=short
    pytest tests/test_e2e/test_render.py -v -k "figure_apoptosis_compression"       # single figure
    pytest tests/test_e2e/test_render.py -v -m "not napari"     # skip napari
"""

import os

import pytest

from scripts.figures import list_registered_figures
from scripts.figures.generate import (
    _apply_no_movies,
    _apply_save_all,
    dispatch_figure,
)

# The registry is ordered by dependency: independent figures first
FIGURE_ORDER = list_registered_figures()


@pytest.fixture(scope="session")
def test_config(config):
    """Config with save-all and no-movies overrides, output to test dir."""
    import copy

    cfg = copy.deepcopy(config)
    _apply_save_all(cfg)
    _apply_no_movies(cfg)
    # Redirect output to test directory
    base = cfg["paths"].get("base_workspace", "")
    cfg["paths"]["output_base"] = os.path.join(base, "figures_raw_test")
    return cfg


@pytest.fixture(scope="session")
def test_output_base(test_config):
    """Return the test output base directory."""
    return test_config["paths"]["output_base"]


@pytest.mark.parametrize("figure_name", FIGURE_ORDER)
def test_figure_dispatch(figure_name, test_config, test_output_base):
    """dispatch_figure() should complete without errors for each figure."""
    fig_cfg = test_config["figures"].get(figure_name, {})
    output_dir = fig_cfg.get("output_dir", figure_name)
    expected_output = os.path.join(test_output_base, output_dir)

    dispatch_figure(figure_name, fig_cfg, test_config)

    # Verify output directory was created
    assert os.path.isdir(expected_output), (
        f"Output directory not created: {expected_output}"
    )

    # Collect produced files
    produced = []
    for root, dirs, files in os.walk(expected_output):
        for f in files:
            fpath = os.path.join(root, f)
            size = os.path.getsize(fpath)
            rel = os.path.relpath(fpath, expected_output)
            produced.append((rel, size))

    print(f"\n  {figure_name}: {len(produced)} files in {expected_output}")
    for rel, size in sorted(produced):
        print(f"    {rel} ({size:,} bytes)")
