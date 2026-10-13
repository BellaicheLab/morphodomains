"""Tier 1: Preprocessing tests for the figure generation framework.

Runs preprocess() for each figure and validates shared_data keys.
Uses existing caches where available. Takes minutes.

Run:
    pytest tests/test_e2e/test_preprocess.py -v --tb=short
"""

import pytest

from scripts.figures import get_figure_module, list_registered_figures
from scripts.figures.config import get_paths

# The registry is ordered by dependency: independent figures first
FIGURE_ORDER = list_registered_figures()


@pytest.mark.parametrize("figure_name", FIGURE_ORDER)
def test_preprocess_returns_dict(figure_name, config):
    """preprocess() should return a non-empty dict for each figure."""
    mod = get_figure_module(figure_name)
    paths = get_paths(config)
    fig_cfg = config["figures"].get(figure_name, {})

    shared_data = mod.preprocess(paths, fig_cfg)

    assert isinstance(shared_data, dict), (
        f"{figure_name}.preprocess() should return a dict, got {type(shared_data)}"
    )
    assert len(shared_data) > 0, (
        f"{figure_name}.preprocess() returned an empty dict"
    )

    # Log keys for debugging
    keys = sorted(k for k in shared_data if not k.startswith("_"))
    print(f"\n  {figure_name} shared_data ({len(keys)} keys): {', '.join(keys)}")
