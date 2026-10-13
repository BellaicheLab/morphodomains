"""Fixtures for E2E figure generation tests."""

import os
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pytest

# Ensure the repo root (morphodomains/) is importable
_REPO_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# Set data root via env var before any framework imports
os.environ.setdefault(
    "MORPHODOMAINS_DATA",
    str(Path(__file__).resolve().parents[4]),   # Zenodo archive root (contains data/)
)

# Headless matplotlib backend — must be set before any figure import
matplotlib.use("Agg")


# ---------------------------------------------------------------------------
# Config fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def config():
    """Load the figure config with resolved paths."""
    from pathlib import Path

    from scripts.figures.config import load_config

    data_root = Path(os.environ["MORPHODOMAINS_DATA"])
    return load_config(base_dir=data_root)


@pytest.fixture(scope="session")
def paths(config):
    """Return the uppercase PATHS dict."""
    from scripts.figures.config import get_paths

    return get_paths(config)


# ---------------------------------------------------------------------------
# Prevent blocking
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def mock_plt_show(monkeypatch):
    """Prevent plt.show() from blocking during tests."""
    monkeypatch.setattr(plt, "show", lambda *a, **kw: None)


@pytest.fixture(autouse=True)
def close_figures():
    """Close all matplotlib figures after each test."""
    yield
    plt.close("all")


# ---------------------------------------------------------------------------
# Temporary output directory
# ---------------------------------------------------------------------------
@pytest.fixture
def figure_output_dir(tmp_path):
    """Return a temporary directory for figure outputs."""
    d = tmp_path / "figure_output"
    d.mkdir()
    return d
