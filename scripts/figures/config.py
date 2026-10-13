"""Configuration loader for the figure generation framework.

Reads figure_config.yaml, resolves ${variable} references in paths,
and provides a PATHS dict with uppercase keys.
"""

import os
from pathlib import Path

import yaml

_SCRIPT_DIR = Path(__file__).parent
_DEFAULT_CONFIG_PATH = _SCRIPT_DIR / "figure_config.yaml"


def load_config(config_path=None, base_dir=None):
    """Load and resolve the figure configuration.

    Parameters
    ----------
    config_path : str or Path, optional
        Path to YAML config file. Defaults to figure_config.yaml
        in the same directory as this module.
    base_dir : str or Path, optional
        Override for ``base_workspace`` when the YAML value is
        ``"__auto__"``.  Typically the Zenodo archive root (contains data/).

    Returns
    -------
    dict
        Parsed config with resolved paths.
    """
    path = Path(config_path) if config_path else _DEFAULT_CONFIG_PATH
    with open(path) as f:
        config = yaml.safe_load(f)
    if config.get("paths", {}).get("base_workspace") == "__auto__":
        if base_dir:
            config["paths"]["base_workspace"] = str(base_dir)
        elif os.environ.get("MORPHODOMAINS_DATA"):
            config["paths"]["base_workspace"] = os.environ["MORPHODOMAINS_DATA"]
    config["paths"] = _resolve_paths(config.get("paths", {}))
    return config


def _resolve_paths(paths_dict):
    """Expand ${variable} references in path values.

    Two-pass resolution: first collect literal values (no ${...}),
    then expand references in remaining values.
    """
    resolved = {}
    deferred = {}

    # Pass 1: collect literal values
    for key, value in paths_dict.items():
        if isinstance(value, str) and "${" in value:
            deferred[key] = value
        else:
            resolved[key] = value

    # Pass 2: expand references (may need multiple iterations for chained refs)
    max_iterations = 5
    for _ in range(max_iterations):
        still_deferred = {}
        for key, value in deferred.items():
            expanded = value
            for ref_key, ref_value in resolved.items():
                expanded = expanded.replace(f"${{{ref_key}}}", str(ref_value))
            if "${" in expanded:
                still_deferred[key] = expanded
            else:
                resolved[key] = expanded
        deferred = still_deferred
        if not deferred:
            break

    # Any remaining unresolved refs are kept as-is
    resolved.update(deferred)
    return resolved


def get_paths(config=None):
    """Return the PATHS dict with uppercase keys.

    Parameters
    ----------
    config : dict, optional
        Pre-loaded config. If None, loads from default config file.

    Returns
    -------
    dict
        Mapping of uppercase path names to resolved path strings.
    """
    if config is None:
        config = load_config()
    paths = config["paths"]

    # String values only (skip nested dicts), uppercase keys
    return {k.upper(): v for k, v in paths.items() if isinstance(v, str)}


_AOA_MOTIONS = ("EULERIAN", "LAGRANGIAN_TRACKING")


def aoa_backup(store, field):
    """Return ``store[<MOTION>]["D2"][field]`` of an across-animal-average store.

    The MOTION group is matched by name so co-located raw ``.mat`` source subfolders
    (e.g. ``raw_mat/``) are ignored.
    """
    motion = next(k for k in store.group_keys() if k in _AOA_MOTIONS)
    return store[motion]["D2"][field]
