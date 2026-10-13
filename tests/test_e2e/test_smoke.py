"""Tier 0: Smoke tests for the figure generation framework.

Tests config loading, module imports, and YAML/code consistency.
No data access — should complete in seconds.

Run:
    pytest tests/test_e2e/test_smoke.py -v
"""

import pytest

from scripts.figures import list_registered_figures


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------
class TestConfigLoading:

    def test_config_has_required_sections(self, config):
        """Config should have paths, figures, enabled_figures, style, napari."""
        for key in ("paths", "figures", "enabled_figures", "style", "napari"):
            assert key in config, f"Config missing top-level key: {key}"

    def test_paths_resolved(self, config):
        """No unresolved ${...} references should remain in paths."""
        for key, value in config["paths"].items():
            if isinstance(value, str):
                assert "${" not in value, (
                    f"Unresolved path reference in paths.{key}: {value}"
                )

    def test_paths_uppercase(self, paths):
        """get_paths() should return uppercase keys."""
        for key in paths:
            assert key == key.upper(), f"Path key not uppercase: {key}"


# ---------------------------------------------------------------------------
# Module registry
# ---------------------------------------------------------------------------
class TestModuleRegistry:

    EXPECTED_FIGURES = list_registered_figures()

    def test_every_figure_has_config(self, config):
        """Every registered figure module should have a YAML config block and an enabled flag."""
        for name in self.EXPECTED_FIGURES:
            assert name in config["figures"], f"Figure missing from figures: {name}"
            assert name in config["enabled_figures"], f"Figure missing from enabled_figures: {name}"

    @pytest.mark.parametrize("figure_name", EXPECTED_FIGURES)
    def test_module_importable(self, figure_name):
        """Each registered figure module should import and have the required API."""
        from scripts.figures import get_figure_module

        mod = get_figure_module(figure_name)
        assert hasattr(mod, "PANELS"), f"{figure_name} missing PANELS dict"
        assert hasattr(mod, "SCENES"), f"{figure_name} missing SCENES dict"
        assert hasattr(mod, "preprocess"), f"{figure_name} missing preprocess()"
        assert callable(mod.preprocess), f"{figure_name}.preprocess is not callable"
        assert isinstance(mod.PANELS, dict), f"{figure_name}.PANELS is not a dict"
        assert isinstance(mod.SCENES, dict), f"{figure_name}.SCENES is not a dict"


# ---------------------------------------------------------------------------
# YAML / Code consistency
# ---------------------------------------------------------------------------
class TestYAMLCodeConsistency:

    def test_every_panel_has_matching_function(self, config):
        """Every panel in YAML should have a matching function in PANELS or SCENES."""
        from scripts.figures import get_figure_module, list_registered_figures

        registered = set(list_registered_figures())
        figures_cfg = config.get("figures", {})

        missing = []
        for fig_name, fig_cfg in figures_cfg.items():
            if fig_name not in registered:
                continue
            mod = get_figure_module(fig_name)
            all_funcs = {**mod.PANELS, **mod.SCENES}

            for panel_name, panel_cfg in fig_cfg.get("panels", {}).items():
                scene_name = panel_cfg.get("scene", panel_name)
                if scene_name not in all_funcs:
                    missing.append(f"{fig_name}/{panel_name} (scene={scene_name})")

        assert not missing, (
            f"Panels in YAML with no matching function:\n"
            + "\n".join(f"  - {m}" for m in missing)
        )

    def test_every_function_has_yaml_panel(self, config):
        """Every function in PANELS/SCENES should have a YAML panel entry."""
        from scripts.figures import get_figure_module, list_registered_figures

        registered = set(list_registered_figures())
        figures_cfg = config.get("figures", {})

        orphans = []
        for fig_name, fig_cfg in figures_cfg.items():
            if fig_name not in registered:
                continue
            mod = get_figure_module(fig_name)
            all_funcs = {**mod.PANELS, **mod.SCENES}

            # Collect all scene names referenced in YAML
            yaml_scenes = set()
            for panel_cfg in fig_cfg.get("panels", {}).values():
                yaml_scenes.add(panel_cfg.get("scene", ""))
            yaml_scenes |= set(fig_cfg.get("panels", {}).keys())

            for func_name in all_funcs:
                if func_name not in yaml_scenes:
                    orphans.append(f"{fig_name}/{func_name}")

        assert not orphans, (
            f"Functions with no YAML panel entry:\n"
            + "\n".join(f"  - {o}" for o in orphans)
        )
