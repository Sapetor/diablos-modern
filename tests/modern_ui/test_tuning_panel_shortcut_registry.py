"""
Focused guard for the "Toggle tuning panel" shortcut metadata.

The keyboard-shortcuts dialog and the command palette both surface the
``shortcut`` field from the registry tables in
``command_palette_manager._VIEW_COMMANDS`` as *display-only* metadata -- the
real key binding is the menu ``QAction`` shortcut built in
``modern_ui/builders/menu_builder.py``. When the two disagree the dialog shows
a blank (or wrong) key for a shortcut that is actually live.

This module pins the registry value against the live action on a real window,
plus a palette-build smoke test (per the project's Qt test recipe).

Run with:
    QT_QPA_PLATFORM=offscreen MPLBACKEND=Agg python -m pytest \
        tests/modern_ui/test_tuning_panel_shortcut_registry.py \
        -p no:cacheprovider -o addopts=""
"""

import pytest

from modern_ui.managers.command_palette_manager import palette_command_groups


@pytest.fixture(autouse=True)
def _qt(qapp):
    """Bind the shared session QApplication (from conftest) for every test."""
    return qapp


class TestRegistryMatchesMenuBinding:
    def test_toggle_tuning_panel_registry_equals_menu_binding(self, qapp):
        from modern_ui.main_window import ModernDiaBloSWindow

        window = ModernDiaBloSWindow()
        try:
            menu_key = window.tuning_panel_action.shortcut().toString()
        finally:
            window.close()
        registry_key = dict(palette_command_groups()["View"])["Toggle tuning panel"]

        assert menu_key == "Ctrl+Shift+T"  # guards the source of truth
        assert registry_key == menu_key  # registry mirrors it (no drift)
        assert registry_key != ""


class TestPaletteStillBuilds:
    """Smoke test (project Qt recipe): the block palette must still construct.

    Not a direct assertion on the registry data, but it exercises the same
    widget tree the central theme handler
    (``ModernBlockPalette._on_theme_changed``) fans out over, catching import
    or construction regressions adjacent to the edited manager module.
    """

    def test_modern_block_palette_constructs(self):
        from lib.lib import DSim
        from modern_ui.widgets.modern_palette import ModernBlockPalette

        d = DSim()
        getattr(d, "menu_blocks_init", lambda: None)()
        if not getattr(d, "menu_blocks", None):
            pytest.skip("menu_blocks empty -- no block library to build a palette")

        palette = ModernBlockPalette(d)
        try:
            # The central theme handler must remain a single callable on the
            # palette (findChildren fan-out), not a per-row connection.
            assert callable(getattr(palette, "_on_theme_changed", None))
        finally:
            palette.deleteLater()
