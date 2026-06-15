"""
Focused guard for the "Toggle tuning panel" shortcut metadata.

The keyboard-shortcuts dialog and the command palette both surface the
``shortcut`` field from the registry tables in
``command_palette_manager._VIEW_COMMANDS`` as *display-only* metadata -- the
real key binding is the menu ``QAction.setShortcut`` in
``modern_ui/builders/menu_builder.py``. When the two disagree the dialog shows
a blank (or wrong) key for a shortcut that is actually live.

This module pins the registry value against the live menu binding two ways:

  * a pure-logic check that parses the ``setShortcut(...)`` string out of
    ``menu_builder.py`` and compares it to the registry entry (no Qt needed);
  * a palette-build smoke test (per the project's Qt test recipe) confirming
    ``ModernBlockPalette`` constructs against a real ``DSim`` so the widgets the
    central theme handler fans out over still exist.

Run with:
    QT_QPA_PLATFORM=offscreen MPLBACKEND=Agg python -m pytest \
        tests/modern_ui/test_tuning_panel_shortcut_registry.py \
        -p no:cacheprovider -o addopts=""
"""

from __future__ import annotations  # PEP 604 `str | None` below must not be evaluated on Python 3.9 (CI)

import re
from pathlib import Path

import pytest

from modern_ui.managers.command_palette_manager import palette_command_groups

# menu_builder.py is the source of the *real* binding (QAction.setShortcut).
_MENU_BUILDER = (
    Path(__file__).resolve().parents[2]
    / "modern_ui" / "builders" / "menu_builder.py"
)


def menu_shortcut_for(label_fragment: str, source: str) -> str | None:
    """Return the ``setShortcut("...")`` key bound near a label fragment.

    Pure helper (no Qt): scans ``source`` for the menu action whose visible
    text contains ``label_fragment`` and returns the literal passed to the
    following ``setShortcut(...)`` call, or ``None`` if either is absent. This
    mirrors how ``menu_builder`` binds keys so the registry's display metadata
    can be checked against the live binding without constructing the window.
    """
    # Locate the addAction line carrying the human label, then the nearest
    # setShortcut("...") after it (menu_builder binds the two together).
    label_idx = source.find(label_fragment)
    if label_idx == -1:
        return None
    tail = source[label_idx:]
    match = re.search(r'setShortcut\(\s*["\']([^"\']+)["\']\s*\)', tail)
    return match.group(1) if match else None


@pytest.fixture(autouse=True)
def _qt(qapp):
    """Bind the shared session QApplication (from conftest) for every test."""
    return qapp


class TestMenuShortcutHelper:
    def test_helper_extracts_following_shortcut(self):
        sample = (
            'action = view_menu.addAction("Parameter &Tuning Panel\\tCtrl+Shift+T", cb)\n'
            '             action.setShortcut("Ctrl+Shift+T")\n'
        )
        assert menu_shortcut_for("Tuning Panel", sample) == "Ctrl+Shift+T"

    def test_helper_returns_none_when_label_absent(self):
        sample = 'action.setShortcut("Ctrl+Shift+T")\n'
        assert menu_shortcut_for("Nonexistent Label", sample) is None

    def test_helper_returns_none_when_no_shortcut_follows(self):
        sample = 'view_menu.addAction("Parameter &Tuning Panel", cb)\n'
        assert menu_shortcut_for("Tuning Panel", sample) is None


class TestRegistryMatchesMenuBinding:
    def test_toggle_tuning_panel_registry_equals_menu_binding(self):
        # The registry's display metadata must equal the live menu binding so
        # the shortcuts dialog and command palette show the real key.
        registry_view = dict(palette_command_groups()["View"])
        registry_key = registry_view["Toggle tuning panel"]

        source = _MENU_BUILDER.read_text(encoding="utf-8")
        menu_key = menu_shortcut_for("Tuning Panel", source)

        assert menu_key == "Ctrl+Shift+T"      # guards the source of truth
        assert registry_key == menu_key        # registry mirrors it (no drift)

    def test_no_view_command_carries_a_blank_for_a_bound_menu_key(self):
        # Defensive: any View command that menu_builder binds a shortcut for
        # must not be exposed with an empty (blank) display key in the registry.
        source = _MENU_BUILDER.read_text(encoding="utf-8")
        registry_view = dict(palette_command_groups()["View"])
        # Only assert on the command this task corrects; other View labels do
        # not all map 1:1 to menu_builder label fragments.
        key = registry_view["Toggle tuning panel"]
        assert key == menu_shortcut_for("Tuning Panel", source)
        assert key != ""


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
