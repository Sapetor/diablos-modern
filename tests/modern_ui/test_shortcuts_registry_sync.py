"""
Drift guard: the keyboard-shortcuts dialog must source its Simulation/View
groups (and the bulk of File) from the command registry so the two cannot
diverge. ``palette_command_groups`` in command_palette_manager is the single
source of truth; ``shortcuts_dialog`` consumes it via ``build_shortcut_groups``.

These tests fail if a future edit re-hardcodes a label/shortcut in the dialog
(the historical "Toggle tuning panel = Ctrl+Shift+T" drift) instead of reading
the registry.

Run with:
    QT_QPA_PLATFORM=offscreen MPLBACKEND=Agg python -m pytest \
        tests/modern_ui/test_shortcuts_registry_sync.py -p no:cacheprovider \
        -o addopts=""
"""

import pytest

from PyQt5.QtWidgets import QLabel

from modern_ui.managers.command_palette_manager import palette_command_groups
from modern_ui.widgets.shortcuts_dialog import (
    KeyboardShortcutsDialog,
    SHORTCUT_GROUPS,
    build_shortcut_groups,
    _FILE_SUPPLEMENT,
)


@pytest.fixture(autouse=True)
def _qt(qapp):
    """Bind the shared session QApplication (from conftest) for every test."""
    return qapp


def _groups(catalogue):
    """Index a [(title, entries)] catalogue by title."""
    return {title: entries for title, entries in catalogue}


class TestRegistryAccessor:
    def test_groups_ordered_sim_view_file(self):
        assert list(palette_command_groups().keys()) == ["Simulation", "View", "File"]

    def test_entries_are_label_shortcut_pairs(self):
        for entries in palette_command_groups().values():
            assert entries
            for entry in entries:
                assert len(entry) == 2
                label, key = entry
                assert isinstance(label, str) and label
                assert isinstance(key, str)  # may be empty (no binding)

    def test_returns_copies_not_shared_tables(self):
        # Mutating the returned mapping must not corrupt the shared source.
        first = palette_command_groups()
        first["Simulation"].append(("Bogus", "Ctrl+Q"))
        second = palette_command_groups()
        assert ("Bogus", "Ctrl+Q") not in second["Simulation"]


class TestDialogSourcedFromRegistry:
    def test_sim_and_view_rows_match_registry_exactly(self):
        registry = palette_command_groups()
        dialog = _groups(build_shortcut_groups())
        assert dialog["Simulation"] == registry["Simulation"]
        assert dialog["View"] == registry["View"]

    def test_file_group_is_registry_plus_supplement(self):
        registry = palette_command_groups()
        dialog = _groups(build_shortcut_groups())
        assert dialog["File"] == registry["File"] + _FILE_SUPPLEMENT

    def test_supplement_rows_are_not_palette_commands(self):
        # Exit / Edit / Help rows must not collide with registry labels.
        registry = palette_command_groups()
        registry_labels = {
            label for entries in registry.values() for label, _ in entries
        }
        catalogue = _groups(build_shortcut_groups())
        supplement_labels = {label for label, _ in _FILE_SUPPLEMENT}
        supplement_labels |= {label for label, _ in catalogue["Edit"]}
        supplement_labels |= {label for label, _ in catalogue["Help"]}
        assert supplement_labels.isdisjoint(registry_labels)

    def test_toggle_tuning_panel_matches_menu_binding(self):
        # Regression: the registry briefly carried an empty key here while the
        # menu (menu_builder) binds and displays Ctrl+Shift+T, so the dialog
        # showed a blank key for a live shortcut. The registry must mirror the
        # menu binding, and the dialog must surface it (no drift).
        view = dict(palette_command_groups()["View"])
        assert view["Toggle tuning panel"] == "Ctrl+Shift+T"
        catalogue = _groups(build_shortcut_groups())
        assert dict(catalogue["View"])["Toggle tuning panel"] == "Ctrl+Shift+T"

    def test_module_constant_matches_builder(self):
        # SHORTCUT_GROUPS is built from build_shortcut_groups at import time.
        assert SHORTCUT_GROUPS == build_shortcut_groups()


class TestDialogRendersRegistryRows:
    def test_registry_shortcuts_appear_in_widget_tree(self):
        dialog = KeyboardShortcutsDialog()
        try:
            texts = {lbl.text() for lbl in dialog.findChildren(QLabel)}
            # A binding from the live Simulation and View groups must render.
            sim = dict(palette_command_groups()["Simulation"])
            assert sim["Run simulation"] in texts  # F5
            view = dict(palette_command_groups()["View"])
            assert view["Toggle theme"] in texts   # Ctrl+T
        finally:
            dialog.close()
