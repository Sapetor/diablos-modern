"""
Construction tests for KeyboardShortcutsDialog.

The dialog is purely a read-only reference listing, so these tests verify it
builds without error, exposes its shortcut catalogue, and renders the known
key bindings (e.g. F1 / Ctrl+S) into the widget tree.
"""

import pytest

from PyQt5.QtWidgets import QDialog, QLabel


@pytest.fixture(autouse=True)
def _qt(qapp):
    """Bind the shared session QApplication (from conftest) for every test."""
    return qapp


@pytest.fixture
def dialog():
    from modern_ui.widgets.shortcuts_dialog import KeyboardShortcutsDialog
    d = KeyboardShortcutsDialog()
    yield d
    d.close()


class TestKeyboardShortcutsDialog:
    def test_constructs(self, dialog):
        assert isinstance(dialog, QDialog)
        assert dialog.windowTitle() == "Keyboard Shortcuts"

    def test_catalogue_has_expected_groups(self):
        from modern_ui.widgets.shortcuts_dialog import SHORTCUT_GROUPS
        titles = [title for title, _ in SHORTCUT_GROUPS]
        assert titles == ["File", "Edit", "Simulation", "View", "Help"]

    def test_catalogue_entries_are_label_key_pairs(self):
        from modern_ui.widgets.shortcuts_dialog import SHORTCUT_GROUPS
        for _title, entries in SHORTCUT_GROUPS:
            assert entries  # each group is non-empty
            for entry in entries:
                assert len(entry) == 2
                label, key = entry
                assert isinstance(label, str) and label
                assert isinstance(key, str)  # may be empty (no binding)

    def test_f1_help_binding_present(self):
        from modern_ui.widgets.shortcuts_dialog import SHORTCUT_GROUPS
        help_entries = dict(
            e for _t, group in SHORTCUT_GROUPS for e in group
        )
        assert help_entries.get("Keyboard shortcuts") == "F1"

    def test_renders_known_keys_into_widget_tree(self, dialog):
        texts = {lbl.text() for lbl in dialog.findChildren(QLabel)}
        # A binding from each major group should appear as a kbd label.
        assert "F1" in texts
        assert "Ctrl+S" in texts
        assert "F5" in texts

    def test_empty_binding_renders_dash_not_blank(self, dialog):
        # Entries with no default key (e.g. "Show plots") render an em-dash
        # placeholder rather than an empty kbd label.
        texts = [lbl.text() for lbl in dialog.findChildren(QLabel)]
        assert "—" in texts
