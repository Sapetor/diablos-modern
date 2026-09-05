"""GUI tests for live retranslation of the main window (View ▸ Language)."""

import pytest

from lib import i18n
from modern_ui.main_window import ModernDiaBloSWindow


@pytest.fixture
def window(qapp):
    """A real main window, restored to English afterwards."""
    win = ModernDiaBloSWindow()
    yield win
    i18n.set_language("en")
    win.retranslate_ui()
    win.close()
    win.deleteLater()


def _menu_titles(win):
    return [action.text() for action in win.menuBar().actions()]


@pytest.mark.qt
class TestLanguageMenu:
    def test_language_menu_is_built_from_catalogs(self, window):
        assert hasattr(window, "language_menu")
        labels = [a.text() for a in window.language_menu.actions() if a.text()]
        # English plus every locales/*.json, listed by its native _meta.name.
        assert "English" in labels
        assert "Español" in labels

    def test_language_menu_has_a_system_entry(self, window):
        labels = [a.text() for a in window.language_menu.actions() if a.text()]
        assert any(label.lower().startswith(("system", "sistema")) for label in labels)


@pytest.mark.qt
class TestRetranslation:
    def test_menu_bar_retranslates(self, window):
        before = _menu_titles(window)
        assert "&File" in before

        i18n.set_language("es")
        window.retranslate_ui()
        after = _menu_titles(window)

        assert after != before
        assert "&File" not in after
        assert after[0] == i18n.tr("&File")

    def test_dock_titles_retranslate(self, window):
        assert window.variable_editor_dock.windowTitle() == "Variable Editor"

        i18n.set_language("es")
        window.retranslate_ui()

        assert window.variable_editor_dock.windowTitle() == i18n.tr("Variable Editor")
        assert window.variable_editor_dock.windowTitle() != "Variable Editor"
        assert window.workspace_editor_dock.windowTitle() == i18n.tr("Workspace Variables")

    def test_panel_titles_retranslate(self, window):
        assert window.palette_panel_title.text() == "Block Palette"

        i18n.set_language("es")
        window.retranslate_ui()

        assert window.palette_panel_title.text() == i18n.tr("Block Palette")
        assert window.palette_panel_title.text() != "Block Palette"
        assert window.properties_panel_title.text() == i18n.tr("Properties")

    def test_toolbar_retranslates_without_losing_actions(self, window):
        actions_before = len(window.toolbar.actions())

        i18n.set_language("es")
        window.retranslate_ui()

        assert len(window.toolbar.actions()) == actions_before
        assert window.toolbar.new_action.text() == i18n.tr("New")
        assert window.toolbar.new_action.text() != "New"

    def test_palette_retranslates(self, window):
        i18n.set_language("es")
        window.retranslate_ui()

        assert window.block_palette.title.text() == i18n.tr("Library")

    def test_switching_back_restores_english(self, window):
        i18n.set_language("es")
        window.retranslate_ui()
        i18n.set_language("en")
        window.retranslate_ui()

        assert "&File" in _menu_titles(window)
        assert window.palette_panel_title.text() == "Block Palette"

    def test_set_language_persists_and_retranslates(self, window, monkeypatch):
        stored = {}
        monkeypatch.setattr(i18n, "store_language_setting", lambda code: stored.update(code=code))

        window.set_language("es")

        assert stored["code"] == "es"
        assert i18n.current_language() == "es"
        assert "&File" not in _menu_titles(window)
