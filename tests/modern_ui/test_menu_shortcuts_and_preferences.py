"""Menu structure, QAction shortcut ownership and the Preferences dialog.

Covers the 2026-10 menu rework: library/mask management moved to a Library
menu, persistent settings moved to File > Preferences, and shortcuts bound as
real ``QAction`` shortcuts (no ``"\\tCtrl+X"`` label suffixes) with exactly one
owner per key.
"""

import pytest
from PyQt6.QtCore import QEvent, Qt
from PyQt6.QtGui import QAction, QKeySequence
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication

from lib import i18n
from modern_ui.main_window import ModernDiaBloSWindow


@pytest.fixture
def window(qapp):
    win = ModernDiaBloSWindow()
    win.resize(1200, 800)
    win.show()
    win.activateWindow()
    QTest.qWaitForWindowActive(win, 2000)
    QTest.qWait(30)
    qapp.processEvents()
    yield win
    i18n.set_language("en")
    win.close()
    win.deleteLater()
    # Flush the deletion now: a half-dead window's actions would otherwise stay
    # in Qt's shortcut map and make the next window's keys ambiguous.
    QApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    qapp.processEvents()


def _menu(win, title):
    for top in win.menuBar().actions():
        if top.text().replace("&", "") == title:
            return top.menu()
    raise AssertionError(f"menu {title!r} not found")


def _walk(menu):
    for action in menu.actions():
        yield action
        if action.menu() is not None:
            yield from _walk(action.menu())


def _labels(menu):
    return {a.text() for a in _walk(menu) if a.text()}


@pytest.mark.qt
class TestMenuStructure:
    def test_top_level_order(self, window):
        titles = [a.text().replace("&", "") for a in window.menuBar().actions()]
        assert titles == ["File", "Edit", "Library", "Simulation", "Analysis", "View", "Help"]

    def test_library_menu_has_the_moved_actions(self, window):
        labels = _labels(_menu(window, "Library"))
        for expected in (
            "Edit &Mask...",
            "&Look Under Mask",
            "Save as &Library Block...",
            "Reload from Li&brary",
            "Refresh Block Librar&y",
            "Reload &User Blocks",
            "Open User Blocks &Folder...",
        ):
            assert expected in labels
            assert expected not in _labels(_menu(window, "Edit"))

    def test_view_menu_keeps_only_view_state(self, window):
        labels = {t.replace("&", "") for t in _labels(_menu(window, "View"))}
        for moved in ("UI Scale", "Language", "Default Connection Routing", "Solid Block Fills"):
            assert moved not in labels
        assert "Block Palette" not in labels
        assert {"Minimap", "Show Grid", "Toggle Theme", "Zoom In"} <= labels

    def test_preferences_in_file_menu(self, window):
        assert "&Preferences..." in _labels(_menu(window, "File"))
        assert window.preferences_action.menuRole() == QAction.MenuRole.PreferencesRole

    def test_no_label_carries_a_text_shortcut(self, window):
        for top in window.menuBar().actions():
            for action in _walk(top.menu()):
                assert "\t" not in action.text(), action.text()

    def test_compiled_solver_label_and_tooltip(self, window):
        action = window.fast_solver_action
        assert action.text() == "Use Compiled Solver"
        assert action.isCheckable() and action.isChecked()
        assert "compiled" in action.toolTip() and "interpreter" in action.toolTip()
        action.trigger()
        assert window.use_fast_solver is False and window.dsim.use_fast_solver is False


@pytest.mark.qt
class TestShortcutOwnership:
    def _all_actions(self, win):
        seen = []
        for top in win.menuBar().actions():
            seen.extend(a for a in _walk(top.menu()) if not a.shortcuts() == [])
        seen.extend(a for a in win.canvas.actions() if a.shortcuts())
        seen.extend(a for a in win.toolbar.actions() if a.shortcuts())
        unique = []
        for a in seen:
            if a not in unique:
                unique.append(a)
        return unique

    def test_no_key_is_bound_by_two_actions(self, window):
        owners = {}
        for action in self._all_actions(window):
            for seq in action.shortcuts():
                key = seq.toString(QKeySequence.SequenceFormat.PortableText)
                assert key not in owners, f"{key} bound by {owners[key]!r} and {action.text()!r}"
                owners[key] = action.text()

    def test_no_duplicate_after_language_rebuild(self, window):
        window.set_language("es")
        window.retranslate_ui()
        self.test_no_key_is_bound_by_two_actions(window)

    @pytest.mark.parametrize(
        "key,mods",
        [
            (Qt.Key.Key_Z, Qt.KeyboardModifier.ControlModifier),
            (Qt.Key.Key_Y, Qt.KeyboardModifier.ControlModifier),
            (Qt.Key.Key_A, Qt.KeyboardModifier.ControlModifier),
            (Qt.Key.Key_C, Qt.KeyboardModifier.ControlModifier),
            (Qt.Key.Key_F, Qt.KeyboardModifier.ControlModifier),
            (Qt.Key.Key_G, Qt.KeyboardModifier.ControlModifier),
            (Qt.Key.Key_F6, Qt.KeyboardModifier.NoModifier),
            (Qt.Key.Key_F8, Qt.KeyboardModifier.NoModifier),
            (
                Qt.Key.Key_U,
                Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.ShiftModifier,
            ),
            (
                Qt.Key.Key_A,
                Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.ShiftModifier,
            ),
        ],
    )
    def test_each_key_fires_its_action_exactly_once(self, window, qapp, key, mods):
        window.canvas.setFocus()
        qapp.processEvents()
        target = QKeySequence(key | mods)
        action = next(
            a
            for a in self._all_actions(window)
            if any(
                s.matches(target) == QKeySequence.SequenceMatch.ExactMatch for s in a.shortcuts()
            )
        )
        hits = []
        action.triggered.connect(lambda *_: hits.append(1))
        QTest.keyClick(window.canvas, key, mods)
        qapp.processEvents()
        assert len(hits) == 1, f"{action.text()!r} fired {len(hits)} times"

    def test_ctrl_z_undoes_one_step(self, window, qapp):
        history = window.canvas.history_manager
        for i in range(3):
            history.push_undo(f"step {i}")
        before = len(history.undo_stack)
        window.canvas.setFocus()
        QTest.keyClick(window.canvas, Qt.Key.Key_Z, Qt.KeyboardModifier.ControlModifier)
        qapp.processEvents()
        assert len(history.undo_stack) == before - 1
        # Redo restores exactly one.
        QTest.keyClick(window.canvas, Qt.Key.Key_Y, Qt.KeyboardModifier.ControlModifier)
        qapp.processEvents()
        assert len(history.undo_stack) == before

    def test_canvas_no_longer_double_handles_ctrl_keys(self, window):
        import inspect

        from modern_ui.widgets.modern_canvas import ModernCanvas

        src = inspect.getsource(ModernCanvas.keyPressEvent)
        for stale in ("Key_Z", "Key_Y", "Key_C", "Key_V", "Key_A", "Key_G", "Key_F5"):
            assert stale not in src

    def test_ctrl_shift_u_aligns_tops(self, window, qapp):
        # Ctrl+Shift+T is the tuning panel, so Align Top uses U ("up").
        from PyQt6.QtCore import QPoint

        canvas = window.canvas
        menu_block = next(m for m in window.dsim.menu_blocks if m.block_fn == "Gain")
        a = canvas.add_block_from_palette(menu_block, QPoint(200, 200))
        b = canvas.add_block_from_palette(menu_block, QPoint(400, 320))
        for blk in window.dsim.blocks_list:
            blk.selected = blk in (a, b)
        canvas.setFocus()
        qapp.processEvents()
        QTest.keyClick(
            canvas,
            Qt.Key.Key_U,
            Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.ShiftModifier,
        )
        qapp.processEvents()
        assert a.top == b.top

    def test_analysis_actions_have_keys(self, window):
        menu = _menu(window, "Analysis")
        assert all(a.shortcuts() for a in menu.actions() if a.text())


@pytest.mark.qt
class TestShortcutsDialogLive:
    def test_dialog_is_generated_from_actions(self, window):
        from modern_ui.widgets.shortcuts_dialog import build_live_shortcut_groups

        groups = dict(build_live_shortcut_groups(window))
        assert "Library" not in groups  # no keys there
        edit = dict(groups["Edit"])
        assert edit["Undo"]
        assert "Linearize & Analyze..." in dict(groups["Analysis"])
        assert dict(groups["Simulation"])["Run"] == "F5"


@pytest.mark.qt
class TestCommandPaletteReach:
    def test_moved_and_new_actions_are_in_the_palette(self, window):
        names = {c["name"] for c in window.command_palette._commands}
        for expected in (
            "Preferences…",
            "Edit mask…",
            "Save as library block…",
            "Reload user blocks",
            "Linearize & analyze…",
            "Monte Carlo…",
            "Toggle compiled solver",
        ):
            assert expected in names

    def test_palette_shortcuts_exist_as_real_bindings(self, window):
        bound = set()
        for top in window.menuBar().actions():
            for a in _walk(top.menu()):
                bound |= {
                    s.toString(QKeySequence.SequenceFormat.PortableText) for s in a.shortcuts()
                }
        for cmd in window.command_palette._commands:
            key = cmd.get("shortcut")
            if key and cmd["type"] in ("sim", "view", "file"):
                if key == "Ctrl+,":  # StandardKey.Preferences on macOS; Ctrl+, elsewhere
                    continue
                assert key in bound or QKeySequence(key).toString() in bound, cmd["name"]


@pytest.mark.qt
class TestPreferencesDialog:
    @pytest.fixture
    def dialog(self, window):
        from modern_ui.widgets.preferences_dialog import PreferencesDialog

        d = PreferencesDialog(window)
        yield d
        d.close()

    def test_opens_with_all_sections(self, dialog):
        titles = [dialog.sections.item(i).text() for i in range(dialog.sections.count())]
        assert titles == ["Appearance", "Language", "Editing", "Simulation"]

    def test_solid_fills_uses_the_appearance_manager_path(self, window, dialog, monkeypatch):
        from modern_ui.themes.theme_manager import theme_manager

        saved = []
        monkeypatch.setattr(window.appearance_manager, "save_preferences", lambda: saved.append(1))
        original = theme_manager.solid_fills
        try:
            dialog.solid_fills_check.setChecked(not original)
            assert theme_manager.solid_fills is (not original)
            assert saved, "preference must persist via AppearanceManager.save_preferences"
        finally:
            theme_manager.set_solid_fills(original)

    def test_palette_and_theme_persist_through_the_same_calls(self, window, dialog, monkeypatch):
        from modern_ui.themes.theme_manager import PALETTE_DISPLAY_NAMES, theme_manager

        saved = []
        monkeypatch.setattr(window.appearance_manager, "save_preferences", lambda: saved.append(1))
        original_palette = theme_manager.current_palette
        original_theme = theme_manager.current_theme
        try:
            other = next(k for k in PALETTE_DISPLAY_NAMES if k != original_palette)
            dialog.palette_combo.setCurrentIndex(dialog.palette_combo.findData(other))
            assert theme_manager.current_palette == other and saved
            saved.clear()
            dialog.theme_combo.setCurrentIndex(1 - dialog.theme_combo.currentIndex())
            assert theme_manager.current_theme != original_theme and saved
        finally:
            theme_manager.set_palette(original_palette)
            theme_manager.set_theme(original_theme)

    def test_language_goes_through_set_language_and_qsettings_key(
        self, window, dialog, monkeypatch
    ):
        calls = []
        monkeypatch.setattr(i18n, "store_language_setting", lambda code: calls.append(code))
        codes = [dialog.language_combo.itemData(i) for i in range(dialog.language_combo.count())]
        target = next(c for c in codes if c not in ("system", dialog.language_combo.currentData()))
        dialog.language_combo.setCurrentIndex(dialog.language_combo.findData(target))
        assert calls == [target]
        assert i18n.SETTINGS_KEY == "ui/language"

    def test_scale_uses_window_setter(self, window, dialog, monkeypatch):
        seen = []
        monkeypatch.setattr(window, "_set_scaling", lambda f: seen.append(f))
        dialog.scale_combo.setCurrentIndex(dialog.scale_combo.findData(1.5))
        assert seen == [1.5]

    def test_routing_updates_canvas_default(self, window, dialog):
        dialog.routing_combo.setCurrentIndex(dialog.routing_combo.findData("orthogonal"))
        assert window.default_routing_mode == "orthogonal"
        assert (
            window.canvas.connection_manager.connection_state.default_routing_mode == "orthogonal"
        )

    def test_ask_before_run_persists_in_the_sim_prefs_key(
        self, window, dialog, monkeypatch, tmp_path
    ):
        from PyQt6.QtCore import QSettings

        from lib import sim_prefs

        store = QSettings(str(tmp_path / "prefs.ini"), QSettings.Format.IniFormat)
        monkeypatch.setattr(sim_prefs, "ui_settings", lambda: store)
        dialog.ask_before_run_check.setChecked(True)
        assert sim_prefs.ask_before_run() is True
        assert store.value("simulation/ask_before_run") is not None
        assert window.dsim.ask_before_run is True
        dialog.ask_before_run_check.setChecked(False)
        assert sim_prefs.ask_before_run() is False
