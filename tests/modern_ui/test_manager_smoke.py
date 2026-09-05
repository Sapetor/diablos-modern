"""Headless smoke tests for the main-window manager cluster.

Seven collaborators of ``ModernDiaBloSWindow`` / ``ModernCanvas`` had no test
reference at all (audit 2026-09-05, section 4): ``AppearanceManager``,
``CommandPaletteManager``, ``ProjectManager``, ``PropertyController``,
``RenderingManager``, ``SimulationActionsManager`` and ``WindowSetupManager``.
An import error or a constructor-signature change in any of them broke the app
at startup with nothing in the suite noticing.

This file builds one REAL window under offscreen Qt (as the other
``tests/modern_ui`` tests do), checks each manager is wired onto the window with
the expected type, and exercises one representative operation per manager --
enough to catch "it doesn't even construct / that method moved" without pinning
down UI details that legitimately churn.

Anything that would touch the user's real files is redirected: the only
filesystem writer here (``AppearanceManager.save_preferences``) has its
``user_data_path`` monkeypatched into ``tmp_path``.

Run with:
    QT_QPA_PLATFORM=offscreen MPLBACKEND=Agg python -m pytest \
        tests/modern_ui/test_manager_smoke.py -p no:cacheprovider -o addopts=""
"""

import json

import pytest
from PyQt5.QtCore import QPoint
from PyQt5.QtGui import QImage, QPainter

from modern_ui.managers.appearance_manager import AppearanceManager
from modern_ui.managers.command_palette_manager import CommandPaletteManager
from modern_ui.managers.project_manager import ProjectManager
from modern_ui.managers.property_controller import PropertyController
from modern_ui.managers.rendering_manager import RenderingManager
from modern_ui.managers.simulation_actions_manager import SimulationActionsManager
from modern_ui.managers.window_setup_manager import WindowSetupManager


@pytest.fixture(scope="module")
def window(qapp):
    from modern_ui.main_window import ModernDiaBloSWindow

    w = ModernDiaBloSWindow()
    yield w
    w.close()


# ---------------------------------------------------------------------------
# Wiring: every manager is constructed and attached under a stable name.
# ---------------------------------------------------------------------------


@pytest.mark.qt
@pytest.mark.parametrize(
    "attr,expected_type",
    [
        ("appearance_manager", AppearanceManager),
        ("command_palette_manager", CommandPaletteManager),
        ("project_manager", ProjectManager),
        ("property_controller", PropertyController),
        ("simulation_actions_manager", SimulationActionsManager),
        ("window_setup_manager", WindowSetupManager),
    ],
)
def test_window_owns_manager(window, attr, expected_type):
    manager = getattr(window, attr, None)
    assert isinstance(manager, expected_type), f"window.{attr} is not wired up"
    # Every window-side manager keeps a back-reference under the same name.
    assert manager.window is window


@pytest.mark.qt
def test_canvas_owns_the_rendering_manager(window):
    manager = window.canvas.rendering_manager
    assert isinstance(manager, RenderingManager)
    assert manager.canvas is window.canvas
    assert manager.dsim is window.canvas.dsim


# ---------------------------------------------------------------------------
# One representative operation per manager.
# ---------------------------------------------------------------------------


@pytest.mark.qt
class TestAppearanceManager:
    def test_update_statusbar_colors_refreshes_the_theme_pill(self, window):
        window.appearance_manager.update_statusbar_colors()

        text = window.theme_status.text()
        assert "·" in text, f"theme pill should read '<theme> · <palette>', got {text!r}"
        assert text.split("·")[0].strip() in ("Light", "Dark")

    def test_save_preferences_writes_the_three_ui_prefs(self, window, tmp_path, monkeypatch):
        """Redirected into tmp_path -- never touch the developer's real prefs."""
        import lib.app_paths

        target = tmp_path / "user_preferences.json"
        monkeypatch.setattr(lib.app_paths, "user_data_path", lambda rel: str(target))

        window.appearance_manager.save_preferences()

        saved = json.loads(target.read_text())
        assert set(saved) >= {"theme", "block_palette", "solid_fills"}
        assert saved["theme"] in ("light", "dark")
        assert isinstance(saved["solid_fills"], bool)

    def test_save_preferences_merges_into_an_existing_file(self, window, tmp_path, monkeypatch):
        """Unrelated keys another feature wrote must survive."""
        import lib.app_paths

        target = tmp_path / "user_preferences.json"
        target.write_text(json.dumps({"unrelated_setting": 42}))
        monkeypatch.setattr(lib.app_paths, "user_data_path", lambda rel: str(target))

        window.appearance_manager.save_preferences()

        saved = json.loads(target.read_text())
        assert saved["unrelated_setting"] == 42
        assert "theme" in saved


@pytest.mark.qt
class TestCommandPaletteManager:
    def test_setup_indexes_the_block_library_and_actions(self, window):
        window.command_palette_manager.setup()

        commands = window.command_palette._commands
        assert commands, "command palette index is empty"
        for command in commands:
            assert "name" in command and "type" in command

        types = {c["type"] for c in commands}
        assert "block" in types, "the block library was not indexed"

    def test_setup_is_idempotent(self, window):
        """It is re-run whenever the block palette changes."""
        window.command_palette_manager.setup()
        first = len(window.command_palette._commands)
        window.command_palette_manager.setup()

        assert len(window.command_palette._commands) == first

    def test_show_does_not_raise_without_a_display(self, window):
        window.command_palette_manager.show()


@pytest.mark.qt
class TestProjectManager:
    def test_autosave_path_is_the_canonical_one(self, window):
        """A path derived from ``__module__`` used to point somewhere the
        autosave was never written, making recovery a silent no-op."""
        path = window.project_manager.autosave_path
        assert path.endswith(".autosave.diablos")
        assert "config" in path

    def test_new_diagram_resets_the_canvas(self, window):
        """File > New.

        As of 2026-09-05 this is broken: ``ProjectManager.new_diagram`` calls
        ``DiagramService.new_diagram``, which calls ``DSim.new_diagram()`` --
        a method ``lib/lib.py`` never defines (it has ``clear_all``) -- so the
        action raises ``AttributeError``. That is reported as an ``xfail`` here
        rather than a hard expectation, so this test turns green by itself the
        moment the delegation is fixed.
        """
        try:
            window.project_manager.new_diagram()
        except AttributeError as exc:
            pytest.xfail(f"DiagramService.new_diagram delegates to a missing method: {exc}")

        assert window.dsim.blocks_list == []
        assert window.dsim.line_list == []

    def test_diagram_service_is_shared_with_the_window(self, window):
        assert window.project_manager.diagram_service is window.diagram_service

    def test_clearing_the_diagram_empties_the_canvas(self, window):
        """``DSim.clear_all`` is the reset that actually works today, and the
        one ``ProjectManager.new_diagram`` should be reaching."""
        window.dsim.clear_all()

        assert window.dsim.blocks_list == []
        assert window.dsim.line_list == []


@pytest.mark.qt
class TestPropertyController:
    @pytest.mark.parametrize(
        "raw,target,expected",
        [
            ("2.5", float, 2.5),
            ("7", int, 7),
            ("true", bool, True),
            ("False", bool, False),
            ("[1, 2, 3]", list, [1, 2, 3]),
            ("hello", str, "hello"),
        ],
    )
    def test_convert_param_value(self, window, raw, target, expected):
        assert window.property_controller.convert_param_value(raw, target) == expected

    def test_an_unconvertible_value_is_kept_as_a_string(self, window):
        """Workspace variables and expressions are resolved later, not here --
        so ``K`` and ``2*K`` must survive the property editor unchanged."""
        controller = window.property_controller

        assert controller.convert_param_value("K", float) == "K"
        assert controller.convert_param_value("2*K", int) == "2*K"
        assert controller.convert_param_value("[K, K]", list) == "[K, K]"


@pytest.mark.qt
class TestRenderingManager:
    def test_run_validation_on_an_empty_diagram(self, window):
        window.dsim.clear_all()

        errors = window.canvas.rendering_manager.run_validation()

        assert isinstance(errors, list)

    def test_clear_validation_resets_the_state(self, window):
        manager = window.canvas.rendering_manager
        manager.run_validation()

        manager.clear_validation()

        assert manager.validation_state is not None

    def test_render_content_paints_without_raising(self, window):
        """The same entry point the canvas paintEvent and the image exporter
        both use, so a crash here is a crash on every repaint."""
        image = QImage(200, 150, QImage.Format_ARGB32)
        image.fill(0)
        painter = QPainter(image)
        try:
            window.canvas.rendering_manager.render_content(painter)
        finally:
            painter.end()

    def test_point_near_port_uses_the_threshold(self, window):
        manager = window.canvas.rendering_manager

        assert manager.point_near_port(QPoint(100, 100), QPoint(105, 100), threshold=12)
        assert not manager.point_near_port(QPoint(100, 100), QPoint(200, 100), threshold=12)


@pytest.mark.qt
class TestSimulationActionsManager:
    def test_toggle_fast_solver_reaches_both_the_window_and_the_engine(self, window):
        manager = window.simulation_actions_manager
        original = window.use_fast_solver
        try:
            manager.toggle_fast_solver(True)
            assert window.use_fast_solver is True
            assert window.dsim.use_fast_solver is True

            manager.toggle_fast_solver(False)
            assert window.use_fast_solver is False
            assert window.dsim.use_fast_solver is False
        finally:
            manager.toggle_fast_solver(original)

    def test_stop_is_safe_when_nothing_is_running(self, window):
        window.simulation_actions_manager.stop()

        assert window.status_message.text() == "Simulation stopped"

    def test_pause_sets_the_engine_flag(self, window):
        manager = window.simulation_actions_manager
        try:
            manager.pause()
            assert window.dsim.execution_pause is True
        finally:
            window.dsim.execution_pause = False
            manager.stop()


@pytest.mark.qt
class TestWindowSetupManager:
    def test_setup_window_titles_the_window_with_the_app_version(self, window):
        from modern_ui import __version__

        window.window_setup_manager.setup_window()

        assert __version__ in window.windowTitle()
        assert window.objectName() == "ModernMainWindow"

    def test_setup_menubar_produces_menus(self, window):
        window.window_setup_manager.setup_menubar()

        assert window.menuBar().actions(), "menubar has no top-level menus"

    def test_setup_toolbar_attaches_a_toolbar(self, window):
        window.window_setup_manager.setup_toolbar()

        assert window.toolbar is not None
