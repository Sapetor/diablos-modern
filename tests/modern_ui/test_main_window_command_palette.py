"""
Characterization tests for ModernDiaBloSWindow's command-palette cluster.

main_window.py historically had zero coverage. These tests build a REAL
``ModernDiaBloSWindow`` under offscreen Qt and pin down the observable behavior
of the command-palette cluster before it is extracted into a dedicated manager:

  * ``show_command_palette``          (open the palette)
  * ``_setup_command_palette``        (build the command index: blocks, sim,
                                       view, file, examples, recent)
  * ``_add_block_from_palette_menu``  (place a block at the cursor/centre)
  * ``_on_command_executed``          (log hook)

The built index is read back from ``command_palette._commands`` (set_commands
stores there). An autouse fixture rebuilds the index after each test so a
monkeypatched recent-files list can't leak into later tests.

Run with:
    QT_QPA_PLATFORM=offscreen MPLBACKEND=Agg python -m pytest \
        tests/modern_ui/test_main_window_command_palette.py -p no:cacheprovider \
        -o addopts="" --timeout=60 --timeout-method=signal
"""

import types

import pytest


@pytest.fixture(scope="module")
def window(qapp):
    from modern_ui.main_window import ModernDiaBloSWindow
    w = ModernDiaBloSWindow()
    yield w
    w.close()


@pytest.fixture(autouse=True)
def _rebuild_after(window):
    yield
    # Restore the palette index to its natural state after any monkeypatching.
    window._setup_command_palette()


def _commands(window):
    window._setup_command_palette()
    return window.command_palette._commands


def _by_name(cmds, name):
    return next((c for c in cmds if c.get('name') == name), None)


# ---------------------------------------------------------------------------
# _setup_command_palette
# ---------------------------------------------------------------------------

class TestSetupCommandPalette:
    def test_sim_commands_present_and_typed(self, window):
        cmds = _commands(window)
        for label in ("Run simulation", "Pause simulation", "Stop simulation",
                      "Step simulation", "Toggle fast solver"):
            c = _by_name(cmds, label)
            assert c is not None, f"missing sim command: {label}"
            assert c['type'] == 'sim'

    def test_view_commands_present(self, window):
        cmds = _commands(window)
        toggle_theme = _by_name(cmds, "Toggle theme")
        assert toggle_theme is not None and toggle_theme['type'] == 'view'

    def test_file_commands_present(self, window):
        cmds = _commands(window)
        for label in ("New diagram", "Open diagram", "Save diagram"):
            c = _by_name(cmds, label)
            assert c is not None and c['type'] == 'file'

    def test_run_simulation_callback_is_window_method(self, window):
        cmds = _commands(window)
        run = _by_name(cmds, "Run simulation")
        assert run['callback'] == window.start_simulation

    def test_block_commands_match_menu_blocks(self, window):
        cmds = _commands(window)
        block_cmds = [c for c in cmds if c['type'] == 'block']
        menu_blocks = list(getattr(window.canvas.dsim, 'menu_blocks', []) or [])
        assert len(block_cmds) == len(menu_blocks)
        for c in block_cmds:
            assert c['name'].startswith("Add ")

    def test_examples_indexed(self, window):
        cmds = _commands(window)
        example_cmds = [c for c in cmds if c.get('name', '').startswith("examples / ")]
        # The repo ships example diagrams, so at least one should be indexed.
        assert len(example_cmds) >= 1
        assert all(c['type'] == 'file' for c in example_cmds)

    def test_recent_files_included_and_capped(self, window, monkeypatch):
        fake = [f"/dir/file_{i}.diablos" for i in range(10)]
        monkeypatch.setattr(window, "_load_recent_files", lambda: fake)
        cmds = _commands(window)
        recent_cmds = [c for c in cmds if c['type'] == 'recent']
        assert len(recent_cmds) == 6  # capped at 6
        assert recent_cmds[0]['name'] == "file_0.diablos"
        assert recent_cmds[0]['description'] == "/dir/file_0.diablos"

    def test_all_commands_have_callbacks(self, window):
        # set_commands drops callback-less rows; every built command must have one.
        cmds = _commands(window)
        assert cmds and all(callable(c.get('callback')) for c in cmds)


# ---------------------------------------------------------------------------
# show_command_palette
# ---------------------------------------------------------------------------

class TestShowCommandPalette:
    def test_calls_show_palette(self, window, monkeypatch):
        called = {}
        monkeypatch.setattr(window.command_palette, "show_palette",
                            lambda: called.setdefault("shown", True))
        window.show_command_palette()
        assert called.get("shown") is True


# ---------------------------------------------------------------------------
# _add_block_from_palette_menu
# ---------------------------------------------------------------------------

class TestAddBlockFromPaletteMenu:
    def test_places_block_and_toasts(self, window, monkeypatch):
        from PyQt5.QtCore import QPoint
        captured = {}

        def fake_add(menu_block, position):
            captured['block'] = menu_block
            captured['position'] = position

        monkeypatch.setattr(window.canvas, "add_block_from_palette", fake_add)
        monkeypatch.setattr(window.toast, "show_message",
                            lambda msg, **kw: captured.setdefault('toast', msg))

        menu_block = types.SimpleNamespace(block_fn="Gain", fn_name="gain")
        window._add_block_from_palette_menu(menu_block)

        assert captured['block'] is menu_block
        assert isinstance(captured['position'], QPoint)
        assert "Gain" in captured['toast']


# ---------------------------------------------------------------------------
# _on_command_executed
# ---------------------------------------------------------------------------

class TestOnCommandExecuted:
    def test_does_not_raise(self, window):
        # Pure log hook; must accept (type, data) without error.
        window._on_command_executed("sim", {"foo": "bar"})
