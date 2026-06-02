"""
Characterization tests for ModernDiaBloSWindow's theme/palette/appearance
cluster.

main_window.py historically had zero coverage. These tests build a REAL
``ModernDiaBloSWindow`` under offscreen Qt and pin down the observable behavior
of the appearance methods before they are extracted into a dedicated
controller:

  * ``_save_user_preferences``   (atomic JSON persistence, key-merge)
  * ``toggle_theme`` /
    ``_set_palette`` /
    ``_toggle_solid_fills``       (mutate theme_manager + persist)
  * ``on_theme_changed`` /
    ``_update_statusbar_colors`` /
    ``_update_menubar_colors``    (restyle window widgets)

Preferences persist to ``user_data_path('user_preferences.json')`` which
resolves through ``lib.app_paths.get_user_data_dir()``; every test redirects
that to a fresh ``tmp_path`` so the user's real prefs are never touched.

Because ``theme_manager`` is a process-global singleton, an autouse fixture
snapshots and restores its state around each test so ordering can't leak.

Run with:
    QT_QPA_PLATFORM=offscreen MPLBACKEND=Agg python -m pytest \
        tests/modern_ui/test_main_window_theme.py -p no:cacheprovider \
        -o addopts="" --timeout=60 --timeout-method=signal
"""

import json
import os

import pytest

import lib.app_paths
from modern_ui.themes.theme_manager import theme_manager, ThemeType


@pytest.fixture(scope="module")
def window(qapp):
    from modern_ui.main_window import ModernDiaBloSWindow
    w = ModernDiaBloSWindow()
    yield w
    w.close()


@pytest.fixture
def cfg_dir(tmp_path, monkeypatch):
    """Redirect the user-data dir so preference writes hit tmp, not real config."""
    monkeypatch.setattr(lib.app_paths, "get_user_data_dir", lambda: str(tmp_path))
    return tmp_path


@pytest.fixture(autouse=True)
def _restore_theme_state():
    """Snapshot/restore the global theme_manager around each test."""
    saved = (
        theme_manager.current_theme,
        theme_manager.current_palette,
        theme_manager.solid_fills,
    )
    yield
    (theme_manager.current_theme,
     theme_manager.current_palette,
     theme_manager.solid_fills) = saved


def _prefs_path(cfg_dir):
    return os.path.join(str(cfg_dir), "user_preferences.json")


def _read_prefs(cfg_dir):
    with open(_prefs_path(cfg_dir)) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# _save_user_preferences
# ---------------------------------------------------------------------------

class TestSaveUserPreferences:
    def test_writes_current_theme_state(self, window, cfg_dir):
        theme_manager.set_theme(ThemeType.DARK)
        theme_manager.set_palette("tailwind")
        theme_manager.set_solid_fills(False)
        window._save_user_preferences()
        prefs = _read_prefs(cfg_dir)
        assert prefs["theme"] == theme_manager.current_theme.value
        assert prefs["block_palette"] == "tailwind"
        assert prefs["solid_fills"] is False

    def test_preserves_unrelated_keys(self, window, cfg_dir):
        # Pre-existing prefs file with a key the window doesn't manage.
        with open(_prefs_path(cfg_dir), "w") as f:
            json.dump({"window_geometry": "1024x768", "theme": "light"}, f)
        window._save_user_preferences()
        prefs = _read_prefs(cfg_dir)
        assert prefs["window_geometry"] == "1024x768"  # untouched
        assert "theme" in prefs and "block_palette" in prefs and "solid_fills" in prefs

    def test_corrupt_file_is_overwritten_cleanly(self, window, cfg_dir):
        with open(_prefs_path(cfg_dir), "w") as f:
            f.write("{ not valid json")
        window._save_user_preferences()  # JSONDecodeError -> start fresh
        prefs = _read_prefs(cfg_dir)
        assert "theme" in prefs

    def test_no_tmp_left_behind(self, window, cfg_dir):
        window._save_user_preferences()
        assert not os.path.exists(_prefs_path(cfg_dir) + ".tmp")


# ---------------------------------------------------------------------------
# toggle / set actions mutate theme_manager AND persist
# ---------------------------------------------------------------------------

class TestToggleAndSet:
    def test_toggle_theme_flips_and_persists(self, window, cfg_dir):
        theme_manager.set_theme(ThemeType.DARK)
        window.toggle_theme()
        assert theme_manager.current_theme == ThemeType.LIGHT
        assert _read_prefs(cfg_dir)["theme"] == theme_manager.current_theme.value
        window.toggle_theme()
        assert theme_manager.current_theme == ThemeType.DARK
        assert _read_prefs(cfg_dir)["theme"] == theme_manager.current_theme.value

    def test_set_palette_switches_and_persists(self, window, cfg_dir):
        window._set_palette("catppuccin")
        assert theme_manager.current_palette == "catppuccin"
        assert _read_prefs(cfg_dir)["block_palette"] == "catppuccin"

    def test_toggle_solid_fills_persists(self, window, cfg_dir):
        window._toggle_solid_fills(True)
        assert theme_manager.solid_fills is True
        assert _read_prefs(cfg_dir)["solid_fills"] is True
        window._toggle_solid_fills(False)
        assert theme_manager.solid_fills is False
        assert _read_prefs(cfg_dir)["solid_fills"] is False


# ---------------------------------------------------------------------------
# restyle methods run cleanly and apply non-empty stylesheets
# ---------------------------------------------------------------------------

class TestRestyle:
    def test_update_statusbar_colors_applies_stylesheet(self, window):
        window._update_statusbar_colors()
        ss = window.statusBar().styleSheet()
        assert "QStatusBar" in ss

    def test_update_menubar_colors_applies_stylesheet(self, window):
        window._update_menubar_colors()
        ss = window.menuBar().styleSheet()
        assert "QMenuBar" in ss and "QMenu" in ss

    def test_on_theme_changed_updates_theme_status_text(self, window):
        theme_manager.set_theme(ThemeType.DARK)
        window.on_theme_changed()
        # theme_status pill should reflect the active theme name.
        assert window.theme_status.text() != ""
        # canvas_area gets a non-empty stylesheet applied.
        assert window.canvas_area.styleSheet() != ""
