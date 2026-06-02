"""
Characterization tests for ModernDiaBloSWindow's window/menubar/toolbar setup
cluster.

main_window.py historically had zero coverage. These tests build a REAL
``ModernDiaBloSWindow`` under offscreen Qt and pin down the observable behavior
of the window-chrome setup cluster before/after extraction:

  * ``_setup_window``    (title, object name, font, screen-aware sizing math)
  * ``_setup_menubar``   (delegates to MenuBuilder)
  * ``_setup_toolbar``   (creates window.toolbar + wires its signals)

Sizing is characterized by spying on setMinimumSize/resize (the math) rather
than reading back the offscreen window geometry, which is unreliable headless.

Run with:
    QT_QPA_PLATFORM=offscreen MPLBACKEND=Agg python -m pytest \
        tests/modern_ui/test_main_window_setup.py -p no:cacheprovider \
        -o addopts="" --timeout=60 --timeout-method=signal
"""

import pytest
from PyQt5.QtCore import QRect
from PyQt5.QtGui import QFont

from modern_ui.widgets.modern_toolbar import ModernToolBar
from modern_ui.platform_config import get_platform_config


@pytest.fixture(scope="module")
def window(qapp):
    from modern_ui.main_window import ModernDiaBloSWindow
    w = ModernDiaBloSWindow()
    yield w
    w.close()


@pytest.fixture(autouse=True)
def _restore_screen_geometry(window):
    saved = window.screen_geometry
    yield
    window.screen_geometry = saved


# ---------------------------------------------------------------------------
# _setup_window
# ---------------------------------------------------------------------------

class TestSetupWindow:
    def test_title_objectname_font(self, window):
        window._setup_window()
        assert window.windowTitle() == "DiaBloS - Modern Block Diagram Simulator"
        assert window.objectName() == "ModernMainWindow"
        assert window.font().family() == QFont("Segoe UI", 10).family()

    def test_fallback_sizing_when_no_screen_geometry(self, window, monkeypatch):
        captured = {}
        monkeypatch.setattr(window, "setMinimumSize",
                            lambda w, h: captured.__setitem__("min", (w, h)))
        monkeypatch.setattr(window, "resize",
                            lambda w, h: captured.__setitem__("resize", (w, h)))
        window.screen_geometry = None
        window._setup_window()
        assert captured["min"] == (1200, 800)
        assert captured["resize"] == (1600, 1000)

    def test_screen_geometry_sizing_math(self, window, monkeypatch):
        captured = {}
        monkeypatch.setattr(window, "setMinimumSize",
                            lambda w, h: captured.__setitem__("min", (w, h)))
        monkeypatch.setattr(window, "resize",
                            lambda w, h: captured.__setitem__("resize", (w, h)))
        window.screen_geometry = QRect(0, 0, 3000, 2000)
        window._setup_window()

        config = get_platform_config()
        tw = int(3000 * config.window_width_percent)
        th = int(2000 * config.window_height_percent)
        if config.should_cap_window_size:
            tw = min(tw, 1600)
            th = min(th, 1000)
        mw = max(int(tw * 0.70), config.window_min_width)
        mh = max(int(th * 0.70), config.window_min_height)
        assert captured["resize"] == (tw, th)
        assert captured["min"] == (mw, mh)


# ---------------------------------------------------------------------------
# _setup_menubar
# ---------------------------------------------------------------------------

class TestSetupMenubar:
    def test_delegates_to_menu_builder(self, window, monkeypatch):
        called = {}
        monkeypatch.setattr(window.menu_builder, "setup_menubar",
                            lambda: called.setdefault("built", True))
        window._setup_menubar()
        assert called.get("built") is True


# ---------------------------------------------------------------------------
# _setup_toolbar
# ---------------------------------------------------------------------------

class TestSetupToolbar:
    def test_toolbar_exists_and_type(self, window):
        assert isinstance(window.toolbar, ModernToolBar)

    def test_command_palette_signal_wired(self, window, monkeypatch):
        # Patches a *downstream* call (show_palette), not the connected slot
        # itself -- a Qt connection captures the bound method at connect time,
        # so patching window.show_command_palette afterwards would not affect it.
        called = {}
        monkeypatch.setattr(window.command_palette, "show_palette",
                            lambda: called.setdefault("shown", True))
        window.toolbar.command_palette_requested.emit()
        assert called.get("shown") is True
