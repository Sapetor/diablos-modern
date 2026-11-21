"""Window layout under a UI scale (QT_SCALE_FACTOR, see lib/ui_scale.py).

The screen's devicePixelRatio includes the UI scale, so at 150% a standard
screen read as high-DPI and PlatformConfig widened the Properties panel by
1.3 on top of the scale. And the fixed 1000x700 minimum window size is larger
than a 1440x900 laptop at 150% (960x600 logical pixels).
"""

from types import SimpleNamespace

import pytest
from PyQt6.QtCore import QRect
from PyQt6.QtWidgets import QApplication, QMainWindow

from modern_ui import platform_config
from modern_ui.managers import window_setup_manager

pytestmark = pytest.mark.qt


def _fake_screen(ratio, width, height):
    return SimpleNamespace(
        devicePixelRatio=lambda: ratio,
        availableGeometry=lambda: QRect(0, 0, width, height),
    )


def _config(monkeypatch, scale, native_ratio, native_w, native_h):
    """PlatformConfig as Qt would report it under ``scale``."""
    monkeypatch.setenv("QT_SCALE_FACTOR", f"{scale:g}")
    screen = _fake_screen(native_ratio * scale, int(native_w / scale), int(native_h / scale))
    monkeypatch.setattr(QApplication, "primaryScreen", staticmethod(lambda: screen))
    return platform_config.PlatformConfig()


@pytest.mark.parametrize("scale", [1.0, 1.25, 1.5])
def test_ui_scale_does_not_change_the_dpi_heuristics(qapp, monkeypatch, scale):
    standard = _config(monkeypatch, scale, 1.0, 1920, 1080)
    assert not standard.is_high_dpi
    assert standard.property_panel_min_width == 260
    assert standard.should_cap_window_size

    retina = _config(monkeypatch, scale, 2.0, 1512, 982)
    assert retina.is_high_dpi and not retina.is_retina_small


def test_minimum_window_size_fits_a_small_scaled_screen(qapp, monkeypatch):
    cfg = _config(monkeypatch, 1.5, 1.0, 1440, 900)
    monkeypatch.setattr(window_setup_manager, "get_platform_config", lambda: cfg)
    win = QMainWindow()
    win.screen_geometry = QRect(0, 0, 960, 600)
    window_setup_manager.WindowSetupManager(win).setup_window()
    assert win.minimumWidth() <= 960
    assert win.minimumHeight() <= 600
    win.deleteLater()
