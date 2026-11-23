"""Window layout under a UI scale (QT_SCALE_FACTOR, see src/diablos/gui/ui_scale.py).

The screen's devicePixelRatio includes the UI scale, so at 150% a standard
screen read as high-DPI and PlatformConfig widened the Properties panel by
1.3 on top of the scale. And the fixed 1000x700 minimum window size is larger
than a 1440x900 laptop at 150% (960x600 logical pixels).
"""

from types import SimpleNamespace

import pytest
from PyQt6.QtCore import QRect
from PyQt6.QtWidgets import QApplication, QDialogButtonBox, QMainWindow

from diablos.gui import platform_config
from diablos.gui.managers import window_setup_manager

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
    assert not retina.should_cap_window_size


@pytest.mark.parametrize("scale", [1.0, 1.25, 1.5])
@pytest.mark.parametrize("native_ratio", [1.5, 2.0])
def test_native_dpi_does_not_inflate_logical_layout(qapp, monkeypatch, scale, native_ratio):
    standard = _config(monkeypatch, scale, 1.0, 1920, 1080)
    dense = _config(monkeypatch, scale, native_ratio, 1920, 1080)
    assert dense.property_panel_min_width == standard.property_panel_min_width
    assert dense.property_panel_max_width == standard.property_panel_max_width
    assert dense.canvas_min_width == standard.canvas_min_width
    assert dense.canvas_min_height == standard.canvas_min_height


def test_minimum_window_size_fits_a_small_scaled_screen(qapp, monkeypatch):
    cfg = _config(monkeypatch, 1.5, 1.0, 1440, 900)
    monkeypatch.setattr(window_setup_manager, "get_platform_config", lambda: cfg)
    win = QMainWindow()
    win.screen_geometry = QRect(0, 0, 960, 600)
    window_setup_manager.WindowSetupManager(win).setup_window()
    assert win.minimumWidth() <= 960
    assert win.minimumHeight() <= 600
    win.deleteLater()


@pytest.mark.parametrize("width,height", [(1440, 900), (1280, 720)])
def test_main_canvas_fits_a_small_scaled_window(qtbot, monkeypatch, width, height):
    from diablos.gui.main_window import ModernDiaBloSWindow

    cfg = _config(monkeypatch, 1.5, 1.0, width, height)
    monkeypatch.setattr(platform_config, "_platform_config", cfg)
    window = ModernDiaBloSWindow(QRect(0, 0, int(width / 1.5), int(height / 1.5)))
    qtbot.addWidget(window)
    window.show()
    QApplication.processEvents()

    central = window.centralWidget()
    canvas_rect = QRect(
        window.canvas.mapTo(central, window.canvas.rect().topLeft()), window.canvas.size()
    )
    assert central.rect().contains(canvas_rect)


def test_tikz_export_dialog_fits_a_small_scaled_screen(qtbot, monkeypatch):
    from diablos.gui.widgets.tikz_export_dialog import TikZExportDialog

    available = QRect(0, 0, 853, 480)
    monkeypatch.setattr(TikZExportDialog, "screen", lambda self: _fake_screen(1.5, 853, 480))
    dialog = TikZExportDialog([], [])
    qtbot.addWidget(dialog)
    dialog.show()
    QApplication.processEvents()

    assert dialog.frameGeometry().width() <= available.width()
    assert dialog.frameGeometry().height() <= available.height()
    button_box = dialog.findChild(QDialogButtonBox)
    assert dialog.rect().contains(button_box.geometry())
