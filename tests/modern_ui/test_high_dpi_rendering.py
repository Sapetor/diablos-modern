"""High-DPI rendering for the toolbar and palette drag previews."""

from types import SimpleNamespace

import pytest
from PyQt6.QtCore import QPoint, QSize
from PyQt6.QtGui import QIcon

from diablos.gui.widgets import modern_palette
from diablos.gui.widgets.modern_toolbar import _make_icon

pytestmark = pytest.mark.qt


@pytest.mark.parametrize("ratio", [1.0, 1.25, 1.5, 2.0, 2.5, 3.0])
def test_toolbar_icon_renders_at_requested_dpr(qapp, ratio):
    icon = _make_icon("play", size=20, color="#ff0000")
    pixmap = icon.pixmap(QSize(20, 20), ratio)
    assert pixmap.size() == QSize(int(20 * ratio), int(20 * ratio))
    assert pixmap.devicePixelRatio() == ratio
    center = pixmap.rect().center()
    assert pixmap.toImage().pixelColor(center).alpha() == 255
    assert pixmap.toImage().pixelColor(0, 0).alpha() == 0

    disabled = icon.pixmap(QSize(20, 20), ratio, QIcon.Mode.Disabled)
    assert 0 < disabled.toImage().pixelColor(center).alpha() < 255


@pytest.mark.parametrize("ratio", [1.0, 1.25, 1.5, 3.0])
@pytest.mark.parametrize("fallback", [False, True])
def test_palette_drag_uses_physical_pixels_and_logical_hotspot(qapp, monkeypatch, ratio, fallback):
    block = SimpleNamespace(
        block_fn="Gain",
        fn_name="gain",
        b_color="#ff0000",
        ins=1,
        outs=1,
        b_type=2,
        io_edit="none",
        params={"gain": 1.0},
        external=False,
    )
    row = modern_palette.CompactBlockRow(block, "Math", None)
    monkeypatch.setattr(row, "devicePixelRatioF", lambda: ratio)
    if fallback:
        del block.block_fn  # trigger the existing renderer fallback

    captured = {}

    def record_drag(drag, *_args):
        captured.update(pixmap=drag.pixmap(), hotspot=drag.hotSpot())

    monkeypatch.setattr(modern_palette.QDrag, "exec", record_drag)
    row._start_drag(None)
    pixmap = captured["pixmap"]
    width, height = (72, 28) if fallback else (80, 48)
    assert pixmap.size() == QSize(int(width * ratio), int(height * ratio))
    assert pixmap.devicePixelRatio() == ratio
    assert captured["hotspot"] == QPoint(width // 2, height // 2)
    assert pixmap.toImage().pixelColor(pixmap.rect().center()).alpha() > 0
