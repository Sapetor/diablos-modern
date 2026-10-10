"""
Toolbar buttons must draw their icon at full size.

``QToolBar#ModernToolBar QToolButton`` gives every toolbar button
``padding: 4px 6px``. That selector is more specific than an id rule such as
``QToolButton#ZoomRockerBtn``, so a padding override written that way is
ignored. On the 22px zoom buttons the padding left about 8px for a 14px icon,
and Qt shrank the zoom -/+ glyphs to 2-3px dots (reported on Windows,
2026-10-10).

Each button is rendered with the real stylesheet, and the drawn glyph is
compared with the icon's own pixmap at the button's icon size.
"""

import pytest


@pytest.fixture(autouse=True)
def _qt(qapp):
    return qapp


@pytest.fixture
def toolbar(qapp):
    from modern_ui.styles.qss_styles import ModernStyles
    from modern_ui.widgets.modern_toolbar import ModernToolBar

    tb = ModernToolBar()
    tb.setStyleSheet(ModernStyles.get_complete_stylesheet())
    tb.resize(1400, 44)
    tb.show()
    qapp.processEvents()
    yield tb
    tb.close()
    tb.deleteLater()


def _ink_width(img, background, inset=0):
    """Width of the bounding box of pixels that differ from ``background``."""
    xs = []
    for y in range(inset, img.height() - inset):
        for x in range(inset, img.width() - inset):
            c = img.pixelColor(x, y)
            if c.alpha() < 40:
                continue
            if background is not None:
                d = sum(abs(a - b) for a, b in zip(c.getRgb()[:3], background.getRgb()[:3]))
                if d < 60:
                    continue
            xs.append(x)
    return (max(xs) - min(xs) + 1) if xs else 0


def _drawn_vs_icon(btn):
    icon_img = btn.icon().pixmap(btn.iconSize()).toImage()
    expected = _ink_width(icon_img, None)
    img = btn.grab().toImage()
    background = img.pixelColor(img.width() // 2, 2)
    drawn = _ink_width(img, background, inset=3)
    return drawn, expected


def _buttons(tb):
    r = tb.zoom_rocker
    t = tb.transport
    return {
        "zoom out": r.minus_btn,
        "zoom in": r.plus_btn,
        "play": t.play_btn,
        "stop": t.stop_btn,
    }


@pytest.mark.qt
@pytest.mark.parametrize("name", ["zoom out", "zoom in", "play", "stop"])
def test_toolbar_button_draws_its_icon_full_size(toolbar, name):
    btn = _buttons(toolbar)[name]
    drawn, expected = _drawn_vs_icon(btn)
    assert expected > 0
    assert drawn >= expected - 1, f"{name}: glyph drawn {drawn}px wide, icon is {expected}px"


@pytest.mark.qt
def test_zoom_glyphs_are_legible(toolbar):
    for btn in (toolbar.zoom_rocker.minus_btn, toolbar.zoom_rocker.plus_btn):
        drawn, _ = _drawn_vs_icon(btn)
        assert drawn >= 8, f"zoom glyph only {drawn}px wide"
