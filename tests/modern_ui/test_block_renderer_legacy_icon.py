"""Regression tests for BlockRenderer legacy-icon overdraw.

_draw_legacy_icon runs after the polymorphic draw_icon. For Constant and
RateLimiter the legacy branch drew content on top of the draw_icon path -- a
"K" overlapping Constant's flat-line shape, and a verbatim re-stroke of
RateLimiter's slew path. Both are now guarded with path.isEmpty() (the same
convention Step/Ramp/Sine already use) so draw_icon owns the shape and the
legacy branch only acts as a fallback.
"""
import pytest
from PyQt5.QtGui import QPainterPath, QPixmap, QPainter, QColor

from modern_ui.renderers.block_renderer import BlockRenderer


@pytest.fixture(autouse=True)
def _qt(qapp):
    return qapp


class _Blk:
    """Minimal stand-in -- _draw_legacy_icon only reads block_fn for these branches."""

    def __init__(self, block_fn):
        self.block_fn = block_fn
        self.left = 0
        self.top = 0
        self.width = 80
        self.height = 60


def _painter():
    pm = QPixmap(120, 120)
    pm.fill(QColor(0, 0, 0))
    return QPainter(pm), pm


def _nonempty_path():
    p = QPainterPath()
    p.moveTo(0.0, 0.0)
    p.lineTo(1.0, 1.0)
    return p


@pytest.mark.qt
class TestConstantLegacyIcon:
    def test_K_not_drawn_when_draw_icon_supplied_path(self, monkeypatch):
        r = BlockRenderer()
        calls = []
        monkeypatch.setattr(r, "_draw_centered_text", lambda *a, **k: calls.append(a))
        painter, _pm = _painter()
        try:
            r._draw_legacy_icon(_Blk("Constant"), painter, _nonempty_path())
        finally:
            painter.end()
        assert calls == []  # no "K" overlaid on the draw_icon flat line

    def test_K_drawn_as_fallback_when_path_empty(self, monkeypatch):
        r = BlockRenderer()
        calls = []
        monkeypatch.setattr(r, "_draw_centered_text", lambda *a, **k: calls.append(a))
        painter, _pm = _painter()
        try:
            r._draw_legacy_icon(_Blk("Constant"), painter, QPainterPath())
        finally:
            painter.end()
        assert len(calls) == 1
        assert "K" in calls[0]


@pytest.mark.qt
class TestRateLimiterLegacyIcon:
    def test_slew_not_restroked_when_path_present(self, monkeypatch):
        r = BlockRenderer()
        monkeypatch.setattr(r, "_draw_corner_label", lambda *a, **k: None)
        painter, _pm = _painter()
        try:
            path = _nonempty_path()
            before = path.elementCount()
            r._draw_legacy_icon(_Blk("RateLimiter"), painter, path)
            added = path.elementCount() - before
        finally:
            painter.end()
        # Only the tick (moveTo + 2 lineTo = 3) is added; the 4-element slew is
        # NOT re-stroked because draw_icon already supplied it.
        assert added == 3

    def test_full_shape_drawn_when_path_empty(self, monkeypatch):
        r = BlockRenderer()
        monkeypatch.setattr(r, "_draw_corner_label", lambda *a, **k: None)
        painter, _pm = _painter()
        try:
            path = QPainterPath()
            r._draw_legacy_icon(_Blk("RateLimiter"), painter, path)
            count = path.elementCount()
        finally:
            painter.end()
        # Fallback path: slew (4) + tick (3) = 7 elements.
        assert count == 7
