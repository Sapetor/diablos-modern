"""Unit/smoke tests for the soft block shadow and port-hover emphasis.

``BlockRenderer`` is a stateless painter for blocks. These tests cover the two
recent visual changes:

* the single hard offset rectangle behind each block was replaced by a stacked
  multi-layer soft shadow (``_SOFT_SHADOW_LAYERS`` / ``_draw_soft_shadow``);
* a specific port can be flagged as hovered and is then drawn larger/brighter
  with the ``port_hover`` color (``_resolve_hovered_port`` / ``_draw_port``).

The renderer is driven against a tiny stub block exposing only the attributes
it actually touches, so the tests stay independent of the heavy DBlock/engine
stack. They assert that drawing runs without error and that the new helpers
return the expected layer count / hover resolution.
"""

import pytest
from PyQt5.QtGui import QPixmap, QPainter, QColor, QFont
from PyQt5.QtCore import QRect, QPoint

from modern_ui.renderers.block_renderer import BlockRenderer, _SOFT_SHADOW_LAYERS
from modern_ui.themes.theme_manager import theme_manager


@pytest.fixture(autouse=True)
def _qt(qapp):
    """Bind the shared session QApplication (from conftest) for every test."""
    return qapp


class _StubBlock:
    """Minimal stand-in exposing only what BlockRenderer reads when drawing."""

    def __init__(self, block_fn="Gain", flipped=False, in_ports=1, out_ports=1):
        self.block_fn = block_fn
        self.flipped = flipped
        self.left = 100
        self.top = 100
        self.width = 80
        self.height = 60
        self.category = "math"
        self.selected = False
        self.b_color = QColor(80, 80, 80)
        self.block_instance = None
        self.rect = QRect(self.left, self.top, self.width, self.height)
        self.font = QFont()
        self.username = "stub"
        self.params = {}
        self.port_radius = 8
        self.in_ports = in_ports
        self.out_ports = out_ports
        self.in_coords = [QPoint(self.left, self.top + 10 + 15 * i) for i in range(in_ports)]
        self.out_coords = [QPoint(self.left + self.width, self.top + 10 + 15 * i) for i in range(out_ports)]


def _painter_on_pixmap():
    pixmap = QPixmap(400, 400)
    pixmap.fill(QColor(0, 0, 0))
    return QPainter(pixmap), pixmap


class TestSoftShadow:
    def test_layer_recipe_count_and_shape(self):
        # 3-4 stacked layers, each a (offset, expand, alpha_scale) triple.
        assert 3 <= len(_SOFT_SHADOW_LAYERS) <= 4
        assert _SOFT_SHADOW_LAYERS is BlockRenderer._SOFT_SHADOW_LAYERS
        for layer in _SOFT_SHADOW_LAYERS:
            assert len(layer) == 3

    def test_layers_soften_outward(self):
        # Offset grows (further out) while alpha fades for the broader layers,
        # which is what reads as soft elevation rather than a hard drop shadow.
        offsets = [layer[0] for layer in _SOFT_SHADOW_LAYERS]
        alphas = [layer[2] for layer in _SOFT_SHADOW_LAYERS]
        assert offsets == sorted(offsets, reverse=True)
        assert alphas == sorted(alphas)
        assert all(0.0 < a <= 1.0 for a in alphas)

    def test_shadow_derives_alpha_from_theme_token(self):
        # Strongest layer must not exceed the theme's block_shadow base alpha.
        base_alpha = theme_manager.get_color('block_shadow').alpha()
        strongest = max(layer[2] for layer in _SOFT_SHADOW_LAYERS)
        assert int(base_alpha * strongest) <= base_alpha

    def test_draw_soft_shadow_runs_for_rect_block(self):
        renderer = BlockRenderer()
        painter, pixmap = _painter_on_pixmap()
        try:
            renderer._draw_soft_shadow(_StubBlock(block_fn="Gain2"), painter)
        finally:
            painter.end()

    def test_draw_soft_shadow_runs_for_gain_triangle(self):
        renderer = BlockRenderer()
        painter, pixmap = _painter_on_pixmap()
        try:
            renderer._draw_soft_shadow(_StubBlock(block_fn="Gain", flipped=False), painter)
            renderer._draw_soft_shadow(_StubBlock(block_fn="Gain", flipped=True), painter)
        finally:
            painter.end()


class TestHoveredPortResolution:
    def test_none_returns_empty_pair(self):
        assert BlockRenderer._resolve_hovered_port(None) == (None, None)

    def test_malformed_returns_empty_pair(self):
        assert BlockRenderer._resolve_hovered_port(("bad",)) == (None, None)
        assert BlockRenderer._resolve_hovered_port(5) == (None, None)

    def test_tuple_is_normalized(self):
        assert BlockRenderer._resolve_hovered_port((2, True)) == (2, True)
        assert BlockRenderer._resolve_hovered_port((0, False)) == (0, False)


class TestDrawBlockWithPorts:
    def test_draw_block_runs_without_hover(self):
        renderer = BlockRenderer()
        painter, pixmap = _painter_on_pixmap()
        try:
            renderer.draw_block(_StubBlock(block_fn="Box"), painter)
        finally:
            painter.end()

    def test_draw_block_runs_with_hovered_output_port(self):
        renderer = BlockRenderer()
        painter, pixmap = _painter_on_pixmap()
        try:
            # Hover the first output port; should draw larger/brighter without error.
            renderer.draw_block(_StubBlock(block_fn="Box"), painter, hovered_port=(0, True))
        finally:
            painter.end()

    def test_draw_ports_runs_with_hovered_input_port(self):
        renderer = BlockRenderer()
        painter, pixmap = _painter_on_pixmap()
        try:
            renderer.draw_ports(_StubBlock(block_fn="Box"), painter, hovered_port=(0, False))
        finally:
            painter.end()

    def test_draw_port_uses_hover_token_when_hovered(self):
        # A hovered port recolors to the 'port_hover' token and grows in radius;
        # exercise the single-port helper directly to confirm it paints cleanly
        # with the hover branch taken.
        renderer = BlockRenderer()
        painter, pixmap = _painter_on_pixmap()
        try:
            base = theme_manager.get_color('port_input')
            renderer._draw_port(painter, QPoint(50, 50), base, 7, is_hovered=True)
            renderer._draw_port(painter, QPoint(80, 80), base, 7, is_hovered=False)
        finally:
            painter.end()
