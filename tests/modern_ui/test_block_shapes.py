"""Block outline shapes (BaseBlock.shape) and the decorations drawn on them."""

import pytest
from PyQt5.QtCore import QPoint, QRect
from PyQt5.QtGui import QColor, QFont, QPainter, QPixmap

from modern_ui.renderers.block_renderer import (
    BlockRenderer,
    block_outline_path,
    resolve_block_shape,
)


@pytest.fixture(autouse=True)
def _qt(qapp):
    return qapp


class _StubBlock:
    def __init__(
        self, block_fn, instance=None, in_ports=1, out_ports=1, flipped=False, params=None
    ):
        self.block_fn = block_fn
        self.block_instance = instance
        self.flipped = flipped
        self.left, self.top, self.width, self.height = 100, 100, 80, 60
        self.category = "math"
        self.selected = False
        self.b_color = QColor(80, 80, 80)
        self.rect = QRect(self.left, self.top, self.width, self.height)
        self.font = QFont()
        self.username = "stub"
        self.params = params or {}
        self.port_radius = 8
        self.in_ports = in_ports
        self.out_ports = out_ports
        self.in_coords = [
            QPoint(self.left, self.top + int(self.height * (i + 1) / (in_ports + 1)))
            for i in range(in_ports)
        ]
        self.out_coords = [
            QPoint(self.left + self.width, self.top + int(self.height * (i + 1) / (out_ports + 1)))
            for i in range(out_ports)
        ]


def _paint(fn):
    pixmap = QPixmap(400, 400)
    pixmap.fill(QColor(0, 0, 0))
    painter = QPainter(pixmap)
    try:
        fn(painter)
    finally:
        painter.end()


class TestShapeResolution:
    def test_shape_hook_is_read_from_the_block_instance(self):
        from blocks.gain import GainBlock
        from blocks.matrix_gain import MatrixGainBlock
        from blocks.sum import SumBlock
        from blocks.product import ProductBlock
        from blocks.goto import GotoBlock
        from blocks.from_block import FromBlock
        from blocks.abs_block import AbsBlock

        assert resolve_block_shape(_StubBlock("Gain", GainBlock())) == "triangle"
        assert resolve_block_shape(_StubBlock("MatrixGain", MatrixGainBlock())) == "triangle"
        assert resolve_block_shape(_StubBlock("Sum", SumBlock(), in_ports=2)) == "circle"
        assert resolve_block_shape(_StubBlock("Product", ProductBlock(), in_ports=2)) == "circle"
        assert resolve_block_shape(_StubBlock("Goto", GotoBlock(), out_ports=0)) == "tag"
        assert resolve_block_shape(_StubBlock("From", FromBlock(), in_ports=0)) == "tag"
        assert resolve_block_shape(_StubBlock("Abs", AbsBlock())) == "rect"

    def test_circle_falls_back_to_rect_with_many_ports(self):
        from blocks.sum import SumBlock

        assert resolve_block_shape(_StubBlock("Sum", SumBlock(), in_ports=3)) == "circle"
        assert resolve_block_shape(_StubBlock("Sum", SumBlock(), in_ports=4)) == "rect"

    def test_fallback_by_name_without_instance(self):
        assert resolve_block_shape(_StubBlock("Gain")) == "triangle"
        assert resolve_block_shape(_StubBlock("MatrixGain")) == "triangle"
        assert resolve_block_shape(_StubBlock("Scope")) == "rect"

    def test_unknown_shape_token_is_rect(self):
        class Weird:
            shape = "hexagon"

        assert resolve_block_shape(_StubBlock("X", Weird())) == "rect"


class TestOutlinePath:
    @pytest.mark.parametrize("shape", ["rect", "triangle", "circle", "tag"])
    def test_outline_fits_block_rect(self, shape):
        block = _StubBlock("X")
        rect = block_outline_path(block, shape).boundingRect()
        assert abs(rect.left() - block.left) < 1
        assert abs(rect.top() - block.top) < 1
        assert abs(rect.width() - block.width) < 1
        assert abs(rect.height() - block.height) < 1

    @pytest.mark.parametrize("shape", ["rect", "triangle", "circle", "tag"])
    def test_expand_grows_outline(self, shape):
        block = _StubBlock("X")
        base = block_outline_path(block, shape).boundingRect()
        grown = block_outline_path(block, shape, expand=3).boundingRect()
        assert grown.width() - base.width() == pytest.approx(6, abs=0.5)
        assert grown.height() - base.height() == pytest.approx(6, abs=0.5)

    def test_triangle_and_tag_mirror_when_flipped(self):
        for shape in ("triangle", "tag"):
            normal = block_outline_path(_StubBlock("X"), shape)
            flipped = block_outline_path(_StubBlock("X", flipped=True), shape)
            # The apex is at the right edge normally, at the left edge flipped;
            # the corner beside the apex lies outside the outline.
            assert normal.contains(QPoint(179, 130))
            assert not normal.contains(QPoint(179, 102))
            assert flipped.contains(QPoint(101, 130))
            assert not flipped.contains(QPoint(101, 102))


class TestGainLabel:
    @pytest.mark.parametrize(
        "value, expected",
        [
            (2.5, "2.5"),
            (1, "1"),
            ("3", "3"),
            (-0.125, "-0.125"),
            ("[[1, 0], [0, 2]]", "[K]"),
            ([1, 2], "K"),
            ("[1, 2, 3]", "K"),
            ("not a number", "K"),
            ("", "K"),
            (1234567.0, "K"),
            ({"default": 4.0}, "4"),
        ],
    )
    def test_labels(self, value, expected):
        assert BlockRenderer.gain_label(value) == expected

    def test_fewer_digits(self):
        assert BlockRenderer.gain_label(3.14159, digits=3) == "3.14"


class TestDrawing:
    def test_draws_every_shape_without_error(self):
        from blocks.gain import GainBlock
        from blocks.matrix_gain import MatrixGainBlock
        from blocks.sum import SumBlock
        from blocks.product import ProductBlock
        from blocks.goto import GotoBlock

        renderer = BlockRenderer()
        blocks = [
            _StubBlock("Gain", GainBlock(), params={"gain": 2.5}),
            _StubBlock("Gain", GainBlock(), params={"gain": -0.125}, flipped=True),
            _StubBlock("MatrixGain", MatrixGainBlock(), params={"gain": "[[1,0],[0,1]]"}),
            _StubBlock("Sum", SumBlock(), in_ports=2, params={"sign": "+-"}),
            _StubBlock("Sum", SumBlock(), in_ports=4, params={"sign": "++-+"}),
            _StubBlock("Product", ProductBlock(), in_ports=2, params={"ops": "*/"}),
            _StubBlock("Goto", GotoBlock(), out_ports=0, params={"tag": "A"}),
        ]

        def paint(painter):
            for block in blocks:
                renderer.draw_block(block, painter)
                block.selected = True
                renderer.draw_block(block, painter)

        _paint(paint)

    def test_sum_signs_and_product_ops(self):
        assert BlockRenderer._sum_signs(_StubBlock("Sum", params={"sign": "+-x"})) == [
            "+",
            "-",
            "+",
        ]
        assert BlockRenderer._product_ops(_StubBlock("Product", params={"ops": "*/"})) == ["×", "÷"]
        assert BlockRenderer._tag_text(_StubBlock("Goto", params={"tag": " B "})) == "[B]"
        assert BlockRenderer._tag_text(_StubBlock("Goto", params={})) == "[A]"
