"""Every registered block must supply its own ``draw_icon``.

``BaseBlock.draw_icon`` returns ``None`` -- the "use the renderer's legacy
switch" fallback. A block that never overrides it is invisible on the canvas
beyond its coloured rounded rectangle, so this module pins two things:

1. every class returned by :func:`lib.block_loader.load_blocks` overrides
   ``draw_icon`` (it is not ``BaseBlock``'s method object); and
2. calling it paints something -- the returned ``QPainterPath`` is mapped with
   the same transform ``BlockRenderer.draw_block`` uses and stroked onto a
   ``QPixmap``, which must come back with pixels that differ from the fill.

A small set of blocks deliberately returns ``None`` from an *overriding*
``draw_icon`` because their glyph is text that only ``QPainter`` can lay out
(``1/s``, ``B(s)/A(s)``, ``PID`` ...); ``LEGACY_TEXT_ICONS`` names them so the
set cannot grow silently.
"""

import pytest
from PyQt5.QtCore import QRect
from PyQt5.QtGui import QColor, QPainter, QPainterPath, QPixmap, QPen, QTransform

from blocks.base_block import BaseBlock
from lib.block_loader import load_blocks

# Blocks whose icon is painter-drawn text in BlockRenderer._draw_legacy_icon.
# They override draw_icon (returning None) to document that on purpose.
LEGACY_TEXT_ICONS = {
    "Deriv",
    "DiscreteTranFn",
    "Display",
    "From",
    "Function",
    "Gain",
    "Goto",
    "Integrator",
    "MathFunction",
    "MatrixGain",
    "PID",
    "Product",  # per-port x / / glyphs, like Sum's signs
    "StateSpace",
    "Sum",
    "TranFn",
}

# Representative block geometries: the default, a small block, and a wide one.
BLOCK_RECTS = (QRect(0, 0, 80, 60), QRect(10, 20, 40, 30), QRect(0, 0, 160, 50))

_FILL = QColor(255, 255, 255)


def _block_classes():
    return sorted(load_blocks(), key=lambda c: (c.__module__, c.__name__))


def _block_ids(classes):
    return [c.__name__ for c in classes]


_CLASSES = _block_classes()


@pytest.fixture(autouse=True)
def _qt(qapp):
    """draw_icon builds QPainterPath objects, which need a QApplication."""
    return qapp


def _render(path, rect):
    """Stroke ``path`` onto a pixmap exactly as BlockRenderer.draw_block does.

    Returns the QImage so callers can inspect pixels.
    """
    pixmap = QPixmap(rect.width() + rect.left(), rect.height() + rect.top())
    pixmap.fill(_FILL)
    painter = QPainter(pixmap)
    try:
        painter.setPen(QPen(QColor(0, 0, 0), 2))
        margin = rect.width() * 0.2
        transform = QTransform()
        transform.translate(rect.left() + margin, rect.top() + margin)
        transform.scale(rect.width() - 2 * margin, rect.height() - 2 * margin)
        painter.drawPath(transform.map(path))
    finally:
        painter.end()
    return pixmap.toImage()


def _painted_pixels(image):
    fill = _FILL.rgb()
    return sum(
        1 for y in range(image.height()) for x in range(image.width()) if image.pixel(x, y) != fill
    )


@pytest.mark.unit
@pytest.mark.qt
class TestEveryBlockHasAnIcon:
    def test_registry_is_not_empty(self):
        assert len(_CLASSES) > 50, "block registry looks truncated"

    @pytest.mark.parametrize("block_cls", _CLASSES, ids=_block_ids(_CLASSES))
    def test_overrides_base_draw_icon(self, block_cls):
        assert block_cls.draw_icon is not BaseBlock.draw_icon, (
            f"{block_cls.__name__} does not define draw_icon; it would render "
            "as a bare rounded rectangle unless the renderer's legacy switch "
            "happens to cover it"
        )

    def test_legacy_text_icon_set_is_exact(self):
        """The painter-text fallback set must not grow without a decision."""
        returns_none = set()
        for block_cls in _CLASSES:
            block = block_cls()
            if block.draw_icon(BLOCK_RECTS[0]) is None:
                returns_none.add(block.block_name)
        assert returns_none == LEGACY_TEXT_ICONS


@pytest.mark.unit
@pytest.mark.qt
class TestIconsPaint:
    @pytest.mark.parametrize("block_cls", _CLASSES, ids=_block_ids(_CLASSES))
    def test_draw_icon_paints_visible_geometry(self, block_cls):
        block = block_cls()
        for rect in BLOCK_RECTS:
            path = block.draw_icon(rect)
            if path is None:
                assert block.block_name in LEGACY_TEXT_ICONS
                continue
            assert isinstance(path, QPainterPath)
            assert not path.isEmpty(), f"{block.block_name}: empty icon path"
            image = _render(path, rect)
            assert _painted_pixels(image) > 0, (
                f"{block.block_name}: icon painted nothing at {rect.width()}x{rect.height()}"
            )

    @pytest.mark.parametrize("block_cls", _CLASSES, ids=_block_ids(_CLASSES))
    def test_icon_stays_inside_the_block(self, block_cls):
        """Icons are drawn in 0..1 normalized coordinates (guide: margin 0.1-0.9)."""
        block = block_cls()
        path = block.draw_icon(BLOCK_RECTS[0])
        if path is None:
            return
        bounds = path.boundingRect()
        assert -0.1 <= bounds.left() <= 1.1, f"{block.block_name}: {bounds}"
        assert -0.1 <= bounds.top() <= 1.1, f"{block.block_name}: {bounds}"
        assert -0.1 <= bounds.right() <= 1.1, f"{block.block_name}: {bounds}"
        assert -0.1 <= bounds.bottom() <= 1.1, f"{block.block_name}: {bounds}"
