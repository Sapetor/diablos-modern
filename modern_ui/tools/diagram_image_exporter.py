"""Export the diagram as an image (PNG/SVG) or copy it to the clipboard.

Renders only the diagram *content* -- blocks, connections, ports -- against the
theme's canvas background, with no UI chrome: no grid, selection highlights,
hover effects, alignment guides, temp connection line, empty-canvas hint, or
error indicators. Content is laid out from the bounding rect of all blocks and
connection paths plus a comfortable margin, independent of the current viewport
zoom/pan.

Lives under ``modern_ui`` (not ``lib/``) because it needs Qt and the canvas
renderers; the rendering itself goes through ``RenderingManager.render_content``
-- the same pass the canvas ``paintEvent`` uses -- so exported output can never
drift from the on-screen look.
"""

import logging
import math

from PyQt5.QtCore import QRectF, QSize
from PyQt5.QtGui import QImage, QPainter

from modern_ui.themes.theme_manager import theme_manager

logger = logging.getLogger(__name__)

# Comfortable margin (scene pixels) around the diagram bounding rect.
MARGIN = 20


def diagram_bounding_rect(canvas, margin=MARGIN):
    """Return the diagram's bounding rect (scene coords) including ``margin``.

    Covers every block rectangle and every connection path's true extent
    (Bezier wires bow outside their anchor points, so the painter path's
    bounding rect is used rather than the waypoint list). Returns ``None``
    for an empty diagram so callers can no-op gracefully.
    """
    blocks = canvas.dsim.blocks_list
    lines = canvas.dsim.line_list
    if not blocks and not lines:
        return None

    xs, ys = [], []
    for b in blocks:
        xs.extend((b.left, b.left + b.width))
        ys.extend((b.top, b.top + b.height))
    for line in lines:
        if not line.path.isEmpty():
            r = line.path.boundingRect()
            xs.extend((r.left(), r.right()))
            ys.extend((r.top(), r.bottom()))

    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    return QRectF(
        min_x - margin,
        min_y - margin,
        (max_x - min_x) + 2 * margin,
        (max_y - min_y) + 2 * margin,
    )


def _paint_content(canvas, painter, rect):
    """Paint the diagram content into ``painter``, offset so ``rect``'s
    top-left lands at the painter origin. Assumes any scaling is already set."""
    painter.setRenderHint(QPainter.Antialiasing)
    painter.translate(-rect.x(), -rect.y())
    with canvas.rendering_manager.decorations_suppressed():
        canvas.rendering_manager.render_content(painter)


def render_diagram_image(canvas, scale=3):
    """Render the diagram to a QImage at ``scale``x for crispness.

    Returns ``None`` for an empty diagram.
    """
    rect = diagram_bounding_rect(canvas)
    if rect is None:
        return None

    width = max(1, int(math.ceil(rect.width() * scale)))
    height = max(1, int(math.ceil(rect.height() * scale)))
    image = QImage(width, height, QImage.Format_ARGB32)
    image.fill(theme_manager.get_color("canvas_background"))

    painter = QPainter(image)
    try:
        painter.scale(scale, scale)
        _paint_content(canvas, painter, rect)
    finally:
        painter.end()
    return image


def render_diagram_svg(canvas, path):
    """Render the diagram to an SVG file at ``path`` (vector, via QSvgGenerator).

    Returns ``True`` on success, ``False`` for an empty diagram.
    """
    from PyQt5.QtSvg import QSvgGenerator

    rect = diagram_bounding_rect(canvas)
    if rect is None:
        return False

    width = int(math.ceil(rect.width()))
    height = int(math.ceil(rect.height()))

    generator = QSvgGenerator()
    generator.setFileName(path)
    generator.setSize(QSize(width, height))
    generator.setViewBox(QRectF(0, 0, width, height))
    generator.setTitle("DiaBloS diagram")

    painter = QPainter(generator)
    try:
        # SVG has no implicit background; paint the canvas color first.
        painter.fillRect(QRectF(0, 0, width, height), theme_manager.get_color("canvas_background"))
        _paint_content(canvas, painter, rect)
    finally:
        painter.end()
    return True


def export_diagram_to_file(canvas, path):
    """Export the diagram to ``path``, choosing PNG or SVG by extension.

    Returns ``True`` on success, ``False`` if the diagram is empty or the write
    failed.
    """
    if path.lower().endswith(".svg"):
        return render_diagram_svg(canvas, path)

    image = render_diagram_image(canvas)
    if image is None:
        return False
    return image.save(path, "PNG")


def copy_diagram_to_clipboard(canvas):
    """Render the diagram and place it on the system clipboard.

    Returns ``True`` if an image was copied, ``False`` for an empty diagram.
    """
    from PyQt5.QtWidgets import QApplication

    image = render_diagram_image(canvas)
    if image is None:
        return False
    QApplication.clipboard().setImage(image)
    return True
