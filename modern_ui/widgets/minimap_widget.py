"""
Minimap Widget for DiaBloS Modern UI
Provides a scaled overview of the entire diagram with viewport indicator.
"""

import logging
from PyQt5.QtWidgets import QWidget
from PyQt5.QtCore import Qt, QPoint, QRect, QRectF, pyqtSignal
from PyQt5.QtGui import QPainter, QPen, QBrush, QColor

from modern_ui.themes.theme_manager import theme_manager

logger = logging.getLogger(__name__)


class MinimapWidget(QWidget):
    """
    Minimap widget showing a scaled overview of the diagram.

    Features:
    - Shows all blocks at reduced scale
    - Highlights current viewport as a rectangle
    - Click-to-pan: clicking on minimap pans main canvas to that location
    """

    # Signal emitted when user clicks on minimap to pan
    pan_requested = pyqtSignal(QPoint)

    def __init__(self, canvas, parent=None):
        super().__init__(parent)
        self.canvas = canvas
        self.dsim = canvas.dsim

        # Minimap settings
        self.setMinimumSize(150, 100)
        self.setMaximumSize(300, 200)

        # Cached bounds
        self._diagram_bounds = None
        self._scale = 1.0
        self._offset = QPoint(0, 0)

        # Enable mouse tracking for click-to-pan
        self.setMouseTracking(True)

        # Connect to theme changes
        theme_manager.theme_changed.connect(self.update)

        logger.info("MinimapWidget initialized")

    def _calculate_diagram_bounds(self):
        """Calculate the bounding box of all blocks in the diagram."""
        blocks = self.dsim.blocks_list
        if not blocks:
            return QRect(0, 0, 100, 100)

        min_x = min(b.left for b in blocks)
        min_y = min(b.top for b in blocks)
        max_x = max(b.left + b.width for b in blocks)
        max_y = max(b.top + b.height for b in blocks)

        # Add padding
        padding = 50
        return QRect(
            min_x - padding,
            min_y - padding,
            (max_x - min_x) + 2 * padding,
            (max_y - min_y) + 2 * padding
        )

    def _calculate_scale_and_offset(self, diagram_bounds):
        """Calculate scale factor and offset to fit diagram in minimap."""
        if diagram_bounds.width() <= 0 or diagram_bounds.height() <= 0:
            return 1.0, QPoint(0, 0)

        # Calculate scale to fit diagram in widget
        scale_x = self.width() / diagram_bounds.width()
        scale_y = self.height() / diagram_bounds.height()
        scale = min(scale_x, scale_y) * 0.95  # Leave some margin

        # Calculate offset to center the diagram
        scaled_width = diagram_bounds.width() * scale
        scaled_height = diagram_bounds.height() * scale
        offset_x = (self.width() - scaled_width) / 2 - diagram_bounds.left() * scale
        offset_y = (self.height() - scaled_height) / 2 - diagram_bounds.top() * scale

        return scale, QPoint(int(offset_x), int(offset_y))

    def paintEvent(self, event):
        """Paint the minimap."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Background
        bg_color = theme_manager.get_color('surface_secondary')
        painter.fillRect(self.rect(), bg_color)

        # Border
        border_color = theme_manager.get_color('border_primary')
        painter.setPen(QPen(border_color, 1))
        painter.drawRect(self.rect().adjusted(0, 0, -1, -1))

        # Calculate bounds and scale
        self._diagram_bounds = self._calculate_diagram_bounds()
        self._scale, self._offset = self._calculate_scale_and_offset(self._diagram_bounds)

        # Apply transformation
        painter.translate(self._offset)
        painter.scale(self._scale, self._scale)

        # Draw blocks
        self._draw_blocks(painter)

        # Draw connections (simplified)
        self._draw_connections(painter)

        # Reset transformation for viewport rectangle
        painter.resetTransform()

        # Draw viewport rectangle
        self._draw_viewport_rect(painter)

        painter.end()

    def _draw_blocks(self, painter):
        """Draw simplified block representations."""
        for block in self.dsim.blocks_list:
            # Block fill color
            if block.selected:
                fill_color = theme_manager.get_color('accent_primary')
            else:
                fill_color = QColor(block.b_color) if hasattr(block, 'b_color') else theme_manager.get_color('block_fill')

            # Draw block rectangle
            painter.setBrush(QBrush(fill_color))
            painter.setPen(QPen(theme_manager.get_color('border_primary'), 1 / self._scale))
            painter.drawRect(block.left, block.top, block.width, block.height)

    def _draw_connections(self, painter):
        """Draw simplified connection lines."""
        line_color = theme_manager.get_color('connection_line')
        painter.setPen(QPen(line_color, 1 / self._scale))
        painter.setBrush(Qt.NoBrush)

        for line in self.dsim.line_list:
            if getattr(line, 'hidden', False):
                continue

            if hasattr(line, 'points') and len(line.points) >= 2:
                # Draw simplified line (just start to end)
                start = line.points[0]
                end = line.points[-1]
                painter.drawLine(start, end)

    def _draw_viewport_rect(self, painter):
        """Draw the current viewport rectangle."""
        if self._diagram_bounds is None:
            return

        # Calculate canvas viewport in diagram coordinates
        canvas_width = self.canvas.width()
        canvas_height = self.canvas.height()
        pan_offset = self.canvas.pan_offset
        zoom_factor = self.canvas.zoom_factor

        # Viewport bounds in diagram coordinates
        viewport_left = -pan_offset.x() / zoom_factor
        viewport_top = -pan_offset.y() / zoom_factor
        viewport_width = canvas_width / zoom_factor
        viewport_height = canvas_height / zoom_factor

        # Convert to minimap coordinates
        mini_left = viewport_left * self._scale + self._offset.x()
        mini_top = viewport_top * self._scale + self._offset.y()
        mini_width = viewport_width * self._scale
        mini_height = viewport_height * self._scale

        # Draw viewport rectangle
        viewport_color = theme_manager.get_color('accent_primary')
        viewport_color.setAlpha(100)

        painter.setBrush(QBrush(viewport_color))
        painter.setPen(QPen(theme_manager.get_color('accent_primary'), 2))
        painter.drawRect(QRectF(mini_left, mini_top, mini_width, mini_height))

    def mousePressEvent(self, event):
        """Handle mouse press for click-to-pan functionality."""
        if event.button() == Qt.LeftButton:
            self._pan_to_minimap_pos(event.pos())

    def mouseMoveEvent(self, event):
        """Handle mouse drag for continuous panning."""
        if event.buttons() & Qt.LeftButton:
            self._pan_to_minimap_pos(event.pos())

    def _pan_to_minimap_pos(self, minimap_pos):
        """Pan the main canvas to center on the clicked minimap position."""
        if self._diagram_bounds is None or self._scale <= 0:
            return

        # Convert minimap position to diagram coordinates
        diagram_x = (minimap_pos.x() - self._offset.x()) / self._scale
        diagram_y = (minimap_pos.y() - self._offset.y()) / self._scale

        # Calculate new pan offset to center canvas on this point
        canvas_width = self.canvas.width()
        canvas_height = self.canvas.height()
        zoom_factor = self.canvas.zoom_factor

        new_pan_x = canvas_width / 2 - diagram_x * zoom_factor
        new_pan_y = canvas_height / 2 - diagram_y * zoom_factor

        # Update canvas pan offset
        self.canvas.pan_offset = QPoint(int(new_pan_x), int(new_pan_y))
        self.canvas.update()

        # Emit signal for any additional handling
        self.pan_requested.emit(QPoint(int(diagram_x), int(diagram_y)))

    def refresh(self):
        """Refresh the minimap display."""
        self.update()
