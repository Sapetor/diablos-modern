"""
Canvas Renderer Module
Responsible for drawing canvas elements like grid, selection rectangle, and overlays.
Separates rendering logic from the ModernCanvas widget.
"""

import logging
from PyQt5.QtGui import QPainter, QPen, QColor, QPainterPath
from PyQt5.QtCore import Qt, QPoint, QRect, QRectF
from modern_ui.themes.theme_manager import theme_manager

logger = logging.getLogger(__name__)

class CanvasRenderer:
    """Renderer for general canvas elements."""

    def draw_grid(self, painter: QPainter, rect: QRect, width: int, height: int, visible: bool):
        """Draw a sophisticated grid system with dots at intervals."""
        if not visible:
            return

        try:
            # Grid configuration
            small_grid_size = 20  # Small dot spacing (20px)
            large_grid_size = 100  # Large dot spacing (100px for emphasis)

            # Get theme colors
            small_dot_color = theme_manager.get_color('grid_dots')
            large_dot_color = theme_manager.get_color('grid_dots')
            large_dot_color.setAlpha(180)  # Make large dots slightly more visible

            # Draw small dots
            painter.setPen(Qt.NoPen)
            painter.setBrush(small_dot_color)
            for x in range(0, width, small_grid_size):
                for y in range(0, height, small_grid_size):
                    # Only draw small dots if not on a large grid intersection
                    if x % large_grid_size != 0 or y % large_grid_size != 0:
                        painter.drawEllipse(QPoint(x, y), 1, 1)

            # Draw larger dots at major grid intersections
            painter.setBrush(large_dot_color)
            for x in range(0, width, large_grid_size):
                for y in range(0, height, large_grid_size):
                    painter.drawEllipse(QPoint(x, y), 2, 2)

        except Exception as e:
            logger.error(f"Error drawing grid: {str(e)}")

    def draw_selection_rect(self, painter: QPainter, start: QPoint, end: QPoint):
        """Draw the selection rectangle."""
        if not start or not end:
            return

        painter.save()
        try:
            # Calculate normalized rectangle
            x1 = min(start.x(), end.x())
            y1 = min(start.y(), end.y())
            x2 = max(start.x(), end.x())
            y2 = max(start.y(), end.y())

            selection_rect = QRect(x1, y1, x2 - x1, y2 - y1)

            # Draw semi-transparent fill
            fill_color = theme_manager.get_color('selection_rectangle')
            fill_color.setAlpha(50)
            painter.fillRect(selection_rect, fill_color)

            # Draw border
            border_pen = QPen(theme_manager.get_color('selection_rectangle'), 2, Qt.DashLine)
            painter.setPen(border_pen)
            painter.setBrush(Qt.NoBrush)
            painter.drawRect(selection_rect)
        finally:
            painter.restore()

    def draw_temp_line(self, painter: QPainter, start: QPoint, end: QPoint, is_valid_target: bool):
        """Draw the temporary connection line during drag-and-drop."""
        # Choose color based on validity
        if is_valid_target:
            line_color = theme_manager.get_color('success')  # Green for valid
        else:
            line_color = theme_manager.get_color('accent_primary')  # Blue for dragging

        painter.save()
        try:
            # Enable antialiasing for smooth preview
            painter.setRenderHint(QPainter.Antialiasing, True)

            # Draw with solid line (not dashed) to avoid shadow artifacts
            pen = QPen(line_color, 2, Qt.SolidLine)
            pen.setCapStyle(Qt.RoundCap)
            painter.setPen(pen)

            # Explicitly set no brush to prevent fill artifacts
            painter.setBrush(Qt.NoBrush)

            # Draw curved Bezier preview
            path = QPainterPath()
            path.moveTo(start)

            # Calculate control points for smooth curve
            dx = end.x() - start.x()
            dy = end.y() - start.y()
            distance = (dx * dx + dy * dy) ** 0.5

            # Control point offset based on distance
            offset = min(distance * 0.5, 100)

            cp1 = QPoint(int(start.x() + offset), start.y())
            cp2 = QPoint(int(end.x() - offset), end.y())

            path.cubicTo(cp1, cp2, end)
            painter.drawPath(path)

            # Draw endpoint indicator
            painter.setBrush(line_color)
            painter.setPen(Qt.NoPen)
            painter.drawEllipse(end, 4, 4)
        finally:
            painter.restore()

    def draw_hover_effects(self, painter: QPainter, hovered_port=None, hovered_block=None, hovered_line=None):
        """Draw hover effects for ports, blocks, and connections."""
        painter.save()
        try:
            # Draw hovered port (highest priority)
            if hovered_port:
                block, port_idx, is_output = hovered_port
                port_list = block.out_coords if is_output else block.in_coords
                if port_idx < len(port_list):
                    port_pos = port_list[port_idx]

                    # Draw pulsing glow around hovered port
                    glow_color = theme_manager.get_color('accent_primary')
                    glow_color.setAlpha(100)
                    painter.setBrush(glow_color)
                    painter.setPen(Qt.NoPen)
                    painter.drawEllipse(port_pos, 12, 12)

                    # Draw brighter center
                    center_color = theme_manager.get_color('accent_primary')
                    center_color.setAlpha(180)
                    painter.setBrush(center_color)
                    painter.drawEllipse(port_pos, 8, 8)

            # Draw hovered block outline
            elif hovered_block and not getattr(hovered_block, 'selected', False):
                hover_color = theme_manager.get_color('accent_secondary')
                hover_color.setAlpha(120)

                # Draw glowing outline
                painter.setPen(QPen(hover_color, 2.5, Qt.SolidLine))
                painter.setBrush(Qt.NoBrush)
                
                # Access block geometry - handle potential differences in block object
                left = getattr(hovered_block, 'left', 0)
                top = getattr(hovered_block, 'top', 0)
                width = getattr(hovered_block, 'width', 0)
                height = getattr(hovered_block, 'height', 0)
                
                painter.drawRoundedRect(
                    left - 2,
                    top - 2,
                    width + 4,
                    height + 4,
                    8, 8
                )

            # Draw hovered connection highlight
            elif hovered_line and not getattr(hovered_line, 'selected', False):
                path = getattr(hovered_line, 'path', None)
                if path and not path.isEmpty():
                    hover_color = theme_manager.get_color('accent_secondary')
                    hover_color.setAlpha(150)

                    # Draw thicker line underneath
                    painter.setPen(QPen(hover_color, 3.5, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
                    painter.setBrush(Qt.NoBrush)
                    painter.drawPath(path)

        except Exception as e:
            logger.error(f"Error drawing hover effects: {str(e)}")
        finally:
            painter.restore()

    def draw_tag_hud(self, painter: QPainter, dsim_instance):
        """Draw a compact HUD summarizing Goto/From tags."""
        try:
            # Only show if there is at least one routing block
            tags = {}
            # Access logic safely
            blocks = getattr(dsim_instance, 'blocks_list', [])
            
            for block in blocks:
                # Assuming block object has block_fn and params
                block_fn = getattr(block, 'block_fn', '')
                params = getattr(block, 'params', {})
                
                if block_fn in ('Goto', 'From'):
                    tag = str(params.get('tag', '')).strip()
                    entry = tags.setdefault(tag, {"goto": 0, "from": 0})
                    if block_fn == 'Goto':
                        entry["goto"] += 1
                    else:
                        entry["from"] += 1

            if not tags:
                return

            # Reset transform so HUD is screen-aligned
            painter.save()
            painter.resetTransform()

            # Prepare text lines
            lines = ["Routing tags"]
            for tag, counts in sorted(tags.items()):
                label = tag if tag else "(empty)"
                lines.append(f"{label}: G{counts['goto']} → F{counts['from']}")

            fm = painter.fontMetrics()
            # horizontalAdvance not in Qt 5.9, fallback to width
            width_func = getattr(fm, "horizontalAdvance", fm.width)
            max_w = max(width_func(line) for line in lines) + 12
            line_h = fm.height() + 2
            box_h = line_h * len(lines) + 8

            # Position top-left margin
            margin = 12
            rect = QRect(margin, margin, max_w, box_h)

            # Background
            bg = theme_manager.get_color('surface')
            bg.setAlpha(200)
            painter.setBrush(bg)
            painter.setPen(Qt.NoPen)
            painter.drawRoundedRect(rect, 6, 6)

            # Text
            painter.setPen(theme_manager.get_color('text_primary'))
            y = margin + 6 + fm.ascent()
            for line in lines:
                painter.drawText(rect.left() + 6, y, line)
                y += line_h

            painter.restore()
        except Exception as e:
            logger.error(f"Error drawing tag HUD: {e}")

    def draw_validation_errors(self, painter: QPainter, blocks_with_errors, blocks_with_warnings):
        """Draw visual indicators for validation errors."""
        try:
            painter.save()

            # Note: The actual drawing of the indicator (red/orange dot/icon) 
            # might depend on block geometry. 
            # We iterate here but the drawing logic needs to know HOW to draw the indicator.
            # Assuming _draw_block_error_indicator logic is needed here or we implement it.
            
            # Since _draw_block_error_indicator was internal to Canvas, we should implement it here.
            
            # Draw error indicators on blocks
            for block in blocks_with_errors:
                self._draw_error_indicator_on_block(painter, block, is_error=True)

            for block in blocks_with_warnings:
                if block not in blocks_with_errors:  # Don't double-draw
                    self._draw_error_indicator_on_block(painter, block, is_error=False)

            painter.restore()

        except Exception as e:
            logger.error(f"Error drawing validation errors: {str(e)}")

    def _draw_error_indicator_on_block(self, painter: QPainter, block, is_error: bool):
        """Helper to draw error/warning icon on a block."""
        # Calculate position (top-right corner of block)
        left = getattr(block, 'left', 0)
        top = getattr(block, 'top', 0)
        width = getattr(block, 'width', 0)
        
        indicator_size = 14
        x = left + width - indicator_size / 2
        y = top - indicator_size / 2
        
        rect = QRectF(x, y, indicator_size, indicator_size)
        
        color = theme_manager.get_color('error') if is_error else theme_manager.get_color('warning')
        
        painter.setBrush(color)
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(rect)
        
        # Draw symbol
        painter.setPen(QColor("white"))
        font = painter.font()
        font.setBold(True)
        font.setPixelSize(10)
        painter.setFont(font)
        
        text = "!" if is_error else "?"
        painter.drawText(rect, Qt.AlignCenter, text)
