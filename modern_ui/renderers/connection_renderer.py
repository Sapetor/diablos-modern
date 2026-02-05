"""
Connection Renderer Module
Responsible for drawing connection lines, arrowheads, and labels.
Separates rendering logic from the DLine data model.
"""

import math
import logging
from PyQt5.QtGui import QPainter, QPen, QColor, QPolygonF, QFont, QFontMetrics
from PyQt5.QtCore import Qt, QPoint, QRect
from modern_ui.themes.theme_manager import theme_manager

logger = logging.getLogger(__name__)

class ConnectionRenderer:
    """Stateless renderer for connection lines."""

    def draw_line(self, line, painter: QPainter):
        """
        Draw the connection line with smooth Bezier curves and modern styling.
        
        Args:
            line: DLine object containing path and properties
            painter: QPainter instance
        """
        if not painter or not painter.isActive():
            return
            
        if not line.path or line.path.isEmpty():
            return

        # Save painter state
        painter.save()

        try:
            # Enable antialiasing for smooth curves
            painter.setRenderHint(QPainter.Antialiasing, True)

            # Use theme_manager for connection colors
            default_connection_color = theme_manager.get_color('connection_default')
            active_connection_color = theme_manager.get_color('connection_active')

            # Determine line color and width based on selection state and signal width
            # Note: selected_segment logic handled below for specific highlighting
            is_selected = line.selected and line.selected_segment == -1
            pen_color = active_connection_color if is_selected else default_connection_color
            
            # Base line width - thicker for vector signals (MIMO indicator)
            base_width = 2.0 if getattr(line, 'signal_width', 1) <= 1 else 3.5
            line_width = base_width + 0.5 if is_selected else base_width

            # Draw subtle glow/shadow for selected connections
            if is_selected:
                # Draw outer glow
                glow_color = QColor(active_connection_color)
                glow_color.setAlpha(40)
                glow_pen = QPen(glow_color, line_width + 4, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
                painter.setPen(glow_pen)
                painter.setBrush(Qt.NoBrush)
                painter.drawPath(line.path)

            # Determine line style based on signal type
            # Discrete signals use dashed lines
            is_discrete = getattr(line, 'discrete_signal', False)
            line_style = Qt.DashLine if is_discrete else Qt.SolidLine

            # Draw main connection line
            pen = QPen(pen_color, line_width, line_style, Qt.RoundCap, Qt.RoundJoin)
            if is_discrete:
                # Set dash pattern for discrete signals: dash-space-dash-space
                pen.setDashPattern([6, 3])  # 6 pixels dash, 3 pixels space
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)
            painter.drawPath(line.path)

            # If a specific segment is selected, highlight it
            if line.selected and line.selected_segment != -1:
                if line.selected_segment < len(line.points) - 1:
                    p1 = line.points[line.selected_segment]
                    p2 = line.points[line.selected_segment + 1]

                    # Draw glow for segment
                    segment_glow_color = QColor(active_connection_color)
                    segment_glow_color.setAlpha(60)
                    glow_pen = QPen(segment_glow_color, 6, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
                    painter.setPen(glow_pen)
                    painter.drawLine(p1, p2)

                    # Draw segment highlight
                    highlight_pen = QPen(active_connection_color, 3, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
                    painter.setPen(highlight_pen)
                    painter.drawLine(p1, p2)

            # Draw intermediate points if the whole line is selected
            if line.selected and line.selected_segment == -1:
                painter.setBrush(active_connection_color)
                # Skip first and last points (ports)
                for point in line.points[1:-1]:
                    painter.drawEllipse(point, 4, 4)

            # Draw connection label if present
            if getattr(line, 'label', ''):
                self._draw_label(line, painter)

            # Draw arrowhead
            self._draw_arrowhead(line, painter, active_connection_color if line.selected else default_connection_color)

        finally:
            # Restore painter state
            painter.restore()

    def _draw_label(self, line, painter: QPainter):
        """Draw the connection label at the midpoint of the line."""
        if len(line.points) < 2:
            return

        # Find midpoint of the line
        mid_index = len(line.points) // 2
        label_pos = line.points[mid_index]

        # Draw label background
        font = QFont("Arial", 9)
        painter.setFont(font)

        metrics = QFontMetrics(font)
        text = str(line.label)
        # Qt5.9 compatibility: use width() as fallback
        text_width = metrics.horizontalAdvance(text) if hasattr(metrics, "horizontalAdvance") else metrics.width(text)
        text_height = metrics.height()

        # Background rectangle
        padding = 4
        bg_rect = QRect(
            label_pos.x() - text_width // 2 - padding,
            label_pos.y() - text_height // 2 - padding,
            text_width + 2 * padding,
            text_height + 2 * padding
        )

        # Draw semi-transparent background
        bg_color = theme_manager.get_color('canvas_background')
        bg_color.setAlpha(230)
        painter.setBrush(bg_color)
        painter.setPen(QPen(theme_manager.get_color('border_primary'), 1))
        painter.drawRoundedRect(bg_rect, 3, 3)

        # Draw text
        text_color = theme_manager.get_color('text_primary')
        painter.setPen(text_color)
        painter.drawText(bg_rect, Qt.AlignCenter, text)

    def _draw_arrowhead(self, line, painter: QPainter, color: QColor):
        """Draw arrowhead at the end of the line."""
        if not line.path or line.path.isEmpty():
            return
            
        arrow_size = 10

        end_point = line.path.pointAtPercent(1.0)
        if line.path.length() > 0:
            point_before_end = line.path.pointAtPercent(1.0 - (arrow_size / line.path.length()))
        else:
            point_before_end = end_point

        # Calculate angle
        dx = end_point.x() - point_before_end.x()
        dy = end_point.y() - point_before_end.y()
        angle = math.atan2(dy, dx)  # Angle in radians

        # Arrowhead points
        arrow_p1 = end_point + QPoint(int(-arrow_size * math.cos(angle - math.pi / 6)), int(-arrow_size * math.sin(angle - math.pi / 6)))
        arrow_p2 = end_point + QPoint(int(-arrow_size * math.cos(angle + math.pi / 6)), int(-arrow_size * math.sin(angle + math.pi / 6)))

        arrow_polygon = QPolygonF([end_point, arrow_p1, arrow_p2])

        painter.setBrush(color)  # Fill arrowhead with line color
        painter.setPen(Qt.NoPen)  # No border for arrowhead
        painter.drawPolygon(arrow_polygon)
