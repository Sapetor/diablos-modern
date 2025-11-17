"""
DLine (connection) class - represents connections between blocks.
"""

import math
import logging
from typing import List, Tuple, Optional, Union
from PyQt5.QtGui import QPainterPath, QColor, QPen, QPolygonF, QPainter
from PyQt5.QtCore import Qt, QPoint, QRect
from modern_ui.themes.theme_manager import theme_manager

logger = logging.getLogger(__name__)


class DLine:
    """
    Represents a connection/line between two blocks.

    Attributes:
        name: Unique identifier for this line
        sid: Sequential ID number
        srcblock: Name of source block
        srcport: Port number on source block
        dstblock: Name of destination block
        dstport: Port number on destination block
        points: List of QPoints defining the line path
        path: QPainterPath for rendering
        segments: List of QRects for collision detection
        selected: Whether line is currently selected
        modified: Whether line path has been manually modified
        selected_segment: Index of selected segment (-1 if whole line selected)
    """

    def __init__(self, sid: int, srcblock: str, srcport: int, dstblock: str, dstport: int,
                 points: Union[List[QPoint], List[Tuple[int, int]]]) -> None:
        """
        Initialize a connection line between two blocks.

        Args:
            sid: Sequential ID for this line
            srcblock: Name of the source block
            srcport: Port number on source block
            dstblock: Name of the destination block
            dstport: Port number on destination block
            points: Initial points defining the line (start and end, or custom path)
        """
        self.name: str = "Line" + str(sid)
        self.sid: int = sid
        self.srcblock: str = srcblock
        self.srcport: int = srcport
        self.dstblock: str = dstblock
        self.dstport: int = dstport
        self.total_srcports: int = 1
        self.total_dstports: int = 1
        self.srcbottom: int = 0
        self.dstbottom: int = 0
        self.points: List[QPoint] = [QPoint(p.x(), p.y()) if isinstance(p, QPoint) else QPoint(p[0], p[1]) for p in points]
        self.cptr: int = 0
        self.selected: bool = False
        self.modified: bool = False
        self.selected_segment: int = -1
        self.path: QPainterPath
        self.segments: List[QRect]
        self.path, self.points, self.segments = self.create_trajectory(self.points[0], self.points[1], [])
        self.color: QColor = QColor(0, 0, 0)  # Default to black

    def toggle_selection(self) -> None:
        """Toggle the selection state of this line."""
        self.selected = not self.selected

    def create_trajectory(self, start: QPoint, finish: QPoint, blocks_list: List,
                         points: Optional[List[QPoint]] = None) -> Tuple[QPainterPath, List[QPoint], List[QRect]]:
        """
        Create a smooth Bezier curve trajectory between start and finish points.

        Args:
            start: Starting point (output port)
            finish: Ending point (input port)
            blocks_list: List of blocks for collision detection
            points: Optional custom waypoints for modified paths

        Returns:
            Tuple of (QPainterPath, waypoints, collision segments)
        """
        all_points = []
        if self.modified and points and len(points) > 1:
            all_points = points
        else:
            is_feedback = start.x() > finish.x()

            if is_feedback:
                src_block = None
                for block in blocks_list:
                    if block.name == self.srcblock:
                        src_block = block
                        break

                if src_block:
                    p1 = QPoint(start.x() + 20, start.y())
                    p2 = QPoint(p1.x(), src_block.rect.bottom() + 30)
                    p3 = QPoint(finish.x() - 20, p2.y())
                    p4 = QPoint(p3.x(), finish.y())
                    all_points = [start, p1, p2, p3, p4, finish]
                else:  # fallback for feedback if src_block not found
                    mid_x = int((start.x() + finish.x()) / 2)
                    all_points = [start, QPoint(mid_x, start.y()), QPoint(mid_x, finish.y()), finish]
            else:
                mid_x = int((start.x() + finish.x()) / 2)
                all_points = [start, QPoint(mid_x, start.y()), QPoint(mid_x, finish.y()), finish]

        # Clean up collinear points
        clean_points = []
        if len(all_points) > 0:
            clean_points.append(all_points[0])
            for i in range(1, len(all_points) - 1):
                p1 = all_points[i-1]
                p2 = all_points[i]
                p3 = all_points[i+1]
                if (p1.x() == p2.x() == p3.x()) or (p1.y() == p2.y() == p3.y()):
                    continue
                clean_points.append(p2)
            clean_points.append(all_points[-1])
        all_points = clean_points

        # Create smooth Bezier curve path
        path = QPainterPath(all_points[0])
        segments = []

        # Curve radius for smooth transitions (adjust for smoothness)
        curve_radius = 25

        for i in range(len(all_points) - 1):
            p1 = all_points[i]
            p2 = all_points[i + 1]

            # Calculate direction of this segment
            dx = p2.x() - p1.x()
            dy = p2.y() - p1.y()

            # Determine if this is a horizontal or vertical segment
            is_horizontal = abs(dx) > abs(dy)

            if i == 0:
                # First segment - start from port, add initial straight section
                if is_horizontal:
                    # Horizontal start
                    start_straight = min(abs(dx) * 0.3, curve_radius)
                    path.lineTo(p1.x() + (start_straight if dx > 0 else -start_straight), p1.y())
                else:
                    # Vertical start
                    start_straight = min(abs(dy) * 0.3, curve_radius)
                    path.lineTo(p1.x(), p1.y() + (start_straight if dy > 0 else -start_straight))

            # Check if we need a curve (direction changes at next point)
            if i < len(all_points) - 2:
                next_p = all_points[i + 2]
                next_dx = next_p.x() - p2.x()
                next_dy = next_p.y() - p2.y()
                next_is_horizontal = abs(next_dx) > abs(next_dy)

                # If direction changes, create a smooth curve around p2
                if is_horizontal != next_is_horizontal:
                    # Calculate control points for smooth transition
                    control_distance = min(curve_radius, abs(dx) * 0.5, abs(dy) * 0.5)

                    if is_horizontal:
                        # Currently horizontal, turning vertical
                        before_corner = QPoint(
                            p2.x() - (control_distance if dx > 0 else -control_distance),
                            p2.y()
                        )
                        after_corner = QPoint(
                            p2.x(),
                            p2.y() + (control_distance if next_dy > 0 else -control_distance)
                        )
                    else:
                        # Currently vertical, turning horizontal
                        before_corner = QPoint(
                            p2.x(),
                            p2.y() - (control_distance if dy > 0 else -control_distance)
                        )
                        after_corner = QPoint(
                            p2.x() + (control_distance if next_dx > 0 else -control_distance),
                            p2.y()
                        )

                    # Draw to point before corner
                    path.lineTo(before_corner)

                    # Create cubic Bezier curve around the corner
                    # Control points pull the curve towards p2
                    control1 = QPoint(p2.x(), before_corner.y()) if is_horizontal else QPoint(before_corner.x(), p2.y())
                    control2 = QPoint(after_corner.x(), p2.y()) if is_horizontal else QPoint(p2.x(), after_corner.y())

                    path.cubicTo(control1, control2, after_corner)

                    segments.append(QRect(p1, p2).normalized())
                else:
                    # Same direction, just continue
                    path.lineTo(p2)
                    segments.append(QRect(p1, p2).normalized())
            else:
                # Last segment - smooth arrival at destination port
                if is_horizontal:
                    # Horizontal end
                    end_straight = min(abs(dx) * 0.3, curve_radius)
                    end_point = QPoint(p2.x() - (end_straight if dx > 0 else -end_straight), p2.y())
                    path.lineTo(end_point)
                    path.lineTo(p2)
                else:
                    # Vertical end
                    end_straight = min(abs(dy) * 0.3, curve_radius)
                    end_point = QPoint(p2.x(), p2.y() - (end_straight if dy > 0 else -end_straight))
                    path.lineTo(end_point)
                    path.lineTo(p2)

                segments.append(QRect(p1, p2).normalized())

        return path, all_points, segments

    def update_line(self, blocks_list: List) -> None:
        logger.debug(f"Updating line {self.name}")
        if blocks_list:
            start, end = None, None
            src_found, dst_found = False, False
            for block in blocks_list:
                if block.name == self.srcblock:
                    start = block.out_coords[self.srcport]
                    self.total_srcports = block.out_ports
                    self.srcbottom = block.top + block.height
                    src_found = True
                if block.name == self.dstblock:
                    end = block.in_coords[self.dstport]
                    self.total_dstports = block.in_ports
                    self.dstbottom = block.top + block.height
                    dst_found = True
            logger.debug(f"src_found: {src_found}, dst_found: {dst_found}")
            if start and end:
                logger.debug(f"start: {start}, end: {end}")
                self.points[0] = start
                self.points[-1] = end
                self.path, self.points, self.segments = self.create_trajectory(start, end, blocks_list)
                self.modified = False

    def draw_line(self, painter: QPainter) -> None:
        """Draw the connection line with smooth Bezier curves and modern styling."""
        if self.path and not self.path.isEmpty():
            # Save painter state
            painter.save()

            # Enable antialiasing for smooth curves
            painter.setRenderHint(QPainter.Antialiasing, True)

            # Use theme_manager for connection colors
            default_connection_color = theme_manager.get_color('connection_default')
            active_connection_color = theme_manager.get_color('connection_active')

            # Determine line color and width based on selection state
            is_selected = self.selected and self.selected_segment == -1
            pen_color = active_connection_color if is_selected else default_connection_color
            line_width = 2.5 if is_selected else 2.0

            # Draw subtle glow/shadow for selected connections
            if is_selected:
                # Draw outer glow
                glow_color = QColor(active_connection_color)
                glow_color.setAlpha(40)
                glow_pen = QPen(glow_color, line_width + 4, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
                painter.setPen(glow_pen)
                painter.setBrush(Qt.NoBrush)
                painter.drawPath(self.path)

            # Draw main connection line
            pen = QPen(pen_color, line_width, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)
            painter.drawPath(self.path)

            # If a specific segment is selected, highlight it
            if self.selected and self.selected_segment != -1:
                if self.selected_segment < len(self.points) - 1:
                    p1 = self.points[self.selected_segment]
                    p2 = self.points[self.selected_segment + 1]

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

            # Restore painter state
            painter.restore()

            # Draw intermediate points if the whole line is selected
            if self.selected and self.selected_segment == -1:
                painter.setBrush(active_connection_color)
                for point in self.points[1:-1]:
                    painter.drawEllipse(point, 4, 4)

            # Draw arrowhead
            arrow_size = 10

            end_point = self.path.pointAtPercent(1.0)
            if self.path.length() > 0:
                point_before_end = self.path.pointAtPercent(1.0 - (arrow_size / self.path.length()))
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

            arrow_color = active_connection_color if self.selected else default_connection_color
            painter.setBrush(arrow_color)  # Fill arrowhead with line color
            painter.setPen(Qt.NoPen)  # No border for arrowhead
            painter.drawPolygon(arrow_polygon)

    def collision(self, m_coords: Union[QPoint, Tuple[int, int]], point_radius: int = 5,
                  line_threshold: int = 5) -> Optional[Tuple[str, int]]:
        if isinstance(m_coords, tuple):
            m_coords = QPoint(*m_coords)

        # Check for point collision
        for i, point in enumerate(self.points):
            if (m_coords - point).manhattanLength() <= point_radius:
                return ("point", i)

        # Check for segment collision
        for i in range(len(self.points) - 1):
            p1 = self.points[i]
            p2 = self.points[i+1]

            # Bounding box check
            if not QRect(p1, p2).normalized().adjusted(-line_threshold, -line_threshold, line_threshold, line_threshold).contains(m_coords):
                continue

            # Distance from point to line segment
            v = p2 - p1
            u = m_coords - p1

            length_squared = v.x()**2 + v.y()**2
            if length_squared == 0:
                dist_sq = u.x()**2 + u.y()**2
            else:
                t = max(0, min(1, QPoint.dotProduct(u, v) / length_squared))
                projection = p1 + t * v
                dist_sq = (m_coords - projection).manhattanLength()

            if dist_sq <= line_threshold:
                return ("segment", i)

        return None

    def change_color(self, color: QColor) -> None:
        """
        Change the line color.

        Args:
            color: New QColor for this line
        """
        self.color = color
