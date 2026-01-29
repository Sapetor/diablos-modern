"""
DLine (connection) class - represents connections between blocks.
"""

import logging
import copy
from typing import List, Tuple, Optional, Union
from PyQt5.QtGui import QPainterPath, QColor
from PyQt5.QtCore import QPoint, QRect

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
                 points: Union[List[QPoint], List[Tuple[int, int]]], hidden: bool = False) -> None:
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
        self.hidden: bool = hidden
        self.total_srcports: int = 1
        self.total_dstports: int = 1
        self.srcbottom: int = 0
        self.dstbottom: int = 0
        self.points: List[QPoint] = [
            QPoint(int(p.x()), int(p.y())) if isinstance(p, QPoint) else QPoint(int(p[0]), int(p[1]))
            for p in points
        ]
        self.cptr: int = 0
        self.selected: bool = False
        self.modified: bool = False
        self.selected_segment: int = -1
        self.path: QPainterPath
        self.segments: List[QRect]
        self.routing_mode: str = "bezier"  # "bezier" or "orthogonal"
        self.path, self.points, self.segments = self.create_trajectory(self.points[0], self.points[1], [])
        self.color: QColor = QColor(0, 0, 0)  # Default to black
        self.label: str = ""  # Connection label for signal names
        self.signal_width: int = 1  # Signal width for MIMO (1=scalar, >1=vector)

    def toggle_selection(self) -> None:
        """Toggle the selection state of this line."""
        self.selected = not self.selected

    def _create_orthogonal_route(self, start: QPoint, finish: QPoint, blocks_list: List) -> List[QPoint]:
        """
        Create an orthogonal (Manhattan-style) route from start to finish, avoiding blocks.

        Args:
            start: Starting point (output port)
            finish: Ending point (input port)
            blocks_list: List of blocks to avoid

        Returns:
            List of waypoints forming an orthogonal path
        """
        # Simple orthogonal routing with collision avoidance
        waypoints = [start]

        # Create expanded block rectangles for collision detection (add margin)
        margin = 20
        obstacles = []
        for block in blocks_list:
            # Don't avoid source and destination blocks
            if block.name in [self.srcblock, self.dstblock]:
                continue
            obstacles.append(QRect(
                block.left - margin,
                block.top - margin,
                block.width + 2 * margin,
                block.height + 2 * margin
            ))

        # Check if straight orthogonal path is clear
        is_feedback = start.x() > finish.x()

        if is_feedback:
            # Feedback connection - route around source block
            src_block = None
            for block in blocks_list:
                if block.name == self.srcblock:
                    src_block = block
                    break

            if src_block:
                # Route to the right, down, left, then to destination
                clearance = 30
                right_point = QPoint(start.x() + clearance, start.y())
                bottom_point = QPoint(right_point.x(), src_block.top + src_block.height + clearance)
                left_point = QPoint(finish.x() - clearance, bottom_point.y())

                waypoints.extend([right_point, bottom_point, left_point, QPoint(left_point.x(), finish.y())])
            else:
                # Fallback: simple mid-point routing
                mid_x = int((start.x() + finish.x()) / 2)
                waypoints.extend([QPoint(mid_x, start.y()), QPoint(mid_x, finish.y())])
        else:
            # Forward connection
            mid_x = int((start.x() + finish.x()) / 2)

            # Check if vertical segment at mid_x would collide with any blocks
            collision = False
            min_y = min(start.y(), finish.y())
            max_y = max(start.y(), finish.y())

            for obstacle in obstacles:
                # Check if vertical line at mid_x intersects obstacle
                if obstacle.left() <= mid_x <= obstacle.right():
                    if obstacle.top() <= max_y and obstacle.bottom() >= min_y:
                        collision = True
                        break

            if not collision:
                # Simple two-segment path
                waypoints.extend([QPoint(mid_x, start.y()), QPoint(mid_x, finish.y())])
            else:
                # Route around the obstacle
                # Try routing above or below
                route_above = True
                min_obstacle_top = float('inf')
                max_obstacle_bottom = float('-inf')

                for obstacle in obstacles:
                    if obstacle.left() <= mid_x <= obstacle.right():
                        min_obstacle_top = min(min_obstacle_top, obstacle.top())
                        max_obstacle_bottom = max(max_obstacle_bottom, obstacle.bottom())

                # Decide whether to route above or below
                if min_obstacle_top != float('inf'):
                    space_above = min_obstacle_top - start.y()
                    space_below = finish.y() - max_obstacle_bottom
                    route_above = space_above > space_below

                if route_above and min_obstacle_top != float('inf'):
                    # Route above obstacles
                    detour_y = min_obstacle_top - margin
                    waypoints.extend([
                        QPoint(start.x() + 20, start.y()),
                        QPoint(start.x() + 20, detour_y),
                        QPoint(finish.x() - 20, detour_y),
                        QPoint(finish.x() - 20, finish.y())
                    ])
                elif max_obstacle_bottom != float('-inf'):
                    # Route below obstacles
                    detour_y = max_obstacle_bottom + margin
                    waypoints.extend([
                        QPoint(start.x() + 20, start.y()),
                        QPoint(start.x() + 20, detour_y),
                        QPoint(finish.x() - 20, detour_y),
                        QPoint(finish.x() - 20, finish.y())
                    ])
                else:
                    # Fallback to simple routing
                    waypoints.extend([QPoint(mid_x, start.y()), QPoint(mid_x, finish.y())])

        waypoints.append(finish)
        return waypoints

    def create_trajectory(self, start: QPoint, finish: QPoint, blocks_list: List,
                         points: Optional[List[QPoint]] = None) -> Tuple[QPainterPath, List[QPoint], List[QRect]]:
        """
        Create a trajectory between start and finish points using the selected routing mode.

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
        elif self.routing_mode == "orthogonal":
            # Use orthogonal routing
            all_points = self._create_orthogonal_route(start, finish, blocks_list)
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

        # Create path based on routing mode
        path = QPainterPath(all_points[0])
        segments = []

        # Radius for smooth corner curves (only used in bezier mode)
        corner_radius = 20 if self.routing_mode == "bezier" else 0

        for i in range(len(all_points) - 1):
            current = all_points[i]
            next_point = all_points[i + 1]

            # Calculate segment direction
            dx = next_point.x() - current.x()
            dy = next_point.y() - current.y()
            segment_length = max(abs(dx), abs(dy))

            # Determine if we should add a curve at the END of this segment
            # (i.e., if the next segment changes direction)
            should_curve_at_end = False
            if i < len(all_points) - 2:
                # There's another segment after this one
                next_next = all_points[i + 2]
                next_dx = next_next.x() - next_point.x()
                next_dy = next_next.y() - next_point.y()

                # Check if direction changes (horizontal to vertical or vice versa)
                this_is_horizontal = abs(dx) > abs(dy)
                next_is_horizontal = abs(next_dx) > abs(next_dy)
                should_curve_at_end = (this_is_horizontal != next_is_horizontal)

            if should_curve_at_end:
                # Calculate how much space we have for the curve
                available_space = min(segment_length / 2, corner_radius)

                # Draw straight line to point before corner
                if abs(dx) > abs(dy):
                    # Horizontal segment
                    pre_corner = QPoint(
                        next_point.x() - int(available_space * (1 if dx > 0 else -1)),
                        next_point.y()
                    )
                else:
                    # Vertical segment
                    pre_corner = QPoint(
                        next_point.x(),
                        next_point.y() - int(available_space * (1 if dy > 0 else -1))
                    )

                path.lineTo(pre_corner)

                # Now create curved transition to next segment
                # Peek at next segment to determine curve direction
                next_segment_dx = next_dx
                next_segment_dy = next_dy
                next_available = min(max(abs(next_segment_dx), abs(next_segment_dy)) / 2, corner_radius)

                if abs(next_segment_dx) > abs(next_segment_dy):
                    # Next segment is horizontal
                    post_corner = QPoint(
                        next_point.x() + int(next_available * (1 if next_segment_dx > 0 else -1)),
                        next_point.y()
                    )
                else:
                    # Next segment is vertical
                    post_corner = QPoint(
                        next_point.x(),
                        next_point.y() + int(next_available * (1 if next_segment_dy > 0 else -1))
                    )

                # Create smooth quadratic curve through the corner
                path.quadTo(next_point, post_corner)
            else:
                # No curve needed, just draw straight line
                path.lineTo(next_point)

            # Add segment for collision detection
            segments.append(QRect(current, next_point).normalized())

        return path, all_points, segments

    def update_line(self, blocks_list: List) -> None:
        """Update line coordinates based on current block positions."""
        if self.hidden:
            return
            
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

    def set_routing_mode(self, mode: str) -> None:
        """
        Set the routing mode for this connection.

        Args:
            mode: Either "bezier" or "orthogonal"
        """
        if mode in ["bezier", "orthogonal"]:
            self.routing_mode = mode
            # Force recalculation of path when mode changes
            self.modified = False

    def toggle_routing_mode(self) -> None:
        """Toggle between bezier and orthogonal routing modes."""
        if self.routing_mode == "bezier":
            self.routing_mode = "orthogonal"
        else:
            self.routing_mode = "bezier"
        # Force recalculation of path when mode changes
        self.modified = False

    def __deepcopy__(self, memo):
        """
        Custom deepcopy to exclude QPainterPath (not pickleable).
        """
        # Create a new instance
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        
        for k, v in self.__dict__.items():
            if k == 'path':
                # Recreate path later or set to default
                setattr(result, k, QPainterPath())
            elif k == 'segments':
                # Segments are lists of QRect, which are picklable with PyQt5?
                # QRect pickles fine.
                try:
                    setattr(result, k, copy.deepcopy(v, memo))
                except Exception:
                     # Fallback 
                     setattr(result, k, [])
            else:
                 try:
                    setattr(result, k, copy.deepcopy(v, memo))
                 except Exception as e:
                    # Ignore unpickleable UI assets
                    setattr(result, k, None)
                    
        return result
