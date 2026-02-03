"""
Connection Manager for DiaBloS Modern Canvas.
Handles connection/wire creation, editing, and deletion.
"""

import logging
from typing import Optional, Tuple, List, Any, TYPE_CHECKING
from PyQt5.QtCore import QPoint
from PyQt5.QtWidgets import QApplication, QInputDialog
from PyQt5.QtCore import Qt

if TYPE_CHECKING:
    from modern_ui.widgets.modern_canvas import ModernCanvas

logger = logging.getLogger(__name__)


class ConnectionManager:
    """
    Manages connection (line/wire) operations on the canvas.
    Extracted from ModernCanvas to reduce file size and improve maintainability.
    """

    def __init__(self, canvas: 'ModernCanvas'):
        self.canvas = canvas
        self.dsim = canvas.dsim

    # ==================== Port Click Detection ====================

    def check_port_clicks(self, pos: QPoint) -> bool:
        """Check for port clicks to create connections. Returns True if a port was clicked."""
        try:
            # Check all blocks for port collisions
            for block in getattr(self.dsim, 'blocks_list', []):
                if hasattr(block, 'port_collision'):
                    # Convert QPoint to tuple for collision detection
                    point_tuple = (pos.x(), pos.y())
                    port_result = block.port_collision(point_tuple)
                    if port_result != (-1, -1):
                        port_type, port_index = port_result
                        logger.debug(f"Port clicked: {port_type}{port_index} on block {getattr(block, 'name', 'Unknown')}")
                        self.handle_port_click(block, port_type, port_index, pos)
                        return True  # Port was clicked
            return False  # No port was clicked
        except Exception as e:
            logger.error(f"Error in check_port_clicks: {str(e)}")
            return False

    def handle_port_click(self, block: Any, port_type: str, port_index: int, pos: QPoint) -> None:
        """Handle port click for connection creation."""
        from modern_ui.interactions.interaction_manager import State

        try:
            block_name = getattr(block, 'name', 'Unknown')
            logger.debug(f"Port clicked on block {block_name}, port: {port_type}{port_index}")

            if self.canvas.line_creation_state is None:
                if port_type == 'o':  # Start line from output port
                    self.canvas.state = State.CONNECTING
                    self.canvas.line_creation_state = 'start'
                    self.canvas.line_start_block = block
                    self.canvas.line_start_port = port_index
                    # Get the output port coordinates
                    if hasattr(block, 'out_coords') and port_index < len(block.out_coords):
                        start_point = block.out_coords[port_index]
                        self.canvas.temp_line = (start_point, pos)
                    logger.info(f"Started line creation from {block_name} output port {port_index}")
            elif self.canvas.line_creation_state == 'start':
                if port_type == 'i':  # End line at input port
                    logger.info(f"Completing line to {block_name} input port {port_index}")
                    self.finish_line_creation(block, port_index)
                else:
                    logger.info("Canceling line creation - clicked on output port")
                    self.cancel_line_creation()
            self.canvas.update()
        except Exception as e:
            logger.error(f"Error in handle_port_click: {str(e)}")

    # ==================== Line Creation ====================

    def finish_line_creation(self, end_block: Any, end_port: int) -> None:
        """Complete line creation between two blocks."""
        try:
            start_block_name = getattr(self.canvas.line_start_block, 'name', 'Unknown')
            end_block_name = getattr(end_block, 'name', 'Unknown')
            logger.debug(f"Finishing line creation from {start_block_name} to {end_block_name}")

            if hasattr(self.dsim, 'add_line'):
                # Get coordinates for the line
                start_coords = None
                end_coords = None
                if (hasattr(self.canvas.line_start_block, 'out_coords') and
                    self.canvas.line_start_port < len(self.canvas.line_start_block.out_coords)):
                    start_coords = self.canvas.line_start_block.out_coords[self.canvas.line_start_port]
                if (hasattr(end_block, 'in_coords') and
                    end_port < len(end_block.in_coords)):
                    end_coords = end_block.in_coords[end_port]

                if start_coords and end_coords:
                    # Validate connection before creating
                    is_valid, validation_errors = self.canvas._validate_connection(
                        self.canvas.line_start_block, self.canvas.line_start_port,
                        end_block, end_port
                    )
                    if not is_valid:
                        error_msg = "\n".join(validation_errors)
                        logger.warning(f"Connection validation failed: {error_msg}")
                        self.canvas.simulation_status_changed.emit(f"Connection invalid: {error_msg}")
                        self.cancel_line_creation()
                        return

                    # Push undo state before creating connection
                    self.canvas._push_undo("Connect")

                    # Create line using DSim's add_line method
                    new_line = self.dsim.add_line(
                        (start_block_name, self.canvas.line_start_port, start_coords),
                        (end_block_name, end_port, end_coords)
                    )
                    if new_line:
                        # Set the default routing mode for the new connection
                        new_line.routing_mode = self.canvas.default_routing_mode
                        logger.info(f"Line created: {start_block_name} -> {end_block_name} (routing: {self.canvas.default_routing_mode})")
                        # If Goto/From involved, relink to sync labels/virtual lines
                        if getattr(self.canvas.line_start_block, "block_fn", "") in ("Goto", "From") or getattr(end_block, "block_fn", "") in ("Goto", "From"):
                            try:
                                self.dsim.model.link_goto_from()
                            except Exception as e:
                                logger.warning(f"Could not relink Goto/From after connection: {e}")
                        self.update_line_positions()
                        self.canvas.connection_created.emit(self.canvas.line_start_block, end_block)
                    else:
                        logger.warning("Failed to create line")
                else:
                    logger.error("Could not get port coordinates for line creation")
            self.cancel_line_creation()
        except Exception as e:
            logger.error(f"Error in finish_line_creation: {str(e)}")
            self.cancel_line_creation()

    def cancel_line_creation(self) -> None:
        """Cancel line creation process."""
        from modern_ui.interactions.interaction_manager import State

        try:
            self.canvas.line_creation_state = None
            self.canvas.line_start_block = None
            self.canvas.line_start_port = None
            self.canvas.temp_line = None
            self.canvas.state = State.IDLE
            self.canvas.update()
            logger.debug("Line creation cancelled")
        except Exception as e:
            logger.error(f"Error cancelling line creation: {str(e)}")

    # ==================== Line Click Detection ====================

    def get_clicked_line(self, pos: QPoint) -> Tuple[Optional[Any], Optional[Any]]:
        """Get the line at the given position."""
        for line in getattr(self.dsim, 'line_list', []):
            if getattr(line, "hidden", False):
                continue
            if hasattr(line, 'collision'):
                result = line.collision(pos)
                if result:
                    return line, result
        return None, None

    def check_line_clicks(self, pos: QPoint) -> None:
        """Check for clicks on connection lines."""
        try:
            # Check for clicks near existing lines
            for line in getattr(self.dsim, 'line_list', []):
                if hasattr(line, 'points') and self.point_near_line(pos, line):
                    self.handle_line_click(line, pos)
                    return
        except Exception as e:
            logger.error(f"Error in check_line_clicks: {str(e)}")

    def point_near_line(self, pos: QPoint, line: Any) -> bool:
        """Check if a point is near a line."""
        try:
            if not hasattr(line, 'points') or len(line.points) < 2:
                return False

            threshold = 10  # pixels
            point_tuple = (pos.x(), pos.y())

            # Check each segment of the line
            for i in range(len(line.points) - 1):
                start = line.points[i]
                end = line.points[i + 1]

                # Convert QPoint to tuple if needed
                if hasattr(start, 'x'):
                    start = (start.x(), start.y())
                if hasattr(end, 'x'):
                    end = (end.x(), end.y())

                # Calculate distance from point to line segment
                distance = self.point_to_line_distance(point_tuple, start, end)
                if distance <= threshold:
                    return True
            return False
        except Exception as e:
            logger.error(f"Error in point_near_line: {str(e)}")
            return False

    def point_to_line_distance(self, point: Tuple[float, float],
                                line_start: Tuple[float, float],
                                line_end: Tuple[float, float]) -> float:
        """Calculate minimum distance from point to line segment."""
        try:
            x0, y0 = point
            x1, y1 = line_start
            x2, y2 = line_end

            # Calculate the distance
            A = x0 - x1
            B = y0 - y1
            C = x2 - x1
            D = y2 - y1

            dot = A * C + B * D
            len_sq = C * C + D * D

            if len_sq == 0:  # Line start and end are the same point
                return ((x0 - x1) ** 2 + (y0 - y1) ** 2) ** 0.5

            param = dot / len_sq

            if param < 0:
                xx, yy = x1, y1
            elif param > 1:
                xx, yy = x2, y2
            else:
                xx = x1 + param * C
                yy = y1 + param * D

            dx = x0 - xx
            dy = y0 - yy
            return (dx * dx + dy * dy) ** 0.5
        except Exception as e:
            logger.error(f"Error calculating point to line distance: {str(e)}")
            return float('inf')

    def handle_line_click(self, line: Any, collision_result: Tuple[str, int], pos: QPoint) -> None:
        """Handle clicking on a connection line."""
        from modern_ui.interactions.interaction_manager import State

        try:
            line_name = getattr(line, 'name', 'Unknown')
            logger.info(f"Line clicked: {line_name}")

            collision_type, collision_index = collision_result

            if not (QApplication.keyboardModifiers() & Qt.ControlModifier):
                self.canvas._clear_selections()

            line.selected = True
            line.modified = True  # Always allow modification on click

            if collision_type == "point":
                self.canvas.state = State.DRAGGING_LINE_POINT
                self.canvas.dragging_item = (line, collision_index)
                self.canvas.drag_offset = pos
                line.selected_segment = -1  # A point is selected, not a segment
                logger.info(f"Dragging point {collision_index} of line {line_name}")
            elif collision_type == "segment":
                self.canvas.state = State.DRAGGING_LINE_SEGMENT
                self.canvas.dragging_item = (line, collision_index)
                self.canvas.drag_offset = pos
                line.selected_segment = collision_index  # A segment is selected
                logger.info(f"Dragging segment {collision_index} of line {line_name}")
            else:  # "line" or None
                line.selected_segment = -1  # The whole line is selected

            self.canvas.update()
        except Exception as e:
            logger.error(f"Error in handle_line_click: {str(e)}")

    # ==================== Line Operations ====================

    def delete_line(self, line: Any) -> None:
        """Delete a specific connection line."""
        try:
            if line in self.dsim.line_list:
                # Push undo state before deleting line
                self.canvas._push_undo("Delete Connection")

                self.dsim.line_list.remove(line)
                logger.info(f"Deleted connection: {line.name}")
                self.canvas.update()
        except Exception as e:
            logger.error(f"Error deleting line: {str(e)}")

    def highlight_connection_path(self, line: Any) -> None:
        """Temporarily highlight a connection path."""
        # This could be enhanced with animation
        line.selected = True
        self.canvas.update()

    def edit_connection_label(self, line: Any) -> None:
        """Edit the label of a connection."""
        # Get current label
        current_label = line.label if hasattr(line, 'label') else ""

        # Show input dialog
        text, ok = QInputDialog.getText(
            self.canvas,
            "Edit Connection Label",
            f"Enter label for connection {line.srcblock} -> {line.dstblock}:",
            text=current_label
        )

        if ok:
            line.label = str(text)
            self.canvas.update()
            logger.info(f"Updated connection label: {line.name} -> '{text}'")

    def set_connection_routing_mode(self, line: Any, mode: str) -> None:
        """Change the routing mode for a connection."""
        if mode in ["bezier", "orthogonal"]:
            line.set_routing_mode(mode)
            # Force update of the line path
            line.update_line(self.dsim.blocks_list)
            self.canvas._capture_state()  # Capture state for undo
            self.canvas.update()
            logger.info(f"Changed routing mode for {line.name} to {mode}")

    def update_line_positions(self) -> None:
        """Update line positions after block movement.

        This replaces DSim.update_lines() - line position logic belongs in canvas.
        """
        for line in self.dsim.line_list:
            line.update_line(self.dsim.blocks_list)
