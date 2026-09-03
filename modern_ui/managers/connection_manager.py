"""
Connection Manager for DiaBloS Modern Canvas.
Handles connection/wire creation, editing, and deletion.
"""

import logging
import types
from typing import Optional, Tuple, Any, TYPE_CHECKING
from PyQt5.QtCore import QPoint
from PyQt5.QtWidgets import QApplication, QInputDialog
from PyQt5.QtCore import Qt

from lib.improvements import ValidationHelper
from modern_ui.widgets.canvas_state import ConnectionState

if TYPE_CHECKING:
    from modern_ui.widgets.modern_canvas import ModernCanvas

logger = logging.getLogger(__name__)


class ConnectionManager:
    """
    Manages connection (line/wire) operations on the canvas.
    Extracted from ModernCanvas to reduce file size and improve maintainability.
    """

    def __init__(self, canvas: "ModernCanvas"):
        self.canvas = canvas
        self.dsim = canvas.dsim
        # Connection-creation state is owned here; callers reach it as
        # ``canvas.connection_manager.connection_state``.
        self.connection_state = ConnectionState()

    def end_connection(self) -> None:
        """Clear connection-creation state and re-evaluate the idle-glow gate.

        The manager-level entry point for clearing the state: ``creation_state``
        gates the canvas glow timer, so clearing it must re-evaluate that gate.
        """
        self.connection_state.end_connection()
        self.canvas._evaluate_animation_state()

    # ==================== Port Click Detection ====================

    def check_port_clicks(self, pos: QPoint) -> bool:
        """Check for port clicks to create connections. Returns True if a port was clicked."""
        try:
            # Check all blocks for port collisions
            for block in getattr(self.dsim, "blocks_list", []):
                if hasattr(block, "port_collision"):
                    # Convert QPoint to tuple for collision detection
                    point_tuple = (pos.x(), pos.y())
                    port_result = block.port_collision(point_tuple)
                    if port_result != (-1, -1):
                        port_type, port_index = port_result
                        logger.debug(
                            f"Port clicked: {port_type}{port_index} on block {getattr(block, 'name', 'Unknown')}"
                        )
                        self.handle_port_click(block, port_type, port_index, pos)
                        return True  # Port was clicked
            return False  # No port was clicked
        except Exception as e:
            logger.error(f"Error in check_port_clicks: {str(e)}")
            return False

    def handle_port_click(self, block: Any, port_type: str, port_index: int, pos: QPoint) -> None:
        """Handle a press on a port.

        With no wire in progress, an output port starts a forward wire and a
        free input port starts a *reverse* wire (finished on an output port).
        With a wire in progress, a port of the opposite kind completes it and
        any other port cancels it. Either way the gesture can be finished by
        a second click or by releasing the mouse over the target port (see
        ``try_finish_at``).
        """
        from modern_ui.interactions.interaction_manager import State

        try:
            block_name = getattr(block, "name", "Unknown")
            logger.debug(f"Port clicked on block {block_name}, port: {port_type}{port_index}")
            state = self.connection_state

            if state.creation_state is None:
                if port_type == "o":
                    coords = getattr(block, "out_coords", [])
                    reverse = False
                elif port_type == "i":
                    if self._input_is_connected(block, port_index):
                        self.canvas.simulation_status_changed.emit(
                            "Input port already connected -- delete the existing wire first"
                        )
                        return
                    coords = getattr(block, "in_coords", [])
                    reverse = True
                else:
                    return
                if port_index >= len(coords):
                    return
                self.canvas.state = State.CONNECTING
                state.creation_state = "start"
                state.start_block = block
                state.start_port = port_index
                state.source_block = block
                state.reverse = reverse
                state.press_pos = QPoint(pos)
                state.dragged = False
                state.temp_line = (coords[port_index], pos)
                # creation_state gates the idle-glow timer; re-evaluate it.
                self.canvas._evaluate_animation_state()
                logger.info(
                    f"Started {'reverse ' if reverse else ''}wire from {block_name} "
                    f"{'input' if reverse else 'output'} port {port_index}"
                )
            elif state.creation_state == "start":
                wanted = "o" if state.reverse else "i"
                if port_type == wanted:
                    logger.info(f"Completing wire at {block_name} port {port_type}{port_index}")
                    self.finish_line_creation(block, port_index)
                else:
                    logger.info("Canceling wire creation - clicked a port of the wrong kind")
                    self.cancel_line_creation()
            self.canvas.update()
        except Exception as e:
            logger.error(f"Error in handle_port_click: {str(e)}")

    def _input_is_connected(self, block: Any, port_index: int) -> bool:
        name = getattr(block, "name", "")
        for line in getattr(self.dsim, "line_list", []):
            if (
                getattr(line, "dstblock", None) == name
                and getattr(line, "dstport", -1) == port_index
            ):
                return True
        return False

    def port_at(self, pos: QPoint):
        """Return ``(block, port_type, port_index)`` for the port under ``pos``."""
        point_tuple = (pos.x(), pos.y())
        for block in getattr(self.dsim, "blocks_list", []):
            if hasattr(block, "port_collision"):
                result = block.port_collision(point_tuple)
                if result != (-1, -1):
                    return block, result[0], result[1]
        return None

    def target_validity(self, block: Any, port_type: str, port_index: int):
        """Classify a hovered port while a wire is in progress.

        Returns ``True`` when dropping here would create a valid wire,
        ``False`` when it is a port of the right kind but the wire would be
        rejected (already connected, same block, ...), and ``None`` when it is
        not a candidate at all (wrong kind, or the port the wire started on).
        """
        state = self.connection_state
        if state.creation_state != "start" or state.start_block is None:
            return None
        wanted = "o" if state.reverse else "i"
        if port_type != wanted:
            return None
        if state.reverse:
            src_block, src_port, dst_block, dst_port = (
                block,
                port_index,
                state.start_block,
                state.start_port,
            )
        else:
            src_block, src_port, dst_block, dst_port = (
                state.start_block,
                state.start_port,
                block,
                port_index,
            )
        ok, _errors = self.validate_connection(src_block, src_port, dst_block, dst_port)
        return bool(ok)

    def note_drag(self, pos: QPoint) -> None:
        """Record that the cursor moved away from the press that started the wire."""
        state = self.connection_state
        if state.creation_state != "start" or state.dragged or state.press_pos is None:
            return
        if (pos - state.press_pos).manhattanLength() > 6:
            state.dragged = True

    def try_finish_at(self, pos: QPoint) -> bool:
        """Complete or cancel an in-progress wire on mouse release.

        Releasing over a port of the wanted kind commits the wire (drag-to-
        connect). Releasing elsewhere after a real drag cancels; releasing at
        the press position (a plain click) leaves the wire pending so the
        classic click-source-then-click-target gesture still works. Returns
        True when the release was consumed by wire creation.
        """
        state = self.connection_state
        if state.creation_state != "start":
            return False
        hit = self.port_at(pos)
        if hit is not None:
            block, port_type, port_index = hit
            wanted = "o" if state.reverse else "i"
            is_origin = block is state.start_block and port_index == state.start_port
            if port_type == wanted and not is_origin:
                self.finish_line_creation(block, port_index)
                return True
            if is_origin or not state.dragged:
                return True  # still pending: click-click gesture
        if state.dragged:
            logger.info("Wire drag released away from any port - cancelled")
            self.cancel_line_creation()
            return True
        return True

    # ==================== Connection Validation ====================

    def validate_connection(
        self, start_block: Any, start_port: int, end_block: Any, end_port: int
    ) -> Tuple[bool, list]:
        """Validate a proposed connection between two blocks.

        Returns (is_valid, validation_errors).
        """
        try:
            validation_errors = []

            # Basic validation checks
            if start_block == end_block:
                validation_errors.append("Cannot connect a block to itself")

            # BodeMag and RootLocus connections logic
            allowed_bode_blocks = [
                "TranFn",
                "DiscreteTranFn",
                "StateSpace",
                "DiscreteStateSpace",
                "PID",
            ]

            if (
                end_block.block_fn in ["BodeMag", "BodePhase", "Nyquist"]
                and start_block.block_fn not in allowed_bode_blocks
            ):
                validation_errors.append(
                    f"{end_block.block_fn} block can only be connected to: {', '.join(allowed_bode_blocks)}"
                )

            if end_block.block_fn == "RootLocus" and start_block.block_fn != "TranFn":
                validation_errors.append(
                    "RootLocus block can only be connected to a Transfer Function."
                )

            # Check if the destination input port is already connected.
            # An exact-duplicate connection (same src+srcport+dst+dstport) is a
            # strict subset of this case, so a single pass over the destination
            # port covers both without emitting two overlapping messages.
            existing_lines = getattr(self.dsim, "line_list", [])
            end_name = getattr(end_block, "name", "")
            for line in existing_lines:
                if hasattr(line, "dstblock") and hasattr(line, "dstport"):
                    if line.dstblock == end_name and line.dstport == end_port:
                        validation_errors.append("Input port already connected")
                        break

            # Use ValidationHelper if available
            try:
                all_blocks = getattr(self.dsim, "blocks_list", [])
                all_lines = getattr(self.dsim, "line_list", [])
                # Create a temporary line list for validation
                temp_lines = list(all_lines)
                # Add our proposed connection for validation
                temp_line = types.SimpleNamespace(
                    srcblock=getattr(start_block, "name", ""),
                    srcport=start_port,
                    dstblock=getattr(end_block, "name", ""),
                    dstport=end_port,
                )
                temp_lines.append(temp_line)

                is_valid, helper_errors = ValidationHelper.validate_block_connections(
                    all_blocks, temp_lines
                )
                if not is_valid:
                    validation_errors.extend(helper_errors)
            except AttributeError as e:
                # Helper genuinely unavailable (method missing) — expected on
                # builds without the extended validator; keep it quiet.
                logger.debug(f"ValidationHelper not available: {str(e)}")
            except Exception as e:
                # The helper exists but raised while validating: that's a real
                # bug in the validator, not an absent feature. Surface it so a
                # broken validator isn't mistaken for a passing connection.
                logger.warning(f"ValidationHelper execution failed: {str(e)}")

            return len(validation_errors) == 0, validation_errors
        except Exception as e:
            logger.error(f"Error validating connection: {str(e)}")
            return False, [f"Validation error: {str(e)}"]

    # ==================== Line Creation ====================

    def finish_line_creation(self, end_block: Any, end_port: int) -> None:
        """Complete line creation between two blocks.

        ``end_block``/``end_port`` is the port the gesture ended on: an input
        port for a forward wire, an output port for a reverse one. The
        committed DLine always runs output -> input.
        """
        try:
            state = self.connection_state
            if state.reverse:
                src_block, src_port = end_block, end_port
                dst_block, dst_port = state.start_block, state.start_port
            else:
                src_block, src_port = state.start_block, state.start_port
                dst_block, dst_port = end_block, end_port
            src_name = getattr(src_block, "name", "Unknown")
            dst_name = getattr(dst_block, "name", "Unknown")
            logger.debug(f"Finishing line creation from {src_name} to {dst_name}")

            if hasattr(self.dsim, "add_line"):
                # Get coordinates for the line
                start_coords = None
                end_coords = None
                if hasattr(src_block, "out_coords") and src_port < len(src_block.out_coords):
                    start_coords = src_block.out_coords[src_port]
                if hasattr(dst_block, "in_coords") and dst_port < len(dst_block.in_coords):
                    end_coords = dst_block.in_coords[dst_port]

                if start_coords and end_coords:
                    # Validate connection before creating
                    is_valid, validation_errors = self.validate_connection(
                        src_block, src_port, dst_block, dst_port
                    )
                    if not is_valid:
                        error_msg = "\n".join(validation_errors)
                        logger.warning(f"Connection validation failed: {error_msg}")
                        self.canvas.simulation_status_changed.emit(
                            f"Connection invalid: {error_msg}"
                        )
                        self.cancel_line_creation()
                        return

                    # Push undo state before creating connection
                    self.canvas._push_undo("Connect")

                    # Create line using DSim's add_line method
                    new_line = self.dsim.add_line(
                        (src_name, src_port, start_coords),
                        (dst_name, dst_port, end_coords),
                    )
                    if new_line:
                        # Route the new wire for the default mode right away
                        # (DLine.__init__ has no block knowledge, so feedback
                        # and obstacle-avoiding routes need this pass).
                        self.route_line_for_mode(new_line, state.default_routing_mode)
                        logger.info(
                            f"Line created: {src_name} -> {dst_name} (routing: {state.default_routing_mode})"
                        )
                        # If Goto/From involved, relink to sync labels/virtual lines
                        if getattr(src_block, "block_fn", "") in ("Goto", "From") or getattr(
                            dst_block, "block_fn", ""
                        ) in ("Goto", "From"):
                            try:
                                self.dsim.model.link_goto_from()
                            except Exception as e:
                                logger.warning(f"Could not relink Goto/From after connection: {e}")
                        self.update_line_positions()
                        self.canvas.connection_created.emit(src_block, dst_block)
                    else:
                        logger.warning("Failed to create line")
                else:
                    logger.error("Could not get port coordinates for line creation")
            self.cancel_line_creation()
        except Exception as e:
            logger.error(f"Error in finish_line_creation: {str(e)}")
            self.cancel_line_creation()

    def route_line_for_mode(self, line: Any, mode: str) -> None:
        """Apply ``mode`` to ``line`` and compute its path from live block positions.

        Orthogonal wires go through the A* router (obstacle-avoiding, marked
        ``auto_routed`` so they follow their blocks); bezier wires get the
        default curve. Custom waypoints are discarded.
        """
        blocks = getattr(self.dsim, "blocks_list", [])
        line.set_routing_mode(mode if mode in ("bezier", "orthogonal") else "bezier")
        if line.routing_mode == "orthogonal":
            self.auto_route_line(line, push_undo=False)
        else:
            line.reroute(blocks)

    def auto_route_line(self, line: Any, push_undo: bool = True) -> bool:
        """Re-route one wire with the A* router. Returns True on success."""
        from lib.simulation.wire_router import route_all_lines

        blocks = getattr(self.dsim, "blocks_list", [])
        if push_undo:
            self.canvas._push_undo("Auto-route wire")
        routed = route_all_lines([line], blocks) > 0
        if not routed:
            line.reroute(blocks)
        self.dsim.dirty = True
        self.canvas.update()
        return routed

    def reset_line_routing(self, line: Any) -> None:
        """Drop manual bends / auto-routed waypoints and use the mode's default route."""
        self.canvas._push_undo("Reset wire routing")
        line.reset_routing(getattr(self.dsim, "blocks_list", []))
        self.dsim.dirty = True
        self.canvas.update()

    def remove_waypoint(self, line: Any, index: int) -> bool:
        """Delete an interior waypoint of a hand-bent wire. Returns True if removed."""
        if not (0 < index < len(line.points) - 1):
            return False
        self.canvas._push_undo("Remove wire bend")
        del line.points[index]
        blocks = getattr(self.dsim, "blocks_list", [])
        if len(line.points) > 2:
            line.mark_manual_edit()
            line.path, line.points, line.segments = line.create_trajectory(
                line.points[0], line.points[-1], blocks, points=line.points
            )
        else:
            line.reset_routing(blocks)
        line.selected_segment = -1
        self.dsim.dirty = True
        self.canvas.update()
        return True

    def cancel_line_creation(self) -> None:
        """Cancel line creation process."""
        from modern_ui.interactions.interaction_manager import State

        try:
            self.end_connection()
            self.canvas.state = State.IDLE
            self.canvas.update()
            logger.debug("Line creation cancelled")
        except Exception as e:
            logger.error(f"Error cancelling line creation: {str(e)}")

    # ==================== Line Click Detection ====================

    def get_clicked_line(self, pos: QPoint) -> Tuple[Optional[Any], Optional[Any]]:
        """Get the line at the given position."""
        for line in getattr(self.dsim, "line_list", []):
            if getattr(line, "hidden", False):
                continue
            if hasattr(line, "collision"):
                result = line.collision(pos)
                if result:
                    return line, result
        return None, None

    def handle_line_click(self, line: Any, collision_result: Tuple[str, int], pos: QPoint) -> None:
        """Handle clicking on a connection line."""
        from modern_ui.interactions.interaction_manager import State

        try:
            line_name = getattr(line, "name", "Unknown")
            logger.info(f"Line clicked: {line_name}")

            collision_type, collision_index = collision_result

            if not (QApplication.keyboardModifiers() & Qt.ControlModifier):
                self.canvas._clear_selections()

            line.selected = True
            line.modified = True  # Always allow modification on click

            if collision_type == "point":
                self.canvas.state = State.DRAGGING_LINE_POINT
                self.canvas.dragging_item = (line, collision_index)
                self.canvas.interaction_manager.drag.offset = pos
                self.canvas.interaction_manager.begin_line_bend(line)
                line.selected_segment = -1  # A point is selected, not a segment
                logger.info(f"Dragging point {collision_index} of line {line_name}")
            elif collision_type == "segment":
                self.canvas.state = State.DRAGGING_LINE_SEGMENT
                self.canvas.dragging_item = (line, collision_index)
                self.canvas.interaction_manager.drag.offset = pos
                self.canvas.interaction_manager.begin_line_bend(line)
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
        current_label = line.label if hasattr(line, "label") else ""

        # Show input dialog
        text, ok = QInputDialog.getText(
            self.canvas,
            "Edit Connection Label",
            f"Enter label for connection {line.srcblock} -> {line.dstblock}:",
            text=current_label,
        )

        if ok:
            line.label = str(text)
            self.canvas.update()
            logger.info(f"Updated connection label: {line.name} -> '{text}'")

    def set_connection_routing_mode(self, line: Any, mode: str) -> None:
        """Change the routing mode for a connection and re-route it."""
        if mode in ["bezier", "orthogonal"]:
            self.canvas._push_undo("Change wire routing")
            self.route_line_for_mode(line, mode)
            self.dsim.dirty = True
            self.canvas.update()
            logger.info(f"Changed routing mode for {line.name} to {mode}")

    def update_line_positions(self) -> None:
        """Update line positions after block movement.

        This replaces DSim.update_lines() - line position logic belongs in canvas.
        """
        for line in self.dsim.line_list:
            line.update_line(self.dsim.blocks_list)
