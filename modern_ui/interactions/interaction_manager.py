import logging
from enum import Enum
from PyQt5.QtCore import Qt, QPoint, QRect

from modern_ui.renderers.canvas_renderer import compute_alignment_guides
from modern_ui.widgets.canvas_state import (
    DragState,
    HoverState,
    ResizeState,
    SelectionState,
)

logger = logging.getLogger(__name__)

# Screen-space snap tolerance for smart-alignment guides. Divided by the zoom
# factor before use so the snap feels equally "sticky" at any zoom level.
_ALIGNMENT_THRESHOLD_PX = 4


class State(Enum):
    """State enumeration for canvas interactions."""

    IDLE = "idle"
    DRAGGING = "dragging"
    DRAGGING_LINE_POINT = "dragging_line_point"
    DRAGGING_LINE_SEGMENT = "dragging_line_segment"
    CONNECTING = "connecting"
    RESIZING = "resizing"


class InteractionManager:
    """
    Manages mouse interactions and state for the ModernCanvas, including the
    block drag + resize state machine (formerly a separate DragResizeManager —
    the two were one gesture pipeline split across two files).
    Decouples event handling from rendering logic.
    """

    def __init__(self, canvas):
        self.canvas = canvas
        # Transient gesture state is owned here; callers reach it as
        # ``canvas.interaction_manager.selection/hover/drag/resize`` (there is no
        # longer a flat canvas proxy layer). The interaction FSM enum still lives
        # on the canvas as ``canvas.state``, exposed below via ``self.state``.
        self.selection = SelectionState()
        self.hover = HoverState()
        self.drag = DragState()
        self.resize = ResizeState()

    @property
    def state(self):
        return self.canvas.state

    @state.setter
    def state(self, value):
        self.canvas.state = value

    def set_state(self, new_state):
        """Transition to a new state."""
        logger.debug(f"State transition: {self.state} -> {new_state}")
        self.state = new_state

    def reset_gesture_state(self):
        """Clear all transient gesture state (selection/hover/drag/resize).

        This is the single place a full gesture reset happens now that the
        gesture slices are owned here rather than on a shared canvas state.
        """
        self.selection.clear()
        self.hover.clear()
        self.drag.end_drag()
        self.resize.end_resize()

    def clear_hover(self):
        """Clear hover tracking (block/port/line).

        Mirrors the side effect of the canvas ``hovered_port`` proxy setter by
        re-evaluating the idle-glow animation gate once the hover clears.
        """
        self.hover.clear()
        self.canvas._evaluate_animation_state()

    def set_hovered_port(self, value):
        """Set the hovered port and re-evaluate the idle-glow animation gate.

        The hovered port is one of the states that gates the canvas glow timer,
        so writing it must re-evaluate that gate (this replaces the side effect
        the old canvas ``hovered_port`` proxy setter carried).
        """
        self.hover.port = value
        self.canvas._evaluate_animation_state()

    def handle_mouse_press(self, event):
        """Handle mouse press events from the canvas."""
        if event.button() == Qt.MiddleButton:
            zp = self.canvas.zoom_pan_manager.state
            zp.is_panning = True
            zp.last_pan_pos = event.pos()
            self.canvas.setCursor(Qt.ClosedHandCursor)
            return

        pos = self.canvas.screen_to_world(event.pos())

        if event.button() == Qt.LeftButton:
            self._handle_left_click(pos, event.modifiers())
        elif event.button() == Qt.RightButton:
            self.canvas._handle_right_click(pos)  # Delegate back for context menu for now

        self.canvas.update()

    def _handle_left_click(self, pos, modifiers):
        """Internal handler for left clicks."""

        # 1. Check for resize handles FIRST on selected blocks
        for block in self.canvas.dsim.blocks_list:
            if block.selected:
                handle = self.canvas.block_renderer.get_resize_handle_at(block, pos)
                if handle:
                    self.canvas._start_resize(block, handle, pos)
                    return

        # 2. Check for port clicks (before block clicks) -- Delegating to canvas to keep complex logic there for now
        # Ideally this logic moves here too, but step-by-step
        if self.canvas._check_port_clicks(pos):
            return

        # 3. Check for Block Clicks
        clicked_block = self.canvas._get_clicked_block(pos)
        if clicked_block:
            self.canvas._handle_block_click(clicked_block, pos)
            return

        # 4. Check for Line Clicks
        clicked_line, collision_result = self.canvas._get_clicked_line(pos)
        if clicked_line:
            self.canvas._handle_line_click(clicked_line, collision_result, pos)
            return

        # 5. Cancel any ongoing line creation if clicking on empty area
        if self.canvas.connection_manager.connection_state.creation_state:
            self.canvas._cancel_line_creation()
        else:
            # 6. Rectangle Selection
            # Start rectangle selection
            self.selection.is_selecting = True
            self.selection.rect_start = pos
            self.selection.rect_end = pos

            # Clear existing selection unless Shift is held
            if not (modifiers & Qt.ShiftModifier):
                self.canvas._clear_selections()

            logger.debug(f"Started rectangle selection at ({pos.x()}, {pos.y()})")

    def handle_mouse_move(self, event):
        """Handle mouse move events with hover detection."""
        try:
            zp = self.canvas.zoom_pan_manager.state
            if zp.is_panning:
                delta = event.pos() - zp.last_pan_pos
                zp.pan_offset += delta
                zp.last_pan_pos = event.pos()
                self.canvas.update()
                return

            pos = self.canvas.screen_to_world(event.pos())

            # Update hover states (only when not dragging/selecting)
            if self.canvas.state == State.IDLE and not self.selection.is_selecting:
                self.canvas._update_hover_states(pos)

            # Update rectangle selection
            if self.selection.is_selecting:
                self.selection.rect_end = pos
                self.canvas.update()
                return

            if self.canvas.state == State.DRAGGING and self.canvas.dragging_block:
                # Update drag
                new_x = pos.x() - self.drag.offset.x()
                new_y = pos.y() - self.drag.offset.y()

                # Snap to grid
                if getattr(self.canvas, "snap_enabled", True):
                    snapped_x = round(new_x / self.canvas.grid_size) * self.canvas.grid_size
                    snapped_y = round(new_y / self.canvas.grid_size) * self.canvas.grid_size
                else:
                    snapped_x = new_x
                    snapped_y = new_y

                # Move clicked block
                self.canvas.dragging_block.relocate_Block(QPoint(int(snapped_x), int(snapped_y)))

                # Move other selected blocks relative to it
                if len(self.drag.offsets) > 1:
                    for block, relative_offset in self.drag.offsets.items():
                        if block is not self.canvas.dragging_block:
                            block_x = snapped_x + relative_offset.x()
                            block_y = snapped_y + relative_offset.y()
                            block.relocate_Block(QPoint(int(block_x), int(block_y)))

                self.canvas._update_line_positions()
                self.canvas.update()

            elif self.canvas.state == State.RESIZING and self.resize.block:
                self.canvas._perform_resize(pos)

            elif self.canvas.state == State.DRAGGING_LINE_POINT and self.canvas.dragging_item:
                line, point_index = self.canvas.dragging_item
                line.points[point_index] = pos
                line.path, line.points, line.segments = line.create_trajectory(
                    line.points[0], line.points[-1], self.canvas.dsim.blocks_list, line.points
                )
                self.canvas.update()

            elif self.canvas.state == State.DRAGGING_LINE_SEGMENT and self.canvas.dragging_item:
                line, segment_index = self.canvas.dragging_item
                p1 = line.points[segment_index]
                p2 = line.points[segment_index + 1]
                is_horizontal = abs(p1.x() - p2.x()) > abs(p1.y() - p2.y())

                # Split segment logic
                if len(line.points) == 2:
                    if is_horizontal:
                        new_point = QPoint(int(p1.x() + (p2.x() - p1.x()) // 2), int(pos.y()))
                        new_index = 1 if pos.y() > p1.y() else 0
                    else:
                        new_point = QPoint(int(pos.x()), int(p1.y() + (p2.y() - p1.y()) // 2))
                        new_index = 1 if pos.x() > p1.x() else 0
                    line.points.insert(1, new_point)
                    self.canvas.dragging_item = (line, new_index)
                    segment_index = new_index

                # Defensive guard: dragging_item may originate elsewhere with a
                # stale index after points were inserted/removed.
                if not (0 <= segment_index < len(line.points) - 1):
                    return

                if is_horizontal:
                    is_first_segment = segment_index == 0
                    is_last_segment = segment_index == len(line.points) - 2
                    if not is_first_segment:
                        line.points[segment_index].setY(pos.y())
                    if not is_last_segment:
                        line.points[segment_index + 1].setY(pos.y())
                else:
                    is_first_segment = segment_index == 0
                    is_last_segment = segment_index == len(line.points) - 2
                    if not is_first_segment:
                        line.points[segment_index].setX(pos.x())
                    if not is_last_segment:
                        line.points[segment_index + 1].setX(pos.x())

                line.path, line.points, line.segments = line.create_trajectory(
                    line.points[0], line.points[-1], self.canvas.dsim.blocks_list, line.points
                )
                self.canvas.update()

            elif (
                self.canvas.connection_manager.connection_state.creation_state == "start"
                and self.canvas.connection_manager.connection_state.temp_line
            ):
                conn_state = self.canvas.connection_manager.connection_state
                conn_state.temp_line = (conn_state.temp_line[0], pos)
                self.canvas.update()
        except Exception as e:
            # Reset interaction state so a failure mid-drag does not leave the
            # canvas stuck in a dragging/resizing state.
            logger.error(f"Error in handle_mouse_move: {str(e)}", exc_info=True)
            self.canvas.state = State.IDLE
            self.canvas.dragging_item = None

    def handle_mouse_release(self, event):
        """Handle mouse release events."""
        try:
            if event.button() == Qt.MiddleButton:
                self.canvas.zoom_pan_manager.state.is_panning = False
                self.canvas.setCursor(Qt.ArrowCursor)

            # Finalize rectangle selection
            if self.selection.is_selecting and event.button() == Qt.LeftButton:
                self.canvas._finalize_rect_selection()
                return

            if self.canvas.state == State.DRAGGING:
                self.canvas._finish_drag()
            elif self.canvas.state == State.RESIZING:
                self.canvas._finish_resize()
            elif self.canvas.state in [State.DRAGGING_LINE_POINT, State.DRAGGING_LINE_SEGMENT]:
                if self.canvas.dragging_item:
                    line, _ = self.canvas.dragging_item
                    if hasattr(line, "_stub_created"):
                        del line._stub_created
                self.canvas.state = State.IDLE
                self.canvas.dragging_item = None
                self.canvas.update()

        except Exception as e:
            # Reset interaction state so a failure during release does not leave
            # the canvas stuck mid-drag.
            logger.error(f"Error in handle_mouse_release: {str(e)}", exc_info=True)
            self.canvas.state = State.IDLE
            self.canvas.dragging_item = None

    # ==================== Block drag & resize ====================
    # Persistent drag/resize state lives in self.drag / self.resize (owned here);
    # the canvas re-exposes it through its drag_*/resize_* property proxies.

    def start_drag(self, block, pos):
        """Start dragging a block (or multiple selected blocks)."""
        try:
            self.canvas.state = State.DRAGGING
            self.canvas.dragging_block = block
            # Calculate drag offset based on block's top-left corner
            self.drag.offset = QPoint(pos.x() - block.left, pos.y() - block.top)

            # Store RELATIVE offsets from the clicked block to all other selected blocks
            # This maintains relative positions when dragging multiple blocks
            self.drag.offsets = {}
            self.drag.start_positions = {}  # Track starting positions for undo threshold
            for b in self.canvas.dsim.blocks_list:
                if b.selected:
                    # Store offset from clicked block to this block
                    self.drag.offsets[b] = QPoint(b.left - block.left, b.top - block.top)
                    # Store starting position
                    self.drag.start_positions[b] = (b.left, b.top)

            # Smart-alignment guides are recomputed on every move; start clean.
            self.canvas._alignment_guides = []

            logger.debug(f"Started dragging {len(self.drag.offsets)} block(s)")
        except Exception as e:
            logger.error(f"Error starting drag: {str(e)}")

    def update_drag_alignment(self):
        """Stash the active alignment guides and (when grid snap is off) nudge.

        Called once per drag move (after handle_mouse_move has relocated the
        block to the cursor) so the snap visibly nudges the block onto a
        shared edge. Computes purely from canvas-space rects via
        ``compute_alignment_guides``; the resulting guide lines are stashed on
        the canvas for ``paintEvent`` so they DISPLAY in every drag.

        The snap delta is only APPLIED when grid snap is OFF: there it relocates
        the dragged block onto the shared edge and shifts any co-selected blocks
        by the same delta so the group moves rigidly. When grid snap is ON the
        move handler has already grid-quantized the block before we run,
        so the guides act as pure visual feedback (they light up when the
        grid-aligned edges happen to line up with a neighbour) and we skip the
        nudge — otherwise the two snaps would fight and could knock the block
        off the grid.
        """
        try:
            block = self.canvas.dragging_block
            if block is None:
                return

            self.canvas._alignment_guides = []

            moving_rect = (block.left, block.top, block.width, block.height)
            other_rects = [
                (b.left, b.top, b.width, b.height)
                for b in self.canvas.dsim.blocks_list
                if b is not block and b not in self.drag.offsets
            ]
            if not other_rects:
                return

            # Keep the snap tolerance constant in screen pixels across zoom.
            zoom = self.canvas.zoom_pan_manager.state.zoom_factor or 1.0
            threshold = _ALIGNMENT_THRESHOLD_PX / zoom

            snap_dx, snap_dy, guide_lines = compute_alignment_guides(
                moving_rect, other_rects, threshold=threshold
            )

            # Apply the nudge only when grid snap is off; with grid snap on the
            # block must stay on the grid, so the guides are feedback-only.
            grid_snap_on = getattr(self.canvas, "snap_enabled", True)
            if not grid_snap_on and (snap_dx or snap_dy):
                # Move the dragged block onto the snapped edge, then shift the
                # rest of the selection by the same delta to stay rigid.
                block.relocate_Block(QPoint(int(block.left + snap_dx), int(block.top + snap_dy)))
                if len(self.drag.offsets) > 1:
                    for b in self.drag.offsets:
                        if b is not block:
                            b.relocate_Block(QPoint(int(b.left + snap_dx), int(b.top + snap_dy)))

            self.canvas._alignment_guides = guide_lines
        except Exception as e:
            # Alignment is a cosmetic assist; never let it break the drag.
            logger.debug(f"Alignment guide update failed: {e}")
            self.canvas._alignment_guides = []

    def _start_resize(self, block, handle, pos):
        """Start resizing a block."""
        try:
            self.canvas.state = State.RESIZING
            self.resize.block = block
            self.resize.handle = handle
            self.resize.start_pos = pos
            self.resize.start_rect = QRect(block.left, block.top, block.width, block.height)

            logger.debug(f"Started resizing block {block.name} from handle {handle}")
        except Exception as e:
            logger.error(f"Error starting resize: {str(e)}")

    def _perform_resize(self, pos):
        """Perform the resize operation based on current mouse position."""
        try:
            if not self.resize.block or not self.resize.handle:
                return

            block = self.resize.block
            handle = self.resize.handle
            start_rect = self.resize.start_rect

            # Calculate delta from start position
            delta_x = pos.x() - self.resize.start_pos.x()
            delta_y = pos.y() - self.resize.start_pos.y()

            # Calculate new position and size based on handle
            new_left = start_rect.left()
            new_top = start_rect.top()
            new_width = start_rect.width()
            new_height = start_rect.height()

            if "left" in handle:
                new_left = start_rect.left() + delta_x
                new_width = start_rect.width() - delta_x
            elif "right" in handle:
                new_width = start_rect.width() + delta_x

            if "top" in handle:
                new_top = start_rect.top() + delta_y
                new_height = start_rect.height() - delta_y
            elif "bottom" in handle:
                new_height = start_rect.height() + delta_y

            # Apply minimum size constraints
            try:
                from config.block_sizes import MIN_BLOCK_WIDTH, MIN_BLOCK_HEIGHT

                min_width = MIN_BLOCK_WIDTH
                min_height = MIN_BLOCK_HEIGHT
            except ImportError:
                min_width = 50
                min_height = 40

            # Also check block's port-based minimum height (for multi-port blocks)
            if hasattr(block, "calculate_min_size"):
                port_min_height = block.calculate_min_size()
                min_height = max(min_height, port_min_height)

            # Track if we're hitting the resize limit
            at_width_limit = new_width <= min_width
            at_height_limit = new_height <= min_height

            # Ensure minimum size
            if new_width < min_width:
                if "left" in handle:
                    new_left = start_rect.right() - min_width
                new_width = min_width

            if new_height < min_height:
                if "top" in handle:
                    new_top = start_rect.bottom() - min_height
                new_height = min_height

            # Visual feedback: change cursor when at limit
            if at_width_limit or at_height_limit:
                self.canvas.setCursor(Qt.ForbiddenCursor)
                self.resize.at_limit = True
            else:
                # Restore appropriate resize cursor
                cursor_map = {
                    "top_left": Qt.SizeFDiagCursor,
                    "top_right": Qt.SizeBDiagCursor,
                    "bottom_left": Qt.SizeBDiagCursor,
                    "bottom_right": Qt.SizeFDiagCursor,
                    "top": Qt.SizeVerCursor,
                    "bottom": Qt.SizeVerCursor,
                    "left": Qt.SizeHorCursor,
                    "right": Qt.SizeHorCursor,
                }
                self.canvas.setCursor(cursor_map.get(handle, Qt.ArrowCursor))
                self.resize.at_limit = False

            # Update block position and size
            block.left = new_left
            block.top = new_top
            block.resize_Block(new_width, new_height)
            block.rect.moveTo(new_left, new_top)

            # Update connected lines
            self.canvas._update_line_positions()
            self.canvas.update()

        except Exception as e:
            logger.error(f"Error performing resize: {str(e)}")

    def _finish_drag(self):
        """Finish dragging operation."""
        try:
            if self.canvas.dragging_block:
                logger.debug(
                    f"Finished dragging block: {getattr(self.canvas.dragging_block, 'fn_name', 'Unknown')}"
                )

                # Only push undo if blocks actually moved significantly (threshold: 5 pixels)
                moved_significantly = False
                move_threshold = 5  # pixels

                for block, start_pos in self.drag.start_positions.items():
                    start_left, start_top = start_pos
                    distance = abs(block.left - start_left) + abs(block.top - start_top)
                    if distance >= move_threshold:
                        moved_significantly = True
                        break

                if moved_significantly:
                    self.canvas._push_undo("Move")
                    moved_block_names = {b.name for b in self.drag.start_positions}
                else:
                    logger.debug("Block moved less than threshold, not capturing undo")

                # Reset drag state
                self.canvas.state = State.IDLE
                self.canvas.dragging_block = None
                self.drag.offset = None
                self.drag.start_positions = {}
                self.canvas._alignment_guides = []  # clear smart-alignment overlay
                self.canvas._update_line_positions()
                if moved_significantly:
                    self.canvas._reroute_affected_lines(moved_block_names)
                self.canvas.update()
        except Exception as e:
            logger.error(f"Error finishing drag: {str(e)}")

    def _finish_resize(self):
        """Finish resizing operation."""
        try:
            if self.resize.block and self.resize.start_rect:
                logger.debug(
                    f"Finished resizing block: {getattr(self.resize.block, 'fn_name', 'Unknown')}"
                )

                # Only push undo if block actually resized significantly (threshold: 5 pixels)
                block = self.resize.block
                start_rect = self.resize.start_rect
                resize_threshold = 5  # pixels

                # Check if size or position changed significantly
                size_change = abs(block.width - start_rect.width()) + abs(
                    block.height - start_rect.height()
                )
                pos_change = abs(block.left - start_rect.left()) + abs(block.top - start_rect.top())

                resized_significantly = (
                    size_change >= resize_threshold or pos_change >= resize_threshold
                )
                if resized_significantly:
                    self.canvas._push_undo("Resize")
                else:
                    logger.debug("Block resized less than threshold, not capturing undo")

                resized_name = block.name

                # Reset resize state
                self.canvas.state = State.IDLE
                self.resize.block = None
                self.resize.handle = None
                self.resize.start_rect = None
                self.resize.start_pos = None
                self.resize.at_limit = False
                self.canvas.setCursor(Qt.ArrowCursor)

                # Ensure lines are updated after resize
                self.canvas._update_line_positions()
                if resized_significantly:
                    self.canvas._reroute_affected_lines({resized_name})
                self.canvas.update()
        except Exception as e:
            logger.error(f"Error finishing resize: {str(e)}")
