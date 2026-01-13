
import logging
from enum import Enum
from PyQt5.QtCore import Qt, QPoint, QPointF
from PyQt5.QtWidgets import QApplication

logger = logging.getLogger(__name__)

class State(Enum):
    """State enumeration for canvas interactions."""
    IDLE = "idle"
    DRAGGING = "dragging"
    DRAGGING_BLOCK = "dragging_block"
    DRAGGING_LINE_POINT = "dragging_line_point"
    DRAGGING_LINE_SEGMENT = "dragging_line_segment"
    CONNECTING = "connecting"
    CONFIGURING = "configuring"
    RESIZING = "resizing"

class InteractionManager:
    """
    Manages mouse interactions and state for the ModernCanvas.
    Decouples event handling from rendering logic.
    """

    def __init__(self, canvas):
        self.canvas = canvas
        # self.state is now a property delegating to canvas
        
        # Interaction context
        self.drag_start_pos = QPoint()
        self.last_mouse_pos = QPoint()
        
        # Connection context
        self.connection_start_block = None
        self.connection_start_port = -1
        self.connection_end_block = None
        self.connection_end_port = -1
        self.temp_connection_points = []
        
        # Dragging context
        self.dragging_block = None
        self.dragging_line = None
        self.dragging_segment_index = -1
        self.dragging_point_index = -1
        self.drag_offset = QPointF(0, 0)
        
        # Resizing context
        self.resizing_block = None
        self.resize_handle = None
        self.original_block_rect = None

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

    def handle_mouse_press(self, event):
        """Handle mouse press events from the canvas."""
        if event.button() == Qt.MiddleButton:
            self.canvas.panning = True
            self.canvas.last_pan_pos = event.pos()
            self.canvas.setCursor(Qt.ClosedHandCursor)
            return

        pos = self.canvas.screen_to_world(event.pos())
        self.last_mouse_pos = pos
        self.drag_start_pos = pos
        
        if event.button() == Qt.LeftButton:
            self._handle_left_click(pos, event.modifiers())
        elif event.button() == Qt.RightButton:
            self.canvas._handle_right_click(pos) # Delegate back for context menu for now
            
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
        if self.canvas.line_creation_state:
            self.canvas._cancel_line_creation()
        else:
            # 6. Rectangle Selection
            # Start rectangle selection
            self.canvas.is_rect_selecting = True
            self.canvas.selection_rect_start = pos
            self.canvas.selection_rect_end = pos

            # Clear existing selection unless Shift is held
            if not (modifiers & Qt.ShiftModifier):
                self.canvas._clear_selections()

            logger.debug(f"Started rectangle selection at ({pos.x()}, {pos.y()})")

    def handle_mouse_move(self, event):
        """Handle mouse move events with hover detection."""
        try:
            if self.canvas.panning:
                delta = event.pos() - self.canvas.last_pan_pos
                self.canvas.pan_offset += delta
                self.canvas.last_pan_pos = event.pos()
                self.canvas.update()
                return

            pos = self.canvas.screen_to_world(event.pos())

            # Update hover states (only when not dragging/selecting)
            if self.canvas.state == State.IDLE and not self.canvas.is_rect_selecting:
                self.canvas._update_hover_states(pos)

            # Update rectangle selection
            if self.canvas.is_rect_selecting:
                self.canvas.selection_rect_end = pos
                self.canvas.update()
                return

            if self.canvas.state == State.DRAGGING and self.canvas.dragging_block:
                # Update drag
                new_x = pos.x() - self.canvas.drag_offset.x()
                new_y = pos.y() - self.canvas.drag_offset.y()

                # Snap to grid
                if getattr(self.canvas, 'snap_enabled', True):
                    snapped_x = round(new_x / self.canvas.grid_size) * self.canvas.grid_size
                    snapped_y = round(new_y / self.canvas.grid_size) * self.canvas.grid_size
                else:
                    snapped_x = new_x
                    snapped_y = new_y

                # Move clicked block
                self.canvas.dragging_block.relocate_Block(QPoint(int(snapped_x), int(snapped_y)))

                # Move other selected blocks relative to it
                if hasattr(self.canvas, 'drag_offsets') and len(self.canvas.drag_offsets) > 1:
                    for block, relative_offset in self.canvas.drag_offsets.items():
                        if block is not self.canvas.dragging_block:
                            block_x = snapped_x + relative_offset.x()
                            block_y = snapped_y + relative_offset.y()
                            block.relocate_Block(QPoint(int(block_x), int(block_y)))

                self.canvas._update_line_positions() 
                self.canvas.update()
                
            elif self.canvas.state == State.RESIZING and self.canvas.resizing_block:
                self.canvas._perform_resize(pos)
                
            elif self.canvas.state == State.DRAGGING_LINE_POINT and self.canvas.dragging_item:
                line, point_index = self.canvas.dragging_item
                line.points[point_index] = pos
                line.path, line.points, line.segments = line.create_trajectory(line.points[0], line.points[-1], self.canvas.dsim.blocks_list, line.points)
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
                    else:
                        new_point = QPoint(int(pos.x()), int(p1.y() + (p2.y() - p1.y()) // 2))
                    line.points.insert(1, new_point)
                    self.canvas.dragging_item = (line, 1 if pos.y() > p1.y() else 0)
                    segment_index = self.canvas.dragging_item[1]

                if is_horizontal:
                    is_first_segment = (segment_index == 0)
                    is_last_segment = (segment_index == len(line.points) - 2)
                    if not is_first_segment:
                        line.points[segment_index].setY(pos.y())
                    if not is_last_segment:
                        line.points[segment_index + 1].setY(pos.y())
                else:
                    is_first_segment = (segment_index == 0)
                    is_last_segment = (segment_index == len(line.points) - 2)
                    if not is_first_segment:
                        line.points[segment_index].setX(pos.x())
                    if not is_last_segment:
                        line.points[segment_index + 1].setX(pos.x())
                
                line.path, line.points, line.segments = line.create_trajectory(line.points[0], line.points[-1], self.canvas.dsim.blocks_list, line.points)
                self.canvas.update()
                
            elif self.canvas.line_creation_state == 'start' and self.canvas.temp_line:
                self.canvas.temp_line = (self.canvas.temp_line[0], pos)
                self.canvas.update()
        except Exception as e:
            logger.error(f"Error in handle_mouse_move: {str(e)}")

    def handle_mouse_release(self, event):
        """Handle mouse release events."""
        try:
            if event.button() == Qt.MiddleButton:
                self.canvas.panning = False
                self.canvas.setCursor(Qt.ArrowCursor)

            # Finalize rectangle selection
            if self.canvas.is_rect_selecting and event.button() == Qt.LeftButton:
                self.canvas._finalize_rect_selection()
                return

            if self.canvas.state == State.DRAGGING:
                self.canvas._finish_drag()
            elif self.canvas.state == State.RESIZING:
                self.canvas._finish_resize()
            elif self.canvas.state in [State.DRAGGING_LINE_POINT, State.DRAGGING_LINE_SEGMENT]:
                if self.canvas.dragging_item:
                    line, _ = self.canvas.dragging_item
                    if hasattr(line, '_stub_created'):
                        del line._stub_created
                self.canvas.state = State.IDLE
                self.canvas.dragging_item = None
                self.canvas.update()

        except Exception as e:
            logger.error(f"Error in handle_mouse_release: {str(e)}")
