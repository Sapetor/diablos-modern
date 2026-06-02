"""
Drag and Resize Manager for ModernCanvas.

Owns the block drag + resize state machine that was previously inlined in
``modern_ui/widgets/modern_canvas.py``. The canvas keeps thin facade methods
(``start_drag``, ``_start_resize``, ``_perform_resize``, ``_finish_drag``,
``_finish_resize``) that delegate here, so external callers
(InteractionManager, the canvas mouse handlers) are unchanged.

All persistent state lives on the canvas (drag/resize properties backed by
``canvas_state``); this manager reads and writes it through ``self.canvas``.
"""

import logging
from PyQt5.QtCore import Qt, QPoint, QRect

from modern_ui.interactions.interaction_manager import State

logger = logging.getLogger(__name__)


class DragResizeManager:
    """
    Manages the block drag and resize interactions for the canvas.
    """
    def __init__(self, canvas):
        self.canvas = canvas

    def start_drag(self, block, pos):
        """Start dragging a block (or multiple selected blocks)."""
        try:
            self.canvas.state = State.DRAGGING
            self.canvas.dragging_block = block
            # Calculate drag offset based on block's top-left corner
            self.canvas.drag_offset = QPoint(pos.x() - block.left, pos.y() - block.top)

            # Store RELATIVE offsets from the clicked block to all other selected blocks
            # This maintains relative positions when dragging multiple blocks
            self.canvas.drag_offsets = {}
            self.canvas.drag_start_positions = {}  # Track starting positions for undo threshold
            for b in self.canvas.dsim.blocks_list:
                if b.selected:
                    # Store offset from clicked block to this block
                    self.canvas.drag_offsets[b] = QPoint(b.left - block.left, b.top - block.top)
                    # Store starting position
                    self.canvas.drag_start_positions[b] = (b.left, b.top)

            logger.debug(f"Started dragging {len(self.canvas.drag_offsets)} block(s)")
        except Exception as e:
            logger.error(f"Error starting drag: {str(e)}")

    def _start_resize(self, block, handle, pos):
        """Start resizing a block."""
        try:
            self.canvas.state = State.RESIZING
            self.canvas.resizing_block = block
            self.canvas.resize_handle = handle
            self.canvas.resize_start_pos = pos
            self.canvas.resize_start_rect = QRect(block.left, block.top, block.width, block.height)

            logger.debug(f"Started resizing block {block.name} from handle {handle}")
        except Exception as e:
            logger.error(f"Error starting resize: {str(e)}")

    def _perform_resize(self, pos):
        """Perform the resize operation based on current mouse position."""
        try:
            if not self.canvas.resizing_block or not self.canvas.resize_handle:
                return

            block = self.canvas.resizing_block
            handle = self.canvas.resize_handle
            start_rect = self.canvas.resize_start_rect

            # Calculate delta from start position
            delta_x = pos.x() - self.canvas.resize_start_pos.x()
            delta_y = pos.y() - self.canvas.resize_start_pos.y()

            # Calculate new position and size based on handle
            new_left = start_rect.left()
            new_top = start_rect.top()
            new_width = start_rect.width()
            new_height = start_rect.height()

            if 'left' in handle:
                new_left = start_rect.left() + delta_x
                new_width = start_rect.width() - delta_x
            elif 'right' in handle:
                new_width = start_rect.width() + delta_x

            if 'top' in handle:
                new_top = start_rect.top() + delta_y
                new_height = start_rect.height() - delta_y
            elif 'bottom' in handle:
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
            if hasattr(block, 'calculate_min_size'):
                port_min_height = block.calculate_min_size()
                min_height = max(min_height, port_min_height)

            # Track if we're hitting the resize limit
            at_width_limit = new_width <= min_width
            at_height_limit = new_height <= min_height

            # Ensure minimum size
            if new_width < min_width:
                if 'left' in handle:
                    new_left = start_rect.right() - min_width
                new_width = min_width

            if new_height < min_height:
                if 'top' in handle:
                    new_top = start_rect.bottom() - min_height
                new_height = min_height

            # Visual feedback: change cursor when at limit
            if at_width_limit or at_height_limit:
                self.canvas.setCursor(Qt.ForbiddenCursor)
                self.canvas.resize_at_limit = True
            else:
                # Restore appropriate resize cursor
                cursor_map = {
                    'top_left': Qt.SizeFDiagCursor,
                    'top_right': Qt.SizeBDiagCursor,
                    'bottom_left': Qt.SizeBDiagCursor,
                    'bottom_right': Qt.SizeFDiagCursor,
                    'top': Qt.SizeVerCursor,
                    'bottom': Qt.SizeVerCursor,
                    'left': Qt.SizeHorCursor,
                    'right': Qt.SizeHorCursor,
                }
                self.canvas.setCursor(cursor_map.get(handle, Qt.ArrowCursor))
                self.canvas.resize_at_limit = False

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
                logger.debug(f"Finished dragging block: {getattr(self.canvas.dragging_block, 'fn_name', 'Unknown')}")

                # Only push undo if blocks actually moved significantly (threshold: 5 pixels)
                moved_significantly = False
                move_threshold = 5  # pixels

                for block, start_pos in self.canvas.drag_start_positions.items():
                    start_left, start_top = start_pos
                    distance = abs(block.left - start_left) + abs(block.top - start_top)
                    if distance >= move_threshold:
                        moved_significantly = True
                        break

                if moved_significantly:
                    self.canvas._push_undo("Move")
                    moved_block_names = {b.name for b in self.canvas.drag_start_positions}
                else:
                    logger.debug("Block moved less than threshold, not capturing undo")

                # Reset drag state
                self.canvas.state = State.IDLE
                self.canvas.dragging_block = None
                self.canvas.drag_offset = None
                self.canvas.drag_start_positions = {}
                self.canvas._update_line_positions()
                if moved_significantly:
                    self.canvas._reroute_affected_lines(moved_block_names)
                self.canvas.update()
        except Exception as e:
            logger.error(f"Error finishing drag: {str(e)}")

    def _finish_resize(self):
        """Finish resizing operation."""
        try:
            if self.canvas.resizing_block and self.canvas.resize_start_rect:
                logger.debug(f"Finished resizing block: {getattr(self.canvas.resizing_block, 'fn_name', 'Unknown')}")

                # Only push undo if block actually resized significantly (threshold: 5 pixels)
                block = self.canvas.resizing_block
                start_rect = self.canvas.resize_start_rect
                resize_threshold = 5  # pixels

                # Check if size or position changed significantly
                size_change = (abs(block.width - start_rect.width()) +
                              abs(block.height - start_rect.height()))
                pos_change = (abs(block.left - start_rect.left()) +
                             abs(block.top - start_rect.top()))

                resized_significantly = size_change >= resize_threshold or pos_change >= resize_threshold
                if resized_significantly:
                    self.canvas._push_undo("Resize")
                else:
                    logger.debug("Block resized less than threshold, not capturing undo")

                resized_name = block.name

                # Reset resize state
                self.canvas.state = State.IDLE
                self.canvas.resizing_block = None
                self.canvas.resize_handle = None
                self.canvas.resize_start_rect = None
                self.canvas.resize_start_pos = None
                self.canvas.resize_at_limit = False
                self.canvas.setCursor(Qt.ArrowCursor)

                # Ensure lines are updated after resize
                self.canvas._update_line_positions()
                if resized_significantly:
                    self.canvas._reroute_affected_lines({resized_name})
                self.canvas.update()
        except Exception as e:
            logger.error(f"Error finishing resize: {str(e)}")
