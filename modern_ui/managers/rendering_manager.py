"""
Rendering Manager for DiaBloS Modern Canvas.
Orchestrates rendering of blocks, connections, and visual indicators.
"""

import logging
from typing import Optional, Set, List, Any, TYPE_CHECKING
from PyQt5.QtCore import QRect, QPoint, Qt
from PyQt5.QtGui import QPainter, QPen, QColor
from PyQt5.QtWidgets import QToolTip

if TYPE_CHECKING:
    from modern_ui.widgets.modern_canvas import ModernCanvas

logger = logging.getLogger(__name__)


class RenderingManager:
    """
    Manages rendering orchestration and visual feedback on the canvas.
    Extracted from ModernCanvas to reduce file size and improve maintainability.
    """

    def __init__(self, canvas: 'ModernCanvas'):
        self.canvas = canvas
        self.dsim = canvas.dsim

    # ==================== Block Rendering ====================

    def render_blocks(self, painter: QPainter, draw_ports: bool = True) -> None:
        """Render all blocks to canvas."""
        if painter is None:
            return
        for block in self.dsim.blocks_list:
            self.canvas.block_renderer.draw_block(block, painter, draw_ports=draw_ports)
            if block.selected:
                self.canvas.block_renderer.draw_resize_handles(block, painter)

    def render_lines(self, painter: QPainter) -> None:
        """Render all connection lines."""
        if painter is None:
            return
        for line in self.dsim.line_list:
            if not getattr(line, "hidden", False):
                self.canvas.connection_renderer.draw_line(line, painter)

    def render_ports(self, painter: QPainter) -> None:
        """Render all ports on top of lines for better visibility."""
        for block in self.dsim.blocks_list:
            self.canvas.block_renderer.draw_ports(block, painter)
            self.canvas.block_renderer.draw_port_labels(block, painter)

    # ==================== Validation Visualization ====================

    def run_validation(self) -> List[Any]:
        """Run diagram validation and update error visualization."""
        try:
            from lib.diagram_validator import DiagramValidator

            validator = DiagramValidator(self.dsim)
            self.canvas.validation_errors = validator.validate()

            # Update sets of blocks with errors/warnings
            self.canvas.blocks_with_errors = validator.get_blocks_with_errors()
            self.canvas.blocks_with_warnings = validator.get_blocks_with_warnings()

            # Enable error visualization
            self.canvas.show_validation_errors = True

            # Trigger repaint
            self.canvas.update()

            logger.info(f"Validation complete: {len(self.canvas.validation_errors)} issues found")
            return self.canvas.validation_errors

        except Exception as e:
            logger.error(f"Error running validation: {str(e)}")
            return []

    def clear_validation(self) -> None:
        """Clear validation errors and hide indicators."""
        self.canvas.validation_errors = []
        self.canvas.blocks_with_errors = set()
        self.canvas.blocks_with_warnings = set()
        self.canvas.show_validation_errors = False
        self.canvas.update()

    def draw_block_error_indicator(self, painter: QPainter, block: Any, is_error: bool = True) -> None:
        """Draw error/warning indicator on a specific block."""
        try:
            # Choose color based on severity
            if is_error:
                indicator_color = QColor(220, 53, 69)  # Red for errors
                border_width = 3
            else:
                indicator_color = QColor(255, 193, 7)  # Yellow/orange for warnings
                border_width = 2

            # Draw pulsing border around block
            painter.setPen(QPen(indicator_color, border_width, Qt.SolidLine))
            painter.setBrush(Qt.NoBrush)

            # Draw outline around block
            padding = 4
            error_rect = QRect(
                block.left - padding,
                block.top - padding,
                block.width + 2 * padding,
                block.height + 2 * padding
            )
            painter.drawRoundedRect(error_rect, 10, 10)

            # Draw error/warning icon in top-right corner
            icon_size = 16
            icon_x = block.left + block.width - icon_size - 2
            icon_y = block.top + 2

            # Draw icon background circle
            icon_bg = QColor(indicator_color)
            icon_bg.setAlpha(200)
            painter.setBrush(icon_bg)
            painter.setPen(QPen(QColor(255, 255, 255), 1))
            painter.drawEllipse(icon_x, icon_y, icon_size, icon_size)

            # Draw exclamation mark
            painter.setPen(QPen(QColor(255, 255, 255), 2))
            # Vertical line
            painter.drawLine(
                icon_x + icon_size // 2, icon_y + 3,
                icon_x + icon_size // 2, icon_y + icon_size - 6
            )
            # Dot
            painter.drawPoint(icon_x + icon_size // 2, icon_y + icon_size - 3)

        except Exception as e:
            logger.error(f"Error drawing block error indicator: {str(e)}")

    # ==================== Hover State Management ====================

    def update_hover_states(self, pos: QPoint) -> None:
        """Update hover states for blocks, ports, and connections."""
        needs_repaint = False

        # Check for resize handles on selected blocks (highest priority)
        resize_handle = None
        for block in self.dsim.blocks_list:
            if block.selected:
                handle = self.canvas.block_renderer.get_resize_handle_at(block, pos)
                if handle:
                    resize_handle = handle
                    self.set_resize_cursor(handle)
                    break

        # Initialize variables
        new_hovered_port = None

        if not resize_handle:
            # Check for hovered port
            for block in self.dsim.blocks_list:
                # Check output ports
                for i, port_pos in enumerate(block.out_coords):
                    if self.point_near_port(pos, port_pos):
                        new_hovered_port = (block, i, True)  # True = output port
                        break
                # Check input ports
                if not new_hovered_port:
                    for i, port_pos in enumerate(block.in_coords):
                        if self.point_near_port(pos, port_pos):
                            new_hovered_port = (block, i, False)  # False = input port
                            break
                if new_hovered_port:
                    break

            if new_hovered_port != self.canvas.hovered_port:
                self.canvas.hovered_port = new_hovered_port
                needs_repaint = True

                # Show tooltip for hovered port
                if new_hovered_port:
                    block, port_idx, is_output = new_hovered_port
                    input_names, output_names = block.get_port_names()
                    if is_output and port_idx < len(output_names):
                        port_name = output_names[port_idx]
                        tooltip = f"Output: {port_name}"
                    elif not is_output and port_idx < len(input_names):
                        port_name = input_names[port_idx]
                        tooltip = f"Input: {port_name}"
                    else:
                        tooltip = f"Port {port_idx}"

                    # Include block documentation excerpt if available
                    if block.doc:
                        doc_lines = block.doc.split('\n')
                        short_doc = doc_lines[0][:60] + '...' if len(doc_lines[0]) > 60 else doc_lines[0]
                        tooltip = f"{tooltip}\n{short_doc}"

                    QToolTip.showText(self.canvas.mapToGlobal(pos), tooltip, self.canvas)
                else:
                    QToolTip.hideText()

            # Reset cursor if not over resize handle or port
            if not new_hovered_port:
                self.canvas.setCursor(Qt.ArrowCursor)

        # Check for hovered block (if no port is hovered)
        if not new_hovered_port:
            new_hovered_block = self.canvas._get_clicked_block(pos)
            if new_hovered_block != self.canvas.hovered_block:
                self.canvas.hovered_block = new_hovered_block
                needs_repaint = True
        else:
            if self.canvas.hovered_block is not None:
                self.canvas.hovered_block = None
                needs_repaint = True

        # Check for hovered line (if no block/port is hovered)
        if not new_hovered_port and not self.canvas.hovered_block:
            line_result, _ = self.canvas._get_clicked_line(pos)
            if line_result != self.canvas.hovered_line:
                self.canvas.hovered_line = line_result
                needs_repaint = True
        else:
            if self.canvas.hovered_line is not None:
                self.canvas.hovered_line = None
                needs_repaint = True

        if needs_repaint:
            self.canvas.update()

    def point_near_port(self, point: QPoint, port_pos: QPoint, threshold: int = 12) -> bool:
        """Check if a point is near a port position."""
        dx = point.x() - port_pos.x()
        dy = point.y() - port_pos.y()
        return (dx * dx + dy * dy) < (threshold * threshold)

    def set_resize_cursor(self, handle: str) -> None:
        """Set the appropriate cursor for a resize handle."""
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
