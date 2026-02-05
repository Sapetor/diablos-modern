"""
Block Renderer Module
Handles the rendering of DBlock objects on the canvas.
Separates the view/drawing logic from the data model.
"""

import logging
from typing import Optional
from PyQt5.QtGui import QColor, QPen, QPainter, QPolygonF, QLinearGradient, QPainterPath, QRadialGradient, QTransform
from PyQt5.QtCore import Qt, QRect, QPoint, QPointF
from modern_ui.themes.theme_manager import theme_manager

logger = logging.getLogger(__name__)

class BlockRenderer:
    """
    Stateless renderer for DBlock objects.
    """

    def __init__(self):
        pass

    def draw_block(self, block, painter: Optional[QPainter], draw_name: bool = True, draw_ports: bool = True) -> None:
        """
        Draw a block on the canvas with modern styling, shadows, and depth.

        Args:
            block: The DBlock instance to draw
            painter: QPainter instance for rendering
            draw_name: Whether to draw the block name/label
            draw_ports: Whether to draw the input/output port connectors
        """
        if painter is None:
            return

        # Draw shadow for depth (offset slightly down and right)
        shadow_offset = 3
        shadow_color = theme_manager.get_color('block_shadow')
        shadow_color.setAlpha(80)  # Semi-transparent shadow

        painter.setBrush(shadow_color)
        painter.setPen(Qt.NoPen)

        if block.block_fn == "Gain":
            # Shadow for triangle
            shadow_points = QPolygonF()
            if not block.flipped:
                shadow_points.append(QPoint(block.left + shadow_offset, block.top + shadow_offset))
                shadow_points.append(QPoint(block.left + block.width + shadow_offset, int(block.top + block.height / 2) + shadow_offset))
                shadow_points.append(QPoint(block.left + shadow_offset, block.top + block.height + shadow_offset))
            else:
                shadow_points.append(QPoint(block.left + block.width + shadow_offset, block.top + shadow_offset))
                shadow_points.append(QPoint(block.left + shadow_offset, int(block.top + block.height / 2) + shadow_offset))
                shadow_points.append(QPoint(block.left + block.width + shadow_offset, block.top + block.height + shadow_offset))
            painter.drawPolygon(shadow_points)
        else:
            # Shadow for rounded rectangle
            radius = 12
            shadow_rect = QRect(block.left + shadow_offset, block.top + shadow_offset, block.width, block.height)
            painter.drawRoundedRect(shadow_rect, radius, radius)

        # Determine border color based on block category
        category_color = block.b_color
        border_color = theme_manager.get_color('border_primary')

        # Try to get category-specific border color
        if hasattr(block, 'category'):
            category_lower = block.category.lower() if isinstance(block.category, str) else str(block.category).lower()
            if 'source' in category_lower:
                border_color = theme_manager.get_color('block_source_border')
            elif 'math' in category_lower:
                border_color = theme_manager.get_color('block_process_border')
            elif 'control' in category_lower:
                border_color = theme_manager.get_color('block_control_border')
            elif 'sink' in category_lower:
                border_color = theme_manager.get_color('block_sink_border')
            else:
                border_color = theme_manager.get_color('block_other_border')

        # Override border if selected
        if block.selected:
            border_color = theme_manager.get_color('block_selected')
            if not block.selected:
                painter.setBrush(category_color)
            else:
                # Create a lighter version for selected state, or use base color per original logic
                painter.setBrush(block.b_color)

        # Draw main block shape with gradient fill
        gradient = QLinearGradient(block.left, block.top, block.left, block.top + block.height)
        base_color = block.b_color
        lighter_color = QColor(base_color)
        lighter_color.setRed(min(255, base_color.red() + 30))
        lighter_color.setGreen(min(255, base_color.green() + 30))
        lighter_color.setBlue(min(255, base_color.blue() + 30))
        gradient.setColorAt(0, lighter_color)
        gradient.setColorAt(1, base_color)
        
        painter.setBrush(gradient)
        painter.setPen(QPen(border_color, 3 if block.selected else 2))

        if block.block_fn == "Gain":
            # Draw a triangle for the Gain block
            points = QPolygonF()
            if not block.flipped:
                points.append(QPoint(block.left, block.top))
                points.append(QPoint(block.left + block.width, int(block.top + block.height / 2)))
                points.append(QPoint(block.left, block.top + block.height))
            else:
                points.append(QPoint(block.left + block.width, block.top))
                points.append(QPoint(block.left, int(block.top + block.height / 2)))
                points.append(QPoint(block.left + block.width, block.top + block.height))
            painter.drawPolygon(points)
        else:
            radius = 12
            painter.drawRoundedRect(QRect(block.left, block.top, block.width, block.height), radius, radius)

        # Draw block-specific icon
        icon_pen = QPen(QColor('#1F2937'), 2)
        painter.setPen(icon_pen)
        
        path = QPainterPath()
        
        # Try polymorphic draw_icon first
        if block.block_instance and hasattr(block.block_instance, 'draw_icon'):
            try:
                custom_path = block.block_instance.draw_icon(block.rect)
                if custom_path is not None:
                    path = custom_path
            except Exception as e:
                logger.warning(f"draw_icon failed for {block.block_fn}: {e}")
        
        # Fallback to legacy switch statement
        self._draw_legacy_icon(block, painter, path)

        if not path.isEmpty():
            margin = block.width * 0.2
            transform = QTransform()
            if block.flipped:
                transform.translate(block.left + block.width - margin, block.top + margin)
                transform.scale(-(block.width - 2 * margin), block.height - 2 * margin)
            else:
                transform.translate(block.left + margin, block.top + margin)
                transform.scale(block.width - 2 * margin, block.height - 2 * margin)
            
            scaled_path = transform.map(path)
            painter.drawPath(scaled_path)

        # Draw ports
        if draw_ports:
            self.draw_ports(block, painter)
            self.draw_port_labels(block, painter)

        if draw_name:
            text_color = theme_manager.get_color('text_primary')
            painter.setPen(text_color)
            font = block.font
            font.setWeight(400)
            painter.setFont(font)
            text_rect = QRect(block.left, block.top + block.height + 2, block.width, 28)
            painter.drawText(text_rect, Qt.AlignHCenter | Qt.AlignTop, block.username)

        # Enhanced selection visualization
        if block.selected:
            selection_color = theme_manager.get_color('block_selected')
            painter.setPen(QPen(selection_color, 3, Qt.SolidLine))
            painter.setBrush(Qt.NoBrush)
            padding = 4
            selection_rect = QRect(
                block.left - padding,
                block.top - padding,
                block.width + 2 * padding,
                block.height + 2 * padding
            )
            painter.drawRoundedRect(selection_rect, 14, 14)

        # Draw sample rate indicator for discrete blocks
        self._draw_sample_rate_indicator(block, painter)

    def draw_ports(self, block, painter: Optional[QPainter]) -> None:
        """Draw input and output ports with modern styling."""
        if painter is None:
            return

        port_input_color = theme_manager.get_color('port_input')
        port_output_color = theme_manager.get_color('port_output')
        port_draw_radius = block.port_radius - 1

        # Input ports
        for port_in_location in block.in_coords:
            gradient = QRadialGradient(port_in_location.x(), port_in_location.y(), port_draw_radius)
            gradient.setColorAt(0.0, port_input_color.lighter(130))
            gradient.setColorAt(0.7, port_input_color)
            gradient.setColorAt(1.0, port_input_color.darker(110))

            painter.setBrush(gradient)
            painter.setPen(QPen(port_input_color.darker(140), 2.0))
            painter.drawEllipse(port_in_location, port_draw_radius, port_draw_radius)

            # Highlight
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor(255, 255, 255, 50))
            highlight_offset = int(port_draw_radius * 0.3)
            highlight_size = int(port_draw_radius * 0.4)
            painter.drawEllipse(port_in_location.x() - highlight_offset, port_in_location.y() - highlight_offset, highlight_size, highlight_size)

        # Output ports
        for port_out_location in block.out_coords:
            gradient = QRadialGradient(port_out_location.x(), port_out_location.y(), port_draw_radius)
            gradient.setColorAt(0.0, port_output_color.lighter(130))
            gradient.setColorAt(0.7, port_output_color)
            gradient.setColorAt(1.0, port_output_color.darker(110))

            painter.setBrush(gradient)
            painter.setPen(QPen(port_output_color.darker(140), 2.0))
            painter.drawEllipse(port_out_location, port_draw_radius, port_draw_radius)

            # Highlight
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor(255, 255, 255, 50))
            highlight_offset = int(port_draw_radius * 0.3)
            highlight_size = int(port_draw_radius * 0.4)
            painter.drawEllipse(port_out_location.x() - highlight_offset, port_out_location.y() - highlight_offset, highlight_size, highlight_size)

    def draw_port_labels(self, block, painter: Optional[QPainter], show_labels: bool = True) -> None:
        """
        Draw labels next to input and output ports for blocks with multiple ports.

        Args:
            block: The DBlock instance
            painter: QPainter instance for rendering
            show_labels: Whether to show port labels (can be toggled in settings)
        """
        if painter is None or not show_labels:
            return

        # Only show labels for blocks with multiple ports on either side
        if block.in_ports <= 1 and block.out_ports <= 1:
            return

        # Skip blocks where port meaning is obvious from the icon
        skip_blocks = {'Sum', 'SgProd', 'Product', 'Mux', 'Demux'}
        if block.block_fn in skip_blocks:
            return

        # Get port names from block
        input_names, output_names = block.get_port_names()

        # Setup font for labels
        from PyQt5.QtGui import QFont, QFontMetrics
        label_font = QFont("Arial", 8)
        label_font.setBold(False)
        painter.setFont(label_font)
        font_metrics = QFontMetrics(label_font)

        # Qt 5.11+ uses horizontalAdvance, older versions use width
        def get_text_width(text):
            if hasattr(font_metrics, 'horizontalAdvance'):
                return font_metrics.horizontalAdvance(text)
            return font_metrics.width(text)

        # Label colors
        text_color = theme_manager.get_color('text_primary')
        bg_color = theme_manager.get_color('background_secondary')
        bg_color.setAlpha(200)

        port_radius = block.port_radius
        label_offset = port_radius + 4  # Offset from port edge

        # Draw input port labels (on left side, text inside block)
        for i, port_coord in enumerate(block.in_coords):
            if i < len(input_names):
                label = input_names[i]
                # Position label to the right of the port (inside block)
                text_width = get_text_width(label)
                text_height = font_metrics.height()

                x = port_coord.x() + label_offset
                y = port_coord.y() + text_height // 4

                # Draw background for readability
                bg_rect = QRect(int(x - 2), int(y - text_height + 2), int(text_width + 4), int(text_height))
                painter.setPen(Qt.NoPen)
                painter.setBrush(bg_color)
                painter.drawRoundedRect(bg_rect, 2, 2)

                # Draw text
                painter.setPen(text_color)
                painter.drawText(int(x), int(y), label)

        # Draw output port labels (on right side, text inside block)
        for i, port_coord in enumerate(block.out_coords):
            if i < len(output_names):
                label = output_names[i]
                # Position label to the left of the port (inside block)
                text_width = get_text_width(label)
                text_height = font_metrics.height()

                x = port_coord.x() - label_offset - text_width
                y = port_coord.y() + text_height // 4

                # Draw background for readability
                bg_rect = QRect(int(x - 2), int(y - text_height + 2), int(text_width + 4), int(text_height))
                painter.setPen(Qt.NoPen)
                painter.setBrush(bg_color)
                painter.drawRoundedRect(bg_rect, 2, 2)

                # Draw text
                painter.setPen(text_color)
                painter.drawText(int(x), int(y), label)

    def draw_resize_handles(self, block, painter):
        """Draw resize handles on the corners and edges of selected blocks."""
        if not block.selected:
            return

        try:
            from config.block_sizes import RESIZE_HANDLE_SIZE
            handle_size = RESIZE_HANDLE_SIZE
        except ImportError:
            handle_size = 8

        handle_color = theme_manager.get_color('accent_primary')
        border_color = theme_manager.get_color('border_primary')

        painter.save()
        painter.setPen(QPen(border_color, 1))
        painter.setBrush(handle_color)

        half_handle = handle_size // 2
        handles = {
            'top_left': (block.left - half_handle, block.top - half_handle),
            'top_right': (block.left + block.width - half_handle, block.top - half_handle),
            'bottom_left': (block.left - half_handle, block.top + block.height - half_handle),
            'bottom_right': (block.left + block.width - half_handle, block.top + block.height - half_handle),
            'top': (block.left + block.width//2 - half_handle, block.top - half_handle),
            'bottom': (block.left + block.width//2 - half_handle, block.top + block.height - half_handle),
            'left': (block.left - half_handle, block.top + block.height//2 - half_handle),
            'right': (block.left + block.width - half_handle, block.top + block.height//2 - half_handle),
        }

        for _, (x, y) in handles.items():
            painter.drawRect(int(x), int(y), int(handle_size), int(handle_size))

        painter.restore()

    def get_resize_handle_at(self, block, pos: QPoint):
        """
        Check if a position is over a resize handle.

        Args:
            block: The block to check handles for
            pos: QPoint position to check

        Returns:
            Handle name if over a handle, None otherwise
        """
        if not block.selected:
            return None

        try:
            from config.block_sizes import RESIZE_HANDLE_SIZE
            handle_size = RESIZE_HANDLE_SIZE
        except ImportError:
            handle_size = 8

        # Define handle positions (same as draw_resize_handles)
        half_handle = handle_size // 2
        handles = {
            'top_left': (block.left - half_handle, block.top - half_handle),
            'top_right': (block.left + block.width - half_handle, block.top - half_handle),
            'bottom_left': (block.left - half_handle, block.top + block.height - half_handle),
            'bottom_right': (block.left + block.width - half_handle, block.top + block.height - half_handle),
            'top': (block.left + block.width//2 - half_handle, block.top - half_handle),
            'bottom': (block.left + block.width//2 - half_handle, block.top + block.height - half_handle),
            'left': (block.left - half_handle, block.top + block.height//2 - half_handle),
            'right': (block.left + block.width - half_handle, block.top + block.height//2 - half_handle),
        }

        # Check if position is within any handle
        for handle_name, (x, y) in handles.items():
            if (x <= pos.x() <= x + handle_size and
                y <= pos.y() <= y + handle_size):
                return handle_name

        return None

    def _draw_legacy_icon(self, block, painter, path):
        """Helper to draw legacy icons that use direct calls or mess with fonts."""
        # Using if/elif chain copied from original block.py
        if path.isEmpty() and block.block_fn == "Step":
            path.moveTo(0.1, 0.7); path.lineTo(0.5, 0.7); path.lineTo(0.5, 0.3); path.lineTo(0.9, 0.3)
        elif path.isEmpty() and block.block_fn == "Ramp":
            path.moveTo(0.1, 0.9); path.lineTo(0.9, 0.1)
        elif path.isEmpty() and block.block_fn == "Sine":
            path.moveTo(0.1, 0.5); path.quadTo(0.3, 0.1, 0.5, 0.5); path.quadTo(0.7, 0.9, 0.9, 0.5)
        elif block.block_fn == "SgProd":
            path.moveTo(0.2, 0.2); path.lineTo(0.8, 0.8)
            path.moveTo(0.2, 0.8); path.lineTo(0.8, 0.2)
        elif block.block_fn == "TranFn":
            self._draw_text_icon(block, painter, ["B(s)", "A(s)"], italic=True)
        elif block.block_fn == "Demux":
            path.moveTo(0.2, 0.5); path.lineTo(0.4, 0.5)
            path.moveTo(0.4, 0.2); path.lineTo(0.4, 0.8); path.lineTo(0.8, 0.8); path.lineTo(0.8, 0.2); path.lineTo(0.4, 0.2)
            path.moveTo(0.8, 0.3); path.lineTo(1.0, 0.3)
            path.moveTo(0.8, 0.7); path.lineTo(1.0, 0.7)
        elif block.block_fn == "Mux":
            path.moveTo(0.2, 0.3); path.lineTo(0.4, 0.3)
            path.moveTo(0.2, 0.7); path.lineTo(0.4, 0.7)
            path.moveTo(0.4, 0.2); path.lineTo(0.8, 0.4); path.lineTo(0.8, 0.6); path.lineTo(0.4, 0.8); path.lineTo(0.4, 0.2)
            path.moveTo(0.8, 0.5); path.lineTo(1.0, 0.5)
        elif block.block_fn == "BodeMag":
            path.moveTo(0.1, 0.9); path.lineTo(0.9, 0.9)
            path.moveTo(0.1, 0.9); path.lineTo(0.1, 0.1)
            path.moveTo(0.1, 0.4); path.lineTo(0.4, 0.4); path.lineTo(0.6, 0.7); path.lineTo(0.9, 0.7)
        elif block.block_fn == "RootLocus":
            path.moveTo(0.1, 0.5); path.lineTo(0.9, 0.5)
            path.moveTo(0.5, 0.1); path.lineTo(0.5, 0.9)
            p_x, p_y = 0.4, 0.3; path.moveTo(p_x-0.03, p_y-0.03); path.lineTo(p_x+0.03, p_y+0.03); path.moveTo(p_x+0.03, p_y-0.03); path.lineTo(p_x-0.03, p_y+0.03)
            p_x, p_y = 0.4, 0.7; path.moveTo(p_x-0.03, p_y-0.03); path.lineTo(p_x+0.03, p_y+0.03); path.moveTo(p_x+0.03, p_y-0.03); path.lineTo(p_x-0.03, p_y+0.03)
            path.addEllipse(QPointF(0.2*block.width, 0.5*block.height), 3, 3) 
            path.moveTo(0.4, 0.3); path.quadTo(0.3, 0.3, 0.2, 0.5)
            path.moveTo(0.4, 0.7); path.quadTo(0.3, 0.7, 0.2, 0.5)
            path.moveTo(0.2, 0.5); path.lineTo(0.1, 0.5)
        elif block.block_fn == "Deriv":
            self._draw_text_icon(block, painter, ["dy", "dt"], italic=True)
        elif block.block_fn == "DiscreteTranFn":
            self._draw_text_icon(block, painter, ["B(z)", "A(z)"], italic=True)
        elif block.block_fn == "Integrator":
             self._draw_text_icon(block, painter, ["1", "s"], italic=True, size_delta=4)
        elif block.block_fn == "Scope":
            path.moveTo(0.1, 0.9); path.lineTo(0.9, 0.9); path.moveTo(0.1, 0.9); path.lineTo(0.1, 0.1)
            path.moveTo(0.1, 0.6); path.quadTo(0.3, 0.2, 0.5, 0.6); path.quadTo(0.7, 1.0, 0.9, 0.6)
        elif block.block_fn == "Sum":
             sign_text = block.params.get('sign', '++')
             if isinstance(sign_text, dict): sign_text = sign_text.get('default', '++')
             self._draw_centered_text(block, painter, str(sign_text), size_delta=4)
        elif block.block_fn == "Noise":
             path.moveTo(0.1, 0.5); path.lineTo(0.2, 0.3); path.lineTo(0.3, 0.7); path.lineTo(0.4, 0.4)
             path.lineTo(0.5, 0.6); path.lineTo(0.6, 0.2); path.lineTo(0.7, 0.8); path.lineTo(0.8, 0.5); path.lineTo(0.9, 0.6)
        elif block.block_fn == "Exp":
             self._draw_centered_text(block, painter, "eˣ", italic=True, size_delta=4)
        elif block.block_fn == "Display":
             params_source = getattr(block, 'exec_params', block.params) or block.params
             display_val = params_source.get('_display_value_', '---')
             # Dynamic character limit based on block width (approx 8 pixels per char)
             block_width = getattr(block, 'width', 80)
             max_chars = max(10, int(block_width / 8))
             if len(str(display_val)) > max_chars: display_val = str(display_val)[:max_chars-1] + "…"
             self._draw_centered_text(block, painter, str(display_val), bold=True, size_delta=2)
        elif block.block_fn == "Term":
             path.moveTo(0.5, 0.2); path.lineTo(0.5, 0.6)
             path.moveTo(0.2, 0.6); path.lineTo(0.8, 0.6)
             path.moveTo(0.3, 0.75); path.lineTo(0.7, 0.75)
             path.moveTo(0.4, 0.9); path.lineTo(0.6, 0.9)
        elif block.block_fn == "Export":
             path.moveTo(0.2, 0.2); path.lineTo(0.8, 0.2); path.lineTo(0.8, 0.8); path.lineTo(0.2, 0.8); path.lineTo(0.2, 0.2)
             path.moveTo(0.5, 0.5); path.lineTo(1.0, 0.5); path.moveTo(0.8, 0.3); path.lineTo(1.0, 0.5); path.lineTo(0.8, 0.7)
        elif block.block_fn == "ZeroOrderHold":
             path.moveTo(0.1, 0.8); path.lineTo(0.3, 0.8); path.lineTo(0.3, 0.5); path.lineTo(0.6, 0.5)
             path.lineTo(0.6, 0.2); path.lineTo(0.9, 0.2)
        elif block.block_fn == "PRBS":
             path.moveTo(0.1, 0.7); path.lineTo(0.18, 0.7); path.lineTo(0.18, 0.3); path.lineTo(0.32, 0.3)
             path.lineTo(0.32, 0.7); path.lineTo(0.45, 0.7); path.lineTo(0.45, 0.4); path.lineTo(0.6, 0.4)
             path.lineTo(0.6, 0.7); path.lineTo(0.78, 0.7); path.lineTo(0.78, 0.3); path.lineTo(0.9, 0.3); path.lineTo(0.9, 0.7)
        elif block.block_fn == "Hysteresis":
             path.moveTo(0.15, 0.75); path.lineTo(0.75, 0.75); path.lineTo(0.75, 0.25); path.lineTo(0.85, 0.25)
             path.moveTo(0.85, 0.25); path.lineTo(0.25, 0.25); path.lineTo(0.25, 0.75); path.lineTo(0.15, 0.75)
             path.moveTo(0.45, 0.75); path.lineTo(0.45, 0.72); path.lineTo(0.51, 0.75); path.lineTo(0.45, 0.78); path.lineTo(0.45, 0.75)
             path.moveTo(0.55, 0.25); path.lineTo(0.55, 0.22); path.lineTo(0.49, 0.25); path.lineTo(0.55, 0.28); path.lineTo(0.55, 0.25)
        elif block.block_fn == "Deadband":
             path.moveTo(0.15, 0.80); path.lineTo(0.35, 0.50); path.lineTo(0.65, 0.50); path.lineTo(0.85, 0.20)
             path.moveTo(0.2, 0.5); path.lineTo(0.8, 0.5)
        elif block.block_fn == "Switch":
             path.moveTo(0.5, 0.10); path.lineTo(0.5, 0.35)
             path.moveTo(0.47, 0.30); path.lineTo(0.5, 0.35); path.lineTo(0.53, 0.30)
             path.moveTo(0.30, 0.35); path.lineTo(0.70, 0.35); path.lineTo(0.70, 0.75); path.lineTo(0.30, 0.75); path.lineTo(0.30, 0.35)
             path.moveTo(0.30, 0.45); path.lineTo(0.45, 0.45); path.moveTo(0.30, 0.65); path.lineTo(0.45, 0.65)
             path.moveTo(0.45, 0.45); path.lineTo(0.55, 0.55); path.lineTo(0.70, 0.55)
             path.moveTo(0.70, 0.55); path.lineTo(0.90, 0.55)
        elif block.block_fn == "Saturation":
             path.moveTo(0.1, 0.8); path.lineTo(0.9, 0.8); path.moveTo(0.1, 0.2); path.lineTo(0.9, 0.2)
             path.moveTo(0.15, 0.5); path.quadTo(0.3, 0.2, 0.45, 0.2); path.lineTo(0.55, 0.2); path.quadTo(0.7, 0.8, 0.85, 0.8)
        elif block.block_fn == "RateLimiter":
             path.moveTo(0.15, 0.75); path.lineTo(0.35, 0.75); path.lineTo(0.65, 0.25); path.lineTo(0.85, 0.25)
             path.moveTo(0.35, 0.75); path.lineTo(0.35, 0.25); path.lineTo(0.40, 0.25)
             self._draw_corner_label(block, painter, "du/dt")
        elif block.block_fn == "PID":
             self._draw_centered_text(block, painter, "PID", bold=True, size_delta=3)
             self._draw_corner_labels(block, painter, "sp", "pv")
        elif block.block_fn == "StateSpace":
             self._draw_centered_text(block, painter, "x' = Ax+Bu\ny = Cx+Du", size_delta=-1)
        elif block.block_fn == "DiscreteStateSpace":
             self._draw_centered_text(block, painter, "x[k+1]=Ax+Bu\ny[k]=Cx+Du", size_delta=-2)
        elif block.block_fn == "External":
             path.moveTo(0.2, 0.2); path.lineTo(0.8, 0.2); path.moveTo(0.2, 0.5); path.lineTo(0.6, 0.5)
             path.moveTo(0.2, 0.8); path.lineTo(0.8, 0.8); path.moveTo(0.2, 0.2); path.lineTo(0.2, 0.8)
        elif block.block_fn == "Constant":
             self._draw_centered_text(block, painter, "K", bold=True, size_delta=4)
        elif block.block_fn == "Delay":
             self._draw_centered_text(block, painter, "z⁻ⁿ", size_delta=2)
        elif block.block_fn == "Abs":
             self._draw_centered_text(block, painter, "|u|", bold=True, size_delta=4)
        elif block.block_fn == "TransportDelay":
             font = painter.font()
             orig = font.pointSize()
             font.setPointSize(orig + 3); font.setItalic(True); painter.setFont(font)
             painter.setPen(QColor('#1F2937'))
             cx, cy = block.left + block.width // 2, block.top + block.height // 2
             painter.drawText(cx - 12, cy + 4, "e")
             font.setPointSize(orig); painter.setFont(font)
             painter.drawText(cx - 2, cy - 4, "-τs")
             font.setItalic(False); painter.setFont(font)
        elif block.block_fn == "XYGraph":
             path.moveTo(0.15, 0.85); path.lineTo(0.85, 0.85); path.moveTo(0.15, 0.85); path.lineTo(0.15, 0.15)
             path.moveTo(0.82, 0.82); path.lineTo(0.85, 0.85); path.lineTo(0.82, 0.88)
             path.moveTo(0.12, 0.18); path.lineTo(0.15, 0.15); path.lineTo(0.18, 0.18)
             path.moveTo(0.25, 0.75); path.quadTo(0.35, 0.35, 0.55, 0.45); path.quadTo(0.75, 0.55, 0.70, 0.30)
        elif block.block_fn == "Assert":
             self._draw_centered_text(block, painter, "⚠", bold=True, size_delta=6, color='#DC2626')
        elif block.block_fn == "Selector":
             path.moveTo(0.15, 0.3); path.lineTo(0.15, 0.7); path.lineTo(0.35, 0.7); path.lineTo(0.35, 0.3); path.lineTo(0.15, 0.3)
             path.moveTo(0.17, 0.4); path.lineTo(0.33, 0.4); path.moveTo(0.17, 0.5); path.lineTo(0.33, 0.5); path.moveTo(0.17, 0.6); path.lineTo(0.33, 0.6)
             path.moveTo(0.35, 0.5); path.lineTo(0.65, 0.5); path.moveTo(0.60, 0.45); path.lineTo(0.65, 0.5); path.lineTo(0.60, 0.55)
             path.moveTo(0.70, 0.45); path.lineTo(0.85, 0.45); path.lineTo(0.85, 0.55); path.lineTo(0.70, 0.55); path.lineTo(0.70, 0.45)
        elif block.block_fn == "Subsystem":
             # Nested rectangles icon
             path.moveTo(0.2, 0.2); path.lineTo(0.8, 0.2); path.lineTo(0.8, 0.8); path.lineTo(0.2, 0.8); path.lineTo(0.2, 0.2)
             path.moveTo(0.3, 0.3); path.lineTo(0.7, 0.3); path.lineTo(0.7, 0.7); path.lineTo(0.3, 0.7); path.lineTo(0.3, 0.3)
        elif block.block_fn == "Inport":
             self._draw_centered_text(block, painter, "In", bold=True, size_delta=2)
             # Draw arrow?
             # path.moveTo(0.2, 0.5); path.lineTo(0.8, 0.5); path.lineTo(0.6, 0.3)...
        elif block.block_fn == "Outport":
             self._draw_centered_text(block, painter, "Out", bold=True, size_delta=2)
             
        elif block.block_fn == "FFT":
             path.moveTo(0.15, 0.80); path.lineTo(0.15, 0.20); path.moveTo(0.15, 0.80); path.lineTo(0.85, 0.80)
             bar_positions = [0.22, 0.32, 0.42, 0.52, 0.62, 0.72]
             bar_heights = [0.30, 0.55, 0.70, 0.45, 0.25, 0.15]
             bar_w = 0.06
             for x, h in zip(bar_positions, bar_heights):
                 path.moveTo(x, 0.80); path.lineTo(x, 0.80 - h * 0.55); path.lineTo(x + bar_w, 0.80 - h * 0.55); path.lineTo(x + bar_w, 0.80)
        elif block.block_fn == "MathFunction":
             self._draw_centered_text(block, painter, "f(u)", italic=True, size_delta=4)

    # --- Helper methods for common icon patterns ---

    def _draw_text_icon(self, block, painter, lines, italic=False, size_delta=2):
        font = painter.font()
        orig = font.pointSize()
        font.setPointSize(orig + size_delta)
        font.setItalic(italic)
        painter.setFont(font)
        painter.setPen(QColor('#1F2937'))
        
        rect_top = QRect(block.left, block.top, block.width, block.height // 2)
        painter.drawText(rect_top, Qt.AlignCenter, lines[0])
        
        line_y = block.top + block.height // 2
        painter.drawLine(block.left + 10, line_y, block.left + block.width - 10, line_y)
        
        rect_bot = QRect(block.left, block.top + block.height // 2, block.width, block.height // 2)
        painter.drawText(rect_bot, Qt.AlignCenter, lines[1])
        
        font.setItalic(False)
        font.setPointSize(orig)
        painter.setFont(font)

    def _draw_centered_text(self, block, painter, text, bold=False, italic=False, size_delta=0, color='#1F2937'):
        font = painter.font()
        orig = font.pointSize()
        font.setPointSize(orig + size_delta)
        font.setBold(bold)
        font.setItalic(italic)
        painter.setFont(font)
        painter.setPen(QColor(color))
        rect = QRect(block.left, block.top, block.width, block.height)
        painter.drawText(rect, Qt.AlignCenter, text)
        font.setBold(False); font.setItalic(False); font.setPointSize(orig)
        painter.setFont(font)

    def _draw_corner_label(self, block, painter, text):
        font = painter.font()
        orig = font.pointSize()
        font.setPointSize(orig - 1)
        painter.setFont(font)
        painter.setPen(QColor('#1F2937'))
        painter.drawText(QRect(block.left + int(block.width*0.4), block.top + int(block.height*0.4), int(block.width*0.6), int(block.height*0.6)), Qt.AlignCenter, text)
        font.setPointSize(orig)
        painter.setFont(font)

    def _draw_corner_labels(self, block, painter, tl, bl):
        font = painter.font()
        orig = font.pointSize()
        font.setPointSize(orig - 1)
        painter.setFont(font)
        painter.drawText(QRect(block.left + 4, block.top + 2, block.width // 2, block.height // 2), Qt.AlignLeft | Qt.AlignTop, tl)
        painter.drawText(QRect(block.left + 4, block.top + block.height // 2, block.width // 2, block.height // 2), Qt.AlignLeft | Qt.AlignBottom, bl)
        font.setPointSize(orig)
        painter.setFont(font)

    def _draw_sample_rate_indicator(self, block, painter: Optional[QPainter]) -> None:
        """
        Draw a small colored indicator for discrete blocks showing their sample rate.

        Indicator colors:
        - No indicator: Continuous block (sample_time < 0 or not set)
        - Gray dot: Inherited sample time (sample_time = 0)
        - Colored dot: Fixed discrete rate (sample_time > 0)
          - Red = fast (small sample time)
          - Blue = slow (large sample time)
          - Gradient between based on log scale

        Args:
            block: The DBlock instance
            painter: QPainter instance for rendering
        """
        if painter is None:
            return

        # Get sample time from block params
        sample_time = block.params.get('sampling_time',
                      block.params.get('sample_time',
                      block.params.get('output_sample_time', -1.0)))

        try:
            sample_time = float(sample_time)
        except (ValueError, TypeError):
            sample_time = -1.0

        # No indicator for continuous blocks
        if sample_time < 0:
            return

        # Indicator properties
        indicator_radius = 5
        indicator_x = block.left + block.width - indicator_radius - 3
        indicator_y = block.top + indicator_radius + 3

        # Determine indicator color
        if sample_time == 0:
            # Inherited rate - gray
            indicator_color = QColor(128, 128, 128)
        else:
            # Fixed discrete rate - color based on sample time
            # Use log scale: 0.001s (1kHz) = red, 1s (1Hz) = blue
            import math
            log_min = math.log10(0.001)  # 1ms = fast (red)
            log_max = math.log10(1.0)    # 1s = slow (blue)
            log_sample = math.log10(max(0.001, min(1.0, sample_time)))

            # Normalize to 0-1 range
            t = (log_sample - log_min) / (log_max - log_min)
            t = max(0.0, min(1.0, t))

            # Interpolate from red (fast) to blue (slow)
            r = int(255 * (1 - t))
            g = int(100 * (1 - abs(t - 0.5) * 2))  # Green peak in middle
            b = int(255 * t)
            indicator_color = QColor(r, g, b)

        # Draw the indicator dot
        painter.save()
        painter.setPen(QPen(QColor(50, 50, 50), 1))
        painter.setBrush(indicator_color)
        painter.drawEllipse(
            QPoint(int(indicator_x), int(indicator_y)),
            indicator_radius,
            indicator_radius
        )
        painter.restore()
