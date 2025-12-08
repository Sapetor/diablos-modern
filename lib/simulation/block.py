"""
DBlock class - represents a block in the simulation diagram.
"""

import logging
import importlib
import copy
from typing import Dict, List, Optional, Any, Union
import numpy as np
from PyQt5.QtGui import QColor, QPen, QFont, QPixmap, QTransform, QPainterPath, QPolygonF, QPainter
from PyQt5.QtCore import Qt, QRect, QPoint
from modern_ui.themes.theme_manager import theme_manager
from lib.dialogs import ParamDialog, PortDialog
import sys

logger = logging.getLogger(__name__)

class DBlock:
    """
    Represents a functional block in the simulation diagram.

    Attributes:
        name: Unique block identifier (block_fn + sid)
        block_fn: Block function name (e.g., "Integrator", "Gain")
        sid: Sequential ID number
        username: User-defined name for the block
        rect: QRect defining block position and size
        b_color: Block color
        params: Block parameters dictionary
        in_ports: Number of input ports
        out_ports: Number of output ports
        in_coords: List of input port coordinates
        out_coords: List of output port coordinates
        hierarchy: Execution hierarchy level (-1 if not set)
        computed_data: Whether block has been computed in current step
        input_queue: Dictionary mapping port number to received data
    """

    def __init__(self, block_fn: str, sid: int, coords: QRect, color: Union[str, QColor],
                 in_ports: int = 1, out_ports: int = 1, b_type: int = 2,
                 io_edit: Union[bool, str] = True, fn_name: str = 'block',
                 params: Optional[Dict[str, Any]] = None, external: bool = False,
                 username: str = '', block_class: Optional[Any] = None,
                 colors: Optional[Dict[str, QColor]] = None, category: str = 'Other') -> None:
        """
        Initialize a block instance.

        Args:
            block_fn: Block function name (e.g., "Integrator", "Step")
            sid: Sequential ID for this block type
            coords: QRect defining block position and size
            color: Block color (QColor or color name string)
            in_ports: Number of input ports (default 1)
            out_ports: Number of output ports (default 1)
            b_type: Block type identifier (default 2)
            io_edit: Port editability ('input', 'output', 'both', 'none', or bool)
            fn_name: Function name for execution
            params: Block parameters dictionary
            external: Whether block uses external code
            username: User-defined block name
            block_class: Block class for instantiation
            colors: Color palette dictionary
            category: Block category (Sources, Math, Control, Sinks, Other)
        """
        if params is None:
            params = {}

        logger.debug(f"Initializing DBlock {block_fn}{sid}")
        self.name: str = block_fn.lower() + str(sid)
        self.category: str = category  # Store category for theme-aware rendering
        self.flipped: bool = False
        self.block_fn: str = block_fn
        self.sid: int = sid
        self.username: str = self.name if username == '' else username

        self.rect: QRect = coords
        self.left: int = self.rect.left()
        self.top: int = self.rect.top()
        self.width: int = self.rect.width()
        self.height: int = self.rect.height()
        self.height_base: int = self.height

        # Handle color - can be either a QColor object or a string
        if isinstance(color, QColor):
            self.b_color: QColor = color
        elif colors and color in colors:
            self.b_color: QColor = colors[color]
        else:
            self.b_color: QColor = QColor(color)
        self.image: QPixmap = QPixmap()  # Initialize as null QPixmap since no icons are available

        self.params: Dict[str, Any] = params.copy()
        self.initial_params: Dict[str, Any] = params.copy()
        self.exec_params: Dict[str, Any] = {}  # Parameters used during execution (resolved variables)
        self.init_params_list: List[str] = [key for key in params.keys() if not (key.startswith('_') and key.endswith('_'))]
        logger.debug(f"Initialized block {self.name} with params: {self.params}")

        self.fn_name: str = fn_name
        self.external: bool = external

        self.port_radius: int = 8
        self.in_ports: int = in_ports
        self.out_ports: int = out_ports

        self.params.update({'_name_': self.name, '_inputs_': self.in_ports, '_outputs_': self.out_ports})
        self.rectf: QRect = QRect(self.left - self.port_radius, self.top, self.width + 2 * self.port_radius, self.height)

        self.in_coords: List[QPoint] = []
        self.out_coords: List[QPoint] = []
        self.io_edit: Union[bool, str] = io_edit
        self.update_Block()

        self.b_type: int = b_type
        self.dragging: bool = False
        self.selected: bool = False

        self.font_size: int = 11  # Reduced from 14 for better fit
        self.font: QFont = QFont()
        self.font.setPointSize(self.font_size)

        self.hierarchy: int = -1
        self.data_recieved: int = 0
        self.computed_data: bool = False
        self.data_sent: int = 0
        self.input_queue: Dict[int, Any] = {}

        # These should be set to match your DSim class attributes
        self.ls_width = 5
        self.l_width = 5
        self.rectf = QRect(self.left - self.port_radius, self.top, self.width + 2 * self.port_radius, self.height)
        logging.debug(f"Block initialized: {self.name}")

        if block_class:
            self.block_instance = block_class()

            # Check if block supports dynamic port configuration and update accordingly
            if hasattr(self.block_instance, 'get_inputs'):
                try:
                    # Get initial input configuration based on default params
                    initial_inputs = self.block_instance.get_inputs(self.params)
                    initial_input_count = len(initial_inputs)

                    # Update port count if different from default
                    if initial_input_count != self.in_ports:
                        logger.info(f"Setting dynamic input ports for {self.name}: {initial_input_count}")
                        self.in_ports = initial_input_count
                        self.params['_inputs_'] = initial_input_count
                        # Update block geometry and port positions
                        self.update_Block()
                except Exception as e:
                    logger.error(f"Error setting initial dynamic ports for {self.name}: {str(e)}")
        else:
            self.block_instance = None





    def toggle_selection(self) -> None:
        """Toggle the selection state of this block."""
        self.selected = not self.selected

    def update_Block(self) -> None:
        """Update block geometry and port positions based on current state."""
        self.in_coords = []
        self.out_coords = []
        self.input_queue = {}
        for i in range(self.in_ports):
            self.input_queue[i] = None

        # Determine input groups if provided by block instance
        control_group_indices = []
        data_group_indices = []
        if hasattr(self, 'block_instance') and self.block_instance:
            try:
                port_defs = self.block_instance.get_inputs(self.params) if hasattr(self.block_instance, 'get_inputs') else self.block_instance.inputs
            except Exception:
                port_defs = []
            for idx, pdef in enumerate(port_defs or []):
                if isinstance(pdef, dict) and pdef.get("group") == "control":
                    control_group_indices.append(idx)
                else:
                    data_group_indices.append(idx)

        port_height = max(self.out_ports, self.in_ports) * self.port_radius * 2
        if port_height > self.height:
            self.height = port_height + 10
        elif port_height < self.height_base:
            self.height = self.height_base
        elif port_height < self.height:
            self.height = port_height + 10
        self.rectf = QRect(self.left - self.port_radius, self.top, self.width + 2 * self.port_radius, self.height)

        port_spacing = max(self.port_radius * 4, self.height / (max(self.in_ports, self.out_ports) + 1))

        in_x = self.left if not self.flipped else self.left + self.width
        out_x = self.left + self.width if not self.flipped else self.left

        grid_size = 10

        # Check if block wants port grid snapping (property-based, not hardcoded)
        use_grid_snap = True
        if hasattr(self, 'block_instance') and self.block_instance and hasattr(self.block_instance, 'use_port_grid_snap'):
            use_grid_snap = self.block_instance.use_port_grid_snap

        # Helper to compute evenly spaced Y positions
        def spaced_positions(count, top, height):
            return [int(top + height * (i + 1) / (count + 1)) for i in range(count)]

        # Assign Y positions with grouping: control at top section, data below
        port_y_positions = {}
        ctrl_section_height = int(self.height * 0.35) if control_group_indices else 0
        data_section_height = self.height - ctrl_section_height
        ctrl_top = self.top
        data_top = self.top + ctrl_section_height

        if control_group_indices:
            ctrl_positions = spaced_positions(len(control_group_indices), ctrl_top, max(ctrl_section_height, self.port_radius * 2))
            for idx, pos in zip(control_group_indices, ctrl_positions):
                port_y_positions[idx] = pos

        data_indices = [i for i in range(self.in_ports) if i not in control_group_indices]
        if data_indices:
            data_positions = spaced_positions(len(data_indices), data_top, max(data_section_height, self.port_radius * 2))
            for idx, pos in zip(data_indices, data_positions):
                port_y_positions[idx] = pos

        if self.in_ports > 0:
            for i in range(self.in_ports):
                if port_y_positions:
                    port_y_float = port_y_positions.get(i, self.top + self.height * (i + 1) / (self.in_ports + 1))
                else:
                    port_y_float = self.top + self.height * (i + 1) / (self.in_ports + 1)
                if use_grid_snap:
                    port_y = int(round(port_y_float / grid_size) * grid_size)
                else:
                    port_y = int(port_y_float)
                port_in = QPoint(in_x, port_y)
                self.in_coords.append(port_in)
        if self.out_ports > 0:
            for j in range(self.out_ports):
                port_y_float = self.top + self.height * (j + 1) / (self.out_ports + 1)
                if use_grid_snap:
                    port_y = int(round(port_y_float / grid_size) * grid_size)
                else:
                    port_y = int(port_y_float)
                port_out = QPoint(out_x, port_y)
                self.out_coords.append(port_out)

    def draw_Block(self, painter: Optional[QPainter], draw_name: bool = True, draw_ports: bool = True) -> None:
        """
        Draw this block on the canvas with modern styling, shadows, and depth.

        Args:
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

        if self.block_fn == "Gain":
            # Shadow for triangle
            shadow_points = QPolygonF()
            if not self.flipped:
                shadow_points.append(QPoint(self.left + shadow_offset, self.top + shadow_offset))
                shadow_points.append(QPoint(self.left + self.width + shadow_offset, int(self.top + self.height / 2) + shadow_offset))
                shadow_points.append(QPoint(self.left + shadow_offset, self.top + self.height + shadow_offset))
            else:
                shadow_points.append(QPoint(self.left + self.width + shadow_offset, self.top + shadow_offset))
                shadow_points.append(QPoint(self.left + shadow_offset, int(self.top + self.height / 2) + shadow_offset))
                shadow_points.append(QPoint(self.left + self.width + shadow_offset, self.top + self.height + shadow_offset))
            painter.drawPolygon(shadow_points)
        else:
            # Shadow for rounded rectangle
            radius = 12
            shadow_rect = QRect(self.left + shadow_offset, self.top + shadow_offset, self.width, self.height)
            painter.drawRoundedRect(shadow_rect, radius, radius)

        # Determine border color based on block category
        # Get category-specific colors if available
        category_color = self.b_color
        border_color = theme_manager.get_color('border_primary')

        # Try to get category-specific border color
        if hasattr(self, 'category'):
            category_lower = self.category.lower() if isinstance(self.category, str) else str(self.category).lower()
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
        if self.selected:
            border_color = theme_manager.get_color('block_selected')
            # Optionally brighten the background for selected blocks
            if not self.selected:
                painter.setBrush(category_color)
            else:
                # Create a lighter version for selected state
                painter.setBrush(self.b_color)

        # Draw main block shape
        painter.setBrush(self.b_color)
        painter.setPen(QPen(border_color, 3 if self.selected else 2))

        if self.block_fn == "Gain":
            # Draw a triangle for the Gain block
            points = QPolygonF()
            if not self.flipped:
                points.append(QPoint(self.left, self.top))
                points.append(QPoint(self.left + self.width, int(self.top + self.height / 2)))
                points.append(QPoint(self.left, self.top + self.height))
            else:
                points.append(QPoint(self.left + self.width, self.top))
                points.append(QPoint(self.left, int(self.top + self.height / 2)))
                points.append(QPoint(self.left + self.width, self.top + self.height))
            painter.drawPolygon(points)
        else:
            # Draw a rounded rectangle for all other blocks with softer corners
            radius = 12
            painter.drawRoundedRect(QRect(self.left, self.top, self.width, self.height), radius, radius)

        # Draw block-specific icon if available
        # Use dark color for icons to contrast with bright block backgrounds
        # This works well in both light and dark modes since block colors are vibrant
        icon_pen = QPen(QColor('#1F2937'), 2)
        painter.setPen(icon_pen)
        
        path = QPainterPath()
        if self.block_fn == "Step":
            path.moveTo(0.1, 0.7)
            path.lineTo(0.5, 0.7)
            path.lineTo(0.5, 0.3)
            path.lineTo(0.9, 0.3)
        elif self.block_fn == "Ramp":
            path.moveTo(0.1, 0.9)
            path.lineTo(0.9, 0.1)
        elif self.block_fn == "Sine":
            path.moveTo(0.1, 0.5)
            path.quadTo(0.3, 0.1, 0.5, 0.5)
            path.quadTo(0.7, 0.9, 0.9, 0.5)
        elif self.block_fn == "SgProd":
            path.moveTo(0.2, 0.2)
            path.lineTo(0.8, 0.8)
            path.moveTo(0.2, 0.8)
            path.lineTo(0.8, 0.2)
        elif self.block_fn == "TranFn":
            font = painter.font()
            original_size = font.pointSize()
            font.setPointSize(original_size + 2)
            font.setItalic(True)
            painter.setFont(font)
            # Use dark color for text symbols on blocks
            painter.setPen(QColor('#1F2937'))

            # Draw B(s)
            rect_top = QRect(self.left, self.top, self.width, self.height // 2)
            painter.drawText(rect_top, Qt.AlignCenter, "B(s)")

            # Draw divisor line
            line_y = self.top + self.height // 2
            painter.drawLine(self.left + 10, line_y, self.left + self.width - 10, line_y)

            # Draw A(s)
            rect_bottom = QRect(self.left, self.top + self.height // 2, self.width, self.height // 2)
            painter.drawText(rect_bottom, Qt.AlignCenter, "A(s)")

            font.setItalic(False)
            font.setPointSize(original_size)
            painter.setFont(font)
        elif self.block_fn == "Demux":
            # Draw a stylized demultiplexer symbol
            path.moveTo(0.2, 0.5)  # Input line
            path.lineTo(0.4, 0.5)
            path.moveTo(0.4, 0.2)  # Main body
            path.lineTo(0.4, 0.8)
            path.lineTo(0.8, 0.8)
            path.lineTo(0.8, 0.2)
            path.lineTo(0.4, 0.2)
            path.moveTo(0.8, 0.3)  # Output line 1
            path.lineTo(1.0, 0.3)
            path.moveTo(0.8, 0.7)  # Output line 2
            path.lineTo(1.0, 0.7)
        elif self.block_fn == "Mux":
            # Draw a stylized multiplexer symbol
            path.moveTo(0.2, 0.3)  # Input line 1
            path.lineTo(0.4, 0.3)
            path.moveTo(0.2, 0.7)  # Input line 2
            path.lineTo(0.4, 0.7)
            path.moveTo(0.4, 0.2)  # Main body
            path.lineTo(0.8, 0.4)
            path.lineTo(0.8, 0.6)
            path.lineTo(0.4, 0.8)
            path.lineTo(0.4, 0.2)
            path.moveTo(0.8, 0.5)  # Output line
            path.lineTo(1.0, 0.5)
        elif self.block_fn == "BodeMag":
            # Draw axes
            path.moveTo(0.1, 0.9) # x-axis
            path.lineTo(0.9, 0.9)
            path.moveTo(0.1, 0.9) # y-axis
            path.lineTo(0.1, 0.1)
            # Draw plot line
            path.moveTo(0.1, 0.4)
            path.lineTo(0.4, 0.4)
            path.lineTo(0.6, 0.7)
            path.lineTo(0.9, 0.7)
        elif self.block_fn == "RootLocus":
            # Draw axes
            path.moveTo(0.1, 0.5) # horizontal axis
            path.lineTo(0.9, 0.5)
            path.moveTo(0.5, 0.1) # vertical axis
            path.lineTo(0.5, 0.9)
            # Draw root locus path (stylized)
            path.moveTo(0.3, 0.5)
            path.quadTo(0.35, 0.35, 0.5, 0.35)
            path.quadTo(0.65, 0.35, 0.7, 0.5)
            path.moveTo(0.3, 0.5)
            path.quadTo(0.35, 0.65, 0.5, 0.65)
            path.quadTo(0.65, 0.65, 0.7, 0.5)
            # Draw some poles (×) and zeros (○)
            path.moveTo(0.25, 0.45)
            path.lineTo(0.35, 0.55)
            path.moveTo(0.25, 0.55)
            path.lineTo(0.35, 0.45)
        elif self.block_fn == "Deriv":
            font = painter.font()
            original_size = font.pointSize()
            font.setPointSize(original_size + 2)
            font.setItalic(True)
            painter.setFont(font)
            # Use dark color for text symbols on blocks
            painter.setPen(QColor('#1F2937'))

            # Draw dy
            rect_top = QRect(self.left, self.top, self.width, self.height // 2)
            painter.drawText(rect_top, Qt.AlignCenter, "dy")

            # Draw divisor line
            line_y = self.top + self.height // 2
            painter.drawLine(self.left + 10, line_y, self.left + self.width - 10, line_y)

            # Draw dt
            rect_bottom = QRect(self.left, self.top + self.height // 2, self.width, self.height // 2)
            painter.drawText(rect_bottom, Qt.AlignCenter, "dt")

            font.setItalic(False)
            font.setPointSize(original_size)
            painter.setFont(font)
        elif self.block_fn == "TranFn":
            font = painter.font()
            original_size = font.pointSize()
            font.setPointSize(original_size + 2)
            font.setItalic(True)
            painter.setFont(font)
            # Use dark color for text symbols on blocks
            painter.setPen(QColor('#1F2937'))

            # Draw B(s)
            rect_top = QRect(self.left, self.top, self.width, self.height // 2)
            painter.drawText(rect_top, Qt.AlignCenter, "B(s)")

            # Draw divisor line
            line_y = self.top + self.height // 2
            painter.drawLine(self.left + 10, line_y, self.left + self.width - 10, line_y)

            # Draw A(s)
            rect_bottom = QRect(self.left, self.top + self.height // 2, self.width, self.height // 2)
            painter.drawText(rect_bottom, Qt.AlignCenter, "A(s)")

            font.setItalic(False)
            font.setPointSize(original_size)
            painter.setFont(font)
        elif self.block_fn == "DiscreteTranFn":
            font = painter.font()
            original_size = font.pointSize()
            font.setPointSize(original_size + 2)
            font.setItalic(True)
            painter.setFont(font)
            # Use dark color for text symbols on blocks
            painter.setPen(QColor('#1F2937'))

            # Draw B(z)
            rect_top = QRect(self.left, self.top, self.width, self.height // 2)
            painter.drawText(rect_top, Qt.AlignCenter, "B(z)")

            # Draw divisor line
            line_y = self.top + self.height // 2
            painter.drawLine(self.left + 10, line_y, self.left + self.width - 10, line_y)

            # Draw A(z)
            rect_bottom = QRect(self.left, self.top + self.height // 2, self.width, self.height // 2)
            painter.drawText(rect_bottom, Qt.AlignCenter, "A(z)")

            font.setItalic(False)
            font.setPointSize(original_size)
            painter.setFont(font)
        elif self.block_fn == "Integrator":
            # Use 1/s notation (transfer function representation)
            font = painter.font()
            original_size = font.pointSize()
            font.setPointSize(original_size + 4)
            font.setItalic(True)
            painter.setFont(font)
            # Use dark color for text symbols on blocks
            painter.setPen(QColor('#1F2937'))

            # Draw 1
            rect_top = QRect(self.left, self.top, self.width, self.height // 2 - 2)
            painter.drawText(rect_top, Qt.AlignCenter | Qt.AlignBottom, "1")

            # Draw divisor line
            line_y = self.top + self.height // 2
            painter.drawLine(self.left + 15, line_y, self.left + self.width - 15, line_y)

            # Draw s
            rect_bottom = QRect(self.left, self.top + self.height // 2 + 2, self.width, self.height // 2)
            painter.drawText(rect_bottom, Qt.AlignCenter | Qt.AlignTop, "s")

            font.setItalic(False)
            font.setPointSize(original_size)
            painter.setFont(font)
        elif self.block_fn == "Scope":
            path.moveTo(0.1, 0.9)
            path.lineTo(0.9, 0.9) # x-axis
            path.moveTo(0.1, 0.9)
            path.lineTo(0.1, 0.1) # y-axis
            path.moveTo(0.1, 0.6)
            path.quadTo(0.3, 0.2, 0.5, 0.6)
            path.quadTo(0.7, 1.0, 0.9, 0.6)
        elif self.block_fn == "Sum":
            font = painter.font()
            original_size = font.pointSize()
            font.setPointSize(original_size + 4)
            painter.setFont(font)
            sign_text = self.params.get('sign', '++')
            painter.drawText(self.rect, Qt.AlignCenter, sign_text)
            font.setPointSize(original_size)
            painter.setFont(font)
        elif self.block_fn == "SgProd":
            path.moveTo(0.2, 0.2)
            path.lineTo(0.8, 0.8)
            path.moveTo(0.2, 0.8)
            path.lineTo(0.8, 0.2)


        elif self.block_fn == "Noise":
            # Draw a random/noisy signal
            path.moveTo(0.1, 0.5)
            path.lineTo(0.2, 0.3)
            path.lineTo(0.3, 0.7)
            path.lineTo(0.4, 0.4)
            path.lineTo(0.5, 0.6)
            path.lineTo(0.6, 0.2)
            path.lineTo(0.7, 0.8)
            path.lineTo(0.8, 0.5)
            path.lineTo(0.9, 0.6)
        elif self.block_fn == "Exp":
            font = painter.font()
            original_size = font.pointSize()
            font.setPointSize(original_size + 4)
            font.setItalic(True)
            painter.setFont(font)
            # Use dark color for text symbols on blocks
            painter.setPen(QColor('#1F2937'))
            painter.drawText(self.rect, Qt.AlignCenter, "eˣ")
            font.setItalic(False)
            font.setPointSize(original_size)
            painter.setFont(font)
        elif self.block_fn == "Term":
            # Draw a ground/terminator symbol
            path.moveTo(0.5, 0.2)
            path.lineTo(0.5, 0.6)
            path.moveTo(0.2, 0.6)
            path.lineTo(0.8, 0.6)
            path.moveTo(0.3, 0.75)
            path.lineTo(0.7, 0.75)
            path.moveTo(0.4, 0.9)
            path.lineTo(0.6, 0.9)
        elif self.block_fn == "Export":
            # Draw an arrow pointing out of a box
            path.moveTo(0.2, 0.2)
            path.lineTo(0.8, 0.2)
            path.lineTo(0.8, 0.8)
            path.lineTo(0.2, 0.8)
            path.lineTo(0.2, 0.2)
            path.moveTo(0.5, 0.5)
            path.lineTo(1.0, 0.5)
            path.moveTo(0.8, 0.3)
            path.lineTo(1.0, 0.5)
            path.lineTo(0.8, 0.7)
        elif self.block_fn == "ZeroOrderHold":
            # Draw a staircase symbol
            path.moveTo(0.1, 0.8)
            path.lineTo(0.3, 0.8)
            path.lineTo(0.3, 0.5)
            path.lineTo(0.6, 0.5)
            path.lineTo(0.6, 0.2)
            path.lineTo(0.9, 0.2)
        elif self.block_fn == "PRBS":
            # Irregular pulse train (variable widths/heights)
            path.moveTo(0.1, 0.7)
            path.lineTo(0.18, 0.7)
            path.lineTo(0.18, 0.3)
            path.lineTo(0.32, 0.3)
            path.lineTo(0.32, 0.7)
            path.lineTo(0.45, 0.7)
            path.lineTo(0.45, 0.4)
            path.lineTo(0.6, 0.4)
            path.lineTo(0.6, 0.7)
            path.lineTo(0.78, 0.7)
            path.lineTo(0.78, 0.3)
            path.lineTo(0.9, 0.3)
            path.lineTo(0.9, 0.7)
        elif self.block_fn == "Hysteresis":
            # Relay hysteresis: two thresholds and retained output
            # Threshold lines
            path.moveTo(0.1, 0.25); path.lineTo(0.9, 0.25)
            path.moveTo(0.1, 0.75); path.lineTo(0.9, 0.75)
            # Output levels
            path.moveTo(0.15, 0.85); path.lineTo(0.85, 0.85)
            path.moveTo(0.15, 0.15); path.lineTo(0.85, 0.15)
            # Rising transition arrow (at upper)
            path.moveTo(0.3, 0.6); path.lineTo(0.3, 0.25); path.lineTo(0.25, 0.32)
            # Falling transition arrow (at lower)
            path.moveTo(0.6, 0.4); path.lineTo(0.6, 0.75); path.lineTo(0.55, 0.68)
        elif self.block_fn == "Deadband":
            # Center deadband band and zeroed output inside
            # Band markers
            path.moveTo(0.35, 0.25); path.lineTo(0.35, 0.75)
            path.moveTo(0.65, 0.25); path.lineTo(0.65, 0.75)
            # Signal clamped to center inside band
            path.moveTo(0.1, 0.5); path.lineTo(0.35, 0.5)
            path.lineTo(0.65, 0.5)
            # Outside band, rising
            path.lineTo(0.8, 0.2)
            path.lineTo(0.9, 0.2)
            # Outside band, falling
            path.moveTo(0.65, 0.5)
            path.lineTo(0.8, 0.8)
            path.lineTo(0.9, 0.8)
        elif self.block_fn == "Switch":
            # 3-input selector icon
            path.moveTo(0.15, 0.3)  # ctrl arrow to decision
            path.lineTo(0.4, 0.5)
            path.moveTo(0.15, 0.7)  # in_false
            path.lineTo(0.4, 0.6)
            path.moveTo(0.15, 0.9)  # in_true
            path.lineTo(0.4, 0.7)
            # decision diamond
            path.moveTo(0.45, 0.5)
            path.lineTo(0.55, 0.4)
            path.lineTo(0.65, 0.5)
            path.lineTo(0.55, 0.6)
            path.lineTo(0.45, 0.5)
            # output
            path.moveTo(0.65, 0.5)
            path.lineTo(0.9, 0.5)
            # tiny port labels
            font = painter.font()
            orig = font.pointSize()
            font.setPointSize(orig - 2)
            painter.setFont(font)
            painter.drawText(QRect(self.left + 4, self.top + 4, self.width//2, self.height//3), Qt.AlignLeft | Qt.AlignTop, "ctrl")
            painter.drawText(QRect(self.left + 4, self.top + self.height//2, self.width//2, self.height//2), Qt.AlignLeft | Qt.AlignBottom, "in0 / in1")
            font.setPointSize(orig)
            painter.setFont(font)
        elif self.block_fn == "Saturation":
            # Clipped sine-like signal against min/max rails
            path.moveTo(0.1, 0.8)
            path.lineTo(0.9, 0.8)  # upper rail
            path.moveTo(0.1, 0.2)
            path.lineTo(0.9, 0.2)  # lower rail
            path.moveTo(0.15, 0.5)
            path.quadTo(0.3, 0.2, 0.45, 0.2)
            path.lineTo(0.55, 0.2)
            path.quadTo(0.7, 0.8, 0.85, 0.8)
        elif self.block_fn == "RateLimiter":
            # Show desired steep ramp (dashed idea) and limited (solid) ramp
            # Limited ramp
            path.moveTo(0.1, 0.8)
            path.lineTo(0.35, 0.6)
            path.lineTo(0.6, 0.4)
            path.lineTo(0.9, 0.4)
            # Desired (steeper) ramp hint
            path.moveTo(0.1, 0.8)
            path.lineTo(0.5, 0.2)
            # Cap marker
            path.moveTo(0.6, 0.45)
            path.lineTo(0.66, 0.33)
            path.lineTo(0.72, 0.45)
            path.lineTo(0.6, 0.45)
            # Small label
            font = painter.font()
            original_size = font.pointSize()
            font.setPointSize(original_size - 1)
            painter.setFont(font)
            painter.setPen(QColor('#1F2937'))
            painter.drawText(QRect(self.left, self.top + self.height // 2, self.width, self.height // 2), Qt.AlignCenter, "du/dt")
            font.setPointSize(original_size)
            painter.setFont(font)
        elif self.block_fn == "PID":
            # Center "PID" label and small sp/pv hints
            font = painter.font()
            original_size = font.pointSize()
            font.setPointSize(original_size + 3)
            font.setWeight(600)
            painter.setFont(font)
            painter.setPen(QColor('#1F2937'))
            painter.drawText(self.rect, Qt.AlignCenter, "PID")
            font.setPointSize(original_size - 1)
            font.setWeight(400)
            painter.setFont(font)
            painter.drawText(QRect(self.left + 4, self.top + 2, self.width // 2, self.height // 2), Qt.AlignLeft | Qt.AlignTop, "sp")
            painter.drawText(QRect(self.left + 4, self.top + self.height // 2, self.width // 2, self.height // 2), Qt.AlignLeft | Qt.AlignBottom, "pv")
            font.setPointSize(original_size)
            painter.setFont(font)
        elif self.block_fn == "StateSpace":
            # Draw x' = Ax + Bu
            font = painter.font()
            original_size = font.pointSize()
            font.setPointSize(original_size - 1) # Slightly smaller font
            painter.setFont(font)
            painter.setPen(QColor('#1F2937'))
            
            rect = QRect(self.left, self.top, self.width, self.height)
            painter.drawText(rect, Qt.AlignCenter, "x' = Ax+Bu\ny = Cx+Du")
            
            font.setPointSize(original_size)
            painter.setFont(font)
        elif self.block_fn == "DiscreteStateSpace":
            # Draw x[k+1] = Ax[k] + ...
            font = painter.font()
            original_size = font.pointSize()
            font.setPointSize(original_size - 2) # Even smaller font for longer text
            painter.setFont(font)
            painter.setPen(QColor('#1F2937'))
            
            rect = QRect(self.left, self.top, self.width, self.height)
            painter.drawText(rect, Qt.AlignCenter, "x[k+1]=Ax+Bu\ny[k]=Cx+Du")
            
            font.setPointSize(original_size)
            painter.setFont(font)
        elif self.block_fn == "External":
            # Draw a stylized letter 'E'
            path.moveTo(0.2, 0.2)
            path.lineTo(0.8, 0.2)
            path.moveTo(0.2, 0.5)
            path.lineTo(0.6, 0.5)
            path.moveTo(0.2, 0.8)
            path.lineTo(0.8, 0.8)
            path.moveTo(0.2, 0.2)
            path.lineTo(0.2, 0.8)
        if not path.isEmpty():
            margin = self.width * 0.2
            transform = QTransform()
            if self.flipped:
                transform.translate(self.left + self.width - margin, self.top + margin)
                transform.scale(-(self.width - 2 * margin), self.height - 2 * margin)
            else:
                transform.translate(self.left + margin, self.top + margin)
                transform.scale(self.width - 2 * margin, self.height - 2 * margin)
            
            scaled_path = transform.map(path)
            painter.drawPath(scaled_path)

        # Draw ports with modern styling (only if requested)
        if draw_ports:
            self.draw_ports(painter)

        if draw_name:
            # Draw block name below the block with better typography
            text_color = theme_manager.get_color('text_primary')
            painter.setPen(text_color)
            font = self.font
            font.setWeight(400)  # Normal weight (not bold)
            painter.setFont(font)
            text_rect = QRect(self.left, self.top + self.height + 2, self.width, 28)
            painter.drawText(text_rect, Qt.AlignHCenter | Qt.AlignTop, self.username)

        # Enhanced selection visualization
        if self.selected:
            selection_color = theme_manager.get_color('block_selected')

            # Draw selection outline with rounded corners
            painter.setPen(QPen(selection_color, 3, Qt.SolidLine))
            painter.setBrush(Qt.NoBrush)

            # Selection rectangle with padding
            padding = 4
            selection_rect = QRect(
                self.left - padding,
                self.top - padding,
                self.width + 2 * padding,
                self.height + 2 * padding
            )
            painter.drawRoundedRect(selection_rect, 14, 14)

            # Optional: Draw corner handles for resize (can be added later)
            # For now, just the glowing outline effect
    
    def draw_selected(self, painter):
        painter.setPen(QColor('black'))
        painter.setFont(self.font)
        painter.drawText(QRect(self.left, self.top - self.ls_width - 20, self.width, 20), Qt.AlignCenter, self.username)
        
        pen = QPen(QColor('black'), self.l_width)
        painter.setPen(pen)
        painter.drawLine(self.left - self.ls_width, self.top - self.ls_width, self.left + self.width + self.ls_width, self.top - self.ls_width)
        painter.drawLine(self.left - self.ls_width, self.top - self.ls_width, self.left - self.ls_width, self.top + self.height + self.ls_width)
        painter.drawLine(self.left + self.width + self.ls_width, self.top + self.height + self.ls_width, self.left + self.width + 5, self.top - self.ls_width)
        painter.drawLine(self.left + self.width + self.ls_width, self.top + self.height + self.ls_width, self.left - self.ls_width, self.top + self.height + self.ls_width)
        
        if self.external:
            painter.drawText(QRect(self.left, self.top + self.height + 15, self.width, 20), Qt.AlignCenter, self.params['filename'])

    def draw_ports(self, painter: Optional[QPainter]) -> None:
        """
        Draw input and output ports with modern styling.

        Renders ports with radial gradients for depth and glossy highlights.
        Uses theme-aware colors for input and output ports.

        Args:
            painter: QPainter instance for rendering
        """
        if painter is None:
            return

        from PyQt5.QtGui import QRadialGradient

        port_input_color = theme_manager.get_color('port_input')
        port_output_color = theme_manager.get_color('port_output')

        # Make ports slightly smaller for a cleaner look
        port_draw_radius = self.port_radius - 1

        # Input ports with radial gradient and glossy effect
        for port_in_location in self.in_coords:
            # Create radial gradient for depth
            gradient = QRadialGradient(port_in_location.x(), port_in_location.y(), port_draw_radius)

            # Lighter center, darker edge for depth
            lighter_input = port_input_color.lighter(130)
            gradient.setColorAt(0.0, lighter_input)
            gradient.setColorAt(0.7, port_input_color)
            gradient.setColorAt(1.0, port_input_color.darker(110))

            painter.setBrush(gradient)
            painter.setPen(QPen(port_input_color.darker(140), 2.0))
            painter.drawEllipse(port_in_location, port_draw_radius, port_draw_radius)

            # Add subtle highlight for glossy effect
            painter.setPen(Qt.NoPen)
            highlight_color = QColor(255, 255, 255, 50)
            painter.setBrush(highlight_color)
            highlight_offset = int(port_draw_radius * 0.3)
            highlight_size = int(port_draw_radius * 0.4)
            painter.drawEllipse(
                port_in_location.x() - highlight_offset,
                port_in_location.y() - highlight_offset,
                highlight_size,
                highlight_size
            )

        # Output ports with radial gradient and glossy effect
        for port_out_location in self.out_coords:
            # Create radial gradient for depth
            gradient = QRadialGradient(port_out_location.x(), port_out_location.y(), port_draw_radius)

            # Lighter center, darker edge for depth
            lighter_output = port_output_color.lighter(130)
            gradient.setColorAt(0.0, lighter_output)
            gradient.setColorAt(0.7, port_output_color)
            gradient.setColorAt(1.0, port_output_color.darker(110))

            painter.setBrush(gradient)
            painter.setPen(QPen(port_output_color.darker(140), 2.0))
            painter.drawEllipse(port_out_location, port_draw_radius, port_draw_radius)

            # Add subtle highlight for glossy effect
            painter.setPen(Qt.NoPen)
            highlight_color = QColor(255, 255, 255, 50)
            painter.setBrush(highlight_color)
            highlight_offset = int(port_draw_radius * 0.3)
            highlight_size = int(port_draw_radius * 0.4)
            painter.drawEllipse(
                port_out_location.x() - highlight_offset,
                port_out_location.y() - highlight_offset,
                highlight_size,
                highlight_size
            )

    def draw_resize_handles(self, painter):
        """Draw resize handles on the corners and edges of selected blocks."""
        if not self.selected:
            return

        # Import resize handle size from config
        try:
            from config.block_sizes import RESIZE_HANDLE_SIZE
            handle_size = RESIZE_HANDLE_SIZE
        except ImportError:
            handle_size = 8

        # Handle styling
        handle_color = theme_manager.get_color('accent_primary')
        border_color = theme_manager.get_color('border_primary')

        painter.save()
        painter.setPen(QPen(border_color, 1))
        painter.setBrush(handle_color)

        # Define handle positions
        half_handle = handle_size // 2
        handles = {
            'top_left': (self.left - half_handle, self.top - half_handle),
            'top_right': (self.left + self.width - half_handle, self.top - half_handle),
            'bottom_left': (self.left - half_handle, self.top + self.height - half_handle),
            'bottom_right': (self.left + self.width - half_handle, self.top + self.height - half_handle),
            # Edge handles for more precise resizing
            'top': (self.left + self.width//2 - half_handle, self.top - half_handle),
            'bottom': (self.left + self.width//2 - half_handle, self.top + self.height - half_handle),
            'left': (self.left - half_handle, self.top + self.height//2 - half_handle),
            'right': (self.left + self.width - half_handle, self.top + self.height//2 - half_handle),
        }

        # Draw handles
        for handle_name, (x, y) in handles.items():
            painter.drawRect(x, y, handle_size, handle_size)

        painter.restore()

    def get_resize_handle_at(self, pos):
        """
        Check if a position is over a resize handle.

        Args:
            pos: QPoint position to check

        Returns:
            Handle name if over a handle, None otherwise
        """
        if not self.selected:
            return None

        try:
            from config.block_sizes import RESIZE_HANDLE_SIZE
            handle_size = RESIZE_HANDLE_SIZE
        except ImportError:
            handle_size = 8

        # Define handle positions
        half_handle = handle_size // 2
        handles = {
            'top_left': (self.left - half_handle, self.top - half_handle),
            'top_right': (self.left + self.width - half_handle, self.top - half_handle),
            'bottom_left': (self.left - half_handle, self.top + self.height - half_handle),
            'bottom_right': (self.left + self.width - half_handle, self.top + self.height - half_handle),
            'top': (self.left + self.width//2 - half_handle, self.top - half_handle),
            'bottom': (self.left + self.width//2 - half_handle, self.top + self.height - half_handle),
            'left': (self.left - half_handle, self.top + self.height//2 - half_handle),
            'right': (self.left + self.width - half_handle, self.top + self.height//2 - half_handle),
        }

        # Check if position is within any handle
        for handle_name, (x, y) in handles.items():
            if (x <= pos.x() <= x + handle_size and
                y <= pos.y() <= y + handle_size):
                return handle_name

        return None

    def port_collision(self, pos):
        if isinstance(pos, tuple):
            pos = QPoint(*pos)
        
        enlarged_radius = self.port_radius * 2  # Increase the clickable area

        for i, coord in enumerate(self.in_coords):
            if (pos - coord).manhattanLength() <= enlarged_radius:
                return ("i", i)
        
        for i, coord in enumerate(self.out_coords):
            if (pos - coord).manhattanLength() <= enlarged_radius:
                return ("o", i)
        
        return (-1, -1)

    def relocate_Block(self, new_pos):
        logger.debug(f"Relocating block {self.name} to {new_pos}")
        self.left = new_pos.x()
        self.top = new_pos.y()
        self.rect.moveTo(self.left, self.top) # Update the QRect used for collision detection
        self.rectf.moveTopLeft(QPoint(self.left, self.top)) # Update the QRectF
        self.update_Block()

    def resize_Block(self, new_width, new_height):
        """
        Resize the block to new dimensions.

        Args:
            new_width: New width in pixels
            new_height: New height in pixels
        """
        # Clamp to minimum and maximum sizes
        try:
            from config.block_sizes import clamp_block_size
            new_width, new_height = clamp_block_size(new_width, new_height)
        except ImportError:
            # Fallback min/max if config not available
            new_width = max(50, min(new_width, 300))
            new_height = max(40, min(new_height, 300))

        # Update dimensions
        self.width = new_width
        self.height = new_height
        self.height_base = new_height

        # Update rect properties
        self.rect = QRect(self.left, self.top, self.width, self.height)
        self.rectf = QRect(self.left - self.port_radius, self.top,
                          self.width + 2 * self.port_radius, self.height)

        # Update port positions
        self.update_Block()

        logger.debug(f"Resized block {self.name} to {new_width}x{new_height}")

    def change_port_numbers(self):
        logger.debug(f"Changing port numbers for block: {self.name}")
        
        if self.io_edit == 'both':
            # Inputs and outputs can be edited
            dialog = PortDialog(self.name, {'inputs': self.in_ports, 'outputs': self.out_ports})
            if dialog.exec_():
                new_io = dialog.get_values()
                self.in_ports = int(new_io['inputs'])
                self.out_ports = int(new_io['outputs'])
                logger.debug(f"Changed input ports to {self.in_ports} and output ports to {self.out_ports}")

        elif self.io_edit == 'input':
            # Only inputs can be edited
            dialog = PortDialog(self.name, {'inputs': self.in_ports})
            if dialog.exec_():
                new_io = dialog.get_values()
                self.in_ports = int(new_io['inputs'])
                logger.debug(f"Changed input ports to {self.in_ports}")

        elif self.io_edit == 'output':
            # Only outputs can be edited
            dialog = PortDialog(self.name, {'outputs': self.out_ports})
            if dialog.exec_():
                new_io = dialog.get_values()
                self.out_ports = int(new_io['outputs'])
                logger.debug(f"Changed output ports to {self.out_ports}")

        else:
            logger.debug(f"Port number change not allowed for block: {self.name}")
            return

        self.update_Block()

        # To maintain the data in the parameters for the functions
        self.params['_inputs_'] = self.in_ports
        self.params['_outputs_'] = self.out_ports

        logger.debug(f"Port numbers updated for block: {self.name}")

    def saving_params(self):
        ed_dict = {}
        for key in self.params.keys():
            if key in self.init_params_list:
                if isinstance(self.params[key], np.ndarray):
                    arraylist = self.params[key]
                    ed_dict[key] = arraylist.tolist()
                else:
                    ed_dict[key] = self.params[key]
        return ed_dict

    def loading_params(self, new_params):
        """
        Load and normalize parameters, converting lists to numpy arrays.

        Args:
            new_params: Dictionary of parameters to load

        Returns:
            Dictionary with list values converted to numpy arrays
        """
        try:
            for key in new_params.keys():
                if isinstance(new_params[key], list):
                    new_params[key] = np.array(new_params[key])
            return new_params
        except (TypeError, ValueError) as e:
            logger.warning(f"Failed to convert parameter lists to arrays: {e}")
            return new_params

    def change_params(self):
        logger.info(f"change_params called for block {self.name}")
        if not self.initial_params:
            return

        ed_dict = {'Name': self.username}
        for key, value in self.initial_params.items():
            if not (key.startswith('_') and key.endswith('_')):
                ed_dict[key] = self.params.get(key, value)  # Use current value if available

        if len(ed_dict) <= 1:  # Only 'Name' is present
            return

        dialog = ParamDialog(self.name, ed_dict)
        if dialog.exec_():
            new_inputs = dialog.get_values()
            
            if new_inputs['Name'] == '--':
                self.username = self.name
            else:
                self.username = new_inputs['Name']
            new_inputs.pop('Name')

            if new_inputs:
                for key, value in new_inputs.items():
                    if key in self.params:
                        self.params[key] = value
                        self.initial_params[key] = value

                # Check if block supports dynamic port configuration
                if self.block_instance and hasattr(self.block_instance, 'get_inputs'):
                    try:
                        # Get new input configuration based on updated params
                        new_inputs_config = self.block_instance.get_inputs(self.params)
                        new_input_count = len(new_inputs_config)

                        # Update port count if it changed
                        if new_input_count != self.in_ports:
                            logger.info(f"Updating {self.name} input ports from {self.in_ports} to {new_input_count}")
                            self.in_ports = new_input_count
                            self.params['_inputs_'] = new_input_count
                            # Update block geometry and port positions
                            self.update_Block()
                    except Exception as e:
                        logger.error(f"Error updating dynamic ports for {self.name}: {str(e)}")

        if self.external:
            self.load_external_data(params_reset=False)

        if self.block_fn == 'TranFn':
            num = self.params.get('numerator', [])
            den = self.params.get('denominator', [])
            if len(den) > len(num):
                self.b_type = 1
            else:
                self.b_type = 2
            self.params['_init_start_'] = True

        logger.debug(f"Final parameters for {self.name}: {self.params}")
        self.dirty = True

    def load_external_data(self, params_reset=False):
        # Implement this method based on your specific requirements for loading external data
        pass

    def reload_external_data(self):
        # Implement this method based on your specific requirements for reloading external data
        pass

    def update_params(self, new_params):
        if new_params:
            for key, value in new_params.items():
                if key in self.params:
                    self.params[key] = value

        # Check if block supports dynamic port configuration
        if self.block_instance and hasattr(self.block_instance, 'get_inputs'):
            try:
                # Get new input configuration based on updated params
                new_inputs = self.block_instance.get_inputs(self.params)
                new_input_count = len(new_inputs)

                # Update port count if it changed
                if new_input_count != self.in_ports:
                    logger.info(f"Updating {self.name} input ports from {self.in_ports} to {new_input_count}")
                    self.in_ports = new_input_count
                    self.params['_inputs_'] = new_input_count
                    # Update block geometry and port positions
                    self.update_Block()
            except Exception as e:
                logger.error(f"Error updating dynamic ports for {self.name}: {str(e)}")

        if self.block_fn == 'TranFn':
            num = self.params.get('numerator', [])
            den = self.params.get('denominator', [])
            if len(den) > len(num):
                self.b_type = 1
            else:
                self.b_type = 2
            self.params['_init_start_'] = True

        logger.debug(f"Final parameters for {self.name}: {self.params}")
        self.dirty = True
