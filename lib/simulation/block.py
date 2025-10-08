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
                 colors: Optional[Dict[str, QColor]] = None) -> None:
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
        """
        if params is None:
            params = {}

        logger.debug(f"Initializing DBlock {block_fn}{sid}")
        self.name: str = block_fn.lower() + str(sid)
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

        if colors and color in colors:
            self.b_color: QColor = colors[color]
        else:
            self.b_color: QColor = QColor(color)
        self.image: QPixmap = QPixmap()  # Initialize as null QPixmap since no icons are available

        self.params: Dict[str, Any] = params.copy()
        self.initial_params: Dict[str, Any] = params.copy()
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

        self.font_size: int = 14
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
        if self.in_ports > 0:
            for i in range(self.in_ports):
                port_y_float = self.top + self.height * (i + 1) / (self.in_ports + 1)
                port_y = int(round(port_y_float / grid_size) * grid_size)
                port_in = QPoint(in_x, port_y)
                self.in_coords.append(port_in)
        if self.out_ports > 0:
            for j in range(self.out_ports):
                port_y_float = self.top + self.height * (j + 1) / (self.out_ports + 1)
                port_y = int(round(port_y_float / grid_size) * grid_size)
                port_out = QPoint(out_x, port_y)
                self.out_coords.append(port_out)

    def draw_Block(self, painter: Optional[QPainter], draw_name: bool = True) -> None:
        """
        Draw this block on the canvas.

        Args:
            painter: QPainter instance for rendering
            draw_name: Whether to draw the block name/label
        """
        if painter is None:
            return

        painter.setBrush(self.b_color)
        border_color = theme_manager.get_color('border_primary')
        painter.setPen(QPen(border_color, 2))

        # Draw block shape
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
            # Draw a rounded rectangle for all other blocks
            radius = 10
            painter.drawRoundedRect(QRect(self.left, self.top, self.width, self.height), radius, radius)

        # Draw block-specific icon if available
        icon_pen = QPen(theme_manager.get_color('text_primary'), 2)
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
            painter.setPen(theme_manager.get_color('text_primary'))

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
            painter.setPen(theme_manager.get_color('text_primary'))

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
            painter.setPen(theme_manager.get_color('text_primary'))

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
        elif self.block_fn == "Integrator":
            # Use 1/s notation (transfer function representation)
            font = painter.font()
            original_size = font.pointSize()
            font.setPointSize(original_size + 4)
            font.setItalic(True)
            painter.setFont(font)
            painter.setPen(theme_manager.get_color('text_primary'))

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
            painter.setPen(theme_manager.get_color('text_primary'))
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

        # Draw ports
        port_color = theme_manager.get_color('text_primary')
        # Input ports (filled)
        painter.setBrush(port_color)
        painter.setPen(Qt.NoPen)
        for port_in_location in self.in_coords:
            painter.drawEllipse(port_in_location, self.port_radius, self.port_radius)

        # Output ports (outline)
        painter.setBrush(Qt.NoBrush)
        painter.setPen(QPen(port_color, 2))
        for port_out_location in self.out_coords:
            painter.drawEllipse(port_out_location, self.port_radius -1, self.port_radius - 1)
        
        if draw_name:
            # Draw block name below the block
            text_color = theme_manager.get_color('text_primary')
            painter.setPen(text_color)
            painter.setFont(self.font)
            text_rect = QRect(self.left, self.top + self.height, self.width, 25)
            painter.drawText(text_rect, Qt.AlignHCenter | Qt.AlignTop, self.username)

        if self.selected:
            selection_color = theme_manager.get_color('accent_primary')
            painter.setPen(QPen(selection_color, 2))
            painter.setBrush(Qt.NoBrush)
            selection_rect = QRect(self.left - self.port_radius, self.top, self.width + 2 * self.port_radius, self.height + 25)
            painter.drawRect(selection_rect)
    
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

    def resize_Block(self, new_coords):
        self.width = new_coords[0]
        self.height = new_coords[1]
        self.update_Block()

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
        try:
            for key in new_params.keys():
                if isinstance(new_params[key], list):
                    new_params[key] = np.array(new_params[key])
            return new_params
        except:
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

