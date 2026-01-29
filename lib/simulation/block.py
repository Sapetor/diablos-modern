"""
DBlock class - represents a block in the simulation diagram.
"""

import logging
import importlib
import copy
from typing import Dict, List, Optional, Any, Union
import numpy as np
from PyQt5.QtGui import QColor, QFont, QPixmap
from PyQt5.QtCore import QRect, QPoint
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

        # Create block_instance BEFORE update_Block so port positions can use it
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
                except Exception as e:
                    logger.error(f"Error setting initial dynamic ports for {self.name}: {str(e)}")
        else:
            self.block_instance = None

        # Now update_Block can access block_instance for port positioning
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

    @property
    def doc(self) -> str:
        """Get the documentation string from the block instance."""
        if self.block_instance and hasattr(self.block_instance, 'doc'):
            return self.block_instance.doc
        return ""

    def calculate_min_size(self) -> int:
        """
        Calculate the minimum height required for the block based on its ports.
        
        Returns:
            int: The calculated minimum height (usually applied only if greater than default).
        """
        # Constants for layout
        PORT_SPACING = 20  # Vertical spacing between ports
        PORT_MARGIN = 12   # Top/bottom margin
        MIN_HEIGHT = 40    # Absolute minimum height
        
        # Calculate required height for inputs and outputs
        max_ports = max(self.in_ports, self.out_ports)
        
        if max_ports <= 1:
            return self.height_base
            
        required_height = (max_ports * PORT_SPACING) + (PORT_MARGIN * 2)
        
        # Ensure we don't shrink below a reasonable minimum or the base height
        return max(self.height_base, required_height)


        # These should be set to match your DSim class attributes
        self.ls_width = 5
        self.l_width = 5
        self.rectf = QRect(self.left - self.port_radius, self.top, self.width + 2 * self.port_radius, self.height)
        logging.debug(f"Block initialized: {self.name}")





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
        top_port_indices = []  # Ports positioned on top edge
        if hasattr(self, 'block_instance') and self.block_instance:
            try:
                port_defs = self.block_instance.get_inputs(self.params) if hasattr(self.block_instance, 'get_inputs') else self.block_instance.inputs
            except Exception:
                port_defs = []
            for idx, pdef in enumerate(port_defs or []):
                if isinstance(pdef, dict):
                    if pdef.get("position") == "top":
                        top_port_indices.append(idx)
                    elif pdef.get("group") == "control":
                        control_group_indices.append(idx)
                    else:
                        data_group_indices.append(idx)
                else:
                    data_group_indices.append(idx)

        # Calculate minimum height based on ports
        min_height = self.calculate_min_size()
        
        # Ensure block is at least the calculated minimum height
        if self.height < min_height:
            self.height = min_height
        
        # Also respect the base height (initial size) if distinct, though calculate_min_size usually covers this
        if self.height < self.height_base:
            self.height = self.height_base
            
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

        data_indices = [i for i in range(self.in_ports) if i not in control_group_indices and i not in top_port_indices]
        if data_indices:
            data_positions = spaced_positions(len(data_indices), data_top, max(data_section_height, self.port_radius * 2))
            for idx, pos in zip(data_indices, data_positions):
                port_y_positions[idx] = pos

        # Calculate top port positions (horizontal spacing on top edge)
        top_port_positions = {}
        if top_port_indices:
            top_x_positions = [int(self.left + self.width * (i + 1) / (len(top_port_indices) + 1)) for i in range(len(top_port_indices))]
            for idx, x_pos in zip(top_port_indices, top_x_positions):
                top_port_positions[idx] = x_pos

        if self.in_ports > 0:
            for i in range(self.in_ports):
                if i in top_port_indices:
                    # Top port: positioned on top edge
                    port_x = top_port_positions.get(i, self.left + self.width // 2)
                    port_in = QPoint(port_x, self.top)
                else:
                    # Left-side port
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
    def __deepcopy__(self, memo):
        """
        Custom deepcopy implementation to assume QPixmap is not copied (recreate or ignore).
        QPixmap cannot be pickled/copied deeply.
        """
        # Create a new instance
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        
        for k, v in self.__dict__.items():
            if k == 'image' or k == 'pixmap':
                # QPixmap cannot be copied, create a new empty one
                setattr(result, k, QPixmap())
            elif k == 'font':
                # QFont cannot be copied safely in some versions, recreate it
                f = QFont()
                f.setPointSize(self.font_size)
                setattr(result, k, f)
            elif k == 'sub_blocks':
                 # List of blocks, recurse
                 setattr(result, k, copy.deepcopy(v, memo))
            else:
                try:
                    setattr(result, k, copy.deepcopy(v, memo))
                except Exception as e:
                    # Only log if it's not a known safe-to-skip Qt object
                    # QColor and QRect are usually fine.
                    logger.warning(f"Deepcopy failed for key {k}: {e}")
                    setattr(result, k, None) 
        
        return result

    # Add reload_image method if needed to restore pixmap?
    # Usually update_Block handles drawing or re-creating it.
