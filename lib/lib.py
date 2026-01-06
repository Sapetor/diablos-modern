"lib.py - Contains all the core functions and classes for the simulation and execution of the graphs."

import numpy as np
import copy
import time
import json
from pathlib import Path
import os
import sys
from tqdm import tqdm
from PyQt5.QtWidgets import QWidget, QFileDialog, QApplication, QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton, QMessageBox
from PyQt5.QtGui import QColor, QPen, QFont, QPixmap, QPainter
from PyQt5.QtCore import Qt, QRect, QPoint, QEvent, QTimer
import pyqtgraph as pg
from lib.block_loader import load_blocks
from lib.dialogs import ParamDialog, PortDialog
from lib.workspace import WorkspaceManager
from lib.dialogs import ParamDialog, PortDialog, SimulationDialog
import logging
from modern_ui.themes.theme_manager import theme_manager
from lib import functions

# Import refactored classes from new modules
from lib.simulation.block import DBlock
from lib.simulation.connection import DLine
from lib.simulation.menu_block import MenuBlocks
from lib.ui.button import Button

# Import block size configuration
from config.block_sizes import get_block_size

logger = logging.getLogger(__name__)



sys.path.append('./usermodels/')


class DSim:
    """
    Class that manages the simulation interface and main functions.

    :param SCREEN_WIDTH: The width of the window
    :param SCREEN_HEIGHT: The height of the window
    :param canvas_top_limit: Top limit where blocks and lines must be drawn.
    :param canvas_left_limit: Left limit where blocks and lines must be drawn.
    :param colors: List of predefined colors for elements that show in the canvas.
    :param fps: Base frames per seconds for pygame's loop.
    :param l_width: Width of the line when a block or a line is selected.
    :param ls_width: Space between a selected block and the line that indicates the former is selected.
    :param filename: Name of the file that was recently loaded. By default is 'data.dat'.
    :param sim_time: Simulation time for graph execution.
    :param sim_dt: Simulation sampling time for graph execution.
    :param plot_trange: Width in number of elements that must be shown when a graph is getting executed with dynamic plot enabled.
    :type SCREEN_WIDTH: int
    :type SCREEN_HEIGHT: int
    :type canvas_top_limit: int
    :type canvas_left_limit: int
    :type colors: dict
    :type fps: int
    :type l_width: int
    :type ls_width: int
    :type filename: str
    :type sim_time: float
    :type sim_dt: float
    :type plot_trange: int

    """
    

    def __init__(self):
        logger.debug("Initializing DSim with MVC architecture")

        # Initialize MVC components
        from lib.models.simulation_model import SimulationModel
        from lib.engine.simulation_engine import SimulationEngine
        from lib.services.file_service import FileService

        self.model = SimulationModel()
        self.engine = SimulationEngine(self.model)
        self.file_service = FileService(self.model)

        # Screen/UI parameters
        self.SCREEN_WIDTH = 1280
        self.SCREEN_HEIGHT = 720 + 50
        self.canvas_top_limit = 60
        self.canvas_left_limit = 200
        self.FPS = 60
        self.l_width = 5
        self.ls_width = 5

        # Delegate commonly used properties to model for backward compatibility
        self.colors = self.model.colors
        self.menu_blocks = self.model.menu_blocks
        self.blocks_list = self.model.blocks_list
        self.line_list = self.model.line_list

        # UI state
        self.line_creation = 0
        self.only_one = False
        self.enable_line_selection = False
        self.holding_CTRL = False
        self.ss_count = 0

        # Delegate simulation parameters to engine
        self.sim_time = self.engine.sim_time
        self.sim_dt = self.engine.sim_dt
        self.plot_trange = 100

        # Execution state (delegate to engine)
        self.execution_initialized = self.engine.execution_initialized
        self.execution_pause = self.engine.execution_pause
        self.execution_stop = self.engine.execution_stop
        self.error_msg = self.engine.error_msg
        self.real_time = self.engine.real_time
        self.dynamic_plot = False

        # Delegate filename to file service
        self.filename = self.file_service.filename
        self.dirty = self.model.dirty

        self.execution_function = functions

        # Legacy execution tracking (still needed for complex execution methods)
        self.global_computed_list = []
        self.timeline = []
        self.outs = []
        self.plotty = None
        # Run history of scope data (session only)
        self.run_history = []
        self.run_history_limit = 5
        self.run_history_persist_enabled = False
        self.run_history_persist_path = Path("saves/run_history.json")
        self._load_run_history()

    def main_buttons_init(self):
        """
        :purpose: Creates a button list with all the basic functions available
        """
        new =  Button('_new_',     ( 40, 10, 40, 40))
        load = Button('_load_',    (100, 10, 40, 40))
        save = Button('_save_',    (160, 10, 40, 40))
        sim =  Button('_play_',    (220, 10, 40, 40))
        pause = Button('_pause_',  (280, 10, 40, 40))
        stop = Button('_stop_',    (340, 10, 40, 40))
        rplt = Button('_plot_',    (400, 10, 40, 40), active=False)
        capt = Button('_capture_', (460, 10, 40, 40))

        self.buttons_list = [new, load, save, sim, pause, stop, rplt, capt]

    def display_buttons(self, painter):
        """
        :purpose: Displays all the buttons on the screen.
        :param painter: Pygame's layer where the figure is drawn.
        """
        if painter is None:
            return
        pen = QPen(self.colors['black'], 2)
        painter.setPen(pen)
        painter.drawLine(200, 60, 1260, 60)
        for button in self.buttons_list:
            button.draw_button(painter)

    def set_color(self, color):
        """
        :purpose: Defines color for an element drawn in pygame.
        :param color: The color in string or rgb to set.
        :type color: str/(float, float, float)
        """
        if isinstance(color, str):
            return self.colors.get(color, self.colors['gray'])
        elif isinstance(color, (tuple, list)) and len(color) == 3:
            return QColor(*color)
        elif isinstance(color, QColor):
            return color
        else:
            return self.colors['gray']

    def screenshot(self, painter):
        """
        :purpose: Takes a capture of the screen with all elements seen on display.
        :param painter: Pygame's layer where the figures, lines and buttons are drawn.
        """
        filename = self.filename[:-4]
        pixmap = QPixmap(self.SCREEN_WIDTH, self.SCREEN_HEIGHT)
        painter.end()  # End the current painter
        new_painter = QPainter(pixmap)
        self.render(new_painter)
        new_painter.end()
        pixmap.save(f"captures/{filename}-{self.ss_count}.png")
        self.ss_count += 1
        painter.begin(self)  # Restart the original painter

    ##### ADD OR REMOVE BLOCKS AND LINES #####

    def add_block(self, block, m_pos):
        """Add a block to the diagram. Delegates to model."""
        new_block = self.model.add_block(block, m_pos)
        self.dirty = self.model.dirty
        return new_block
    def add_line(self, srcData, dstData):
        """Add a connection line between two blocks. Delegates to model."""
        new_line = self.model.add_line(srcData, dstData)
        self.dirty = self.model.dirty
        return new_line

    def remove_block_and_lines(self, block):
        """Remove a block and its associated lines. Delegates to model."""
        self.model.remove_block(block)
        self.line_list = self.model.line_list  # Sync line_list after removal
        self.dirty = self.model.dirty

    def check_line_block(self, line, b_del_list):
        """
        :purpose: Checks if a line is connected to one or more removed blocks.
        :param line: Line object.
        :param b_del_list: List of recently removed blocks.
        """
        if line.dstblock in b_del_list or line.srcblock in b_del_list:
            return True
        return False

    def check_line_port(self, line, block):
        """
        :purpose: Checks if there are lines left from a removed port (associated to a block).
        :param line: Line object.
        :param block: Block object.
        """
        if line.srcblock == block.name and line.srcport > block.out_ports - 1:
            return True
        elif line.dstblock == block.name and line.dstport > block.in_ports - 1:
            return True
        else:
            return False
        


    def display_lines(self, painter):
        """
        :purpose: Draws lines connecting blocks in the screen.
        :param painter: Pygame's layer where the figure is drawn.
        """
        if painter is None:
            return
        for line in self.line_list:
            if getattr(line, "hidden", False):
                continue
            line.draw_line(painter)

    def update_lines(self):
        """
        :purpose: Updates lines according to the location of blocks if these changed place.
        """
        logger.debug("Updating lines")
        for line in self.line_list:
            line.update_line(self.blocks_list)

    def display_blocks(self, painter, draw_ports=True):
        """
        :purpose: Draws blocks defined in the main list on the screen.
        :param painter: A layer in a pygame canvas where the figure is drawn.
        :param draw_ports: Whether to draw ports (default True for backward compatibility)
        """
        if painter is None:
            return
        for b_elem in self.blocks_list:
            b_elem.draw_Block(painter, draw_ports=draw_ports)
            # Draw resize handles for selected blocks
            if b_elem.selected:
                b_elem.draw_resize_handles(painter)

    def display_ports(self, painter):
        """
        Draw only the ports for all blocks on a separate rendering layer.

        This allows ports to be rendered on top of connections for better visibility.
        Delegates to each block's draw_ports() method for consistent rendering.

        :param painter: QPainter instance for rendering
        :type painter: QPainter
        """
        if painter is None:
            return
        for b_elem in self.blocks_list:
            b_elem.draw_ports(painter)

    def port_availability(self, dst_line):
        """
        :purpose: Checks if an input port is free to get connected with a line to another port.
        :param dst_line: The name of a Line object.
        :type dst_line: str
        """
        for line in self.line_list:
            if line.dstblock == dst_line[0] and line.dstport == dst_line[1]:
                return False
        return True

    ##### MENU BLOCKS #####

    def load_all_blocks(self):
        """
        :purpose: Function that initializes all types of blocks available in the menu.
        :description: From the block files, base blocks are generated. Then they are accumulated in a list so that they are available in the interface menu.
        """
        self.menu_blocks = []
        block_classes = load_blocks()

        # Define color map for categories
        category_colors = {
            "Sources": "blue",
            "Math": "lime_green",
            "Control": "magenta",
            "Filters": "cyan",
            "Sinks": "red",
            "Routing": "orange",
            "Other": "light_gray"
        }

        # Simple categorization based on block names
        source_keywords = ['step', 'ramp', 'sine', 'square', 'constant', 'source']
        math_keywords = ['sum', 'gain', 'multiply', 'add', 'subtract', 'divide', 'abs', 'sqrt']
        control_keywords = ['integr', 'deriv', 'pid', 'controller', 'delay']
        filter_keywords = ['filter', 'lowpass', 'highpass', 'bandpass']
        sink_keywords = ['scope', 'display', 'sink', 'plot', 'output', 'term']

        for block_class in block_classes:
            block = block_class()
            
            b_type = 2
            if not block.inputs:
                b_type = 0
            elif block.block_name.lower() in ['integr', 'integrator']:
                b_type = 1

            io_params = {
                'inputs': len(block.inputs), 
                'outputs': len(block.outputs), 
                'b_type': b_type,
                'io_edit': False
            }
            
            ex_params = {}
            for param_name, param_props in block.params.items():
                ex_params[param_name] = param_props['default']

            # Determine block category and color
            if hasattr(block, 'category'):
                category = block.category
            else:
                block_name_lower = block.block_name.lower()
                category = "Other"
                if any(keyword in block_name_lower for keyword in source_keywords):
                    category = "Sources"
                elif any(keyword in block_name_lower for keyword in math_keywords):
                    category = "Math"
                elif any(keyword in block_name_lower for keyword in control_keywords):
                    category = "Control"
                elif any(keyword in block_name_lower for keyword in filter_keywords):
                    category = "Filters"
                elif any(keyword in block_name_lower for keyword in sink_keywords):
                    category = "Sinks"

            if hasattr(block, 'color'):
                color = block.color
            else:
                color = category_colors.get(category, "light_gray")

            if hasattr(block, 'fn_name'):
                fn_name = block.fn_name
            else:
                fn_name = block.block_name.lower()

            # Get block-specific size from configuration
            block_size = get_block_size(block.block_name)

            menu_block = MenuBlocks(
                block_fn=block.block_name,
                fn_name=fn_name,
                io_params=io_params,
                ex_params=ex_params,
                b_color=color,
                coords=block_size,  # Use configured block size
                block_class=block_class,
                colors=self.colors
            )
            self.menu_blocks.append(menu_block)

    def display_menu_blocks(self, painter):
        """
        :purpose: Draws MenuBlocks objects in the screen.
        :param painter: QtPy layer where the figure is drawn.
        """
        if painter is None:
            return
        pen = QPen(self.colors['black'], 2)
        painter.setPen(pen)
        painter.drawLine(200, 60, 200, 710)
        for i, block in enumerate(self.menu_blocks):
            block.draw_menublock(painter, i)

    ##### LOADING AND SAVING #####

    def save(self, autosave=False, modern_ui_data=None):
        """
        :purpose: Saves blocks, lines and other data in a .dat file.
        :description: Obtaining the location where the file is to be saved, all the important data of the DSim class, each one of the blocks and each one of the lines, are copied into dictionaries, which will then be loaded to the external file by means of the JSON library.
        :param autosave: Flag that defines whether the process to be performed is an autosave or not.
        :type autosave: bool
        :notes: This function is executed automatically when you want to simulate, so as not to lose unsaved information.
        """
        
        if not autosave:
            options = QFileDialog.Options()
            initial_dir = os.path.join(os.path.dirname(__file__), 'saves')
            file, _ = QFileDialog.getSaveFileName(
                None,
                "Save File",
                os.path.join(initial_dir, self.filename),
                "Data Files (*.dat);;All Files (*)",
                options=options
            )

            if not file:
                return 1
            if not file.lower().endswith('.dat'):
                file += '.dat'
        else:  # Option for when a graph is going to run
            if '_AUTOSAVE' not in self.filename:
                file = f'saves/{self.filename[:-4]}_AUTOSAVE.dat'
            else:
                file = f'saves/{self.filename}'

        # Ensure the saves directory exists
        os.makedirs(os.path.dirname(file), exist_ok=True)

        # DSim data (Simulator interface)
        init_dict = {
            "wind_width": self.SCREEN_WIDTH,
            "wind_height": self.SCREEN_HEIGHT,
            "fps": self.FPS,
            "only_one": self.only_one,
            "enable_line_sel": self.enable_line_selection,
            "sim_time": self.sim_time,
            "sim_dt": self.sim_dt,
            "sim_trange": self.plot_trange
        }

        # Blocks parameters data
        blocks_dict = []
        for block in self.blocks_list:
            block_dict = {
                "block_fn": block.block_fn,
                "sid": block.sid,
                "username": block.username,
                "coords_left": block.left,
                "coords_top": block.top,
                "coords_width": block.width,
                "coords_height": block.height,
                "coords_height_base": block.height_base,
                "in_ports": block.in_ports,
                "out_ports": block.out_ports,
                "dragging": block.dragging,
                "selected": block.selected,
                "b_color": block.b_color.name(),  # Convert QColor to string
                "b_type": block.b_type,
                "io_edit": block.io_edit,
                "fn_name": block.fn_name,
                "params": block.saving_params(),
                "external": block.external,
                "flipped": block.flipped
            }
            blocks_dict.append(block_dict)

        # Line parameters data
        lines_dict = []
        for line in self.line_list:
            line_dict = {
                "name": line.name,
                "sid": line.sid,
                "srcblock": line.srcblock,
                "srcport": line.srcport,
                "dstblock": line.dstblock,
                "dstport": line.dstport,
                "points": [(p.x(), p.y()) for p in line.points],
                "cptr": getattr(line, 'cptr', 0),  # Use getattr with a default value
                "selected": line.selected
            }
            lines_dict.append(line_dict)

        main_dict = {"sim_data": init_dict, "blocks_data": blocks_dict, "lines_data": lines_dict}

        if modern_ui_data:
            main_dict["modern_ui_data"] = modern_ui_data
        main_dict["version"] = "2.0"

        with open(file, 'w') as fp:
            json.dump(main_dict, fp, indent=4)

        if not autosave:
            self.filename = os.path.basename(file)  # Keeps the name of the file if you want to save it again later

        logger.info(f"SAVED AS {file}")

    def open(self):
        """
        :purpose: Loads blocks, lines and other data from a .dat file.
        :description: Starting from the .dat file, the data saved in the dictionaries are unpacked, updating the data in DSim, creating new blocks and lines, leaving the canvas and the configurations as they were saved before.
        :notes: The name of the loaded file is saved in the system, in order to facilitate the saving of data in it (overwriting it).
        """
        options = QFileDialog.Options()
        initial_dir = os.path.dirname(os.path.abspath(__file__))
        file, _ = QFileDialog.getOpenFileName(
            None,
            "Open File",
            os.path.join(initial_dir, self.filename),
            "Data Files (*.dat);;All Files (*)",
            options=options
        )

        if not file:  # If no file was selected (dialog was cancelled)
            return

        with open(file) as json_file:
            data = json.load(json_file)

        version = data.get("version", "1.0")
        if version != "2.0":
            logger.warning(f"Loading a file with version {version}, but the current version is 2.0. Some features may not be supported.")

        sim_data = data['sim_data']
        blocks_data = data['blocks_data']
        lines_data = data['lines_data']
        modern_ui_data = data.get("modern_ui_data")

        self.clear_all()
        self.update_sim_data(sim_data)
        self.ss_count = 0
        for block in blocks_data:
            self.update_blocks_data(block)
        for line in lines_data:
            self.update_lines_data(line)

        self.filename = os.path.basename(file)  # Keeps the name of the file if you want to save it again later

        logger.info(f"LOADED FROM {file}")
        return modern_ui_data

    def update_sim_data(self, data):
        """
        :purpose: Updates information related with the main class variables saved in a file to the current simulation.
        :param data: Dictionary with DSim parameters.
        :type data: dict
        """
        self.SCREEN_WIDTH = data.get('wind_width', 1280)
        self.SCREEN_HEIGHT = data.get('wind_height', 770)
        self.FPS = data.get('fps', 60)
        self.line_creation = 0
        self.only_one = data.get('only_one', False)
        self.enable_line_selection = data.get('enable_line_sel', True)
        self.sim_time = data.get('sim_time', 1.0)
        self.sim_dt = data.get('sim_dt', 0.01)
        self.plot_trange = data.get('sim_trange', 100)

    def update_blocks_data(self, block_data):
        """
        :purpose: Updates information related with all the blocks saved in a file to the current simulation.
        :param block_data: Dictionary with Block object id, parameters, variables, etc.
        :type block_data: dict
        """
        menu_block = None
        for mb in self.menu_blocks:
            if mb.block_fn == block_data['block_fn']:
                menu_block = mb
                break

        loaded_params = block_data['params']
        if menu_block:
            default_params = copy.deepcopy(menu_block.params)
            default_params.update(loaded_params)
            loaded_params = default_params

        # Use fn_name from menu_block to ensure we have the correct function name,
        # not the potentially outdated one from saved file
        fn_name = menu_block.fn_name if menu_block else block_data.get('fn_name', block_data['block_fn'].lower())

        coords = QRect(block_data['coords_left'], block_data['coords_top'], block_data['coords_width'], block_data['coords_height_base'])
        block = DBlock(block_fn=block_data['block_fn'],
                      sid=block_data['sid'],
                      coords=coords,
                      color=block_data['b_color'],
                      in_ports=block_data['in_ports'],
                      out_ports=block_data['out_ports'],
                      b_type=block_data['b_type'],
                      io_edit=block_data['io_edit'],
                      fn_name=fn_name,
                      params=loaded_params,
                      external=block_data['external'],
                      username=block_data['username'],
                      block_class=menu_block.block_class if menu_block else None,
                      colors=self.colors)
        block.height = block_data['coords_height']
        block.selected = block_data['selected']
        block.dragging = block_data['dragging']
        block.flipped = block_data.get('flipped', False)
        self.blocks_list.append(block)

    def update_lines_data(self, line_data):
        logger.debug(f"Updating line data: {line_data}")
        """
        :purpose: Updates information related with all the lines saved in a file to the current simulation.
        :param line_data: Dictionary with Line object id, parameters, variables, etc.
        :type line_data: dict
        """
        # Build username→name map for flexible block references
        # This allows files to use usernames (e.g., "err1") instead of auto-generated names ("Sum1")
        srcblock = line_data['srcblock']
        dstblock = line_data['dstblock']
        
        for block in self.blocks_list:
            if block.username == srcblock:
                srcblock = block.name
            if block.username == dstblock:
                dstblock = block.name
        
        line = DLine(sid=line_data['sid'],
                    srcblock=srcblock,
                    srcport=line_data['srcport'],
                    dstblock=dstblock,
                    dstport=line_data['dstport'],
                    points=line_data['points'])
        line.selected = line_data['selected']
        line.update_line(self.blocks_list)
        self.line_list.append(line)

    def clear_all(self):
        """Clear all blocks and lines from the diagram. Delegates to model."""
        self.model.clear_all()
        # Update references
        self.blocks_list = self.model.blocks_list
        self.line_list = self.model.line_list
        self.dirty = self.model.dirty

        # Reset UI state
        self.line_creation = 0
        self.only_one = False
        self.enable_line_selection = False
        self.ss_count = 0
        self.filename = 'data.dat'
        self.sim_time = 1.0
        self.sim_dt = 0.01
        self.plot_trange = 100
        self.dynamic_plot = False


    ##### DIAGRAM EXECUTION #####

    def execution_init_time(self):
        """
        :purpose: Creates a pop-up window to ask for graph simulation setup values.
        :description: The first step in order to be able to perform a network simulation, is to have the execution data. These are mainly simulation time and sampling period, but we also ask for variables needed for the graphs.
        """
        dialog = SimulationDialog(self.sim_time, self.sim_dt, self.plot_trange)
        if dialog.exec_() == QDialog.Accepted:
            try:
                values = dialog.get_values()
                self.sim_time = values['sim_time']
                self.sim_dt = values['sim_dt']
                self.plot_trange = values['plot_trange']
                self.dynamic_plot = values['dynamic_plot']
                self.real_time = values['real_time']
                return self.sim_time
            except ValueError:
                logger.warning("Invalid input. Using default values.")
                return self.sim_time
        else:
            return -1

    def execution_init(self):
        """
        :purpose: Initializes the graph execution.
        :description: This is the first stage of the graph simulation, where variables and vectors are initialized, as well as testing to verify that everything is working properly. A previous autosave is done, as well as a block connection check and possible algebraic loops. If everything goes well, we continue with the loop stage.
        """
        try:
            logger.debug("Starting execution initialization...")
            # The class containing the functions for the execution is called

            self.execution_stop = False                         # Prevent execution from stopping before executing in error
            self.error_msg = ""                                 # Clear any previous error message
            self.time_step = 0                                  # First iteration of the time which will be incrementing self.sim_dt seconds
            self.timeline = np.array([self.time_step])          # List containing the value of all past iterations

            # Some parameters are initialized including the maximum simulation time.
            self.execution_time = self.execution_init_time()

            # To cancel the simulation before running it (having pressed X in the pop up)
            if self.execution_time == -1 or len(self.blocks_list) == 0:
                self.execution_initialized = False
                return False

            # Force save before executing (so as not to lose the diagram)
            if self.save(True) == 1:
                return False

            logger.debug("*****INIT NEW EXECUTION*****")

            workspace_manager = WorkspaceManager()

            for block in self.blocks_list:
                # Resolve parameters using WorkspaceManager
                logger.debug(f"Block {block.name}: params before resolve = {block.params}")
                logger.debug(f"WorkspaceManager variables = {workspace_manager.variables}")
                block.exec_params = workspace_manager.resolve_params(block.params)
                logger.debug(f"Block {block.name}: exec_params after resolve = {block.exec_params}")
                # Copy internal parameters that start with '_'
                block.exec_params.update({k: v for k, v in block.params.items() if k.startswith('_')})

                # Dynamically set b_type for Transfer Functions
                if block.block_fn == 'TranFn':
                    num = block.exec_params.get('numerator', [])
                    den = block.exec_params.get('denominator', [])
                    if len(den) > len(num):
                        block.b_type = 1  # Strictly proper, has memory
                    else:
                        block.b_type = 2  # Not strictly proper, direct feedthrough
                
                elif block.block_fn == 'DiscreteTranFn':
                    num = block.exec_params.get('numerator', [])
                    den = block.exec_params.get('denominator', [])
                    # In discrete transfer function H(z) = N(z)/D(z)
                    # If order(D) > order(N), it's strictly proper (delay) -> Type 1
                    # If order(D) == order(N), it has direct feedthrough -> Type 2
                    # Note: len(den) is order + 1
                    if len(den) > len(num):
                        block.b_type = 1
                    else:
                        block.b_type = 2

                elif block.block_fn == 'DiscreteStateSpace':
                    # Check D matrix for direct feedthrough
                    D = np.array(block.exec_params.get('D', [[0.0]]))
                    # If D is zero matrix, it's strictly proper (Type 1)
                    if np.all(D == 0):
                        block.b_type = 1
                    else:
                        block.b_type = 2

                
                block.exec_params['dtime'] = self.sim_dt
                try:
                    missing_file_flag = block.reload_external_data()
                    if missing_file_flag == 1:
                        logger.error(f"Missing external file for block: {block.name}")
                        return False
                except Exception as e:
                    logger.error(f"Error reloading external data for block {block.name}: {str(e)}")
                    return False

            if not self.check_diagram_integrity():
                logger.error("Diagram integrity check failed")
                return False
            logger.debug("Initializing execution...")

            # Generation of a checklist for the computation of functions
            self.global_computed_list = [{'name': x.name, 'computed_data': x.computed_data, 'hierarchy': x.hierarchy}
                                    for x in self.blocks_list]
            self.reset_execution_data()
            self.execution_time_start = time.time()
            logger.debug("Execution initialization complete")
        except Exception as e:
            logger.error(f"Error during execution initialization: {str(e)}")
            return False

        logger.debug("*****EXECUTION START*****")

        # Initialization of the progress bar
        self.pbar = tqdm(desc='SIMULATION PROGRESS', total=int(self.execution_time/self.sim_dt), unit='itr')
        self.dirty = False

        # Check the existence of algebraic loops (part 1)
        check_loop = self.count_computed_global_list()

        # Identify memory blocks to correctly solve algebraic loops
        self.memory_blocks = set()
        for block in self.blocks_list:
            if block.b_type == 1: # Integrators
                self.memory_blocks.add(block.name)
            elif block.block_fn == 'TranFn': # Strictly proper transfer functions
                num = block.params.get('numerator', [])
                den = block.params.get('denominator', [])
                if len(den) > len(num):
                    self.memory_blocks.add(block.name)
            elif block.block_fn == 'DiscreteTranFn':
                num = block.params.get('numerator', [])
                den = block.params.get('denominator', [])
                if len(den) > len(num):
                    self.memory_blocks.add(block.name)
            elif block.block_fn == 'DiscreteStateSpace':
                D = np.array(block.params.get('D', [[0.0]]))
                if np.all(D == 0):
                    self.memory_blocks.add(block.name)
        logger.debug(f"MEMORY BLOCKS IDENTIFIED: {self.memory_blocks}")

        # Check for integrators using Runge-Kutta 45 and initialize counter
        self.rk45_len = self.count_rk45_ints()
        self.rk_counter = 0

        # Auto-connect Goto/From tags before execution starts
        try:
            self.model.link_goto_from()
            # refresh references
            self.blocks_list = self.model.blocks_list
            self.line_list = self.model.line_list
        except Exception as e:
            logger.warning(f"Goto/From linking failed: {e}")

        # Validate signal dimensions between connected blocks
        self._validate_signal_dimensions()

        # The block diagram starts with looking for source type blocks
        for block in self.blocks_list:
            logger.debug(f"Initial processing of block: {block.name}, b_type: {block.b_type}")
            children = {}
            out_value = {}
            if block.b_type == 0:
                # The function is executed (differentiate between internal and external function first)
                if block.external:
                    try:
                        out_value = getattr(block.file_function, block.fn_name)(time=self.time_step, inputs=block.input_queue, params=block.exec_params)
                    except Exception as e:
                        logger.error(f"ERROR FOUND IN EXTERNAL FUNCTION {block.file_function}: {e}")
                        self.execution_failed(str(e))
                        return False
                else:
                    out_value = block.block_instance.execute(time=self.time_step, inputs=block.input_queue, params=block.exec_params)
                block.computed_data = True
                block.hierarchy = 0
                self.update_global_list(block.name, h_value=0, h_assign=True)
                children = self.get_outputs(block.name)

            elif block.name in self.memory_blocks:
                kwargs = {
                    'time': self.time_step,
                    'inputs': block.input_queue,
                    'params': block.exec_params,
                    'output_only': True
                }
                if block.block_fn == 'Integrator':
                    kwargs['next_add_in_memory'] = False
                    kwargs['dtime'] = self.sim_dt
                
                if block.external:
                    try:
                        out_value = getattr(block.file_function, block.fn_name)(**kwargs)
                    except Exception as e:
                        logger.error(f"ERROR FOUND IN EXTERNAL FUNCTION {block.file_function}: {e}")
                        self.execution_failed(str(e))
                        return False
                else:
                    # Use block_instance if available (refactored blocks), otherwise use execution_function (legacy)
                    if block.block_instance:
                        out_value = block.block_instance.execute(**kwargs)
                    else:
                        out_value = getattr(self.execution_function, block.fn_name)(**kwargs)
                children = self.get_outputs(block.name)
                block.computed_data = True
                self.update_global_list(block.name, h_value=0, h_assign=True)

            if 'E' in out_value.keys() and out_value['E']:
                self.execution_failed(out_value.get('error', 'Unknown error'))
                return False

            for mblock in self.blocks_list:
                is_child, tuple_list = self.children_recognition(block_name=mblock.name, children_list=children)
                if is_child:
                    # Data is sent to each required port of the child block
                    for tuple_child in tuple_list:
                        if tuple_child['dstport'] not in mblock.input_queue:
                            mblock.data_recieved += 1
                        mblock.input_queue[tuple_child['dstport']] = out_value[tuple_child['srcport']]
                        block.data_sent += 1

        # Un-compute memory blocks so they are executed in the while loop to update their state
        for block in self.blocks_list:
            if block.name in self.memory_blocks:
                block.computed_data = False
                for g_block in self.global_computed_list:
                    if g_block['name'] == block.name:
                        g_block['computed_data'] = False

        # The diagram continues to be executed through the following blocks
        h_count = 1
        while not self.check_global_list():
            for block in self.blocks_list:
                # This part is executed only if the received data is equal to the number of input ports and the block has not been computed yet.
                # Special case: From blocks have in_ports=0 but receive data via virtual lines
                can_execute = block.data_recieved == block.in_ports
                if block.block_fn == 'From':
                    # From blocks must wait for virtual line data in input_queue[0]
                    can_execute = 0 in block.input_queue and block.input_queue[0] is not None
                
                if can_execute and not block.computed_data:
                    # The function is executed (differentiate between internal and external function first)
                    if block.external:
                        try:
                            out_value = getattr(block.file_function, block.fn_name)(time=self.time_step, inputs=block.input_queue, params=block.exec_params)
                        except Exception as e:
                            logger.error(f"ERROR FOUND IN EXTERNAL FUNCTION {block.file_function}: {e}")
                            self.execution_failed()
                            return False
                    else:
                        # Use block_instance if available (refactored blocks), otherwise use execution_function (legacy)
                        if block.block_instance:
                            out_value = block.block_instance.execute(time=self.time_step, inputs=block.input_queue, params=block.exec_params)
                        else:
                            out_value = getattr(self.execution_function, block.fn_name)(time=self.time_step, inputs=block.input_queue, params=block.exec_params)

                    # Check if the execution returned None (some blocks may not return anything)
                    if out_value is None:
                        logger.error(f"Block {block.name} ({block.block_fn}) returned None")
                        self.execution_failed(f"Block {block.name} returned None")
                        return False

                    # It is checked that the function has not delivered an error
                    if 'E' in out_value.keys() and out_value['E']:
                        self.execution_failed(out_value.get('error', 'Unknown error'))
                        return False

                    # After execution, for memory blocks, update the 'output' state for the next step
                    if block.name in self.memory_blocks:
                        if block.block_fn == 'Integrator' and 'mem' in block.exec_params:
                            block.exec_params['output'] = block.exec_params['mem']

                    # The computed_data booleans are updated in the global list as well as in the block itself
                    self.update_global_list(block.name, h_value=h_count, h_assign=True)
                    block.computed_data = True

                    if block.b_type not in [1, 3]:  # Elements that do not deliver a result to children (1 is initial cond.)
                        children = self.get_outputs(block.name)
                        for mblock in self.blocks_list:
                            is_child, tuple_list = self.children_recognition(block_name=mblock.name, children_list=children)
                            if is_child:
                                # Data is sent to each required port of the child block
                                for tuple_child in tuple_list:
                                    signal_data = out_value[tuple_child['srcport']]
                                    mblock.input_queue[tuple_child['dstport']] = signal_data
                                    mblock.data_recieved += 1
                                    block.data_sent += 1
                                    
                                    # Update line signal_width for MIMO visual indicator
                                    signal_len = len(np.atleast_1d(signal_data).flatten()) if signal_data is not None else 1
                                    for line in self.line_list:
                                        if line.srcblock == block.name and line.dstblock == mblock.name:
                                            if line.srcport == tuple_child['srcport'] and line.dstport == tuple_child['dstport']:
                                                line.signal_width = signal_len
                                                break

            # The number of executed blocks from the previous stage is compared. If it is zero, there is an algebraic loop.
            computed_count = self.count_computed_global_list()
            if computed_count == check_loop:
                uncomputed_blocks = [b for b in self.blocks_list if not b.computed_data]

                # If there are no uncomputed blocks, we are done with the init loop
                if not uncomputed_blocks:
                    break

                # Perform a topological sort to detect cycles
                try:
                    is_algebraic, cycle_nodes = self.detect_algebraic_loops(uncomputed_blocks)
                    if is_algebraic:
                        self.error_msg = f"Algebraic loop detected involving blocks: {cycle_nodes}"
                        logger.error(self.error_msg)
                        self.execution_failed(self.error_msg)
                        return False
                    else:
                        break
                except Exception as e:
                    self.error_msg = f"Error during algebraic loop detection: {e}"
                    logger.error(self.error_msg)
                    self.execution_failed(self.error_msg)
                    return False
            else:
                check_loop = computed_count

            h_count += 1

        # The highest hierarchy value is determined for the next iterations
        self.max_hier = self.get_max_hierarchy()
        self.execution_initialized = True
        self.rk_counter += 1

        # Enable the plot button if there is at least one scope
        for block in self.blocks_list:
            if block.block_fn == 'Scope':
                self.buttons_list[6].active = True

        # The dynamic plot function is initialized, if the Boolean is active
        self.dynamic_pyqtPlotScope(step=0)

        return True

    def execution_batch(self):
        """Run the entire simulation as fast as possible."""
        while self.execution_initialized:
            self.execution_loop()

        

    def execution_loop(self):
        """
        :purpose: Continues with the execution sequence in loop until time runs out or an special event stops it.
        :description: This is the second stage of the network simulation. Here the reading of the complete graph will be done cyclically until the time is up, the user indicates that it is finished (by pressing Stop) or simply until one of the blocks gives error. At the end, the data saved in blocks like 'Scope' and 'External_data', will be exported to other libraries to perform their functions.
        """
        try:
            if self.execution_pause:
                return

            self.reset_execution_data()

            if self.rk45_len:
                self.rk_counter %= 4
                if self.rk_counter in [1, 3]:
                    self.time_step += self.sim_dt / 2
                    self.pbar.update(1/2)
                elif self.rk_counter == 0:
                    self.time_step += self.sim_dt
                    self.pbar.update(1)
                    self.timeline = np.append(self.timeline, self.time_step)
            else:
                self.time_step += self.sim_dt
                self.pbar.update(1)
                self.timeline = np.append(self.timeline, self.time_step)

            for block in self.blocks_list:
                try:
                    if block.name in self.memory_blocks:
                        kwargs = {
                            'time': self.time_step,
                            'inputs': block.input_queue,
                            'params': block.exec_params,
                            'output_only': True
                        }
                        if block.block_fn == 'Integrator':
                            add_in_memory = not self.rk45_len or self.rk_counter == 3
                            kwargs['next_add_in_memory'] = add_in_memory
                            kwargs['dtime'] = self.sim_dt
                        
                        if block.external:
                            try:
                                out_value = getattr(block.file_function, block.fn_name)(**kwargs)
                            except Exception as e:
                                logger.error(f"ERROR FOUND IN EXTERNAL FUNCTION {block.file_function}: {str(e)}")
                                self.execution_failed()
                                return
                        else:
                            # Use block_instance if available (refactored blocks), otherwise use execution_function (legacy)
                            if block.block_instance:
                                out_value = block.block_instance.execute(**kwargs)
                            else:
                                out_value = getattr(self.execution_function, block.fn_name)(**kwargs)

                        if 'E' in out_value and out_value['E']:
                            self.execution_failed()
                            return

                        children = self.get_outputs(block.name)
                        for mblock in self.blocks_list:
                            is_child, tuple_list = self.children_recognition(mblock.name, children)
                            if is_child:
                                for tuple_child in tuple_list:
                                                                    if tuple_child['dstport'] not in mblock.input_queue:
                                                                        mblock.data_recieved += 1
                                                                    mblock.input_queue[tuple_child['dstport']] = out_value[tuple_child['srcport']]
                                                                    block.data_sent += 1
                    if self.rk45_len and self.rk_counter != 0:
                        block.params['_skip_'] = True
                except Exception as e:
                    logger.error(f"Error executing block {block.name}: {str(e)}")
                    self.execution_failed()
                    return

            # All blocks are executed according to the hierarchy order defined in the first iteration
            for hier in range(self.max_hier + 1):
                for block in self.blocks_list:
                    # The block must have the degree of hierarchy to execute it (and meet the other requirements above)
                    if block.hierarchy == hier and (block.data_recieved == block.in_ports or block.in_ports == 0) and not block.computed_data:
                        # The function is executed (differentiate between internal and external function first)
                        if block.external:
                            try:
                                out_value = getattr(block.file_function, block.fn_name)(self.time_step, block.input_queue, block.exec_params)
                            except Exception as e:
                                logger.error(f"ERROR FOUND IN EXTERNAL FUNCTION {block.file_function}: {e}")
                                self.execution_failed()
                                return
                        else:
                            # Use block_instance if available (refactored blocks), otherwise use execution_function (legacy)
                            if block.block_instance:
                                out_value = block.block_instance.execute(time=self.time_step, inputs=block.input_queue, params=block.exec_params)
                            else:
                                out_value = getattr(self.execution_function, block.fn_name)(self.time_step, block.input_queue, block.exec_params)
                        # After execution, for memory blocks, update the 'output' state for the next step
                        if block.name in self.memory_blocks:
                            if block.block_fn == 'Integrator' and 'mem' in block.exec_params:
                                block.exec_params['output'] = block.exec_params['mem']

                        # It is checked that the function has not delivered an error
                        if 'E' in out_value.keys() and out_value['E']:
                            self.execution_failed()
                            return

                        # The computed_data booleans are updated in the global list as well as in the block itself
                        self.update_global_list(block.name, h_value=0)
                        block.computed_data = True

                        # The blocks that require the processed data from this block are searched
                        children = self.get_outputs(block.name)
                        if block.b_type not in [1, 3]:  # Elements that do not deliver a result to children (1 is initial cond.)
                            for mblock in self.blocks_list:
                                is_child, tuple_list = self.children_recognition(block_name=mblock.name, children_list=children)
                                if is_child:
                                    # Data is sent to each required port of the child block
                                                                    for tuple_child in tuple_list:
                                                                        if tuple_child['dstport'] not in mblock.input_queue:
                                                                            mblock.data_recieved += 1
                                                                        mblock.input_queue[tuple_child['dstport']] = out_value[tuple_child['srcport']]
                                                                        block.data_sent += 1
                hier += 1

            # The dynamic plot function is called to save the new data, if active
            self.dynamic_pyqtPlotScope(step=1)

            # It is checked if the total simulation (execution) time has been exceeded to end the loop
            if self.time_step > self.execution_time + self.sim_dt:  # seconds
                self.execution_initialized = False                  # The execution loop is terminated
                self.pbar.close()                                   # The progress bar ends

                # Export
                self.export_data()

                # Record run history for inspector
                try:
                    self._record_run_history()
                except Exception as e:
                    logger.warning(f"Could not record run history: {e}")

                # Scope
                if not self.dynamic_plot:
                    logger.debug("Calling pyqtPlotScope...")
                    self.pyqtPlotScope()
                    logger.debug("pyqtPlotScope call finished.")

                # Resets the initialization of the blocks with special initial executions
                self.reset_memblocks()
                logger.debug("*****EXECUTION DONE*****")

            self.rk_counter += 1
        
        except Exception as e:
            import traceback
            logger.error(f"Error during execution loop: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            self.execution_failed()

    def execution_failed(self, msg=""):
        """
        :purpose: If an error is found while executing the graph, this function stops all the processes and resets values to the state before execution.
        """
        self.execution_initialized = False   # Finishes the simulation execution
        self.reset_memblocks()               # Restores the initialization of the integrators (in case the error was due to vectors of different dimensions).
        self.pbar.close()                    # Finishes the progress bar
        self.error_msg = msg
        logger.error("*****EXECUTION STOPPED*****")

    def check_diagram_integrity(self):
        logger.debug("*****Checking diagram integrity*****")
        error_trigger = False
        for block in self.blocks_list:
            inputs, outputs = self.get_neighbors(block.name)

            if block.in_ports == 1 and len(inputs) < block.in_ports:
                logger.error(f"ERROR. UNLINKED INPUT IN BLOCK: {block.name}")
                error_trigger = True
            elif block.in_ports > 1:
                in_vector = np.zeros(block.in_ports)
                for tupla in inputs:
                    in_vector[tupla['dstport']] += 1
                finders = np.where(in_vector == 0)
                if len(finders[0]) > 0:
                    logger.error(f"ERROR. UNLINKED INPUT(S) IN BLOCK: {block.name} PORT(S): {finders[0]}")
                    error_trigger = True

            if block.out_ports == 1 and len(outputs) < block.out_ports:
                logger.error(f"ERROR. UNLINKED OUTPUT PORT: {block.name}")
                error_trigger = True
            elif block.out_ports > 1:
                out_vector = np.zeros(block.out_ports)
                for tupla in outputs:
                    out_vector[tupla['srcport']] += 1
                finders = np.where(out_vector == 0)
                if len(finders[0]) > 0:
                    logger.error(f"ERROR. UNLINKED OUTPUT(S) IN BLOCK: {block.name} PORT(S): {finders[0]}")
                    error_trigger = True

        if error_trigger:
            logger.error("Diagram integrity check failed.")
            return False
        logger.debug("NO ISSUES FOUND IN DIAGRAM")
        return True

    def _validate_signal_dimensions(self):
        """
        Validate signal dimensions between connected blocks.
        Logs warnings for potential dimension mismatches (non-fatal).
        """
        warnings = []
        
        for line in self.line_list:
            if line.hidden:
                continue
                
            # Find source and destination blocks
            src_block = next((b for b in self.blocks_list if b.name == line.srcblock), None)
            dst_block = next((b for b in self.blocks_list if b.name == line.dstblock), None)
            
            if not src_block or not dst_block:
                continue
            
            # Get expected output width from source block
            src_width = None
            if hasattr(src_block, 'block_instance') and src_block.block_instance:
                outputs = getattr(src_block.block_instance, 'outputs', [])
                if line.srcport < len(outputs):
                    port_def = outputs[line.srcport]
                    if isinstance(port_def, dict):
                        src_width = port_def.get('width', None)
            
            # Get expected input width from destination block
            dst_width = None
            if hasattr(dst_block, 'block_instance') and dst_block.block_instance:
                try:
                    inputs = (dst_block.block_instance.get_inputs(dst_block.params) 
                              if hasattr(dst_block.block_instance, 'get_inputs') 
                              else getattr(dst_block.block_instance, 'inputs', []))
                    if line.dstport < len(inputs):
                        port_def = inputs[line.dstport]
                        if isinstance(port_def, dict):
                            dst_width = port_def.get('width', None)
                except Exception:
                    pass
            
            # Check for dimension mismatch (only if both specify a width)
            if src_width is not None and dst_width is not None:
                if src_width != dst_width and src_width != -1 and dst_width != -1:
                    warnings.append(
                        f"Dimension mismatch: {src_block.name}[{line.srcport}] → "
                        f"{dst_block.name}[{line.dstport}] (width {src_width} → {dst_width})"
                    )
        
        for warning in warnings:
            logger.warning(f"Signal dimension: {warning}")
        
        if warnings:
            logger.info(f"Signal dimension validation: {len(warnings)} potential mismatch(es) detected")
        else:
            logger.debug("Signal dimension validation: No mismatches detected")

    def count_rk45_ints(self):
        """
        :purpose: Checks all integrators and looks if there's at least one that use 'RK45' as integration method.
        """
        for block in self.blocks_list:
            if block.block_fn == 'Integrator' and block.params['method'] == 'RK45':
                return True
            elif block.block_fn == 'External' and 'method' in block.params.keys() and block.params['method'] == 'RK45':
                return True
        return False

    def update_global_list(self, block_name, h_value, h_assign=False):
        """
        :purpose: Updates the global execution list.
        :param block_name: Block object name id.
        :param h_value: Value in graph hierarchy.
        :param h_assign: Flag that defines if the block gets assigned with h_value or not.
        :type block_name: str
        :type h_value: int
        :type h_assign: bool
        """
        # h_assign is used to assign the degree of hierarchy only in the first iteration.
        for elem in self.global_computed_list:
            if elem['name'] == block_name:
                if h_assign:
                    elem['hierarchy'] = h_value
                elem['computed_data'] = True

    def check_global_list(self):
        """
        :purpose: Checks if there are no blocks of a graph left unexecuted.
        """
        for elem in self.global_computed_list:
            if not elem['computed_data']:
                return False
        return True

    def count_computed_global_list(self):
        """
        :purpose: Counts the number of already computed blocks of a graph.
        """
        return len([x for x in self.global_computed_list if x['computed_data']])

    def reset_execution_data(self):
        """
        :purpose: Resets the execution state for all the blocks of a graph.
        """
        for i in range(len(self.blocks_list)):
            self.global_computed_list[i]['computed_data'] = False
            self.blocks_list[i].computed_data = False
            self.blocks_list[i].data_recieved = 0
            self.blocks_list[i].data_sent = 0
            self.blocks_list[i].input_queue = {}
            self.blocks_list[i].hierarchy = self.global_computed_list[i]['hierarchy']

    def get_max_hierarchy(self):
        """
        :purpose: Finds in the global execution list the max value in hierarchy.
        """
        max_val = 0
        for elem in self.global_computed_list:
            if elem['hierarchy'] >= max_val:
                max_val = elem['hierarchy']
        return max_val

    def detect_algebraic_loops(self, uncomputed_blocks):
        from collections import deque

        block_map = {block.name: block for block in self.blocks_list}
        uncomputed_block_names = {block.name for block in uncomputed_blocks}

        # Build the graph only with uncomputed blocks
        graph = {block.name: [] for block in uncomputed_blocks}
        in_degree = {block.name: 0 for block in uncomputed_blocks}

        for block in uncomputed_blocks:
            children = self.get_outputs(block.name)
            for child_info in children:
                child_name = child_info['dstblock']
                if child_name in uncomputed_block_names:
                    graph[block.name].append(child_name)
                    in_degree[child_name] += 1

        # Find all nodes with an in-degree of 0
        queue = deque([name for name, degree in in_degree.items() if degree == 0])

        # Perform topological sort
        count = 0
        while queue:
            u = queue.popleft()
            count += 1
            for v in graph.get(u, []):
                in_degree[v] -= 1
                if in_degree[v] == 0:
                    queue.append(v)

        # If the count of visited nodes is less than the number of nodes, there is a cycle
        if count < len(uncomputed_blocks):
            # Find the nodes involved in the cycle
            cycle_nodes = {name for name, degree in in_degree.items() if degree > 0}
            
            # Check if the cycle contains any memory blocks
            has_memory_block = any(node in self.memory_blocks for node in cycle_nodes)
            
            if not has_memory_block:
                return True, list(cycle_nodes)

        return False, []

    def get_outputs(self, block_name):
        """
        :purpose: Finds all the blocks that need a "block_name" result as input.
        :param block_name: Block object name id.
        :type block_name: str
        """
        # Returns a list of dictionaries with the outgoing ports for block_name, as well as the incoming blocks and ports
        neighs = []
        for line in self.line_list:
            if line.srcblock == block_name:
                neighs.append({'srcport': line.srcport, 'dstblock': line.dstblock, 'dstport': line.dstport})
        return neighs



    def children_recognition(self, block_name, children_list):
        """
        :purpose: For a block, checks all the blocks that are connected to its outputs and sends a list with them.
        :param block_name: Block object name id.
        :param children_list: List of dictionaries with blocks data that require the output of block 'block_name'.
        :type block_name: str
        :type children_list: list
        """
        child_ports = []
        for child in children_list:
            if block_name in child.values():
                child_ports.append(child)
        if child_ports == []:
            return False, -1
        return True, child_ports

    def get_neighbors(self, block_name):
        inputs = []
        outputs = []
        for signal in self.line_list:
            if signal.dstblock == block_name:
                inputs.append({
                    'srcblock': signal.srcblock,
                    'srcport': signal.srcport,
                    'dstport': signal.dstport
                })
            if signal.srcblock == block_name:
                outputs.append({
                    'dstblock': signal.dstblock,
                    'srcport': signal.srcport,
                    'dstport': signal.dstport
                })
        return inputs, outputs

    def reset_memblocks(self):
        """
        :purpose: Resets the "_init_start_" parameter in all blocks.
        """
        for block in self.blocks_list:
            if '_init_start_' in block.params.keys():
                block.params['_init_start_'] = True
            # Also reset in exec_params if it exists (used during execution)
            if hasattr(block, 'exec_params') and block.exec_params:
                if '_init_start_' in block.exec_params:
                    block.exec_params['_init_start_'] = True
                # Clear any stored state like _prev
                if '_prev' in block.exec_params:
                    del block.exec_params['_prev']

    def plot_again(self):
        """
        :purpose: Plots the data saved in Scope and XYGraph blocks without needing to execute the simulation again.
        """
        if self.dirty:
            logger.error("ERROR: The diagram has been modified. Please run the simulation again.")
            return
        try:
            # Plot Scopes
            vectors = [
                x.params.get('vector')
                for x in self.blocks_list
                if x.block_fn == 'Scope' and 'vector' in x.params
            ]
            valid_vectors = [v for v in vectors if v is not None and hasattr(v, '__len__') and len(v) > 0]
            if valid_vectors:
                self.pyqtPlotScope()
            else:
                logger.info("PLOT: No scope data available to plot.")
            
            # Plot XYGraphs
            for block in self.blocks_list:
                if block.block_fn == 'XYGraph':
                    params = getattr(block, 'exec_params', block.params)
                    x_data = params.get('_x_data_', [])
                    y_data = params.get('_y_data_', [])
                    if x_data and y_data:
                        self._plot_xygraph(block)
            
            # Plot FFT spectrums
            for block in self.blocks_list:
                if block.block_fn == 'FFT':
                    params = getattr(block, 'exec_params', block.params)
                    buffer = params.get('_fft_buffer_', [])
                    if buffer and len(buffer) > 1:
                        self._plot_fft(block)
                        
        except Exception as e:
            logger.info(f"PLOT: Skipping plot; scope data not available yet ({str(e)})")

    def _plot_xygraph(self, block):
        """Plot XY graph data for a single XYGraph block."""
        import matplotlib.pyplot as plt
        
        # Data is stored in exec_params during execution
        params = getattr(block, 'exec_params', block.params)
        x_data = params.get('_x_data_', [])
        y_data = params.get('_y_data_', [])
        
        if not x_data or not y_data:
            return
        
        title = block.params.get('title', 'XY Plot')
        x_label = block.params.get('x_label', 'X')
        y_label = block.params.get('y_label', 'Y')
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(x_data, y_data, 'b-', linewidth=1)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(f"{title} ({block.name})")
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='box')
        
        plt.tight_layout()
        plt.show()

    def _plot_fft(self, block):
        """Plot FFT spectrum for a single FFT block."""
        import matplotlib.pyplot as plt
        
        params = getattr(block, 'exec_params', block.params)
        buffer = params.get('_fft_buffer_', [])
        time_data = params.get('_fft_time_', [])
        
        if not buffer or len(buffer) < 2:
            return
        
        # Convert to numpy array
        signal = np.array(buffer)
        if signal.ndim > 1:
            signal = signal[:, 0]  # Take first channel
        
        # Calculate sample rate
        if len(time_data) > 1:
            dt = np.mean(np.diff(time_data))
            fs = 1.0 / dt if dt > 0 else 1.0
        else:
            fs = 1.0 / self.sim_dt if hasattr(self, 'sim_dt') else 1.0
        
        # Apply window function
        window_type = params.get('window', 'hann')
        n = len(signal)
        if window_type == 'hann':
            window = np.hanning(n)
        elif window_type == 'hamming':
            window = np.hamming(n)
        elif window_type == 'blackman':
            window = np.blackman(n)
        else:
            window = np.ones(n)
        
        windowed_signal = signal * window
        
        # Compute FFT
        fft_result = np.fft.rfft(windowed_signal)
        freqs = np.fft.rfftfreq(n, d=1.0/fs)
        magnitude = np.abs(fft_result)
        
        # Normalize
        if params.get('normalize', True):
            max_mag = np.max(magnitude)
            if max_mag > 0:
                magnitude = magnitude / max_mag
        
        # Convert to dB if requested
        if params.get('log_scale', False):
            magnitude = 20 * np.log10(magnitude + 1e-12)
        
        # Plot
        title = params.get('title', 'FFT Spectrum')
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(freqs, magnitude, 'b-', linewidth=1)
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Magnitude (dB)' if params.get('log_scale', False) else 'Magnitude')
        ax.set_title(f"{title} ({block.name})")
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, fs/2])
        
        plt.tight_layout()
        plt.show()

    def export_data(self):
        """
        :purpose: Exports the data saved in Export blocks in .npz format.
        :description: This function is executed after the simulation has finished or stopped. It looks for export blocks, which have some vectors saved with signal outputs from previous blocks. Then it merge all vectors in one big matrix, which is exported with the time vector to a .npz file, formatted in a way it is ready for graph libraries.
        """
        vec_dict = {}
        export_toggle = False
        for block in self.blocks_list:
            if block.block_fn == 'Export':
                export_toggle = True
                labels = block.params['vec_labels']
                vector = block.params['vector']
                if block.params['vec_dim'] == 1:
                    vec_dict[labels] = vector
                elif block.params['vec_dim'] > 1:
                    for i in range(block.params['vec_dim']):
                        vec_dict[labels[i]] = vector[:, i]
        if export_toggle:
            np.savez('saves/' + self.filename[:-4], t=self.timeline, **vec_dict)
            logger.info("DATA EXPORTED TO " + 'saves/' + self.filename[:-4] + '.npz')

    def _is_discrete_upstream(self, block_name, visited=None):
        """
        Determine if a block (or any of its ancestors) is discrete-time/ZOH.
        """
        if visited is None:
            visited = set()

        if block_name in visited:
            return False
        visited.add(block_name)

        # Lookup the block in the model
        block = None
        try:
            block = self.model.get_block_by_name(block_name)
        except Exception:
            # Fallback in case model is not available for some reason
            block = next((b for b in self.blocks_list if b.name == block_name), None)

        if block is None:
            return False

        discrete_blocks = {'DiscreteTranFn', 'DiscreteStateSpace', 'ZeroOrderHold'}
        continuous_blocks = {'Integrator', 'TranFn', 'StateSpace', 'Derivative'}

        if block.block_fn in discrete_blocks:
            return True
        if block.block_fn in continuous_blocks:
            return False

        inputs, _ = self.get_neighbors(block.name)
        for conn in inputs:
            src_block = conn.get('srcblock') if isinstance(conn, dict) else getattr(conn, 'srcblock', None)
            if src_block and self._is_discrete_upstream(src_block, visited):
                return True
        return False

    def _scope_step_modes(self):
        """
        Build a list of step-mode flags (one per Scope block) for plotting.
        """
        modes = []
        for block in self.blocks_list:
            if block.block_fn == 'Scope':
                inputs, _ = self.get_neighbors(block.name)
                if not inputs:
                    modes.append(False)
                    continue
                step_mode = any(
                    self._is_discrete_upstream(
                        conn.get('srcblock') if isinstance(conn, dict) else getattr(conn, 'srcblock', None)
                    )
                    for conn in inputs
                    if (conn.get('srcblock') if isinstance(conn, dict) else getattr(conn, 'srcblock', None))
                )
                modes.append(step_mode)
        return modes

    def get_scope_traces(self):
        """
        Collect scope data as a flat list of traces for the waveform inspector.
        Returns (timeline, traces) where traces is a list of dicts:
        {'name': str, 'y': np.ndarray, 'step': bool}
        """
        if not hasattr(self, 'timeline') or self.timeline is None or len(self.timeline) == 0:
            return None, []

        step_modes = self._scope_step_modes()
        traces = []
        step_idx = 0
        for block in self.blocks_list:
            if block.block_fn != 'Scope':
                continue
            labels = block.exec_params.get('vec_labels', block.params.get('vec_labels'))
            vec = block.exec_params.get('vector', block.params.get('vector'))
            if vec is None:
                continue
            arr = np.array(vec)
            step_flag = step_modes[step_idx] if step_idx < len(step_modes) else False
            step_idx += 1

            if arr.ndim == 1:
                name = labels if isinstance(labels, str) else block.name
                traces.append({'name': name, 'y': arr, 'step': step_flag})
            elif arr.ndim == 2:
                # Multiple channels
                for i in range(arr.shape[1]):
                    if isinstance(labels, (list, tuple)) and i < len(labels):
                        name = labels[i]
                    else:
                        name = f"{block.name}[{i}]"
                    traces.append({'name': name, 'y': arr[:, i], 'step': step_flag})
            else:
                continue

        return self.timeline, traces

    def _record_run_history(self):
        """
        Save latest scope run into history (session, optionally persisted).
        """
        timeline, traces = self.get_scope_traces()
        if timeline is None or not traces:
            return

        run_entry = {
            "name": f"Run {len(self.run_history)+1}",
            "timeline": np.array(timeline, dtype=float),
            "traces": traces,
            "sim_dt": self.sim_dt,
            "sim_time": self.sim_time,
            "pinned": False,
        }
        self.run_history.append(run_entry)
        # Enforce limit on unpinned runs while keeping pinned entries
        unpinned_count = sum(1 for r in self.run_history if not r.get("pinned", False))
        if unpinned_count > self.run_history_limit:
            drop = unpinned_count - self.run_history_limit
            new_history = []
            for r in self.run_history:
                if not r.get("pinned", False) and drop > 0:
                    drop -= 1
                    continue
                new_history.append(r)
            self.run_history = new_history

        # Persist if enabled
        self.save_run_history()

    def _load_run_history(self):
        """Load persisted run history if enabled."""
        try:
            if not self.run_history_persist_path.exists():
                return
            with open(self.run_history_persist_path, "r") as f:
                payload = json.load(f)
            self.run_history_persist_enabled = bool(payload.get("persist", False))
            if not self.run_history_persist_enabled:
                return
            runs = payload.get("runs", [])
            loaded = []
            for run in runs:
                timeline = np.array(run.get("timeline", []), dtype=float)
                traces = []
                for tr in run.get("traces", []):
                    traces.append({
                        "name": tr.get("name", ""),
                        "y": np.array(tr.get("y", []), dtype=float),
                        "step": bool(tr.get("step", False))
                    })
                loaded.append({
                    "name": run.get("name", "Run"),
                    "timeline": timeline,
                    "traces": traces,
                    "sim_dt": run.get("sim_dt"),
                    "sim_time": run.get("sim_time"),
                    "pinned": bool(run.get("pinned", False))
                })
            self.run_history = loaded[-self.run_history_limit:] if loaded else []
        except Exception as e:
            logger.warning(f"Could not load run history: {e}")

    def save_run_history(self):
        """Persist run history to disk if enabled."""
        try:
            if not self.run_history_persist_enabled:
                return
            self.run_history_persist_path.parent.mkdir(parents=True, exist_ok=True)
            serializable_runs = []
            for run in self.run_history:
                serializable_runs.append({
                    "name": run.get("name", ""),
                    "timeline": run.get("timeline", np.array([])).tolist(),
                    "traces": [
                        {
                            "name": tr.get("name", ""),
                            "y": np.array(tr.get("y", [])).tolist(),
                            "step": bool(tr.get("step", False))
                        }
                        for tr in run.get("traces", [])
                    ],
                    "sim_dt": run.get("sim_dt"),
                    "sim_time": run.get("sim_time"),
                    "pinned": bool(run.get("pinned", False))
                })
            payload = {
                "persist": self.run_history_persist_enabled,
                "runs": serializable_runs
            }
            with open(self.run_history_persist_path, "w") as f:
                json.dump(payload, f)
        except Exception as e:
            logger.warning(f"Could not save run history: {e}")

    def set_run_history_persist(self, enabled: bool):
        """Toggle persistence of waveform run history."""
        self.run_history_persist_enabled = bool(enabled)
        if not enabled:
            # Optionally remove file to avoid stale data
            try:
                if self.run_history_persist_path.exists():
                    self.run_history_persist_path.unlink()
            except Exception:
                pass
        else:
            self.save_run_history()

    # Pyqtgraph functions
    def pyqtPlotScope(self):
        """
        :purpose: Plots the data saved in Scope blocks using pyqtgraph.
        :description: This function is executed while the simulation has stopped. It looks for Scope blocks, from which takes their 'vec_labels' parameter to get the labels of each vector and the 'vector' parameter containing the vector (or matrix if the input for the Scope block was a vector) and initializes a SignalPlot class object that uses pyqtgraph to show a graph.
        """

        logger.debug("Attempting to plot...")
        labels_list = []
        vector_list = []
        for block in self.blocks_list:
            if block.block_fn == 'Scope':
                logger.debug(f"Found Scope block: {block.name}")
                # Use exec_params as that's where simulation data is stored
                b_labels = block.exec_params.get('vec_labels', block.params.get('vec_labels'))
                labels_list.append(b_labels)
                b_vectors = block.exec_params.get('vector', block.params.get('vector'))
                if b_vectors is None:
                    b_vectors = []
                vector_list.append(b_vectors)
                logger.debug(f"Full vector for {block.name}: shape {np.shape(b_vectors)}")
                logger.debug(f"Labels: {b_labels}")
                logger.debug(f"Vector length: {len(b_vectors)}")
                logger.debug(f"Vector type: {type(b_vectors)}")
                logger.debug(f"Vector sample: {b_vectors[:5]}")

        step_modes = self._scope_step_modes()

        if labels_list and vector_list:
            logger.debug("Creating SignalPlot...")
            # Use step mode for discrete/ZOH signals to keep values constant between samples
            self.plotty = SignalPlot(self.sim_dt, labels_list, len(self.timeline), step_mode=step_modes)
            try:
                self.plotty.loop(new_t=self.timeline.astype(float), new_y=[np.array(v).astype(float) for v in vector_list])
                self.plotty.show()
                logger.debug("SignalPlot should be visible now.")
            except Exception as e:
                logger.error(f"Error in plotting: {e}")
        else:
            logger.debug("No data to plot.")

    def dynamic_pyqtPlotScope(self, step):
        """
        :purpose: Plots the data saved in Scope blocks dynamically with pyqtgraph.
        :description: This function is executed while the simulation is running, starting after all the blocks were executed in the first loop. It looks for Scope blocks, from which takes their 'labels' parameter and initializes a SignalPlot class object that uses pyqtgraph to show a graph. Then for each loop completed, it calls those Scope blocks again to get their vectors and update the graph with the new information.
        """
        if not self.dynamic_plot:
            return

        if step == 0:  # init
            labels_list = []
            for block in self.blocks_list:
                if block.block_fn == 'Scope':
                    b_labels = block.params['vec_labels']
                    labels_list.append(b_labels)

            if labels_list != []:
                step_modes = self._scope_step_modes()
                self.plotty = SignalPlot(self.sim_dt, labels_list, self.plot_trange, step_mode=step_modes)

        elif step == 1: # loop
            vector_list = []
            for block in self.blocks_list:
                if block.block_fn == 'Scope':
                    b_vectors = block.params['vector']
                    vector_list.append(b_vectors)
            if len(vector_list) > 0:
                self.plotty.loop(new_t=self.timeline, new_y=vector_list)
            else:
                self.dynamic_plot = False
                logger.info("DYNAMIC PLOT: OFF")


class SignalPlot(QWidget):
    """
    Class that manages the display of dynamic plots through the simulation.
    *WARNING: It uses pyqtgraph as base (MIT license, but interacts with PyQT5 (GPL)).*

    :param dt: Sampling time of the system.
    :param labels: List of names of the vectors.
    :param xrange: Maximum number of elements to plot in axis x.
    :type dt: float
    :type labels: list
    :type xrange: int

    """
    def __init__(self, dt, labels, xrange, step_mode=False):
        super().__init__()
        self.dt = dt
        self.step_mode = step_mode
        self.curve_step_modes = self._expand_step_modes(step_mode, len(labels))
        self.xrange = xrange * self.dt
        self.plot_items = []
        self.curves = []

        # Store data for export
        self.labels = labels
        self.timeline = None
        self.data_vectors = None

        main_layout = QVBoxLayout()
        self.setLayout(main_layout)

        # Create plot area
        plot_layout = QVBoxLayout()
        plot_layout.setSpacing(10)  # Add spacing between plots
        plot_layout.setContentsMargins(0, 0, 0, 10)  # Add bottom margin to plot area
        for idx, label in enumerate(labels):
            plot_widget = pg.PlotWidget(title=label)
            plot_widget.showGrid(x=True, y=True)

            # Configure axes for better visibility
            self._configure_plot_axes(plot_widget)

            # Use stepMode=True for Zero-Order Hold visualization (staircase plot)
            # This requires the x-axis (time) to have one more element than y-axis
            curve = plot_widget.plot(pen='y', stepMode=self.curve_step_modes[idx])
            self.plot_items.append(plot_widget)
            self.curves.append(curve)
            plot_layout.addWidget(plot_widget)

        main_layout.addLayout(plot_layout)

        # Add spacing before the export button to prevent it from overlapping the x-axis
        main_layout.addSpacing(20)

        # Add export button at bottom
        from lib.ui.button import Button
        self.export_button = QPushButton("Export to CSV...")
        self.export_button.setToolTip("Export plot data to CSV file")
        self.export_button.clicked.connect(self.export_to_csv)
        main_layout.addWidget(self.export_button)

        self.resize(800, 600)

    def _configure_plot_axes(self, plot_widget):
        """
        Configure plot widget axes for better visibility and consistent appearance.

        Sets up the x-axis (time) with proper labels, styling, and ensures
        sufficient space for axis labels to be displayed.

        :param plot_widget: The PlotWidget to configure
        :type plot_widget: pg.PlotWidget
        """
        # Configure x-axis to ensure labels are shown
        plot_widget.setLabel('bottom', 'Time')
        plot_widget.getAxis('bottom').setStyle(tickTextOffset=10)
        plot_widget.getAxis('bottom').setPen(pg.mkPen(color='k', width=1))
        plot_widget.getAxis('bottom').enableAutoSIPrefix(False)  # Disable SI scaling

        # Set minimum height for plot widget to ensure x-axis labels have room
        plot_widget.setMinimumHeight(200)

    def _expand_step_modes(self, step_mode, count):
        """
        Normalize step_mode (bool or list) into a list matching the plot count.
        """
        if isinstance(step_mode, (list, tuple, np.ndarray)):
            modes = [bool(mode) for mode in step_mode]
        else:
            modes = [bool(step_mode)] * count

        if len(modes) < count:
            modes.extend([modes[-1] if modes else False] * (count - len(modes)))
        elif len(modes) > count:
            modes = modes[:count]
        return modes

    def pltcolor(self, index, hues=9, hueOff=180, minHue=0, maxHue=360, val=255, sat=255, alpha=255):
        """
        :purpose: Assigns a color to a vector for plotting purposes.
        """
        third = (maxHue - minHue) / 3
        hues = int(hues)
        indc = int(index) // 3
        indr = int(index) % 3

        hsection = indr * third
        hrange = (indc * third / (hues // 3)) % third
        h = (hsection + hrange + hueOff) % 360
        return pg.hsvColor(h/360, sat/255, val/255, alpha/255)

    def plot_config(self, settings_dict={}):
        return

    def loop(self, new_t, new_y):
        """
        :purpose: Updates the time and scope vectors and plot them.
        """
        try:
            # Store data for export
            self.timeline = new_t
            self.data_vectors = new_y

            for i, curve in enumerate(self.curves):
                if i < len(new_y):
                    step_mode = self.curve_step_modes[i] if i < len(self.curve_step_modes) else False
                    if step_mode:
                        # For stepMode=True, x must be len(y) + 1
                        # Ensure t_step is exactly len(new_y[i]) + 1
                        if len(new_t) == len(new_y[i]):
                            t_step = np.append(new_t, new_t[-1] + self.dt)
                        elif len(new_t) > len(new_y[i]):
                            t_step = new_t[:len(new_y[i]) + 1]
                        else:
                            # This case shouldn't happen if logic is correct, but handle safely
                            t_step = np.append(new_t, [new_t[-1] + self.dt] * (len(new_y[i]) - len(new_t) + 1))
                        
                        curve.setData(t_step, new_y[i])
                    else:
                        # For normal plotting, x and y must have same length
                        # Ensure t_step is exactly len(new_y[i])
                        if len(new_t) == len(new_y[i]):
                            t_step = new_t
                        elif len(new_t) > len(new_y[i]):
                            t_step = new_t[:len(new_y[i])]
                        else:
                            # Pad time if needed (shouldn't happen)
                            t_step = np.append(new_t, [new_t[-1] + self.dt] * (len(new_y[i]) - len(new_t)))

                        curve.setData(t_step, new_y[i])
        except Exception as e:
            logger.error(f"Error updating plot: {e}")

    def export_to_csv(self):
        """
        Export plot data to CSV file with user selection of which scopes to include.
        """
        import csv
        from datetime import datetime

        # Check if we have data to export
        if self.timeline is None or self.data_vectors is None:
            QMessageBox.warning(self, "No Data", "No plot data available to export.")
            return

        # Create scope selection dialog
        dialog = QDialog(self)
        dialog.setWindowTitle("Select Scopes to Export")
        dialog_layout = QVBoxLayout()

        # Instructions
        instruction_label = QLabel("Select which scope blocks to include in the CSV export:")
        dialog_layout.addWidget(instruction_label)

        # Create checkboxes for each scope
        checkboxes = []
        for i, label in enumerate(self.labels):
            # labels can be either a string or a list of strings
            if isinstance(label, str):
                scope_name = label
            elif isinstance(label, list):
                scope_name = f"Scope {i} ({', '.join(label[:3])}{'...' if len(label) > 3 else ''})"
            else:
                scope_name = f"Scope {i}"

            checkbox = QWidget()
            checkbox_layout = QHBoxLayout()
            checkbox_layout.setContentsMargins(0, 0, 0, 0)

            from PyQt5.QtWidgets import QCheckBox
            cb = QCheckBox(scope_name)
            cb.setChecked(True)  # Default to all selected
            checkbox_layout.addWidget(cb)
            checkbox.setLayout(checkbox_layout)

            checkboxes.append(cb)
            dialog_layout.addWidget(checkbox)

        # Buttons
        button_layout = QHBoxLayout()
        select_all_btn = QPushButton("Select All")
        deselect_all_btn = QPushButton("Deselect All")
        ok_btn = QPushButton("Export")
        cancel_btn = QPushButton("Cancel")

        select_all_btn.clicked.connect(lambda: [cb.setChecked(True) for cb in checkboxes])
        deselect_all_btn.clicked.connect(lambda: [cb.setChecked(False) for cb in checkboxes])
        ok_btn.clicked.connect(dialog.accept)
        cancel_btn.clicked.connect(dialog.reject)

        button_layout.addWidget(select_all_btn)
        button_layout.addWidget(deselect_all_btn)
        button_layout.addStretch()
        button_layout.addWidget(ok_btn)
        button_layout.addWidget(cancel_btn)

        dialog_layout.addLayout(button_layout)
        dialog.setLayout(dialog_layout)
        dialog.setMinimumWidth(400)

        # Show dialog and get result
        if dialog.exec_() != QDialog.Accepted:
            return

        # Get selected scopes
        selected_indices = [i for i, cb in enumerate(checkboxes) if cb.isChecked()]

        if not selected_indices:
            QMessageBox.warning(self, "No Selection", "Please select at least one scope to export.")
            return

        # Get save file path
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_filename = f"plot_data_{timestamp}.csv"

        filepath, _ = QFileDialog.getSaveFileName(
            self,
            "Export Plot Data to CSV",
            default_filename,
            "CSV Files (*.csv);;All Files (*)"
        )

        if not filepath:
            return

        # Export data to CSV
        try:
            with open(filepath, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)

                # Build header row
                header = ['time']
                column_data = []

                for idx in selected_indices:
                    label = self.labels[idx]
                    vector = self.data_vectors[idx]

                    # Handle multi-dimensional vectors
                    if isinstance(label, list):
                        # Multiple signals in this scope
                        for i, sig_label in enumerate(label):
                            header.append(sig_label)
                            # Extract column from 2D array
                            if len(vector.shape) > 1:
                                column_data.append(vector[:, i])
                            else:
                                column_data.append(vector)
                    else:
                        # Single signal
                        header.append(label)
                        column_data.append(vector.flatten() if hasattr(vector, 'flatten') else vector)

                # Write header
                writer.writerow(header)

                # Write data rows
                num_rows = len(self.timeline)
                for row_idx in range(num_rows):
                    row = [self.timeline[row_idx]]
                    for col_data in column_data:
                        if row_idx < len(col_data):
                            row.append(col_data[row_idx])
                        else:
                            row.append('')  # Handle mismatched lengths
                    writer.writerow(row)

            # Log success and update button text briefly
            logger.info(f"Plot data exported to {filepath} ({num_rows} rows, {len(header)} columns)")

            # Briefly change button text to show success
            original_text = self.export_button.text()
            self.export_button.setText(f"✓ Exported to {os.path.basename(filepath)}")
            self.export_button.setEnabled(False)

            # Reset button after 3 seconds
            QTimer.singleShot(3000, lambda: (
                self.export_button.setText(original_text),
                self.export_button.setEnabled(True)
            ))

        except Exception as e:
            QMessageBox.critical(self, "Export Failed", f"Failed to export data:\n{str(e)}")
            logger.error(f"Failed to export plot data: {e}")

    def sort_labels(self, labels):
        """
        :purpose: Rearranges the list if some elements are lists too.
        """
        self.labels = []
        for elem in labels:
            if isinstance(elem, str):
                self.labels += [elem]
            elif isinstance(elem, list):
                self.labels += elem

    def sort_vectors(self, ny):
        """
        :purpose: Rearranges all vectors in one matrix.
        """
        new_vec = ny[0]
        for i in range(1, len(ny)):
            new_vec = np.column_stack((new_vec, ny[i]))
        return new_vec
