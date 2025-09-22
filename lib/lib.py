"lib.py - Contains all the core functions and classes for the simulation and execution of the graphs."

import numpy as np
import copy
import time
import json
import importlib
import os
import sys
from tqdm import tqdm
from PyQt5.QtWidgets import QWidget, QFileDialog, QApplication, QDialog, QVBoxLayout, QLabel, QLineEdit, QPushButton, QCheckBox, QInputDialog
from PyQt5.QtGui import QColor, QPen, QFont, QPixmap, QPainter, QCursor, QPainterPath, QPolygonF, QTransform
from PyQt5.QtCore import Qt, QRect, QPoint, QTimer, QEvent
import pyqtgraph as pg
from lib.functions import *
from lib.dialogs import ParamDialog, PortDialog, SimulationDialog
import math
import logging
from modern_ui.themes.theme_manager import theme_manager

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
        logger.debug("Initializing DSim")
        self.SCREEN_WIDTH = 1280
        self.SCREEN_HEIGHT = 720 + 50

        self.canvas_top_limit = 60
        self.canvas_left_limit = 200

        self.colors = {
            'black': QColor(0, 0, 0),
            'red': QColor(255, 0, 0),
            'green': QColor(0, 255, 0),
            'blue': QColor(0, 0, 255),
            'yellow': QColor(255, 255, 0),
            'magenta': QColor(255, 0, 255),
            'cyan': QColor(0, 255, 255),
            'purple': QColor(128, 0, 255),
            'orange': QColor(255, 128, 0),
            'aqua': QColor(0, 255, 128),
            'pink': QColor(255, 0, 128),
            'lime_green': QColor(128, 255, 0),
            'light_blue': QColor(0, 128, 255),
            'dark_red': QColor(128, 0, 0),
            'dark_green': QColor(0, 128, 0),
            'dark_blue': QColor(0, 0, 128),
            'dark_gray': QColor(64, 64, 64),
            'gray': QColor(128, 128, 128),
            'light_gray': QColor(192, 192, 192),
            'white': QColor(255, 255, 255)
        }

        self.FPS = 60

        self.l_width = 5                    # Line width in selected mode
        self.ls_width = 5                   # Line-block spacing width in selected mode

        self.filename = 'data.dat'          # Saved Filename by default
        self.sim_time = 1.0                 # Default simulation time
        self.sim_dt = 0.01                  # Base sampling time for simulation (Default: 10ms)
        self.plot_trange = 100              # Window width for dynamic plot (Default: 100 samples)

        self.menu_blocks = []               # List of base blocks
        self.blocks_list = []               # List of blocks
        self.line_list = []                 # List of lines

        self.line_creation = 0              # Boolean (3 states) for creation of a line
        self.only_one = False               # Boolean to prevent more than one block from being able to perform an operation
        self.enable_line_selection = False  # Boolean to indicate whether it is possible to select a line or not
        self.holding_CTRL = False           # Boolean to control the state of the CTRL key
        self.execution_initialized = False  # Boolean to indicate if the graph was executed at least once
        self.ss_count = 0                   # Screenshot counter
        
        self.execution_pause = False        # Boolean indicating whether execution was paused at some point in time or not
        self.execution_stop = False         # Boolean indicating whether the execution was completely stopped or not
        self.dynamic_plot = False           # Boolean indicating whether the plot is displayed dynamically advancing or not
        self.real_time = True               # Boolean indicating whether the simulation runs in real-time or not
        self.error_msg = ""                   # Error message from simulation
        self.dirty = False                  # Boolean indicating whether the diagram has been modified

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
        """
        :purpose: Function that adds a block to the interface, with a unique ID.
        :description: From a visible list of MenuBlocks objects, a complete Block instance is created, which is available for editing its parameters or connecting to other blocks.
        :param block: Base-block containing the base parameters for each type of block.
        :param m_pos: Coordinates (x, y) to locate the upper left corner of the future block.
        :type block: BaseBlock class
        :type m_pos: tuple
        :bugs: Under a wrongly configured MenuBlock, the resulting block may not have the correct qualities or parameters.
        """
        logger.debug(f"Adding new block of type {block.block_fn} at position {m_pos}")
        logger.debug(f"m_pos type: {type(m_pos)}, m_pos.x(): {m_pos.x()} (type: {type(m_pos.x())}), m_pos.y(): {m_pos.y()} (type: {type(m_pos.y())})")
        logger.debug(f"block.side_length: {block.side_length} (types: {type(block.side_length[0])}, {type(block.side_length[1])})")
        
        id_list = [int(b_elem.name[len(b_elem.block_fn):]) for b_elem in self.blocks_list if b_elem.block_fn == block.block_fn]
        sid = max(id_list) + 1 if id_list else 0

        try:
            # Ensure all values are integers
            mouse_x = int(m_pos.x() - block.side_length[0] // 2)
            mouse_y = int(m_pos.y() - block.side_length[1] // 2)
            logger.debug(f"Calculated mouse_x: {mouse_x} (type: {type(mouse_x)}), mouse_y: {mouse_y} (type: {type(mouse_y)})")
            logger.debug(f"block.size: {block.size} (types: {type(block.size[0])}, {type(block.size[1])})")
            
            # Ensure block size values are integers too
            width = int(block.size[0])
            height = int(block.size[1])
            logger.debug(f"Final values - mouse_x: {mouse_x}, mouse_y: {mouse_y}, width: {width}, height: {height}")
            
            block_collision = QRect(mouse_x, mouse_y, width, height)
            logger.debug(f"QRect created successfully: {block_collision}")
        except Exception as e:
            logger.error(f"Error creating QRect: {str(e)}")
            # Fallback with explicit integer conversion
            mouse_x = int(float(m_pos.x()) - float(block.side_length[0]) // 2)
            mouse_y = int(float(m_pos.y()) - float(block.side_length[1]) // 2)
            width = int(float(block.size[0]))
            height = int(float(block.size[1]))
            block_collision = QRect(mouse_x, mouse_y, width, height)

        new_block = DBlock(block.block_fn, sid, block_collision, block.b_color, block.ins, block.outs, block.b_type, block.io_edit, block.fn_name, copy.deepcopy(block.params), block.external)
        self.blocks_list.append(new_block)
        self.dirty = True
        logger.debug(f"New block created: {new_block.name}")
        return new_block
    def add_line(self, srcData, dstData):
        """
        :purpose: Function that adds a line to the interface, with a unique ID.
        :description: Based on the existence of one or more blocks, this function creates a line between the last selected ports.
        :param srcData: Triplet containing 'block name', 'port number', 'port coordinates' of an output port (starting point for the line).
        :param dstData: Triplet containing 'block name', 'port number', 'port coordinates' of an input port (finish point for the line).
        :type srcData: triplet
        :type dstData: triplet
        """
    
        if srcData is None or dstData is None:
            logger.debug("Error: Invalid line data")
            return None

        id_list = [int(line.name[4:]) for line in self.line_list]
        sid = max(id_list) + 1 if id_list else 0

        try:
            line = DLine(sid, srcblock=srcData[0], srcport=srcData[1],
                 dstblock=dstData[0], dstport=dstData[1],
                 points=(srcData[2], dstData[2]))
            #line.color = QColor(list(self.colors.values())[sid % len(self.colors)])  # Assign a color from the color list
            line.color = QColor(255,0,0)  # Assign red to lines

            self.line_list.append(line)
            self.dirty = True
            logger.debug(f"Line created: {line.name}")
            logger.debug(f"Current line_list: {[l.name for l in self.line_list]}")
            return line
        except Exception as e:
            logger.error(f"Error creating line: {e}")
            return None

    def remove_block_and_lines(self,block):
        """
        :purpose: Function to remove blocks or lines.
        :description: Removes a block or a line depending on whether it is selected or not.
        :notes: Lines associated to a block being removed are also removed.
        """
        self.blocks_list.remove(block)
        self.line_list = [line for line in self.line_list if not self.check_line_block(line, [block.name])]
        self.dirty = True
        logger.debug(f"Removed block {block.name} and associated lines")

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
            line.draw_line(painter)

    def update_lines(self):
        """
        :purpose: Updates lines according to the location of blocks if these changed place.
        """
        logger.debug("Updating lines")
        for line in self.line_list:
            line.update_line(self.blocks_list)

    def display_blocks(self, painter):
        """
        :purpose: Draws blocks defined in the main list on the screen.
        :param painter: A layer in a pygame canvas where the figure is drawn.
        """
        if painter is None:
            return
        for b_elem in self.blocks_list:
            b_elem.draw_Block(painter)

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

    def menu_blocks_init(self):
        """
        :purpose: Function that initializes all types of blocks available in the menu.
        :description: From the MenuBlocks class, base blocks are generated for the functions already defined in lib.functions.py. Then they are accumulated in a list so that they are available in the interface menu.
        """
        # block_fn, fn_name, {# inputs, # output, execution hierarchy}, {<specific argument/parameters>}, color, (width, height), allows_io_change

        # source-type blocks
        step = MenuBlocks(block_fn="Step", fn_name='step',
                        io_params={'inputs': 0, 'outputs': 1, 'b_type': 0, 'io_edit': False}, ex_params={'value': 1.0, 'delay': 0.0, 'type': 'up', 'pulse_start_up': True, '_init_start_': True},
                        b_color='blue', coords=(80, 80))

        ramp = MenuBlocks(block_fn="Ramp", fn_name='ramp',
                        io_params={'inputs': 0, 'outputs': 1, 'b_type': 0, 'io_edit': False}, ex_params={'slope': 1.0, 'delay': 0.0},
                        b_color='light_blue', coords=(80, 80))

        sine = MenuBlocks(block_fn="Sine", fn_name='sine',
                        io_params={'inputs': 0, 'outputs': 1, 'b_type': 0, 'io_edit': False}, ex_params={'amplitude': 1.0, 'omega': 1.0, 'init_angle': 0},
                        b_color='cyan', coords=(80, 80))

        noise = MenuBlocks(block_fn="Noise", fn_name='noise',
                        io_params={'inputs': 0, 'outputs': 1, 'b_type': 0, 'io_edit': False}, ex_params={'sigma': 1, 'mu': 0},
                        b_color='purple', coords=(80, 80))


        # N-process-type blocks
        integrator = MenuBlocks(block_fn="Integr", fn_name='integrator',
                        io_params={'inputs': 1, 'outputs': 1, 'b_type': 1, 'io_edit': False}, ex_params={'init_conds': 0.0, 'method': 'FWD_EULER', '_init_start_': True},
                        b_color='magenta', coords=(100, 80))

        transfer_function = MenuBlocks(block_fn="TranFn", fn_name='transfer_function',
                                   io_params={'inputs': 1, 'outputs': 1, 'b_type': 1, 'io_edit': False},
                                   ex_params={'numerator': [1], 'denominator': [1, 1], 'init_conds': 0.0, '_init_start_': True},
                                   b_color='purple', coords=(100, 80))
        

        # Z-process-type blocks
        derivative = MenuBlocks(block_fn="Deriv", fn_name='derivative',
                        io_params={'inputs': 1, 'outputs': 1, 'b_type': 2, 'io_edit': False}, ex_params={'_init_start_': True},
                        b_color=(255, 0, 200), coords=(100, 80))

        adder = MenuBlocks(block_fn="Sum", fn_name='adder',
                        io_params={'inputs': 2, 'outputs': 1, 'b_type': 2, 'io_edit': 'input'}, ex_params={'sign': "++"},
                        b_color='lime_green', coords=(60, 60))

        sigproduct = MenuBlocks(block_fn="SgProd", fn_name='sigproduct',
                        io_params={'inputs': 2, 'outputs': 1, 'b_type': 2, 'io_edit': 'input'}, ex_params={},
                        b_color='green', coords=(90, 70))

        gain = MenuBlocks(block_fn="Gain", fn_name='gain',
                        io_params={'inputs': 1, 'outputs': 1, 'b_type': 2, 'io_edit': False}, ex_params={'gain': 1.0},
                        b_color=(255, 216, 0), coords=(80, 80))

        exponential = MenuBlocks(block_fn="Exp", fn_name='exponential',
                        io_params={'inputs': 1, 'outputs': 1, 'b_type': 2, 'io_edit': False}, ex_params={'a': 1.0, 'b': 1.0},
                        b_color='yellow', coords=(80, 80))  # a*e^bx

        mux = MenuBlocks(block_fn="Mux", fn_name="mux",
                        io_params={'inputs': 2, 'outputs': 1, 'b_type': 2, 'io_edit': 'input'}, ex_params={},
                        b_color=(190, 0, 255), coords=(80, 80))

        demux = MenuBlocks(block_fn="Demux", fn_name="demux",
                        io_params={'inputs': 1, 'outputs': 2, 'b_type': 2, 'io_edit': 'output'}, ex_params={'output_shape': 1},
                        b_color=(170, 0, 255), coords=(80, 80))


        # Terminal-type blocks
        terminator = MenuBlocks(block_fn="Term", fn_name='terminator',
                        io_params={'inputs': 1, 'outputs': 0, 'b_type': 3, 'io_edit': False}, ex_params={},
                        b_color=(255, 106, 0), coords=(80, 80))

        scope = MenuBlocks(block_fn="Scope", fn_name='scope',
                        io_params={'inputs': 1, 'outputs': 0, 'b_type': 3, 'io_edit': False}, ex_params={'labels': 'default', '_init_start_': True},
                        b_color='red', coords=(80, 80))

        export = MenuBlocks(block_fn="Export", fn_name="export",
                        io_params={'inputs': 1, 'outputs': 0, 'b_type': 3, 'io_edit': False}, ex_params={'str_name': 'default', '_init_start_': True},
                        b_color='orange', coords=(90, 80))

        bodemag = MenuBlocks(block_fn="BodeMag", fn_name='bodemag',
                        io_params={'inputs': 1, 'outputs': 0, 'b_type': 3, 'io_edit': False}, ex_params={},
                        b_color='dark_red', coords=(80, 80))


        # External/general use block
        external = MenuBlocks(block_fn="External", fn_name='external',
                        io_params={'inputs': 1, 'outputs': 1, 'b_type': 2, 'io_edit': False}, ex_params={"filename": '<no filename>'},
                        b_color='light_gray', coords=(140, 80), external=True)

        self.menu_blocks = [step, ramp, sine, noise, integrator, transfer_function, derivative, adder, sigproduct, gain, exponential, mux, demux, terminator, scope, export, bodemag, external]

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

    def save(self, autosave=False):
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

        with open(file, 'w') as fp:
            json.dump(main_dict, fp, indent=4)

        if not autosave:
            self.filename = os.path.basename(file)  # Keeps the name of the file if you want to save it again later

        print("SAVED AS", file)

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
        sim_data = data['sim_data']
        blocks_data = data['blocks_data']
        lines_data = data['lines_data']

        self.clear_all()
        self.update_sim_data(sim_data)
        self.ss_count = 0
        for block in blocks_data:
            self.update_blocks_data(block)
        for line in lines_data:
            self.update_lines_data(line)

        self.filename = os.path.basename(file)  # Keeps the name of the file if you want to save it again later

        print("LOADED FROM", file)

    def update_sim_data(self, data):
        """
        :purpose: Updates information related with the main class variables saved in a file to the current simulation.
        :param data: Dictionary with DSim parameters.
        :type data: dict
        """
        self.SCREEN_WIDTH = data['wind_width']
        self.SCREEN_HEIGHT = data['wind_height']
        self.FPS = data['fps']
        self.line_creation = 0
        self.only_one = data['only_one']
        self.enable_line_selection = data['enable_line_sel']
        self.sim_time = data['sim_time']
        self.sim_dt = data['sim_dt']
        self.plot_trange = data['sim_trange']

    def update_blocks_data(self, block_data):
        """
        :purpose: Updates information related with all the blocks saved in a file to the current simulation.
        :param block_data: Dictionary with Block object id, parameters, variables, etc.
        :type block_data: dict
        """
        block = DBlock(block_fn=block_data['block_fn'],
                      sid=block_data['sid'],
                      coords=(block_data['coords_left'], block_data['coords_top'], block_data['coords_width'], block_data['coords_height_base']),
                      color=block_data['b_color'],
                      in_ports=block_data['in_ports'],
                      out_ports=block_data['out_ports'],
                      b_type=block_data['b_type'],
                      io_edit=block_data['io_edit'],
                      fn_name=block_data['fn_name'],
                      params=block_data['params'],
                      external=block_data['external'],
                      username=block_data['username'])
        block.height = block_data['coords_height']
        block.selected = block_data['selected']
        block.dragging = block_data['dragging']
        block.flipped = block_data.get('flipped', False)
        self.blocks_list.append(block)

    def update_lines_data(self, line_data):
        """
        :purpose: Updates information related with all the lines saved in a file to the current simulation.
        :param line_data: Dictionary with Line object id, parameters, variables, etc.
        :type line_data: dict
        """
        line = DLine(sid=line_data['sid'],
                    srcblock=line_data['srcblock'],
                    srcport=line_data['srcport'],
                    dstblock=line_data['dstblock'],
                    dstport=line_data['dstport'],
                    points=line_data['points'],
                    cptr=line_data['cptr'])
        line.selected = line_data['selected']
        line.update_line(self.blocks_list)
        self.line_list.append(line)

    def clear_all(self):
        """
        :purpose: Cleans the screen from all blocks, lines and some main variables.
        """
        self.blocks_list = []
        self.line_list = []
        self.line_creation = 0
        self.only_one = False
        self.enable_line_selection = False
        self.buttons_list[6].active = False  # Disable plot button
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
                print("Invalid input. Using default values.")
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
            self.execution_function = DFunctions()
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

            print("*****INIT NEW EXECUTION*****")

            for block in self.blocks_list:
                block.params['dtime'] = self.sim_dt
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
            print("Initializing execution...")

            # Generation of a checklist for the computation of functions
            self.global_computed_list = [{'name': x.name, 'computed_data': x.computed_data, 'hierarchy': x.hierarchy}
                                    for x in self.blocks_list]
            self.reset_execution_data()
            self.execution_time_start = time.time()
            logger.debug("Execution initialization complete")
        except Exception as e:
            logger.error(f"Error during execution initialization: {str(e)}")
            return False

        print("*****EXECUTION START*****")
        logger.info("*****EXECUTION START*****")

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
        logger.info(f"MEMORY BLOCKS IDENTIFIED: {self.memory_blocks}")

        # Check for integrators using Runge-Kutta 45 and initialize counter
        self.rk45_len = self.count_rk45_ints()
        self.rk_counter = 0

        # The block diagram starts with looking for source type blocks
        for block in self.blocks_list:
            logger.info(f"Initial processing of block: {block.name}, b_type: {block.b_type}")
            children = {}
            out_value = {}
            if block.b_type == 0:
                # The function is executed (differentiate between internal and external function first)
                if block.external:
                    try:
                        out_value = getattr(block.file_function, block.fn_name)(time=self.time_step, inputs=block.input_queue, params=block.params)
                    except:
                        print("ERROR FOUND IN EXTERNAL FUNCTION",block.file_function)
                        self.execution_failed(out_value.get('error', 'Unknown error'))
                        return False
                else:
                    out_value = getattr(self.execution_function, block.fn_name)(time=self.time_step, inputs=block.input_queue, params=block.params)
                block.computed_data = True
                block.hierarchy = 0
                self.update_global_list(block.name, h_value=0, h_assign=True)
                children = self.get_outputs(block.name)

            elif block.name in self.memory_blocks:
                kwargs = {
                    'time': self.time_step,
                    'inputs': block.input_queue,
                    'params': block.params,
                    'output_only': True
                }
                if block.block_fn == 'Integr':
                    kwargs['next_add_in_memory'] = False
                    kwargs['dtime'] = self.sim_dt
                
                if block.external:
                    try:
                        out_value = getattr(block.file_function, block.fn_name)(**kwargs)
                    except:
                        print("ERROR FOUND IN EXTERNAL FUNCTION", block.file_function)
                        self.execution_failed(out_value.get('error', 'Unknown error'))
                        return False
                else:
                    out_value = getattr(self.execution_function, block.fn_name)(**kwargs)
                children = self.get_outputs(block.name)

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
                        logger.info(f"Propagated output from {block.name} to {mblock.name}. {mblock.name} now has {mblock.data_recieved}/{mblock.in_ports} inputs.")

        for b in self.blocks_list:
            logger.info(f"PRE-WHILE CHECK: Block {b.name}, data_recieved: {b.data_recieved}, in_ports: {b.in_ports}, computed: {b.computed_data}")

        # The diagram continues to be executed through the following blocks
        h_count = 1
        while not self.check_global_list():
            logger.info(f"Starting execution iteration, h_count: {h_count}")
            for block in self.blocks_list:
                # This part is executed only if the received data is equal to the number of input ports and the block has not been computed yet.
                if block.data_recieved == block.in_ports and not block.computed_data:
                    logger.info(f"Executing block: {block.name}, data_recieved: {block.data_recieved}, in_ports: {block.in_ports}, computed_data: {block.computed_data}")
                    # The function is executed (differentiate between internal and external function first)
                    if block.external:
                        try:
                            out_value = getattr(block.file_function, block.fn_name)(time=self.time_step, inputs=block.input_queue, params=block.params)
                        except:
                            print("ERROR FOUND IN EXTERNAL FUNCTION", block.file_function)
                            self.execution_failed()
                            return False
                    else:
                        out_value = getattr(self.execution_function, block.fn_name)(time=self.time_step, inputs=block.input_queue, params=block.params)

                    # It is checked that the function has not delivered an error
                    if 'E' in out_value.keys() and out_value['E']:
                        self.execution_failed(out_value.get('error', 'Unknown error'))
                        return False

                    # The computed_data booleans are updated in the global list as well as in the block itself
                    self.update_global_list(block.name, h_value=h_count, h_assign=True)
                    block.computed_data = True

                    # The blocks that require the processed data from this block are searched.
                    children = self.get_outputs(block.name)
                    if block.b_type not in [1, 3]:  # Elements that do not deliver a result to children (1 is initial cond.)
                        for mblock in self.blocks_list:
                            is_child, tuple_list = self.children_recognition(block_name=mblock.name, children_list=children)
                            if is_child:
                                # Data is sent to each required port of the child block
                                for tuple_child in tuple_list:
                                    mblock.input_queue[tuple_child['dstport']] = out_value[tuple_child['srcport']]
                                    mblock.data_recieved += 1
                                    block.data_sent += 1

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
                        logger.info("All remaining uncomputed blocks form a valid loop with memory blocks. Breaking initialization loop.")
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
        #print(f"\nExecuting time step: {self.time_step}")
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
                            'params': block.params,
                            'output_only': True
                        }
                        if block.block_fn == 'Integr':
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
                                out_value = getattr(block.file_function, block.fn_name)(self.time_step, block.input_queue, block.params)
                            except:
                                print("ERROR FOUND IN EXTERNAL FUNCTION", block.file_function)
                                self.execution_failed()
                                return
                        else:
                            out_value = getattr(self.execution_function, block.fn_name)(self.time_step, block.input_queue, block.params)

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
                #print("SIMULATION TIME:", round(time.time() - self.execution_time_start, 5), 'SECONDS')

                # Export
                self.export_data()

                # Scope
                if not self.dynamic_plot:
                    print("Calling pyqtPlotScope...")
                    self.pyqtPlotScope()
                    print("pyqtPlotScope call finished.")

                # Resets the initialization of the blocks with special initial executions
                self.reset_memblocks()
                print("*****EXECUTION DONE*****")

            self.rk_counter += 1
        
        except Exception as e:
            logger.error(f"Error during execution loop: {str(e)}")
            self.execution_failed()

    def execution_failed(self, msg=""):
        """
        :purpose: If an error is found while executing the graph, this function stops all the processes and resets values to the state before execution.
        """
        self.execution_initialized = False   # Finishes the simulation execution
        self.reset_memblocks()               # Restores the initialization of the integrators (in case the error was due to vectors of different dimensions).
        self.pbar.close()                    # Finishes the progress bar
        self.error_msg = msg
        print("*****EXECUTION STOPPED*****")

    def check_diagram_integrity(self):
        print("*****Checking diagram integrity*****")
        error_trigger = False
        for block in self.blocks_list:
            inputs, outputs = self.get_neighbors(block.name)
            
            if block.in_ports == 1 and len(inputs) < block.in_ports:
                print(f"ERROR. UNLINKED INPUT IN BLOCK: {block.name}")
                error_trigger = True
            elif block.in_ports > 1:
                in_vector = np.zeros(block.in_ports)
                for tupla in inputs:
                    in_vector[tupla['dstport']] += 1
                finders = np.where(in_vector == 0)
                if len(finders[0]) > 0:
                    print(f"ERROR. UNLINKED INPUT(S) IN BLOCK: {block.name} PORT(S): {finders[0]}")
                    error_trigger = True
                    
            if block.out_ports == 1 and len(outputs) < block.out_ports:
                print(f"ERROR. UNLINKED OUTPUT PORT: {block.name}")
                error_trigger = True
            elif block.out_ports > 1:
                out_vector = np.zeros(block.out_ports)
                for tupla in outputs:
                    out_vector[tupla['srcport']] += 1
                finders = np.where(out_vector == 0)
                if len(finders[0]) > 0:
                    print(f"ERROR. UNLINKED OUTPUT(S) IN BLOCK: {block.name} PORT(S): {finders[0]}")
                    error_trigger = True
                    
        if error_trigger:
            print("Diagram integrity check failed.")
            return False
        print("NO ISSUES FOUND IN DIAGRAM")
        return True

    def count_rk45_ints(self):
        """
        :purpose: Checks all integrators and looks if there's at least one that use 'RK45' as integration method.
        """
        for block in self.blocks_list:
            if block.block_fn == 'Integr' and block.params['method'] == 'RK45':
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

    def get_neighbors(self, block_name):
        logger.debug(f"get_neighbors called for block_name: {block_name}")
        # Returns two lists of blocks, for blocks connected through lines where the current block is destination or source
        n_inputs = []
        n_outputs = []
        for line in self.line_list:
            logger.debug(f"  Checking line: {line.name}, srcblock: {line.srcblock}, dstblock: {line.dstblock}")
            if line.srcblock == block_name:
                n_outputs.append({'srcport': line.srcport, 'dstblock': line.dstblock, 'dstport': line.dstport})
            if line.dstblock == block_name:
                n_inputs.append({'dstport': line.dstport, 'srcblock': line.dstblock, 'srcport': line.srcport})
        logger.debug(f"  Found inputs: {n_inputs}, outputs: {n_outputs}")
        return n_inputs, n_outputs

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

    def reset_memblocks(self):
        """
        :purpose: Resets the "_init_start_" parameter in all blocks.
        """
        for block in self.blocks_list:
            if '_init_start_' in block.params.keys():
                block.params['_init_start_'] = True

    def plot_again(self):
        """
        :purpose: Plots the data saved in Scope blocks without needing to execute the simulation again.
        """
        if self.dirty:
            print("ERROR: The diagram has been modified. Please run the simulation again.")
            return
        try:
            scope_lengths = [len(x.params['vector']) for x in self.blocks_list if x.block_fn == 'Scope']
            if scope_lengths and scope_lengths[0] > 0:
                self.pyqtPlotScope()
            else:
                print("ERROR: NOT ENOUGH SAMPLES TO PLOT")
        except Exception as e:
            print(f"ERROR: GRAPH HAS NOT BEEN SIMULATED YET: {str(e)}")

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
            print("DATA EXPORTED TO", 'saves/' + self.filename[:-4] + '.npz')

    # Pyqtgraph functions
    def pyqtPlotScope(self):
        """
        :purpose: Plots the data saved in Scope blocks using pyqtgraph.
        :description: This function is executed while the simulation has stopped. It looks for Scope blocks, from which takes their 'vec_labels' parameter to get the labels of each vector and the 'vector' parameter containing the vector (or matrix if the input for the Scope block was a vector) and initializes a SignalPlot class object that uses pyqtgraph to show a graph.
        """
       
        print("Attempting to plot...")
        labels_list = []
        vector_list = []
        for block in self.blocks_list:
            if block.block_fn == 'Scope':
                print(f"Found Scope block: {block.name}")
                b_labels = block.params['vec_labels']
                labels_list.append(b_labels)
                b_vectors = block.params['vector']
                vector_list.append(b_vectors)
                print(f"Labels: {b_labels}")
                print(f"Vector length: {len(b_vectors)}")
                print(f"Vector type: {type(b_vectors)}")
                print(f"Vector sample: {b_vectors[:5]}")

        if labels_list and vector_list:
            print("Creating SignalPlot...")
            self.plotty = SignalPlot(self.sim_dt, labels_list, len(self.timeline))
            try:
                self.plotty.loop(new_t=self.timeline.astype(float), new_y=[np.array(v).astype(float) for v in vector_list])
                self.plotty.show()
                print("SignalPlot should be visible now.")
            except Exception as e:
                print(f"Error in plotting: {e}")
        else:
            print("No data to plot.")

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
                self.plotty = SignalPlot(self.sim_dt, labels_list, self.plot_trange)

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
                print("DYNAMIC PLOT: OFF")


class DBlock:
    def __init__(self, block_fn, sid, coords, color, in_ports=1, out_ports=1, b_type=2, io_edit=True, fn_name='block', params={}, external=False, username=''):
        logger.debug(f"Initializing DBlock {block_fn}{sid}")
        self.name = block_fn.lower() + str(sid)
        self.flipped = False
        self.block_fn = block_fn
        self.sid = sid
        self.username = self.name if username == '' else username

        self.rect = coords
        self.left = self.rect.left()
        self.top = self.rect.top()
        self.width = self.rect.width()
        self.height = self.rect.height()
        self.height_base = self.height

        self.b_color = color if isinstance(color, QColor) else QColor(color)
        self.image = QPixmap() # Initialize as null QPixmap since no icons are available


        self.params = params.copy()
        self.initial_params = params.copy()
        self.init_params_list = [key for key in params.keys() if not (key.startswith('_') and key.endswith('_'))]
        logger.debug(f"Initialized block {self.name} with params: {self.params}")

        self.fn_name = fn_name
        
        self.external = external

        self.port_radius = 8
        self.in_ports = in_ports
        self.out_ports = out_ports

        self.params.update({'_name_': self.name, '_inputs_': self.in_ports, '_outputs_': self.out_ports})
        self.rectf = QRect(self.left - self.port_radius, self.top, self.width + 2 * self.port_radius, self.height)

        self.in_coords = []
        self.out_coords = []
        self.io_edit = io_edit
        self.update_Block()

        self.b_type = b_type
        self.dragging = False
        self.selected = False

        self.font_size = 14
        self.font = QFont()
        self.font.setPointSize(self.font_size)

        self.hierarchy = -1
        self.data_recieved = 0
        self.computed_data = False
        self.data_sent = 0
        self.input_queue = {}

        # These should be set to match your DSim class attributes
        self.ls_width = 5
        self.l_width = 5
        self.rectf = QRect(self.left - self.port_radius, self.top, self.width + 2 * self.port_radius, self.height)
        logging.debug(f"Block initialized: {self.name}")





    def toggle_selection(self):
        self.selected = not self.selected

   
    def update_Block(self):
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

    def draw_Block(self, painter, draw_name=True):
        if painter is None:
            return

        painter.setBrush(self.b_color)
        border_color = theme_manager.get_color('border_primary')
        painter.setPen(QPen(border_color, 2))

        # Special case for Gain block to draw a triangle
        if self.block_fn == "Gain":
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
            # Draw rounded rectangle for all other blocks
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
        elif self.block_fn == "Noise":
            path.moveTo(0.1, 0.5)
            path.lineTo(0.15, 0.2)
            path.lineTo(0.2, 0.8)
            path.lineTo(0.25, 0.3)
            path.lineTo(0.3, 0.7)
            path.lineTo(0.35, 0.4)
            path.lineTo(0.4, 0.6)
            path.lineTo(0.45, 0.5)
            path.lineTo(0.5, 0.2)
            path.lineTo(0.55, 0.8)
            path.lineTo(0.6, 0.3)
            path.lineTo(0.65, 0.7)
            path.lineTo(0.7, 0.4)
            path.lineTo(0.75, 0.6)
            path.lineTo(0.8, 0.5)
            path.lineTo(0.85, 0.2)
            path.lineTo(0.9, 0.8)
        elif self.block_fn == "SgProd":
            path.moveTo(0.2, 0.2)
            path.lineTo(0.8, 0.8)
            path.moveTo(0.2, 0.8)
            path.lineTo(0.8, 0.2)
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
        elif self.block_fn == "Integr":
            font = painter.font()
            original_size = font.pointSize()
            font.setPointSize(original_size + 6)
            font.setItalic(True)
            painter.setFont(font)
            painter.setPen(theme_manager.get_color('text_primary'))
            painter.drawText(self.rect, Qt.AlignCenter, "1/s")
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
            sign_text = self.params.get('sign', '')
            if sign_text:
                font = painter.font()
                original_size = font.pointSize()
                font.setPointSize(original_size + 4)
                painter.setFont(font)

                painter.setPen(theme_manager.get_color('text_primary'))
                painter.drawText(self.rect, Qt.AlignCenter, sign_text)

                font.setPointSize(original_size)
                painter.setFont(font)

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
            text_rect = QRect(self.left, self.top + self.height + 5, self.width, 20)
            painter.drawText(text_rect, Qt.AlignCenter | Qt.TextWordWrap, self.name)

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

class DLine:
    def __init__(self, sid, srcblock, srcport, dstblock, dstport, points):
        self.name = "Line" + str(sid)
        self.sid = sid
        self.srcblock = srcblock
        self.srcport = srcport
        self.dstblock = dstblock
        self.dstport = dstport
        self.total_srcports = 1
        self.total_dstports = 1
        self.srcbottom = 0
        self.dstbottom = 0
        self.points = [QPoint(p.x(), p.y()) if isinstance(p, QPoint) else QPoint(p[0], p[1]) for p in points]
        self.cptr = 0  
        self.selected = False  
        self.modified = False
        self.selected_segment = -1
        self.path, self.points, self.segments = self.create_trajectory(self.points[0], self.points[1], [])
        self.color = QColor(0, 0, 0)  # Default to black

    def toggle_selection(self):
        self.selected = not self.selected
        
    def create_trajectory(self, start, finish, blocks_list, points=None):
        path = QPainterPath(start)
        
        all_points = []
        if self.modified and points and len(points) > 1:
            all_points = points
        else:
            is_feedback = start.x() > finish.x()
            
            if is_feedback:
                src_block = None
                for block in blocks_list:
                    if block.name == self.srcblock:
                        src_block = block
                        break
                
                if src_block:
                    p1 = QPoint(start.x() + 20, start.y())
                    p2 = QPoint(p1.x(), src_block.rect.bottom() + 30)
                    p3 = QPoint(finish.x() - 20, p2.y())
                    p4 = QPoint(p3.x(), finish.y())
                    all_points = [start, p1, p2, p3, p4, finish]
                else: # fallback for feedback if src_block not found
                    mid_x = int((start.x() + finish.x()) / 2)
                    all_points = [start, QPoint(mid_x, start.y()), QPoint(mid_x, finish.y()), finish]
            else:
                mid_x = int((start.x() + finish.x()) / 2)
                all_points = [start, QPoint(mid_x, start.y()), QPoint(mid_x, finish.y()), finish]

        # Clean up collinear points
        clean_points = []
        if len(all_points) > 0:
            clean_points.append(all_points[0])
            for i in range(1, len(all_points) - 1):
                p1 = all_points[i-1]
                p2 = all_points[i]
                p3 = all_points[i+1]
                if (p1.x() == p2.x() == p3.x()) or (p1.y() == p2.y() == p3.y()):
                    continue
                clean_points.append(p2)
            clean_points.append(all_points[-1])
        all_points = clean_points

        segments = []
        path = QPainterPath(all_points[0])
        for i in range(len(all_points) - 1):
            p1 = all_points[i]
            p2 = all_points[i+1]
            path.lineTo(p2)
            segments.append(QRect(p1, p2).normalized())
        
        return path, all_points, segments

    def update_line(self, blocks_list):
        logger.debug(f"Updating line {self.name}")
        if blocks_list:
            start, end = None, None
            for block in blocks_list:
                if block.name == self.srcblock:
                    start = block.out_coords[self.srcport]
                    self.total_srcports = block.out_ports
                    self.srcbottom = block.top + block.height
                if block.name == self.dstblock:
                    end = block.in_coords[self.dstport]
                    self.total_dstports = block.in_ports
                    self.dstbottom = block.top + block.height
            if start and end:
                self.points[0] = start
                self.points[-1] = end
                self.path, self.points, self.segments = self.create_trajectory(start, end, blocks_list)
                self.modified = False

    def draw_line(self, painter):
        if self.path and not self.path.isEmpty(): # self.path is a QPainterPath
            # Use theme_manager for connection colors
            default_connection_color = theme_manager.get_color('connection_default')
            active_connection_color = theme_manager.get_color('connection_active')
            
            # Draw the whole line with default color, or active color if fully selected
            pen_color = active_connection_color if self.selected and self.selected_segment == -1 else default_connection_color
            pen = QPen(pen_color, 2)
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)
            painter.drawPath(self.path)

            # If a segment is selected, draw it highlighted
            if self.selected and self.selected_segment != -1:
                if self.selected_segment < len(self.points) - 1:
                    p1 = self.points[self.selected_segment]
                    p2 = self.points[self.selected_segment + 1]
                    highlight_pen = QPen(active_connection_color, 3, Qt.SolidLine) # Thicker pen
                    painter.setPen(highlight_pen)
                    painter.drawLine(p1, p2)

            # Draw intermediate points if the whole line is selected
            if self.selected and self.selected_segment == -1:
                painter.setBrush(active_connection_color)
                for point in self.points[1:-1]:
                    painter.drawEllipse(point, 4, 4)

            # Draw arrowhead
            arrow_size = 10
            
            end_point = self.path.pointAtPercent(1.0)
            if self.path.length() > 0:
                point_before_end = self.path.pointAtPercent(1.0 - (arrow_size / self.path.length()))
            else:
                point_before_end = end_point

            # Calculate angle
            dx = end_point.x() - point_before_end.x()
            dy = end_point.y() - point_before_end.y()
            angle = math.atan2(dy, dx) # Angle in radians

            # Arrowhead points
            arrow_p1 = end_point + QPoint(int(-arrow_size * math.cos(angle - math.pi / 6)), int(-arrow_size * math.sin(angle - math.pi / 6)))
            arrow_p2 = end_point + QPoint(int(-arrow_size * math.cos(angle + math.pi / 6)), int(-arrow_size * math.sin(angle + math.pi / 6)))

            arrow_polygon = QPolygonF([end_point, arrow_p1, arrow_p2])
            
            arrow_color = active_connection_color if self.selected else default_connection_color
            painter.setBrush(arrow_color) # Fill arrowhead with line color
            painter.setPen(Qt.NoPen) # No border for arrowhead
            painter.drawPolygon(arrow_polygon)

    def collision(self, m_coords, point_radius=5, line_threshold=5):
        if isinstance(m_coords, tuple):
            m_coords = QPoint(*m_coords)

        # Check for point collision
        for i, point in enumerate(self.points):
            if (m_coords - point).manhattanLength() <= point_radius:
                return ("point", i)

        # Check for segment collision
        for i in range(len(self.points) - 1):
            p1 = self.points[i]
            p2 = self.points[i+1]
            
            # Bounding box check
            if not QRect(p1, p2).normalized().adjusted(-line_threshold, -line_threshold, line_threshold, line_threshold).contains(m_coords):
                continue

            # Distance from point to line segment
            v = p2 - p1
            u = m_coords - p1
            
            length_squared = v.x()**2 + v.y()**2
            if length_squared == 0:
                dist_sq = u.x()**2 + u.y()**2
            else:
                t = max(0, min(1, QPoint.dotProduct(u, v) / length_squared))
                projection = p1 + t * v
                dist_sq = (m_coords - projection).manhattanLength()

            if dist_sq <= line_threshold:
                return ("segment", i)

        return None

    def change_color(self, color):
        self.color = color

class MenuBlocks(DSim):
    def __init__(self, block_fn, fn_name, io_params, ex_params, b_color, coords, external=False):
        super().__init__()
        self.block_fn = block_fn
        self.fn_name = fn_name
        self.ins = io_params['inputs']
        self.outs = io_params['outputs']
        self.b_type = io_params['b_type']
        self.io_edit = io_params['io_edit']
        self.params = ex_params
        self.b_color = self.set_color(b_color)
        self.size = coords
        self.side_length = (30, 30)
        self.image = QPixmap(f'./lib/icons/{self.block_fn.lower()}.png').scaled(self.side_length[0], self.side_length[1], Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.external = external
        self.collision = None
        self.font = QFont('Arial', 10)  

    def draw_menublock(self, painter, pos):
        self.collision = QRect(40, 80 + 40*pos, self.side_length[0], self.side_length[1])
        painter.fillRect(self.collision, self.b_color)
        if not self.image.isNull():
            painter.drawPixmap(self.collision.topLeft(), self.image)
        
        painter.setFont(self.font)
        painter.setPen(self.colors['black'])
        text_rect = QRect(90, 80 + 40*pos, 100, 30)
        painter.drawText(text_rect, Qt.AlignLeft | Qt.AlignVCenter, self.fn_name)

class Button:
    def __init__(self, name, coords, active=True):
        self.name = name
        self.text = name.strip('_')  # Remove underscores for display
        self.collision = QRect(*coords) if isinstance(coords, tuple) else coords
        self.pressed = False
        self.active = active
        self.font = QFont()
        self.font.setPointSize(12)  # Adjust font size as needed
        self.collision = QRect(*coords)

    def draw_button(self, painter):
        if painter is None:
            return
        if not self.active:
            bg_color = QColor(240, 240, 240)
            text_color = QColor(128, 128, 128)
        else:
            bg_color = QColor(200, 200, 200) if self.pressed else QColor(220, 220, 220)
            text_color = QColor(0, 0, 0)

        painter.setBrush(bg_color)
        painter.setPen(Qt.NoPen)
        painter.drawRect(self.collision)

        painter.setFont(self.font)
        painter.setPen(text_color)
        painter.drawText(self.collision, Qt.AlignCenter, self.text)

        if self.active:
            painter.setPen(QPen(QColor(100, 100, 100), 1))
            painter.setBrush(Qt.NoBrush)
            painter.drawRect(self.collision)

    def contains(self, point):
        return self.collision.contains(point)

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
    def __init__(self, dt, labels, xrange):
        super().__init__()
        self.dt = dt
        self.xrange = xrange * self.dt
        self.plot_items = []
        self.curves = []

        layout = QVBoxLayout()
        self.setLayout(layout)

        for label in labels:
            plot_widget = pg.PlotWidget(title=label)
            plot_widget.showGrid(x=True, y=True)
            curve = plot_widget.plot(pen='y')
            self.plot_items.append(plot_widget)
            self.curves.append(curve)
            layout.addWidget(plot_widget)

        self.resize(800, 600)


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
            for i, curve in enumerate(self.curves):
                if i < len(new_y):
                    curve.setData(new_t, new_y[i])
        except Exception as e:
            print(f"Error updating plot: {e}")


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

