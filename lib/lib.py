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

# Import refactored classes from new modules
from lib.simulation.block import DBlock
from lib.simulation.connection import DLine
from lib.simulation.menu_block import MenuBlocks
from lib.ui.button import Button
from blocks.subsystem import Subsystem
from blocks.inport import Inport
from blocks.outport import Outport

# Import block size configuration
from config.block_sizes import get_block_size

# Import extracted classes
from lib.plotting.signal_plot import SignalPlot
from lib.plotting.scope_plotter import ScopePlotter

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
        from lib.services.run_history_service import RunHistoryService
        from lib.diagram_validator import DiagramValidator
        self.model = SimulationModel()
        self.engine = SimulationEngine(self.model)
        self.file_service = FileService(self.model)
        self.diagram_validator = DiagramValidator(self.model)

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
        self.connections_list = self.line_list  # Alias for backward/forward compatibility

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

        # Execution state (properties delegate to engine)
        # execution_initialized, execution_stop, error_msg are now properties
        self.execution_pause = self.engine.execution_pause
        self.real_time = self.engine.real_time
        self.dynamic_plot = False

        # Delegate filename to file service
        self.filename = self.file_service.filename
        self.dirty = self.model.dirty

        # Execution tracking (timeline, global_computed_list are now properties)
        self.scope_plotter = ScopePlotter(self)
        self.outs = []
        self.plotty = None
        
        # Run history service
        self.run_history_service = RunHistoryService()
        self.run_history_service.load_history()

        # Navigation Context
        # Stack of (blocks_list, line_list, parent_subsystem_name) tuples
        self.navigation_stack = []
        self.current_subsystem = None # Top level is None

    def enter_subsystem(self, subsystem_block):
        """
        Enter a subsystem block to edit its contents.
        Pushes the current context to the stack and enters the subsystem.
        """
        # Save current context (references to the lists)
        self.navigation_stack.append((self.model.blocks_list, self.model.line_list, self.current_subsystem))
        
        # Switch to subsystem context
        # We perform a swap of the lists in the model.
        # Use the subsystem's internal lists as the active model lists.
        self.model.blocks_list = subsystem_block.sub_blocks
        self.model.line_list = subsystem_block.sub_lines
        
        # Sync DSim references to match model's new active lists
        self.blocks_list = self.model.blocks_list
        self.line_list = self.model.line_list
        self.connections_list = self.line_list
        
        self.current_subsystem = subsystem_block.name
        
        # Reset selection state when changing scope
        for block in self.blocks_list:
             block.selected = False
        for line in self.line_list:
             line.selected = False
             
        # Reset view state variables
        self.ss_count = 0
        self.dirty = False # View change doesn't mean dirty, but usually we care about content
        
        logger.info(f"Entered subsystem: {subsystem_block.name}")

    def exit_subsystem(self):
        """
        Exit the current subsystem and return to the parent scope.
        """
        if not self.navigation_stack:
            logger.warning("Already at top level.")
            return

        # Restore previous context
        (prev_blocks, prev_lines, prev_subsystem) = self.navigation_stack.pop()
        
        # Restore model lists
        self.model.blocks_list = prev_blocks
        self.model.line_list = prev_lines
        
        # Sync DSim references
        self.blocks_list = self.model.blocks_list
        self.line_list = self.model.line_list
        self.connections_list = self.line_list
        
        self.current_subsystem = prev_subsystem
        
        # Reset selection
        for block in self.blocks_list:
             block.selected = False
        for line in self.line_list:
             line.selected = False
             
        logger.info("Exited subsystem, returned to parent scope")


    def get_current_path(self):
        """
        Return the current navigation path as a list of strings.
        Example: ['Top Level', 'Subsystem1', 'Nested2']
        """
        path = ['Top Level']
        for _, _, name in self.navigation_stack:
             if name:
                 path.append(name)
                 
        if self.current_subsystem:
            path.append(self.current_subsystem)
            
        return path

    def get_root_context(self):
        """
        Get the root context (blocks_list, line_list) of the simulation model.
        Used for execution to ensure we always simulate the full system.
        """
        if not self.navigation_stack:
            return self.blocks_list, self.line_list
        else:
            # The bottom of the stack (index 0) contains the references to root lists
            # Stack format: (prev_blocks, prev_lines, prev_subsystem)
            return self.navigation_stack[0][0], self.navigation_stack[0][1]

    # Properties for state shared with SimulationEngine
    @property
    def timeline(self):
        """Timeline array - shared with engine."""
        return self.engine.timeline

    @timeline.setter
    def timeline(self, value):
        self.engine.timeline = value

    @property
    def time_step(self):
        """Current time step - shared with engine."""
        return self.engine.time_step

    @time_step.setter
    def time_step(self, value):
        self.engine.time_step = value

    @property
    def global_computed_list(self):
        """Block computation tracking - shared with engine."""
        return self.engine.global_computed_list

    @global_computed_list.setter
    def global_computed_list(self, value):
        self.engine.global_computed_list = value

    @property
    def execution_initialized(self):
        """Whether simulation is initialized - shared with engine."""
        return self.engine.execution_initialized

    @execution_initialized.setter
    def execution_initialized(self, value):
        self.engine.execution_initialized = value

    @property
    def execution_stop(self):
        """Whether simulation is stopped - shared with engine."""
        return self.engine.execution_stop

    @execution_stop.setter
    def execution_stop(self, value):
        self.engine.execution_stop = value

    @property
    def error_msg(self):
        """Error message - shared with engine."""
        return self.engine.error_msg

    @error_msg.setter
    def error_msg(self, value):
        self.engine.error_msg = value

    @property
    def execution_time_start(self):
        """Execution start time - shared with engine."""
        return self.engine.execution_time_start

    @execution_time_start.setter
    def execution_time_start(self, value):
        self.engine.execution_time_start = value

    @property
    def memory_blocks(self):
        """Memory blocks set - shared with engine."""
        return self.engine.memory_blocks

    @memory_blocks.setter
    def memory_blocks(self, value):
        self.engine.memory_blocks = value

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

    def create_subsystem_from_selection(self, selected_blocks):
        """
        Create a subsystem containing the selected blocks.
        """
        if not selected_blocks:
            return

        # Calculate bounding box of selected blocks
        min_x = min(b.rect.left() for b in selected_blocks)
        min_y = min(b.rect.top() for b in selected_blocks)
        max_x = max(b.rect.right() for b in selected_blocks)
        max_y = max(b.rect.bottom() for b in selected_blocks)
        center_x = (min_x + max_x) // 2
        center_y = (min_y + max_y) // 2

        # Create Subsystem block
        # We need to construct it directly as DBlock logic is complex with MenuBlocks
        # But here we want a specific 'Subsystem' type.
        # We can use self.model.add_block if we have a template, or create manually.
        # Let's create manually for now to avoid menu dependency.
        from PyQt5.QtCore import QRect
        subsys = Subsystem()
        subsys.sid = max([b.sid for b in self.blocks_list] + [0]) + 1
        subsys.rect = QRect(center_x - subsys.width//2, center_y - subsys.height//2, subsys.width, subsys.height)
        subsys.name = f"Subsystem{subsys.sid}"
        
        # Add to current scope
        self.blocks_list.append(subsys)
        
        # Logic to move blocks and handle connections
        # 1. Identify internal lines (both ends in selection)
        # 2. Identify boundary lines
        
        internal_lines = []
        boundary_lines = [] # list of (line, type) where type is 'in' or 'out'
        
        selected_names = {b.name for b in selected_blocks}
        
        # We iterate a copy because we might modify self.line_list
        current_lines = list(self.line_list)
        
        for line in current_lines:
            src_in = line.srcblock in selected_names
            dst_in = line.dstblock in selected_names
            
            if src_in and dst_in:
                internal_lines.append(line)
            elif src_in and not dst_in:
                boundary_lines.append((line, 'out'))
            elif not src_in and dst_in:
                boundary_lines.append((line, 'in'))
                
        # Move blocks and internal lines to subsystem
        for b in selected_blocks:
            if b in self.blocks_list:
                self.blocks_list.remove(b)
            # Normalize positions relative to top-left of selection + margin
            b.rect.translate(-min_x + 100, -min_y + 100)
            subsys.sub_blocks.append(b)
            
        for l in internal_lines:
            if l in self.line_list:
                self.line_list.remove(l)
            # Update line trajectories if necessary? 
            # DLine stores points. We should translate them too.
            # But DLine uses block names. 
            # Coordinate translation is good for visual.
            new_points = []
            for p in l.points:
                 new_points.append(QPoint(p.x() - min_x + 100, p.y() - min_y + 100))
            l.points = tuple(new_points)
            subsys.sub_lines.append(l)

        # Handle Ports
        inport_idx = 1
        outport_idx = 1
        
        # For each boundary line, we need to create a port
        for line, direction in boundary_lines:
            if direction == 'in':
                # External Source -> Subsystem (Inport) -> Internal Dest
                
                # Create Inport inside subsystem
                # We place it to the left of the internal blocks
                inport = Inport(block_name=f"In{inport_idx}")
                inport.sid = max([b.sid for b in subsys.sub_blocks] + [0]) + 1
                # Position it: Find dest block position
                # But multiple lines could go to same block.
                # Just place them vertically on the left.
                inport.rect = QRect(20, 50 * inport_idx, inport.width, inport.height)
                subsys.sub_blocks.append(inport)
                
                # New internal line: Inport -> Internal Dest
                internal_line = DLine(
                    sid=max([l.sid for l in subsys.sub_lines] + [0]) + 1,
                    srcblock=inport.name, srcport=0, # Inport has 1 output (index 0)
                    dstblock=line.dstblock, dstport=line.dstport,
                    points=(inport.out_coords[0], line.points[-1]) # Simplify points
                )
                subsys.sub_lines.append(internal_line)
                
                # Update external line: External Source -> Subsystem
                # We reuse the existing line object but change destination
                # We neeed to know WHICH input port of subsystem this corresponds to.
                # Subsystem generally doesn't define ports until now.
                # We need to add dynamic ports to Subsystem block.
                
                # Add port to subsystem visual definition
                # Subsystem.ports dict needs updates.
                if 'in' not in subsys.ports: subsys.ports['in'] = []
                
                # Calculate port position on Subsystem block
                port_pos = (0, (subsys.height / (len(boundary_lines) + 1)) * inport_idx) # Rough
                subsys.ports['in'].append({'pos': port_pos, 'type': 'input', 'name': str(inport_idx)})
                
                # Map external port index to Inport name
                # Inport name is 'In{idx}'
                # External port index is len(subsys.ports['in']) - 1
                port_idx = len(subsys.ports['in']) - 1
                
                line.dstblock = subsys.name
                line.dstport = port_idx
                # Update line points to end at subsystem
                # This requires recalculating trajectory which happens in update
                
                inport_idx += 1
                
            elif direction == 'out':
                 # Internal Source -> Subsystem (Outport) -> External Dest
                 
                 # Create Outport inside
                 outport = Outport(block_name=f"Out{outport_idx}")
                 outport.sid = max([b.sid for b in subsys.sub_blocks] + [0]) + 1
                 # Place on right
                 max_internal_x = max(b.rect.right() for b in subsys.sub_blocks)
                 outport.rect = QRect(max_internal_x + 50, 50 * outport_idx, outport.width, outport.height)
                 subsys.sub_blocks.append(outport)
                 
                 # New internal line: Internal Source -> Outport
                 internal_line = DLine(
                    sid=max([l.sid for l in subsys.sub_lines] + [0]) + 1,
                    srcblock=line.srcblock, srcport=line.srcport,
                    dstblock=outport.name, dstport=0, # Outport has 1 input (index 0)
                    points=(line.points[0], outport.in_coords[0])
                 )
                 subsys.sub_lines.append(internal_line)
                 
                 # Update external line: Subsystem -> External Dest
                 if 'out' not in subsys.ports: subsys.ports['out'] = []
                 port_pos = (subsys.width, (subsys.height / (len(boundary_lines) + 1)) * outport_idx)
                 subsys.ports['out'].append({'pos': port_pos, 'type': 'output', 'name': str(outport_idx)})
                 
                 port_idx = len(subsys.ports['out']) - 1
                 
                 line.srcblock = subsys.name
                 line.srcport = port_idx
                 
                 outport_idx += 1
                 
        self.dirty = True
        logger.info(f"Created subsystem {subsys.name} with {len(subsys.sub_blocks)} blocks")

    # NOTE: display_lines, display_blocks, display_ports, update_lines moved to ModernCanvas
    # NOTE: Block loading moved to SimulationModel.load_all_blocks()

    ##### LOADING AND SAVING #####

    def save(self, autosave=False, modern_ui_data=None):
        """
        Save diagram to file. Delegates to FileService.
        
        Args:
            autosave: If True, save to autosave location without dialog
            modern_ui_data: Additional UI state data to save
        
        Returns:
            0 on success, 1 if user cancelled
        """
        # Sync parameters to file_service
        sim_params = {
            'sim_time': self.sim_time,
            'sim_dt': self.sim_dt,
            'plot_trange': self.plot_trange
        }
        self.file_service.SCREEN_WIDTH = self.SCREEN_WIDTH
        self.file_service.SCREEN_HEIGHT = self.SCREEN_HEIGHT
        
        result = self.file_service.save(
            autosave=autosave,
            modern_ui_data=modern_ui_data,
            sim_params=sim_params
        )
        
        # Sync filename back for backward compatibility
        if result == 0 and not autosave:
            self.filename = self.file_service.filename
            
        return result

    def serialize(self, modern_ui_data=None):
        """
        Serialize current diagram state to dict.
        Used by DiagramService.
        """
        sim_params = {
            'sim_time': self.sim_time,
            'sim_dt': self.sim_dt,
            'plot_trange': self.plot_trange
        }
        return self.file_service.serialize(modern_ui_data, sim_params)

    def deserialize(self, data):
        """
        Deserialize diagram state from dict.
        Used by DiagramService.
        """
        sim_params = self.file_service.apply_loaded_data(data)
        
        # Sync simulation parameters back to DSim
        self.sim_time = sim_params.get('sim_time', 1.0)
        self.sim_dt = sim_params.get('sim_dt', 0.01)
        self.plot_trange = sim_params.get('plot_trange', 100)
        self.ss_count = 0
        self.filename = self.file_service.filename
        return sim_params


    def open(self):
        """
        Load diagram from file. Delegates to FileService.
        
        Returns:
            modern_ui_data dict if present in file, None otherwise
        """
        data = self.file_service.load()
        
        if data is None:
            return None
            
        version = data.get("version", "1.0")
        if version != "2.0":
            logger.warning(f"Loading file version {version}, current is 2.0")
        
        # Apply loaded data using internal deserialize which handles syncing
        self.deserialize(data)
        
        return data.get("modern_ui_data")

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

            # Helper for recursive parameter resolution
            def resolve_recursive(blocks):
                for block in blocks:
                    # Resolve parameters using WorkspaceManager
                    block.exec_params = workspace_manager.resolve_params(block.params)
                    # Copy internal parameters that start with '_'
                    block.exec_params.update({k: v for k, v in block.params.items() if k.startswith('_')})
                    
                    # Dynamically set b_type for Transfer Functions (delegated to engine)
                    self.engine.set_block_type(block)
                    
                    block.exec_params['dtime'] = self.sim_dt
                    
                    # Reload external data
                    try:
                         if block.block_fn == 'External': # Check explicitly or reload_external_data handles it
                             missing_file_flag = block.reload_external_data()
                             if missing_file_flag == 1:
                                 logger.error(f"Missing external file for block: {block.name}")
                                 return False
                    except Exception as e:
                         logger.error(f"Error reloading external data for block {block.name}: {str(e)}")
                         return False
                         
                    # Recurse if subsystem
                    if getattr(block, 'block_type', '') == 'Subsystem':
                        if resolve_recursive(block.sub_blocks) is False:
                            return False
                return True

            # Get Root Context for execution
            root_blocks, root_lines = self.get_root_context()

            # Resolve params for ALL blocks in hierarchy
            logger.debug(f"Resolving parameters for hierarchy...")
            if not resolve_recursive(root_blocks):
                 return False

            logger.debug("Initializing execution...")

            # Initialize engine with ROOT context (will trigger flattening)
            # Pass lines explicitly!
            if not self.engine.initialize_execution(root_blocks, root_lines):
                self.execution_failed(self.engine.error_msg)
                return False
                
            # Generation of a checklist for the computation of functions (Legacy? Engine handles global_computed_list now)
            # self.global_computed_list is synced from engine via property

            self.reset_execution_data() # Delegates to engine
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

        # Identify memory blocks to correctly solve algebraic loops (delegated to engine)
        self.engine.identify_memory_blocks()

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

        # Initialize execution via engine
        if not self.engine.initialize_execution(self.blocks_list):
            self.execution_failed(self.engine.error_msg)
            return False

        # Retrieve state from engine
        self.max_hier = self.engine.max_hier
        self.rk45_len = self.engine.rk45_len
        self.rk_counter = self.engine.rk_counter
        self.execution_time_start = self.engine.execution_time_start
        self.execution_initialized = self.engine.execution_initialized

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
        # FAST SOLVER CHECK
        # Check if fast solver is enabled (default True if attr missing)
        use_fast = getattr(self, 'use_fast_solver', True)
        
        if use_fast and self.engine.check_compilability(self.blocks_list):
            logger.info("System is compilable. Using Fast Solver.")
            self.last_solver_type = "Fast (Compiled)"
            t_span = (0.0, self.execution_time)
            success = self.engine.run_compiled_simulation(
                self.blocks_list, 
                self.line_list, 
                t_span, 
                self.sim_dt
            )
            if success:
                logger.info("Fast simulation successful.")
                
                # Sync timeline from engine (Required for plotting)
                self.timeline = self.engine.timeline
                
                # Finalize execution state
                self.execution_initialized = False
                
                # Update progress bar to 100%
                if hasattr(self, 'pbar') and self.pbar:
                    self.pbar.n = self.pbar.total
                    self.pbar.last_print_n = self.pbar.total
                    self.pbar.refresh()
                    self.pbar.close()
                
                # Perform post-simulation tasks normally handled by loop
                self.export_data()
                try:
                    self._record_run_history()
                except Exception as e:
                    logger.warning(f"Failed to record run history: {e}")
                    
                return

        logger.info("System not fully compilable. Using Interpreter Mode.")
        self.last_solver_type = "Standard (Interpreter)"
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

            # Use the active list from engine (flattened if needed)
            # Fallback to local list if engine not ready (though it should be)
            current_blocks = self.engine.active_blocks_list if self.engine.active_blocks_list else self.blocks_list
            
            for block in current_blocks:
                try:
                    if block.name in self.memory_blocks:
                        # Execute memory blocks with output_only=True using engine
                        out_value = self.engine.execute_block(block, output_only=True)

                        if out_value is None or ('E' in out_value and out_value['E']):
                            self.execution_failed(out_value.get('error', 'Unknown error') if out_value else 'Block returned None')
                            return

                        # Propagate outputs to children
                        self.engine.propagate_outputs(block, out_value)

                    if self.rk45_len and self.rk_counter != 0:
                        block.params['_skip_'] = True
                except Exception as e:
                    logger.error(f"Error executing block {block.name}: {str(e)}")
                    self.execution_failed()
                    return

            # All blocks are executed according to the hierarchy order defined in the first iteration
            for hier in range(self.max_hier + 1):
                for block in current_blocks:
                    # The block must have the degree of hierarchy to execute it (and meet the other requirements above)
                    if block.hierarchy == hier and (block.data_recieved == block.in_ports or block.in_ports == 0) and not block.computed_data:
                        # Execute using engine (handles external vs internal, kwargs building)
                        out_value = self.engine.execute_block(block)
                        
                        # After execution, for memory blocks, update the 'output' state for the next step
                        if block.name in self.memory_blocks:
                            if block.block_fn == 'Integrator' and 'mem' in block.exec_params:
                                block.exec_params['output'] = block.exec_params['mem']

                        # It is checked that the function has not delivered an error
                        if out_value is None or ('E' in out_value and out_value['E']):
                            self.execution_failed(out_value.get('error', 'Unknown error') if out_value else 'Block returned None')
                            return

                        # The computed_data booleans are updated in the global list as well as in the block itself
                        self.engine.update_global_list(block.name, h_value=0)
                        block.computed_data = True

                        # Propagate outputs to children (engine handles b_type check)
                        if block.b_type not in [1, 3]:
                            self.engine.propagate_outputs(block, out_value)
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
        if hasattr(self, 'pbar'):
            self.pbar.close()                    # Finishes the progress bar
        self.error_msg = msg
        logger.error("*****EXECUTION STOPPED*****")

    def check_diagram_integrity(self):
        """Check diagram integrity. Delegates to SimulationEngine."""
        return self.engine.check_diagram_integrity()

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
        """Check if any integrators use RK45 method. Delegates to engine."""
        return self.engine.count_rk45_integrators()

    def create_subsystem_from_selection(self):
        """
        Create a subsystem from currently selected blocks.
        Moves selected blocks into a new Subsystem block and maintains connections.
        """
        selected_blocks = [b for b in self.blocks_list if b.selected]
        if not selected_blocks:
            return

        logger.info(f"Creating subsystem from {len(selected_blocks)} blocks...")
        
        # 1. Identify Blocks and Lines
        internal_block_names = set(b.name for b in selected_blocks)
        internal_lines = []
        crossing_inputs = [] # (line, src_external, dst_internal)
        crossing_outputs = [] # (line, src_internal, dst_external)
        lines_to_remove = []
        
        # Helper to find line by ID or object (since we iterate line_list)
        for line in list(self.line_list): # Copy list as we might modify
            src_in = line.srcblock in internal_block_names
            dst_in = line.dstblock in internal_block_names
            
            if src_in and dst_in:
                internal_lines.append(line)
                lines_to_remove.append(line)
            elif not src_in and dst_in:
                crossing_inputs.append(line)
                lines_to_remove.append(line)
            elif src_in and not dst_in:
                crossing_outputs.append(line)
                lines_to_remove.append(line)
        
        # 2. Create Subsystem Block
        # Calculate bounding box
        coords = [b.rect for b in selected_blocks]
        if not coords:
             logger.error("No coordinates found for selected blocks!")
             return
             
        min_x = min(r.left() for r in coords)
        min_y = min(r.top() for r in coords)
        max_x = max(r.left() + r.width() for r in coords)
        max_y = max(r.top() + r.height() for r in coords)
        
        center_x = (min_x + max_x) // 2
        center_y = (min_y + max_y) // 2
        
        logger.info(f"Subsystem Bounding Box: ({min_x}, {min_y}) - ({max_x}, {max_y}). Center: ({center_x}, {center_y})")
        
        # Find unique SID/Name for subsystem
        base_name = "Subsystem"
        existing_names = [b.name for b in self.blocks_list]
        idx = 1
        while f"{base_name}{idx}" in existing_names:
            idx += 1
        subsys_name = f"{base_name}{idx}"
        
        # Create Subsystem
        subsys = Subsystem(block_name=subsys_name, sid=idx)
        # Use QRect explicitly
        from PyQt5.QtCore import QRect
        subsys.rect = QRect(center_x - 50, center_y - 40, 100, 80) # Default size
        # Sync shadow attributes
        subsys.left = subsys.rect.left()
        subsys.top = subsys.rect.top()
        subsys.width = subsys.rect.width()
        subsys.height = subsys.rect.height()
        
        # Initialize robust port mapping
        subsys.ports_map = {'input': {}, 'output': {}}
        
        logger.info(f"Subsystem Rect assigned: {subsys.rect}")
        
        # 3. Process Inputs (Crossing In) -> Create Inports
        inport_map = {} # (src_name, src_port) -> InportBlock
        
        for line in crossing_inputs:
            key = (line.srcblock, line.srcport)
            if key not in inport_map:
                # Create Inport
                idx = len(inport_map)
                in_name = f"In{idx+1}"
                inport = Inport(block_name=in_name, sid=idx+1)
                # Position Inport inside: Left side, stacked
                inport.rect = QRect(min_x - 50, min_y + idx*60, 40, 30)
                subsys.sub_blocks.append(inport)
                inport_map[key] = inport
                
                # External connection: Src -> Subsys:Port
                port_idx = subsys.in_ports
                subsys.in_ports += 1
                
                self.add_line(line.srcblock, line.srcport, subsys.name, port_idx)
                
                # Update map
                subsys.ports_map['input'][port_idx] = inport.name
                
            inport = inport_map[key]
            
            # Create Internal Line: Inport:0 -> InternalDst
            from lib.simulation.connection import DLine
            # Use global max SID for safety
            all_sids = [l.sid for l in self.line_list] + [l.sid for l in subsys.sub_lines] + [0]
            lid = max(all_sids) + 1
            
            start_p = (inport.rect.right(), inport.rect.center().y())
            # Find dest block object to get coords
            # Careful: selected_blocks items are moved potentially, but objects persist
            dst_block = next(b for b in selected_blocks if b.name == line.dstblock)
            end_p = dst_block.in_coords[line.dstport]
            
            new_line = DLine(lid, inport.name, 0, line.dstblock, line.dstport, [start_p, end_p])
            subsys.sub_lines.append(new_line)
            
        # 4. Process Outputs (Crossing Out) -> Create Outports
        outport_map = {} # (src_name, src_port) -> OutportBlock
        
        for line in crossing_outputs:
             key = (line.srcblock, line.srcport)
             if key not in outport_map:
                 idx = len(outport_map)
                 out_name = f"Out{idx+1}"
                 outport = Outport(block_name=out_name, sid=idx+1)
                 outport.rect = QRect(max_x + 10, min_y + idx*60, 40, 30)
                 subsys.sub_blocks.append(outport)
                 outport_map[key] = {
                     'block': outport,
                     'port_idx': subsys.out_ports
                 }
                 subsys.out_ports += 1
                 
                 # Create Internal Line: InternalSrc -> Outport:0
                 from lib.simulation.connection import DLine
                 all_sids = [l.sid for l in self.line_list] + [l.sid for l in subsys.sub_lines] + [0]
                 lid = max(all_sids) + 1
                 
                 src_block = next(b for b in selected_blocks if b.name == line.srcblock)
                 start_p = src_block.out_coords[line.srcport]
                 end_p = (outport.rect.left(), outport.rect.center().y())
                 
                 new_line = DLine(lid, line.srcblock, line.srcport, outport.name, 0, [start_p, end_p])
                 subsys.sub_lines.append(new_line)
                 
             # Create External Line: Subsys:Port -> ExternalDst
             out_info = outport_map[key]
             self.add_line(subsys.name, out_info['port_idx'], line.dstblock, line.dstport)
             
             # Update map
             subsys.ports_map['output'][out_info['port_idx']] = outport.name
        
        # 4b. Process Unconnected Internal Ports
        # Use existing sets to track what is connected
        connected_inputs = set() # (block_name, port_idx)
        connected_outputs = set() # (block_name, port_idx)
        
        # Populate with existing lines (both internal and crossing)
        all_lines_involved = internal_lines + crossing_inputs + crossing_outputs
        for line in all_lines_involved:
            connected_inputs.add((line.dstblock, line.dstport))
            connected_outputs.add((line.srcblock, line.srcport))
            
        # Scan blocks for unconnected ports
        for block in selected_blocks:
            # Inputs
            for i in range(block.in_ports):
                if (block.name, i) not in connected_inputs:
                    # Found unconnected input - Create Inport
                    idx = subsys.in_ports # Current count
                    in_name = f"In{idx+1}"
                    inport = Inport(block_name=in_name, sid=max([b.sid for b in subsys.sub_blocks] + [0]) + 1)
                    inport.rect = QRect(min_x - 50, min_y + idx*60, 40, 30)
                    # Sync attributes for inport
                    inport.left = inport.rect.left()
                    inport.top = inport.rect.top()
                    inport.width = inport.rect.width()
                    inport.height = inport.rect.height()
                    
                    subsys.sub_blocks.append(inport) # Restore this!
                    subsys.in_ports += 1
                    subsys.ports_map['input'][idx] = inport.name
                    
                    # Connect Inport -> Block
                    from lib.simulation.connection import DLine
                    all_sids = [l.sid for l in self.line_list] + [l.sid for l in subsys.sub_lines] + [0]
                    lid = max(all_sids) + 1
                    
                    # Start point (Inport right), End point (Block in)
                    start_p = (inport.rect.right(), inport.rect.center().y())
                    end_p = block.in_coords[i] if i < len(block.in_coords) else (block.left, block.top + 10) # Fallback
                    
                    new_line = DLine(lid, inport.name, 0, block.name, i, [start_p, end_p])
                    subsys.sub_lines.append(new_line)
                    logger.info(f"Promoted unconnected input {block.name}:{i} to {in_name}")

            # Outputs
            for i in range(block.out_ports):
                if (block.name, i) not in connected_outputs:
                    # Found unconnected output - Create Outport
                    idx = subsys.out_ports
                    out_name = f"Out{idx+1}"
                    outport = Outport(block_name=out_name, sid=max([b.sid for b in subsys.sub_blocks] + [0]) + 1)
                    outport.rect = QRect(max_x + 10, min_y + idx*60, 40, 30)
                    # Sync attributes for outport
                    outport.left = outport.rect.left()
                    outport.top = outport.rect.top()
                    outport.width = outport.rect.width()
                    outport.height = outport.rect.height()
                    
                    subsys.sub_blocks.append(outport) # Restore this!
                    subsys.out_ports += 1
                    subsys.ports_map['output'][idx] = outport.name
                    
                    # Connect Block -> Outport
                    from lib.simulation.connection import DLine
                    all_sids = [l.sid for l in self.line_list] + [l.sid for l in subsys.sub_lines] + [0]
                    lid = max(all_sids) + 1
                    
                    start_p = block.out_coords[i] if i < len(block.out_coords) else (block.left + block.width, block.top + 10)
                    end_p = (outport.rect.left(), outport.rect.center().y())
                    
                    new_line = DLine(lid, block.name, i, outport.name, 0, [start_p, end_p])
                    subsys.sub_lines.append(new_line)
                    logger.info(f"Promoted unconnected output {block.name}:{i} to {out_name}")

        # 5. Move Internal Lines
        subsys.sub_lines.extend(internal_lines)
        
        # 6. Move Blocks
        for b in selected_blocks:
            if b in self.blocks_list:
                self.blocks_list.remove(b)
            # Deselect them
            b.selected = False
            subsys.sub_blocks.append(b)
            
        # 7. Remove Old Lines
        for l in lines_to_remove:
            if l in self.line_list:
                self.line_list.remove(l)
            if l in self.connections_list:
                self.connections_list.remove(l)
                
        # 8. Add Subsystem to scope
        self.blocks_list.append(subsys)
        subsys.selected = True # Select the new subsystem
        
        # 9. Update canvas
        self.dirty = True
        subsys.update_Block() # Recalculate ports and internal geometry
        logger.info(f"Subsystem {subsys.name} created with {len(subsys.sub_blocks)} blocks.")
        return subsys

    def update_global_list(self, block_name, h_value, h_assign=False):
        """Update the global execution list. Delegates to engine."""
        self.engine.update_global_list(block_name, h_value, h_assign)

    def check_global_list(self):
        """Check if all blocks are computed. Delegates to engine."""
        return self.engine.check_global_list()

    def count_computed_global_list(self):
        """Count computed blocks. Delegates to engine."""
        return self.engine.count_computed_global_list()

    def reset_execution_data(self):
        """Reset execution state for all blocks. Delegates to engine."""
        self.engine.reset_execution_data()

    def get_max_hierarchy(self):
        """Get max hierarchy value. Delegates to engine."""
        return self.engine.get_max_hierarchy()

    def detect_algebraic_loops(self, uncomputed_blocks):
        """Detect algebraic loops in uncomputed blocks. Delegates to engine."""
        return self.engine.detect_algebraic_loops(uncomputed_blocks)

    def get_outputs(self, block_name):
        """Get output connections for a block. Delegates to SimulationEngine."""
        return self.engine.get_outputs(block_name)



    def children_recognition(self, block_name, children_list):
        """Check if block_name is a child in the children_list. Delegates to engine."""
        is_child, ports = self.engine._children_recognition(block_name, children_list)
        # Backward compatibility: return -1 instead of [] when not a child
        return (is_child, ports if is_child else -1)

    def get_neighbors(self, block_name):
        """Get neighbors for a block. Delegates to SimulationEngine."""
        return self.engine.get_neighbors(block_name)

    def reset_memblocks(self):
        """Reset memory blocks. Delegates to engine."""
        self.engine.reset_memblocks()

    def plot_again(self):
        """
        :purpose: Plots the data saved in Scope and XYGraph blocks without needing to execute the simulation again.
        """
        self.scope_plotter.plot_again()

    def _plot_xygraph(self, block):
        """Plot XY graph data for a single XYGraph block. Delegates to ScopePlotter."""
        self.scope_plotter._plot_xygraph(block)

    def _plot_fft(self, block):
        """Plot FFT spectrum for a single FFT block. Delegates to ScopePlotter."""
        self.scope_plotter._plot_fft(block)

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
        Delegates to ScopePlotter.
        """
        return self.scope_plotter._is_discrete_upstream(block_name, visited)

    def _scope_step_modes(self):
        """
        Build a list of step-mode flags (one per Scope block) for plotting.
        Delegates to ScopePlotter.
        """
        return self.scope_plotter._scope_step_modes()

    def get_scope_traces(self):
        """
        Collect scope data as a flat list of traces for the waveform inspector.
        Delegates to ScopePlotter.
        """
        return self.scope_plotter.get_scope_traces()

    def _record_run_history(self):
        """Record the current run data for future inspection."""
        timeline, traces = self.scope_plotter.get_scope_traces()
        if timeline is None or len(timeline) == 0:
            return

        self.run_history_service.record_run(
             timeline=timeline,
             traces=traces,
             sim_dt=self.sim_dt,
             sim_time=self.sim_time
        )


    @property
    def run_history(self):
        """Delegated property for backward compatibility."""
        return self.run_history_service.history
    
    @run_history.setter
    def run_history(self, value):
        self.run_history_service.history = value

    def save_run_history(self):
        """Persist run history to disk if enabled."""
        self.run_history_service.save_history()

    def set_run_history_persist(self, enabled: bool):
        """Toggle persistence of waveform run history."""
        self.run_history_service.set_persist(enabled)

    # Pyqtgraph functions
    def pyqtPlotScope(self):
        """
        :purpose: Plots the data saved in Scope blocks using pyqtgraph.
        Delegates to ScopePlotter.
        """
        self.scope_plotter.pyqtPlotScope()

    def plot_again(self):
        """
        :purpose: Re-plots the scope data and handling other plots.
        Delegates to ScopePlotter.
        """
        self.scope_plotter.plot_again()

    def dynamic_pyqtPlotScope(self, step):
        """
        :purpose: Plots the data saved in Scope blocks dynamically with pyqtgraph.
        Delegates to ScopePlotter.
        """
        self.scope_plotter.dynamic_pyqtPlotScope(step)




