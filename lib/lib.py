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

# Import block size configuration
from config.block_sizes import get_block_size

# Import extracted classes
from lib.plotting.signal_plot import SignalPlot

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

        # Execution state (properties delegate to engine)
        # execution_initialized, execution_stop, error_msg are now properties
        self.execution_pause = self.engine.execution_pause
        self.real_time = self.engine.real_time
        self.dynamic_plot = False

        # Delegate filename to file service
        self.filename = self.file_service.filename
        self.dirty = self.model.dirty

        # Execution tracking (timeline, global_computed_list are now properties)
        self.outs = []
        self.plotty = None
        # Run history of scope data (session only)
        self.run_history = []
        self.run_history_limit = 5
        self.run_history_persist_enabled = False
        self.run_history_persist_path = Path("saves/run_history.json")
        self._load_run_history()

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
        
        # Apply loaded data using FileService
        sim_params = self.file_service.apply_loaded_data(data)
        
        # Sync simulation parameters back to DSim
        self.sim_time = sim_params.get('sim_time', 1.0)
        self.sim_dt = sim_params.get('sim_dt', 0.01)
        self.plot_trange = sim_params.get('plot_trange', 100)
        self.ss_count = 0
        self.filename = self.file_service.filename
        
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

            for block in self.blocks_list:
                # Resolve parameters using WorkspaceManager
                logger.debug(f"Block {block.name}: params before resolve = {block.params}")
                logger.debug(f"WorkspaceManager variables = {workspace_manager.variables}")
                block.exec_params = workspace_manager.resolve_params(block.params)
                logger.debug(f"Block {block.name}: exec_params after resolve = {block.exec_params}")
                # Copy internal parameters that start with '_'
                block.exec_params.update({k: v for k, v in block.params.items() if k.startswith('_')})

                # Dynamically set b_type for Transfer Functions (delegated to engine)
                self.engine.set_block_type(block)
                
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
                    # Use block_instance (all blocks now have execute() methods)
                    out_value = block.block_instance.execute(**kwargs)
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
                        # Use block_instance (all blocks now have execute() methods)
                        out_value = block.block_instance.execute(time=self.time_step, inputs=block.input_queue, params=block.exec_params)

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
                            # Use block_instance (all blocks now have execute() methods)
                            out_value = block.block_instance.execute(**kwargs)

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
                            # Use block_instance (all blocks now have execute() methods)
                            out_value = block.block_instance.execute(time=self.time_step, inputs=block.input_queue, params=block.exec_params)
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
        """Get output connections for a block. Delegates to SimulationEngine."""
        return self.engine.get_outputs(block_name)



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
        """Get neighbors for a block. Delegates to SimulationEngine."""
        return self.engine.get_neighbors(block_name)

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
            # Plot Scopes - scope stores data in exec_params during execution
            vectors = []
            for x in self.blocks_list:
                if x.block_fn == 'Scope':
                    # Check exec_params first (used during execution), fallback to params
                    params = getattr(x, 'exec_params', x.params)
                    vec = params.get('vector')
                    if vec is None:
                        vec = x.params.get('vector')
                    vectors.append(vec)
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




