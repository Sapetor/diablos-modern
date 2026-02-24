"""
SimulationEngine - Execution and analysis logic for DiaBloS.
Handles simulation initialization, execution loops, and diagram analysis.
"""

import logging
import time as time_module
from typing import List, Dict, Tuple, Any, Optional, Union
import numpy as np
from scipy import signal
from lib.simulation.block import DBlock
from lib.simulation.connection import DLine
from lib.workspace import WorkspaceManager
from lib.engine.system_compiler import SystemCompiler
from lib.engine.flattener import Flattener

logger = logging.getLogger(__name__)


class SimulationEngine:
    """
    Simulation engine that manages execution logic.
    Analyzes diagrams, detects algebraic loops, and executes simulations.

    Attributes:
        model: SimulationModel containing diagram data
        execution_initialized: Whether simulation has been initialized
        execution_pause: Whether simulation is paused
        execution_stop: Whether simulation has been stopped
        error_msg: Last error message from simulation
        sim_time: Total simulation time
        sim_dt: Simulation time step
        real_time: Whether to run simulation in real-time
        global_computed_list: Tracking list for block computation
        timeline: Time values for each simulation step
        outs: Output values from simulation
    """

    def __init__(self, model: Any) -> None:
        """
        Initialize simulation engine.

        Args:
            model: SimulationModel instance containing blocks and lines
        """
        self.model = model
        # Execution state
        self.execution_initialized: bool = False
        self.execution_pause: bool = False
        self.execution_stop: bool = False
        self.error_msg: str = ""

        # Simulation parameters
        self.sim_time: float = 1.0
        self.sim_dt: float = 0.01
        self.real_time: bool = True
        self.execution_time: float = 1.0
        self.time_step: float = 0.0

        # Execution tracking
        self.global_computed_list: List[Dict[str, Any]] = []
        self.timeline: np.ndarray = np.array([0.0])
        self.outs: List[Any] = []
        self.memory_blocks: set = set()
        self.max_hier: int = 0
        self.rk45_len: bool = False
        self.rk_counter: int = 0
        self.execution_time_start: float = 0.0
        
        # System Compiler
        self.compiler = SystemCompiler()
        self.flattener = Flattener()
        
        # Active execution lists (may differ from model if flattened)
        self.active_blocks_list = []
        self.active_line_list = []

    def initialize_execution(self, blocks_list: List[DBlock], lines_list: Optional[List[DLine]] = None) -> bool:
        """
        Initialize the execution sequence for the simulation.

        Args:
            blocks_list: List of blocks (Top Level)
            lines_list: List of lines (Top Level). Required for flattening.

        Returns:
            bool: True if block initialization successful
        """
        try:
            import time as _time
            _te0 = _time.time()
            logger.debug("Engine: Initializing execution...")
            
            # 1. Flatten Hierarchy if lines provided
            # If line_list is None, we fallback to model list, but flattening requires consistent lists.
            # DSim calls this. We assume DSim passes lines.
            
            if lines_list is None:
                lines_list = self.model.line_list

            # Check if flattening needed (if any Subsystem block exists)
            has_subsystems = any(getattr(b, 'block_type', '') == 'Subsystem' for b in blocks_list)

            if has_subsystems:
                logger.info("Flattening hierarchical system...")
                self.active_blocks_list, self.active_line_list = self.flattener.flatten(blocks_list, lines_list)
                logger.info(f"Flattening complete. Blocks: {len(self.active_blocks_list)}, Lines: {len(self.active_line_list)}")
            else:
                 self.active_blocks_list = blocks_list
                 self.active_line_list = lines_list
            logger.debug(f"[ENGINE TIMING] flattening check: {_time.time() - _te0:.3f}s")

            # Integrity Check on the Active (Flattened) System
            _te1 = _time.time()
            if not self.check_diagram_integrity():
                 self.error_msg = "Diagram integrity check failed (connections)."
                 logger.error(self.error_msg)
                 return False
            logger.debug(f"[ENGINE TIMING] integrity check: {_time.time() - _te1:.3f}s")

            # Reset temporary lists using ACTIVE list
            self.global_computed_list = [{'name': x.name, 'computed_data': x.computed_data, 'hierarchy': x.hierarchy}
                                       for x in self.active_blocks_list]
            self.reset_execution_data()
            self.execution_time_start = time_module.time()

            # Check for algebraic loops (part 1)
            check_loop = self.count_computed_global_list()

            # Identify memory blocks (on ACTIVE list)
            self.identify_memory_blocks()

            # Propagate sample times for multi-rate simulation
            self.propagate_sample_times()

            # Count RK45 integrators
            self.rk45_len = self.count_rk45_integrators()
            self.rk_counter = 0
            logger.debug(f"[ENGINE TIMING] pre-loop setup: {_time.time() - _te0:.3f}s")

            # Initial validation of signal dimensions happens in DSim or can remain there for now
            # as it relies on line_list which is in DSim/Model.
            # For this refactor, we assume DSim handles the pre-checks on lines.

            # Initialize exec_params for all blocks before execution
            from lib.workspace import WorkspaceManager
            workspace_manager = WorkspaceManager()
            for block in self.active_blocks_list:
                block.exec_params = workspace_manager.resolve_params(block.params)
                # Copy internal parameters (those starting with '_')
                block.exec_params.update({k: v for k, v in block.params.items() if k.startswith('_')})
                block.exec_params['dtime'] = self.sim_dt if hasattr(self, 'sim_dt') else 0.01

            # Loop 1: Execute Source Blocks (b_type=0) and Initialize Memory Blocks
            # Iterate active_blocks_list instead of blocks_list input
            blocks_to_exec = self.active_blocks_list
            logger.info(f"Engine: Initializing execution for {len(blocks_to_exec)} blocks (flattened)")
            _te2 = _time.time()
            
            for block in blocks_to_exec:
                logger.debug(f"Engine: Initial processing of block: {block.name}, b_type: {block.b_type}")
                children = {}
                out_value = {}

                if block.b_type == 0:
                    # Execute source block
                    _tblk = _time.time()
                    out_value = self.execute_block(block)
                    logger.debug(f"[ENGINE TIMING] execute_block({block.name}): {_time.time() - _tblk:.3f}s")
                    if out_value is False: # execute_block handles errors and returns None/False/Dict
                        return False

                    block.computed_data = True
                    block.hierarchy = 0
                    self.update_global_list(block.name, h_value=0, h_assign=True)
                    children = self.get_outputs(block.name)

                elif block.name in self.memory_blocks:
                    # Execute memory block (output_only=True)
                    _tblk = _time.time()
                    out_value = self.execute_block(block, output_only=True)
                    logger.debug(f"[ENGINE TIMING] execute_block({block.name}, memory): {_time.time() - _tblk:.3f}s")
                    if out_value is False:
                         return False

                    children = self.get_outputs(block.name)
                    block.computed_data = True
                    self.update_global_list(block.name, h_value=0, h_assign=True)

                # Check for errors in output
                if out_value and isinstance(out_value, dict) and 'E' in out_value and out_value['E']:
                    self.error_msg = out_value.get('error', 'Unknown error')
                    logger.error(self.error_msg)
                    return False

                # Propagate outputs to children
                if out_value:
                    if block.b_type not in [1, 3]: # Only propagate if valid type logic applies (memory blocks propagate manually here)
                         # Note: The original logic had custom propagation here.
                         pass

                    # We can reuse propagate_outputs but need to be careful about the specific logic used in init
                    # Original logic manually iterated children. Let's replicate or delegate.
                    _tprop = _time.time()
                    self.propagate_outputs(block, out_value)
                    _prop_time = _time.time() - _tprop
                    if _prop_time > 0.01:
                        logger.debug(f"[ENGINE TIMING] propagate_outputs({block.name}): {_prop_time:.3f}s")

            # Note: Memory blocks stay computed - they executed in Loop 1 and will receive
            # feedback via propagation. They don't need to re-execute in Loop 2.
            # Their input_queue is preserved between time steps so feedback is applied
            # at the START of the NEXT time step in Loop 1.
            logger.debug(f"[ENGINE TIMING] Loop 1 (source blocks): {_time.time() - _te2:.3f}s")

            # Loop 2: Hierarchy Resolution Matrix
            _te3 = _time.time()
            h_count = 1
            while not self.check_global_list():
                for block in blocks_to_exec:
                    # Check execution readiness - account for optional inputs
                    optional_inputs = set()
                    if hasattr(block, 'block_instance') and block.block_instance:
                        if hasattr(block.block_instance, 'optional_inputs'):
                            optional_inputs = set(block.block_instance.optional_inputs)

                    required_ports = block.in_ports - len(optional_inputs)
                    can_execute = block.data_recieved >= required_ports or block.in_ports == 0

                    logger.info(f"LOOP {h_count}: {block.name} (computed={block.computed_data}) Ready={can_execute} (Recv={block.data_recieved}/Ports={block.in_ports}, Req={required_ports})")

                    if can_execute and not block.computed_data:
                        # OUT_VALUE execute_block...
                        out_value = self.execute_block(block)
                        if out_value is False:
                            return False
                        # Check for error dict from block
                        if isinstance(out_value, dict) and out_value.get('E') or out_value.get('error'):
                            self.error_msg = out_value.get('error', 'Block returned error')
                            logger.error(f"Block {block.name} error: {self.error_msg}")
                            return False
                            
                        # Memory block special output update
                        if block.name in self.memory_blocks:
                             if block.block_fn == 'Integrator' and 'mem' in block.exec_params:
                                block.exec_params['output'] = block.exec_params['mem']
                        
                        self.update_global_list(block.name, h_value=h_count, h_assign=True)
                        block.computed_data = True

                        if block.name not in self.memory_blocks and block.b_type != 3:
                            self.propagate_outputs(block, out_value)

                        logger.info(f"EXECUTED in LOOP {h_count}: {block.name}")
                            
                # Algebraic Loop Detection
                computed_count = self.count_computed_global_list()
                if computed_count == check_loop:
                    uncomputed_blocks = [b for b in blocks_to_exec if not b.computed_data]
                    if not uncomputed_blocks:
                        break
                        
                    is_algebraic, cycle_nodes = self.detect_algebraic_loops(uncomputed_blocks)
                    if is_algebraic:
                        self.error_msg = f"Algebraic loop detected involving blocks: {cycle_nodes}"
                        logger.error(self.error_msg)
                        return False
                    else:
                        break
                else:
                    check_loop = computed_count
                
                h_count += 1
            logger.debug(f"[ENGINE TIMING] Loop 2 (hierarchy): {_time.time() - _te3:.3f}s")

            # Sync hierarchies back to blocks
            self.reset_execution_data()

            # Calculate max hierarchy
            self.max_hier = self.get_max_hierarchy()

            logger.debug(f"Engine: Execution initialized. Max hierarchy: {self.max_hier}")
            logger.debug(f"[ENGINE TIMING] initialize_execution TOTAL: {_time.time() - _te0:.3f}s")
            self.execution_initialized = True
            return True

        except Exception as e:
            import traceback
            logger.error(f"Engine: Error during execution init: {e}")
            logger.error(traceback.format_exc())
            self.error_msg = str(e)
            return False

    def update_global_list(self, block_name: str, h_value: int = 0, h_assign: bool = False, reset_computed: bool = False) -> None:
        """Update global computed list."""
        for g_block in self.global_computed_list:
            if g_block['name'] == block_name:
                if reset_computed:
                    g_block['computed_data'] = False
                else:
                    g_block['computed_data'] = True
                
                if h_assign:
                    g_block['hierarchy'] = h_value
                break

    def execute_block(self, block: DBlock, output_only: bool = False) -> Union[Dict[int, Any], bool]:
        """
        Execute a single block.
        Returns output value (dict) or False on failure.
        """
        try:
            logger.info(f"ENGINE EXECUTE: {block.name} (b_type={block.b_type})")
            kwargs = {
                'time': self.time_step,
                'inputs': block.input_queue,
                'params': block.exec_params
            }
            
            if output_only:
                kwargs['output_only'] = True
                if block.block_fn == 'Integrator':
                    kwargs['next_add_in_memory'] = False
                    kwargs['dtime'] = self.sim_dt
            
            if block.external:
                try:
                    out_value = getattr(block.file_function, block.fn_name)(**kwargs)
                except Exception as e:
                    logger.error(f"ERROR IN EXTERNAL FUNCTION {block.file_function}: {e}")
                    return False
            else:
                if block.block_instance is None:
                    # Logic for blocks without instance (e.g. Subsystem if not flattened correctly)
                    # If it's a Subsystem, we shouldn't be here unless flattening failed.
                    b_type_logs = getattr(block, 'block_type', 'Unknown')
                    logger.error(f"Block {block.name} (type={b_type_logs}) has no block_instance. Skipping execution.")
                    return False
                    
                out_value = block.block_instance.execute(**kwargs)
                
            if out_value is None:
                 logger.error(f"Block {block.name} returned None")
                 return False
                 
            if isinstance(out_value, dict) and 'E' in out_value and out_value['E']:
                return out_value # Caller checks for error
                
            return out_value
            
        except Exception as e:
            logger.error(f"Error executing block {block.name}: {e}")
            return False

    def check_diagram_integrity(self):
        """
        Verify that all block ports are properly connected.

        Returns:
            bool: True if diagram is valid, False otherwise
        """
        logger.debug("Checking diagram integrity")
        error_trigger = False

        # Use active lists (fallback to model if not initialized, but ideally active)
        blocks_to_check = self.active_blocks_list if self.active_blocks_list else self.model.blocks_list
        # get_neighbors already uses active_line_list

        for block in blocks_to_check:
            inputs, outputs = self.get_neighbors(block.name)

            # Get optional inputs from block instance (if available)
            optional_inputs = set()
            if hasattr(block, 'block_instance') and block.block_instance:
                if hasattr(block.block_instance, 'optional_inputs'):
                    optional_inputs = set(block.block_instance.optional_inputs)

            # Get optional outputs from block instance (if available)
            optional_outputs = set()
            if hasattr(block, 'block_instance') and block.block_instance:
                if hasattr(block.block_instance, 'optional_outputs'):
                    optional_outputs = set(block.block_instance.optional_outputs)
                # Also check requires_outputs property
                if hasattr(block.block_instance, 'requires_outputs'):
                    if not block.block_instance.requires_outputs:
                        optional_outputs = set(range(block.out_ports))

            # Check input ports
            required_in_ports = block.in_ports - len(optional_inputs)
            connected_required_inputs = sum(1 for t in inputs if t['dstport'] not in optional_inputs)

            if required_in_ports == 1 and connected_required_inputs < 1:
                logger.error(f"ERROR. UNLINKED INPUT IN BLOCK: {block.name}")
                error_trigger = True
            elif required_in_ports > 1 or (block.in_ports > 1 and required_in_ports > 0):
                in_vector = np.zeros(block.in_ports)
                for tupla in inputs:
                    in_vector[tupla['dstport']] += 1
                # Find unlinked ports that are NOT optional
                unlinked = [i for i in range(block.in_ports) if in_vector[i] == 0 and i not in optional_inputs]
                if len(unlinked) > 0:
                    logger.error(f"ERROR. UNLINKED INPUT(S) IN BLOCK: {block.name} PORT(S): {unlinked}")
                    error_trigger = True

            # Check output ports
            required_out_ports = block.out_ports - len(optional_outputs)
            connected_required_outputs = sum(1 for t in outputs if t['srcport'] not in optional_outputs)

            if required_out_ports == 1 and connected_required_outputs < 1:
                logger.error(f"ERROR. UNLINKED OUTPUT PORT: {block.name}")
                error_trigger = True
            elif required_out_ports > 1 or (block.out_ports > 1 and required_out_ports > 0):
                out_vector = np.zeros(block.out_ports)
                for tupla in outputs:
                    out_vector[tupla['srcport']] += 1
                # Find unlinked ports that are NOT optional
                unlinked = [i for i in range(block.out_ports) if out_vector[i] == 0 and i not in optional_outputs]
                if len(unlinked) > 0:
                    logger.error(f"ERROR. UNLINKED OUTPUT(S) IN BLOCK: {block.name} PORT(S): {unlinked}")
                    error_trigger = True

        if error_trigger:
            logger.error("Diagram integrity check failed.")
            return False
        logger.debug("NO ISSUES FOUND IN DIAGRAM")
        return True

    def get_neighbors(self, block_name):
        """
        Get all input and output connections for a block.

        Args:
            block_name: Name of the block

        Returns:
            tuple: (inputs, outputs) where each is a list of connection dicts
        """
        inputs = []
        outputs = []

        # Use active_line_list if active_blocks_list is populated (implies execution setup)
        # Otherwise fallback to model list logic
        use_active = len(self.active_blocks_list) > 0
        line_source = self.active_line_list if use_active else self.model.line_list

        for line in line_source:
            if line.dstblock == block_name:
                inputs.append({
                    'srcblock': line.srcblock,
                    'srcport': line.srcport,
                    'dstport': line.dstport
                })
            if line.srcblock == block_name:
                outputs.append({
                    'dstblock': line.dstblock,
                    'srcport': line.srcport,
                    'dstport': line.dstport
                })

        return inputs, outputs

    def get_outputs(self, block_name):
        """
        Get all output connections for a block.

        Args:
            block_name: Name of the block

        Returns:
            list: Output connections
        """
        outputs = []
        # Use active_line_list if active_blocks_list is populated
        use_active = len(self.active_blocks_list) > 0
        line_source = self.active_line_list if use_active else self.model.line_list
        
        for line in line_source:
            if line.srcblock == block_name:
                outputs.append({
                    'dstblock': line.dstblock,
                    'srcport': line.srcport,
                    'dstport': line.dstport
                })
        return outputs

    def get_max_hierarchy(self):
        """
        Find the maximum hierarchy level in the diagram.

        Returns:
            int: Maximum hierarchy value
        """
        max_h = -1
        # Use active_blocks_list
        for block in self.active_blocks_list:
            if block.hierarchy > max_h:
                max_h = block.hierarchy
        return max_h

    # detect_algebraic_loops doesn't access lists directly, uses get_outputs

    # children_recognition uses get_outputs

    def reset_execution_data(self) -> None:
        """Reset execution state for all blocks.

        IMPORTANT: Must update global_computed_list AND restore hierarchy from it.
        Memory blocks preserve their input_queue so feedback from previous step can be used.
        """
        # Safety check - if global_computed_list isn't populated yet, use simple reset
        if not self.global_computed_list or len(self.global_computed_list) != len(self.active_blocks_list):
            for block in self.active_blocks_list:
                block.computed_data = False
                block.data_recieved = 0
                block.data_sent = 0
                block.hierarchy = -1
                # Preserve input_queue for memory blocks (they need feedback from previous step)
                if block.name not in self.memory_blocks:
                    block.input_queue = {}
            return

        for i in range(len(self.active_blocks_list)):
            block = self.active_blocks_list[i]
            self.global_computed_list[i]['computed_data'] = False
            block.computed_data = False
            block.data_recieved = 0
            block.data_sent = 0
            # Preserve input_queue for memory blocks (they need feedback from previous step)
            if block.name not in self.memory_blocks:
                block.input_queue = {}
            block.hierarchy = self.global_computed_list[i]['hierarchy']
    
    # Duplicate definition of count_rk45_integrators here? 
    # Yes, previous edit might have left one. 
    # Let's fix count_rk45_integrators to use active list too.
    
    def count_rk45_integrators(self):
        """
        Check if any integrators use RK45 method.

        Returns:
            bool: True if RK45 integrators exist
        """
        for block in self.active_blocks_list:
            if block.block_fn == 'Integrator' and block.params.get('method') == 'RK45':
                return True
            elif block.block_fn == 'External' and block.params.get('method') == 'RK45':
                return True
        return False

    def reset_memblocks(self) -> None:
        """Reset memory blocks (integrators, transfer functions, etc.).

        Resets _init_start_ in both params and exec_params, and clears _prev state.
        """
        for block in self.active_blocks_list:
            if '_init_start_' in block.params:
                block.params['_init_start_'] = True
            # Also reset in exec_params if it exists (used during execution)
            if hasattr(block, 'exec_params') and block.exec_params:
                if '_init_start_' in block.exec_params:
                    block.exec_params['_init_start_'] = True
                # Clear any stored state like _prev
                if '_prev' in block.exec_params:
                    del block.exec_params['_prev']

    def detect_algebraic_loops(self, uncomputed_blocks):
        """
        Detect if there are algebraic loops in uncomputed blocks using 
        topological sort (Kahn's algorithm).

        Args:
            uncomputed_blocks: List of blocks that haven't been computed

        Returns:
            tuple: (is_algebraic: bool, cycle_nodes: list) - True if loop detected,
                   with list of block names involved in the cycle
        """
        from collections import deque
        
        if len(uncomputed_blocks) == 0:
            return False, []

        logger.debug("Checking for algebraic loops...")
        logger.debug(f"Uncomputed blocks: {[b.name for b in uncomputed_blocks]}")

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
            cycle_nodes = [name for name, degree in in_degree.items() if degree > 0]
            
            # Check if the cycle contains any memory blocks
            has_memory_block = any(node in self.memory_blocks for node in cycle_nodes)
            
            if not has_memory_block:
                logger.error("ALGEBRAIC LOOP DETECTED")
                logger.error(f"Blocks involved: {cycle_nodes}")
                return True, cycle_nodes

        return False, []

    def children_recognition(self, block_name, children_list):
        """
        Recursively find all children (downstream blocks) of a block.

        Args:
            block_name: Name of the parent block
            children_list: List to accumulate children

        Returns:
            list: Updated children list
        """
        outputs = self.get_outputs(block_name)

        for output in outputs:
            child_name = output['dstblock']
            if child_name not in children_list:
                children_list.append(child_name)
                self.children_recognition(child_name, children_list)

        return children_list

    def update_sim_params(self, sim_time: float, sim_dt: float) -> None:
        """
        Update simulation parameters.

        Args:
            sim_time: Total simulation time
            sim_dt: Time step
        """
        self.sim_time = sim_time
        self.sim_dt = sim_dt

    def get_execution_status(self):
        """
        Get current execution status.

        Returns:
            dict: Status information
        """
        return {
            'initialized': self.execution_initialized,
            'paused': self.execution_pause,
            'stopped': self.execution_stop,
            'error': self.error_msg if self.error_msg else None,
            'sim_time': self.sim_time,
            'sim_dt': self.sim_dt
        }

    # =========================================================================
    # Core Execution Methods - Migrated from DSim
    # =========================================================================

    def prepare_execution(self, execution_time: float) -> bool:
        """
        Prepare the simulation for execution by resolving parameters and 
        identifying memory blocks.

        Args:
            execution_time: Total simulation time in seconds

        Returns:
            bool: True if preparation successful, False otherwise
        """
        logger.debug("*****INIT NEW EXECUTION*****")
        
        self.execution_stop = False
        self.error_msg = ""
        self.time_step = 0
        self.timeline = np.array([self.time_step])
        self.execution_time = execution_time

        workspace_manager = WorkspaceManager()

        for block in self.model.blocks_list:
            # Resolve parameters using WorkspaceManager
            logger.debug(f"Block {block.name}: params before resolve = {block.params}")
            block.exec_params = workspace_manager.resolve_params(block.params)
            logger.debug(f"Block {block.name}: exec_params after resolve = {block.exec_params}")
            
            # Copy internal parameters that start with '_'
            block.exec_params.update({k: v for k, v in block.params.items() if k.startswith('_')})

            # Dynamically set b_type for Transfer Functions
            self.set_block_type(block)
            
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

        # Initialize global computed list
        self.global_computed_list = [
            {'name': x.name, 'computed_data': x.computed_data, 'hierarchy': x.hierarchy}
            for x in self.model.blocks_list
        ]
        self.reset_execution_data()
        self.execution_time_start = time_module.time()
        
        # Identify memory blocks
        self.identify_memory_blocks()
        
        # Check for RK45 integrators
        self.rk45_len = self.count_rk45_integrators()
        self.rk_counter = 0

        # Auto-connect Goto/From tags
        try:
            self.model.link_goto_from()
        except Exception as e:
            logger.warning(f"Goto/From linking failed: {e}")

        logger.debug("Execution preparation complete")
        return True

    def set_block_type(self, block: DBlock) -> None:
        """Set block type based on transfer function properness."""
        if block.block_fn == 'TranFn':
            num = block.exec_params.get('numerator', [])
            den = block.exec_params.get('denominator', [])
            block.b_type = 1 if len(den) > len(num) else 2
        elif block.block_fn == 'DiscreteTranFn':
            num = block.exec_params.get('numerator', [])
            den = block.exec_params.get('denominator', [])
            block.b_type = 1 if len(den) > len(num) else 2
        elif block.block_fn == 'DiscreteStateSpace':
            D = np.array(block.exec_params.get('D', [[0.0]]))
            block.b_type = 1 if np.all(D == 0) else 2

    def identify_memory_blocks(self) -> None:
        """Identify blocks with memory (integrators, strictly proper TFs, state variables)."""
        self.memory_blocks = set()
        for block in self.active_blocks_list:
            # Check if block has requires_inputs=False property (can output without inputs)
            block_class = getattr(block, 'block_class', None)
            if block_class:
                try:
                    instance = block_class()
                    if hasattr(instance, 'requires_inputs') and not instance.requires_inputs:
                        self.memory_blocks.add(block.name)
                        continue
                except Exception:
                    pass

            # Check for known memory block types by name
            if block.block_fn in ('Integrator', 'StateVariable', 'TransportDelay', 'Delay'):
                self.memory_blocks.add(block.name)
            elif block.block_fn == 'TranFn':
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

    def propagate_sample_times(self) -> None:
        """
        Propagate sample times through the diagram.

        Resolves effective sample times for all blocks based on:
        - Explicit sample_time parameter (>0 = fixed discrete rate)
        - Inherited rate (0 = inherit from fastest connected input)
        - Continuous (-1 = execute every timestep, default)

        Must be called after identify_memory_blocks() during initialization.
        """
        logger.debug("Propagating sample times...")

        # Build connection map for efficient lookup
        # Maps block_name -> list of source block names
        input_sources: Dict[str, List[str]] = {b.name: [] for b in self.active_blocks_list}
        for line in self.active_line_list:
            if line.dstblock in input_sources:
                input_sources[line.dstblock].append(line.srcblock)

        # Create block lookup
        block_map = {b.name: b for b in self.active_blocks_list}

        # Phase 1: Resolve explicit sample times from parameters
        for block in self.active_blocks_list:
            declared_rate = block.resolve_sample_time()
            block.effective_sample_time = declared_rate
            # Reset execution state for new simulation
            block.reset_sample_time_state()

        # Phase 2: Propagate inherited rates (sample_time = 0)
        # Use iterative propagation until no changes occur
        max_iterations = len(self.active_blocks_list) + 1
        for iteration in range(max_iterations):
            changed = False

            for block in self.active_blocks_list:
                # Only process blocks that inherit (sample_time = 0)
                if block.effective_sample_time != 0.0:
                    continue

                # Find fastest (smallest positive) sample time from inputs
                fastest_rate = -1.0  # Default to continuous if no discrete inputs
                for src_name in input_sources.get(block.name, []):
                    src_block = block_map.get(src_name)
                    if src_block and src_block.effective_sample_time > 0:
                        if fastest_rate < 0 or src_block.effective_sample_time < fastest_rate:
                            fastest_rate = src_block.effective_sample_time

                # Apply inherited rate
                if fastest_rate != block.effective_sample_time:
                    block.effective_sample_time = fastest_rate
                    changed = True

            if not changed:
                break

        # Phase 3: Mark connections as discrete based on source block sample time
        for line in self.active_line_list:
            src_block = block_map.get(line.srcblock)
            if src_block and src_block.effective_sample_time > 0:
                line.discrete_signal = True
            else:
                line.discrete_signal = False

        # Log resolved sample times
        discrete_blocks = [(b.name, b.effective_sample_time)
                          for b in self.active_blocks_list if b.effective_sample_time > 0]
        if discrete_blocks:
            logger.info(f"DISCRETE BLOCKS: {discrete_blocks}")
        discrete_lines = sum(1 for line in self.active_line_list if getattr(line, 'discrete_signal', False))
        if discrete_lines:
            logger.info(f"DISCRETE CONNECTIONS: {discrete_lines}")
        logger.debug("Sample time propagation complete")

    def propagate_outputs(self, block: DBlock, out_value: Dict[int, Any]) -> None:
        """
        Propagate block outputs to connected downstream blocks.

        Args:
            block: Source block
            out_value: Output values from the block
        """
        children = self.get_outputs(block.name)
        # Use active blocks if execution initialized (flattened copies), otherwise model (fallback)
        target_blocks = self.active_blocks_list if len(self.active_blocks_list) > 0 else self.model.blocks_list
        
        logger.info(f"ENGINE PROPAGATE: {block.name} -> {[c['dstblock'] for c in children]}")
        
        for mblock in target_blocks:
            is_child, tuple_list = self._children_recognition(mblock.name, children)
            if is_child:
                for tuple_child in tuple_list:
                    mblock.data_recieved += 1
                    mblock.input_queue[tuple_child['dstport']] = out_value[tuple_child['srcport']]
                    block.data_sent += 1

    def _children_recognition(self, block_name: str, children_list: List[Dict]) -> Tuple[bool, List[Dict]]:
        """
        Check if block_name is in the children list.

        Returns:
            Tuple of (is_child, matching_connections)
        """
        child_ports = []
        for child in children_list:
            if block_name in child.values():
                child_ports.append(child)
        if not child_ports:
            return False, []
        return True, child_ports



    def check_global_list(self) -> bool:
        """Check if all blocks have been computed."""
        return all(elem['computed_data'] for elem in self.global_computed_list)

    def count_computed_global_list(self) -> int:
        """Count the number of computed blocks."""
        return sum(1 for x in self.global_computed_list if x['computed_data'])

    def execution_failed(self, msg: str = "") -> None:
        """
        Handle execution failure.

        Args:
            msg: Error message
        """
        self.execution_initialized = False
        self.reset_memblocks()
        self.error_msg = msg
        logger.error("*****EXECUTION STOPPED*****")

    def check_compilability(self, blocks: List[DBlock]) -> bool:
        """Check if the system can be compiled."""
        return self.compiler.check_compilability(blocks)

    def run_compiled_simulation(self, blocks: List[DBlock], lines: List[Any], t_span: Tuple[float, float], dt: float) -> bool:
        """
        Run the simulation using the compiled fast solver.
        """
        try:
            from scipy.integrate import solve_ivp
            from lib.workspace import WorkspaceManager

            # Check if already initialized (by DSim.execution_init)
            # Skip redundant initialization to avoid 2x overhead
            if len(self.active_blocks_list) == 0:
                # Not yet initialized - do it now
                if not self.initialize_execution(blocks, lines):
                    logger.error("Failed to initialize execution (algebraic loop or error).")
                    return False
            else:
                logger.debug("Engine already initialized, skipping redundant initialization")

            # Use the FLATTENED lists for checking and compilation
            current_blocks = self.active_blocks_list
            current_lines = self.active_line_list if self.active_line_list else lines

            # CRITICAL: Resolve parameters before compilation
            # This populates exec_params with resolved values (workspace variables, etc.)
            # Without this, PDE blocks get default/zero initial conditions
            workspace_manager = WorkspaceManager()
            for block in current_blocks:
                block.exec_params = workspace_manager.resolve_params(block.params)
                # Copy internal parameters (those starting with '_')
                block.exec_params.update({k: v for k, v in block.params.items() if k.startswith('_')})
                block.exec_params['dtime'] = dt

            # Final check on flattened system
            if not self.compiler.check_compilability(current_blocks):
                logger.error("Flattened system contains uncompilable blocks.")
                return False

            # Topological sort via hierarchy
            sorted_blocks = sorted(current_blocks, key=lambda b: b.hierarchy)

            logger.info("Compiling system...")
            model_func, y0, state_map, block_matrices = self.compiler.compile_system(current_blocks, sorted_blocks, current_lines)
            
            logger.info(f"Solving IVP over {t_span} with {len(y0)} states...")
            t_eval = np.arange(t_span[0], t_span[1] + dt, dt)
            # Clip to avoid floating-point overshoot past t_span[1]
            t_eval = t_eval[t_eval <= t_span[1] + 1e-12]
            t_eval[-1] = min(t_eval[-1], t_span[1])
            
            if len(y0) == 0:
                # Purely algebraic system
                # Create a dummy solution object for compatibility
                class MockSol:
                    pass
                sol = MockSol()
                sol.t = t_eval
                sol.y = np.zeros((0, len(t_eval)))
                sol.success = True
                sol.message = "Algebraic system computed successfully"
                logger.info("System is algebraic (0 states). Skipping solver.")
            else:
                # Using RK45 by default with strict tolerances for long-duration stability
                sol = solve_ivp(model_func, t_span, y0, t_eval=t_eval, method='RK45', rtol=1e-9, atol=1e-12)
            
            if not sol.success:
                logger.error(f"Solver failed: {sol.message}")
                return False
                
            logger.info("Simulation finished. Processing results...")
            
            # 4. Distribute results
            self.outs = sol.y
            self.timeline = sol.t
            
            if sol.y.size > 0:
                logger.info(f"Solver output range: min={np.min(sol.y):.6f}, max={np.max(sol.y):.6f}")

            # For Scope visualization, we need to populate block outputs.
            # We "replay" the simulation using the solution to capture all signals.
            num_steps = len(sol.t)
            
            # Replay Sort: Topological sort respecting Direct Feedthrough
            # 1. Build Graph
            replay_order = []
            in_degree = {b.name: 0 for b in current_blocks}
            adj = {b.name: [] for b in current_blocks}
            
            # 2. Add edges only for Direct Feedthrough connections
            for line in current_lines:
                src = line.srcblock
                dst = line.dstblock
                
                # Check Direct Feedthrough
                is_feedthrough = True
                dst_block = next((b for b in current_blocks if b.name == dst), None)
                if dst_block:
                    fn = dst_block.block_fn.title() if dst_block.block_fn else ''
                    if fn == 'Statespace': fn = 'StateSpace'
                    if fn in ('Transferfcn', 'Tranfn'): fn = 'TransferFcn'
                    
                    if fn == 'Integrator':
                        is_feedthrough = False
                    elif fn == 'TransferFcn':
                        # Check strictly proper (num < den)
                        num = dst_block.params.get('numerator', [])
                        den = dst_block.params.get('denominator', [])
                        if len(den) > len(num):
                            is_feedthrough = False
                    elif fn == 'StateSpace':
                         # Check D=0
                         D = np.array(dst_block.params.get('D', [[0.0]]))
                         if np.all(D == 0):
                             is_feedthrough = False
                
                if is_feedthrough:
                    if src in adj:
                        adj[src].append(dst)
                        in_degree[dst] += 1
            
            # 3. Kahn's Algorithm
            queue = [b for b in current_blocks if in_degree[b.name] == 0]
            # stable sort for determinism (e.g. step before sine if independent)
            queue.sort(key=lambda b: b.name) 
            
            while queue:
                u = queue.pop(0)
                replay_order.append(u)
                
                if u.name in adj:
                    for v_name in adj[u.name]:
                        in_degree[v_name] -= 1
                        if in_degree[v_name] == 0:
                            v_block = next((b for b in current_blocks if b.name == v_name), None)
                            if v_block:
                                queue.append(v_block)
            
            # If cycles remain (algebraic loops), append leftovers (best effort)
            if len(replay_order) < len(current_blocks):
                leftovers = [b for b in current_blocks if b not in replay_order]
                # logger.warning(f"Replay Sort: Algebraic loop detected or sort failed. Leftovers: {[b.name for b in leftovers]}")
                replay_order.extend(leftovers)
                
            sorted_blocks = replay_order
            
            # Replay Loop
            for i in range(num_steps):
                t = sol.t[i]
                y_step = sol.y[:, i] if sol.y.ndim > 1 else sol.y
                
                # 1. State Map - Populate 'current_states' first
                # Output 'signals' populate diffently based on block type.
                current_signals = {}
                current_states = {} # b_name -> x
                
                for b_name, (start, size) in state_map.items():
                     x_val = y_step[start : start + size]
                     current_states[b_name] = x_val
                     
                     # For Integrator, y = x
                     # For SS/TF, y != x. We calculate y later.
                     # We can pe-fill generic "Integrator" assumption if we verify type?
                     # No, let's rely on block loop.
                     
                # 2. Block Logic Replay
                # Execute blocks in topological order
                for block in sorted_blocks:
                    b_name = block.name
                    # Normalize function name (matches SystemCompiler logic)
                    fn = block.block_fn.title() if block.block_fn else ''
                    if fn == 'Statespace': fn = 'StateSpace'
                    if fn in ('Transferfcn', 'Tranfn'): fn = 'TransferFcn'
                    if block.block_fn == 'PID': fn = 'PID'
                    if fn == 'Ratelimiter': fn = 'RateLimiter'
                    
                    # Collect inputs
                    inputs = {}
                    for line in current_lines:
                        if line.dstblock == b_name:
                            # Direct feedthrough lookup - handle multi-output blocks
                            src_port = getattr(line, 'srcport', 0) or 0
                            if src_port == 0:
                                val = current_signals.get(line.srcblock, 0.0)
                            else:
                                # Secondary output - use suffix naming convention
                                # Check for common suffixes used by multi-output blocks
                                src_key = f"{line.srcblock}_out{src_port}"
                                val = current_signals.get(src_key, current_signals.get(line.srcblock, 0.0))
                            inputs[line.dstport] = val
                    
                    out_val = 0.0
                    
                    if fn == 'Integrator':
                        # Valid because Integrator state output is just the state
                        if b_name in current_states:
                             val = current_states[b_name]
                             out_val = val if val.size > 1 else val.item()
                        
                    elif fn in ('StateSpace', 'TransferFcn'):
                        if b_name in block_matrices and b_name in current_states:
                             A, B, C, D = block_matrices[b_name]
                             x = current_states[b_name].reshape(-1, 1)
                             
                             u_val = inputs.get(0, 0.0)
                             u = np.atleast_1d(u_val).reshape(-1, 1)
                             
                             if B.shape[1] > 0 and u.shape[0] != B.shape[1]: 
                                 if B.shape[1] == 1:
                                     u = np.array([[float(u_val)]])
                                 else:
                                     u = np.full((B.shape[1], 1), float(u_val))
                             elif B.shape[1] == 0:
                                 u = np.array([[]])

                             y_out = C @ x + D @ u
                             out_val = y_out if y_out.size > 1 else y_out.item()

                    # ==================== PDE BLOCKS ====================
                    elif fn == 'Heatequation1D':
                        # HeatEquation1D: output is the temperature field (state vector)
                        if b_name in current_states:
                            T = current_states[b_name]
                            out_val = T
                            current_signals[b_name + '_out1'] = float(np.mean(T))  # T_avg
                        else:
                            out_val = np.zeros(int(block.params.get('N', 20)))

                    elif fn == 'Waveequation1D':
                        # WaveEquation1D: state is [u, v], output primary is u (displacement)
                        N = int(block.params.get('N', 50))
                        if b_name in current_states:
                            state = current_states[b_name]
                            u = state[:N]   # Displacement field
                            v = state[N:]   # Velocity field
                            out_val = u
                            current_signals[b_name + '_out1'] = v  # v_field
                            # Energy = 0.5 * (kinetic + potential)
                            L = float(block.params.get('L', 1.0))
                            dx = L / (N - 1)
                            energy = 0.5 * np.sum(v**2) * dx  # Simplified
                            current_signals[b_name + '_out2'] = float(energy)
                        else:
                            out_val = np.zeros(N)

                    elif fn == 'Advectionequation1D':
                        # AdvectionEquation1D: output is concentration field
                        if b_name in current_states:
                            c = current_states[b_name]
                            out_val = c
                            L = float(block.params.get('L', 1.0))
                            N = len(c)
                            dx = L / (N - 1) if N > 1 else 1.0
                            current_signals[b_name + '_out1'] = float(np.sum(c) * dx)  # c_total
                        else:
                            out_val = np.zeros(int(block.params.get('N', 50)))

                    elif fn == 'Diffusionreaction1D':
                        # DiffusionReaction1D: output is concentration field
                        if b_name in current_states:
                            c = current_states[b_name]
                            out_val = c
                            L = float(block.params.get('L', 1.0))
                            N = len(c)
                            dx = L / (N - 1) if N > 1 else 1.0
                            current_signals[b_name + '_out1'] = float(np.sum(c) * dx)  # c_total
                            k = float(block.params.get('k', 0.1))
                            n_order = int(block.params.get('n', 1))
                            reaction = np.sum(k * np.power(np.maximum(c, 0), n_order)) * dx
                            current_signals[b_name + '_out2'] = float(reaction)  # reaction_rate
                        else:
                            out_val = np.zeros(int(block.params.get('N', 50)))

                    # ==================== 2D PDE BLOCKS ====================
                    elif fn == 'Heatequation2D':
                        # HeatEquation2D: output is 2D temperature field
                        Nx = int(block.params.get('Nx', 20))
                        Ny = int(block.params.get('Ny', 20))
                        if b_name in current_states:
                            state = current_states[b_name]
                            T_field = state.reshape((Ny, Nx))
                        else:
                            T_field = np.zeros((Ny, Nx))
                        out_val = T_field
                        # Store secondary outputs for multi-port access
                        current_signals[b_name + '_out1'] = float(np.mean(T_field))  # T_avg
                        current_signals[b_name + '_out2'] = float(np.max(T_field))   # T_max

                    elif fn == 'Fieldprobe2D':
                        # FieldProbe2D: bilinear interpolation from 2D field
                        field = inputs.get(0, None)
                        if field is None or not isinstance(field, np.ndarray) or field.ndim != 2:
                            out_val = 0.0
                        else:
                            Ny_f, Nx_f = field.shape
                            x_pos = float(block.params.get('x_position', 0.5))
                            y_pos = float(block.params.get('y_position', 0.5))
                            x_norm = max(0, min(1, x_pos))
                            y_norm = max(0, min(1, y_pos))
                            i_float = x_norm * (Nx_f - 1)
                            j_float = y_norm * (Ny_f - 1)
                            i0 = int(np.floor(i_float))
                            i1 = min(i0 + 1, Nx_f - 1)
                            j0 = int(np.floor(j_float))
                            j1 = min(j0 + 1, Ny_f - 1)
                            di = i_float - i0
                            dj = j_float - j0
                            out_val = (field[j0, i0] * (1 - di) * (1 - dj) +
                                      field[j0, i1] * di * (1 - dj) +
                                      field[j1, i0] * (1 - di) * dj +
                                      field[j1, i1] * di * dj)

                    elif fn == 'Fieldscope2D':
                        # FieldScope2D: pass through 2D field
                        field = inputs.get(0, np.zeros((1, 1)))
                        out_val = np.atleast_2d(field)

                    elif fn == 'Fieldslice':
                        # FieldSlice: extract 1D slice from 2D field
                        field = inputs.get(0, None)
                        if field is None or not isinstance(field, np.ndarray) or field.ndim != 2:
                            out_val = np.array([0.0])
                        else:
                            Ny_f, Nx_f = field.shape
                            direction = block.params.get('slice_direction', 'x')
                            position = float(block.params.get('slice_position', 0.5))
                            if direction.lower() == 'x':
                                j = int(position * (Ny_f - 1))
                                j = max(0, min(Ny_f - 1, j))
                                out_val = field[j, :]
                            else:
                                i = int(position * (Nx_f - 1))
                                i = max(0, min(Nx_f - 1, i))
                                out_val = field[:, i]

                    elif fn == 'PID':
                         # Inputs
                         sp = float(inputs.get(0, 0.0))
                         meas = float(inputs.get(1, 0.0))
                         e = sp - meas
                         
                         # Params
                         Kp = float(block.params.get('Kp', 1.0))
                         Ki = float(block.params.get('Ki', 0.0))
                         Kd = float(block.params.get('Kd', 0.0))
                         N = float(block.params.get('N', 20.0))
                         
                         # States
                         states = current_states[b_name]
                         x_i = states[0]
                         x_d = states[1]
                         
                         # Derivatives (need for D-term calc)
                         # d_term = Kd * dx_d
                         # dx_d = N * (e - x_d)
                         dx_d = N * (e - x_d)
                         
                         d_term = Kd * dx_d
                         i_term = Ki * x_i
                         p_term = Kp * e
                         
                         u_unsat = p_term + i_term + d_term
                         
                         # Saturation
                         u_min = float(block.params.get('u_min', -np.inf))
                         u_max = float(block.params.get('u_max', np.inf))
                         
                         out_val = np.clip(u_unsat, u_min, u_max)

                    elif fn == 'Sine':
                        amp = float(block.params.get('amplitude', 1.0))
                        freq = float(block.params.get('frequency', block.params.get('omega', 1.0)))
                        phase = float(block.params.get('phase', block.params.get('init_angle', 0.0)))
                        bias = float(block.params.get('bias', 0.0))
                        out_val = amp * np.sin(freq * t + phase) + bias
                        
                    elif fn == 'Constant':
                        raw_val = block.params.get('value', 0.0)
                        if isinstance(raw_val, (list, tuple)):
                            out_val = np.atleast_1d(raw_val)
                        elif hasattr(raw_val, '__iter__') and not isinstance(raw_val, str):
                            out_val = np.atleast_1d(raw_val)
                        else:
                            out_val = float(raw_val)
                        
                    elif fn == 'Gain':
                        out_val = inputs.get(0, 0.0) * float(block.params.get('gain', 1.0))
                        
                    elif fn == 'Sum':
                         signs = block.params.get('sign', block.params.get('inputs', '++'))
                         res = 0.0
                         for idx, char in enumerate(signs):
                             val = inputs.get(idx, 0.0)
                             if char == '+': res += val
                             elif char == '-': res -= val
                         out_val = res
                    
                    elif fn == 'Step':
                        step_t = float(block.params.get('delay', 0.0))
                        val = float(block.params.get('value', 1.0))
                        out_val = val if t >= step_t else 0.0

                    elif fn == 'SgProd':
                         res = 1.0
                         if inputs:
                             for val in inputs.values():
                                 res *= val
                         else:
                             res = 1.0
                         out_val = res

                    elif fn == 'Product':
                         # Product block with configurable * and / operations
                         ops = block.params.get('ops', '**')
                         res = 1.0
                         for idx, val in sorted(inputs.items()):
                             op = ops[idx] if idx < len(ops) else '*'
                             if op == '*':
                                 res *= float(val)
                             elif op == '/':
                                 res = res / float(val) if val != 0 else 1e308
                         out_val = res

                    elif fn in ('Statevariable', 'StateVariable'):
                         # StateVariable: manage discrete state across iterations
                         # State is stored in block.params for persistence across replay steps
                         # Key insight: We must update state from PREVIOUS iteration's computed input
                         # before outputting, not after.
                         if '_replay_state_' not in block.params:
                             initial = block.params.get('initial_value', [1.0])
                             if isinstance(initial, str):
                                 try:
                                     initial = eval(initial)
                                 except Exception:
                                     initial = [1.0]
                             # Preserve full vector state, not just first element
                             block.params['_replay_state_'] = np.atleast_1d(initial).copy()
                             block.params['_replay_pending_'] = None  # Input from previous step

                         # First: Apply pending update from previous iteration
                         if block.params['_replay_pending_'] is not None:
                             block.params['_replay_state_'] = block.params['_replay_pending_']
                             block.params['_replay_pending_'] = None

                         # Output current state (preserve vector or return scalar if 1D)
                         state = block.params['_replay_state_']
                         out_val = state if np.atleast_1d(state).size > 1 else float(np.atleast_1d(state)[0])

                         # Store input for next iteration (will be applied next time step)
                         if 0 in inputs:
                             new_val = inputs[0]
                             # Preserve full vector, not just first element
                             block.params['_replay_pending_'] = np.atleast_1d(new_val).copy()

                    elif fn == 'Exponential':
                        # y = a * exp(b * x)
                        a = float(block.params.get('a', 1.0))
                        b = float(block.params.get('b', 1.0))
                        x_in = float(inputs.get(0, 0.0))
                        out_val = a * np.exp(b * x_in)
                        
                    elif fn == 'Deadband':
                        val = float(inputs.get(0, 0.0))
                        start = float(block.params.get('start', -0.5))
                        end = float(block.params.get('end', 0.5))
                        
                        if val < start:
                            out_val = val - start
                        elif val > end:
                            out_val = val - end
                        else:
                            out_val = 0.0

                    elif fn == 'Saturation':
                        val = inputs.get(0, 0.0)
                        lower = float(block.params.get('min', -np.inf))
                        upper = float(block.params.get('max', np.inf))
                        out_val = np.clip(val, lower, upper)
                        
                    elif fn in ('Abs', 'Absblock'):
                        out_val = np.abs(inputs.get(0, 0.0))
                        
                    elif fn == 'Ramp':
                        slope = float(block.params.get('slope', 1.0))
                        delay = float(block.params.get('delay', 0.0))
                        if slope > 0:
                            out_val = np.maximum(0.0, slope * (t - delay))
                        elif slope < 0:
                            out_val = np.minimum(0.0, slope * (t - delay))
                        else:
                            out_val = 0.0
                            
                    elif fn == 'Switch':
                        ctrl = float(inputs.get(0, 0.0))
                        mode = block.params.get('mode', 'threshold')
                        n_inputs = int(block.params.get('n_inputs', 2))
                        
                        if mode == 'index':
                            sel = int(round(ctrl))
                        else:
                            threshold = float(block.params.get('threshold', 0.0))
                            sel = 0 if ctrl >= threshold else 1
                            
                        sel = max(0, min(n_inputs - 1, sel))
                        out_val = inputs.get(sel + 1, 0.0)
                        
                    elif fn == 'RateLimiter':
                        if b_name in current_states:
                            out_val = current_states[b_name][0]
                        else:
                            out_val = 0.0

                    elif fn == 'Wavegenerator':
                        waveform = block.params.get('waveform', 'Sine')
                        amp = float(block.params.get('amplitude', 1.0))
                        freq = float(block.params.get('frequency', 1.0))
                        phase = float(block.params.get('phase', 0.0))
                        bias = float(block.params.get('bias', 0.0))
                        arg = 2 * np.pi * freq * t + phase

                        if waveform == 'Sine':
                            out_val = bias + amp * np.sin(arg)
                        elif waveform == 'Square':
                            out_val = bias + amp * signal.square(arg)
                        elif waveform == 'Triangle':
                            out_val = bias + amp * signal.sawtooth(arg, width=0.5)
                        elif waveform == 'Sawtooth':
                            out_val = bias + amp * signal.sawtooth(arg, width=1.0)
                        else:
                            out_val = bias + amp * np.sin(arg)

                    elif fn == 'Noise':
                        mu = float(block.params.get('mu', 0.0))
                        sigma = float(block.params.get('sigma', 1.0))
                        out_val = mu + sigma * np.random.randn()

                    elif fn == 'Mathfunction':
                        val = float(inputs.get(0, 0.0))
                        # Check both 'function' and 'expression' keys for backward compatibility
                        func_raw = block.params.get('function', block.params.get('expression', 'sin'))
                        func = str(func_raw).lower()

                        try:
                            if func == 'sin':
                                out_val = np.sin(val)
                            elif func == 'cos':
                                out_val = np.cos(val)
                            elif func == 'tan':
                                out_val = np.tan(val)
                            elif func == 'asin':
                                out_val = np.arcsin(val) if -1 <= val <= 1 else 0.0
                            elif func == 'acos':
                                out_val = np.arccos(val) if -1 <= val <= 1 else 0.0
                            elif func == 'atan':
                                out_val = np.arctan(val)
                            elif func == 'exp':
                                out_val = np.exp(val)
                            elif func == 'log':
                                out_val = np.log(val) if val > 0 else 0.0
                            elif func == 'log10':
                                out_val = np.log10(val) if val > 0 else 0.0
                            elif func == 'sqrt':
                                out_val = np.sqrt(val) if val >= 0 else 0.0
                            elif func == 'square':
                                out_val = val * val
                            elif func == 'sign':
                                out_val = np.sign(val)
                            elif func == 'abs':
                                out_val = np.abs(val)
                            elif func == 'ceil':
                                out_val = np.ceil(val)
                            elif func == 'floor':
                                out_val = np.floor(val)
                            elif func == 'reciprocal':
                                out_val = 1.0 / val if val != 0 else 0.0
                            elif func == 'cube':
                                out_val = val * val * val
                            else:
                                # Python expression fallback
                                context = {
                                    "u": val, "t": t,
                                    "sin": np.sin, "cos": np.cos, "tan": np.tan,
                                    "asin": np.arcsin, "acos": np.arccos, "atan": np.arctan,
                                    "exp": np.exp, "log": np.log, "log10": np.log10,
                                    "sqrt": np.sqrt, "abs": np.abs, "sign": np.sign,
                                    "ceil": np.ceil, "floor": np.floor,
                                    "pi": np.pi, "e": np.e, "np": np
                                }
                                out_val = float(eval(str(func_raw), {"__builtins__": None}, context))
                        except (ValueError, ZeroDivisionError):
                            out_val = 0.0

                    elif fn == 'Selector':
                        val = inputs.get(0, 0.0)
                        u = np.atleast_1d(val).flatten()
                        indices_str = str(block.params.get('indices', '0'))
                        max_len = len(u)

                        result = []
                        for part in indices_str.split(','):
                            part = part.strip()
                            if ':' in part:
                                parts = part.split(':')
                                start_idx = int(parts[0]) if parts[0] else 0
                                end_idx = int(parts[1]) if len(parts) > 1 and parts[1] else max_len
                                result.extend(u[start_idx:min(end_idx, max_len)])
                            else:
                                try:
                                    idx = int(part)
                                    if idx < 0:
                                        idx = max_len + idx
                                    if 0 <= idx < max_len:
                                        result.append(u[idx])
                                except ValueError:
                                    pass

                        if len(result) == 1:
                            out_val = result[0]
                        else:
                            out_val = np.array(result) if result else 0.0

                    elif fn == 'Hysteresis':
                        val = float(inputs.get(0, 0.0))
                        upper = float(block.params.get('upper', 0.5))
                        lower = float(block.params.get('lower', -0.5))
                        high_val = float(block.params.get('high', 1.0))
                        low_val = float(block.params.get('low', 0.0))

                        # Get or initialize persistent state for replay
                        if not hasattr(block, '_replay_hyst_state'):
                            block._replay_hyst_state = low_val

                        if val >= upper:
                            block._replay_hyst_state = high_val
                        elif val <= lower:
                            block._replay_hyst_state = low_val

                        out_val = block._replay_hyst_state

                    elif fn == 'Mux':
                        # Collect all inputs into array
                        vals = []
                        port_idx = 0
                        while port_idx in inputs:
                            vals.append(inputs[port_idx])
                            port_idx += 1
                        out_val = np.array(vals) if vals else 0.0

                    elif fn == 'Demux':
                        # Pass through - demux output depends on which port is connected
                        out_val = inputs.get(0, 0.0)

                    elif fn == 'Fieldprobe':
                        # FieldProbe: Extract value at position from field array
                        field = inputs.get(0, np.array([0.0]))
                        field = np.atleast_1d(field).flatten()

                        position = float(block.params.get('position', 0.5))
                        mode = block.params.get('position_mode', 'normalized')
                        L = float(block.params.get('L', 1.0))
                        N = len(field)

                        if N == 0:
                            out_val = 0.0
                        else:
                            if mode == 'normalized':
                                idx_float = position * (N - 1)
                            else:
                                idx_float = (position / L) * (N - 1)

                            idx_float = max(0, min(N - 1, idx_float))
                            idx_low = int(np.floor(idx_float))
                            idx_high = min(idx_low + 1, N - 1)
                            frac = idx_float - idx_low

                            out_val = field[idx_low] * (1 - frac) + field[idx_high] * frac

                    elif fn == 'Fieldscope':
                        # FieldScope: Store field for 2D visualization
                        field = inputs.get(0, np.array([0.0]))
                        out_val = np.atleast_1d(field).flatten()

                    elif fn in ('Terminator', 'Display'):
                        pass # Do nothing

                    else:
                        # Fallback: call block.execute() for unhandled block types
                        # This handles optimization primitives and custom blocks
                        if block.block_instance is not None:
                            try:
                                result = block.block_instance.execute(
                                    time=t,
                                    inputs=inputs,
                                    params=block.params
                                )
                                if result and 0 in result:
                                    out_val = result[0]
                            except Exception as e:
                                logger.debug(f"Replay fallback execute failed for {b_name}: {e}")

                    # Store
                    current_signals[b_name] = out_val
                    # logger.info(f"DEBUG Replay {b_name} t={t:.2f} out={out_val}") # Uncomment for verbose debug
                    
                    # Store in Block History for Scopes
                    # ScopePlotter expects `block.out_history` list? Or `block.params['vector']`?
                    # DSim.execution_loop doesn't seem to append to `out_history` explicitly?
                    # Ah, `Scope` blocks have internal `execute` that saves to `vector`.
                    # Standard blocks don't save history unless probed.
                    # But Scopes DO.
                    if fn == 'Scope':
                        # Scope can have multiple inputs - collect all of them
                        # Ensure we write to exec_params as ScopePlotter prioritizes it
                        if not hasattr(block, 'exec_params'):
                            block.exec_params = block.params.copy()

                        # Get number of input ports
                        n_inputs = block.in_ports if hasattr(block, 'in_ports') else 1

                        # Collect and flatten all input values (matching Scope.execute() behavior)
                        # Each port value is flattened to 1D so vector signals (e.g. StateSpace
                        # with 4 outputs) are properly expanded into individual components.
                        combined = []
                        for port in range(n_inputs):
                            val = inputs.get(port, 0.0)
                            combined.append(np.atleast_1d(val).flatten())
                        new_sample = np.concatenate(combined) if combined else np.array([0.0])
                        vec_dim = len(new_sample)

                        # Initialize vector list and labels on first timestep
                        if i == 0:
                            block.exec_params['vector'] = []
                            block.exec_params['vec_dim'] = vec_dim
                            # Set vec_labels from 'labels' param (Scope uses 'labels', plotter reads 'vec_labels')
                            labels_raw = block.params.get('labels', block.exec_params.get('labels', ''))
                            if labels_raw and labels_raw != 'default':
                                labels_list = [l.strip() for l in labels_raw.replace(' ', '').split(',') if l.strip()]
                                # Pad or trim to match actual signal dimension
                                while len(labels_list) < vec_dim:
                                    labels_list.append(f"{b_name}-{len(labels_list)}")
                                labels_list = labels_list[:vec_dim]
                            else:
                                labels_list = [f"{b_name}-{j}" for j in range(vec_dim)]
                            block.exec_params['vec_labels'] = labels_list

                        block.exec_params['vector'].append(new_sample)

                        if i == num_steps - 1:
                            vec = block.exec_params['vector']
                            logger.info(f"Replay Scope {b_name}: vec_dim={vec_dim}, samples={len(vec)}, labels={block.exec_params.get('vec_labels')}")

                    if fn == 'Fieldscope':
                        # FieldScope: Store field history for 2D heatmap
                        field = inputs.get(0, np.array([0.0]))
                        field = np.atleast_1d(field).flatten()

                        if not hasattr(block, 'exec_params'):
                            block.exec_params = block.params.copy()

                        if i == 0:
                            block.exec_params['_field_history_'] = []
                            block.exec_params['_time_history_'] = []

                        block.exec_params['_field_history_'].append(field.copy())
                        block.exec_params['_time_history_'].append(t)

                        if i == num_steps - 1:
                            logger.info(f"DEBUG Replay FieldScope {b_name}: field_len={len(field)}, history_len={len(block.exec_params['_field_history_'])}")

                    if fn == 'Fieldscope2D':
                        # FieldScope2D: Store 2D field history for animated heatmap
                        field = inputs.get(0, np.zeros((1, 1)))
                        field = np.atleast_2d(field)

                        if not hasattr(block, 'exec_params'):
                            block.exec_params = block.params.copy()

                        if i == 0:
                            block.exec_params['_field_history_2d_'] = []
                            block.exec_params['_time_history_'] = []

                        # Store every N frames to reduce memory
                        sample_interval = int(block.params.get('sample_interval', 5))
                        if i % sample_interval == 0:
                            block.exec_params['_field_history_2d_'].append(field.copy())
                            block.exec_params['_time_history_'].append(t)

                        if i == num_steps - 1:
                            logger.info(f"DEBUG Replay FieldScope2D {b_name}: field_shape={field.shape}, history_len={len(block.exec_params['_field_history_2d_'])}")

            # Finalize Scope Vectors (convert to numpy)
            for block in current_blocks:
                if block.block_fn == 'Scope':
                     if hasattr(block, 'exec_params') and 'vector' in block.exec_params:
                        block.exec_params['vector'] = np.array(block.exec_params['vector'])
                elif block.block_fn == 'FieldScope':
                    if hasattr(block, 'exec_params') and '_field_history_' in block.exec_params:
                        block.exec_params['_field_history_'] = np.array(block.exec_params['_field_history_'])
                    if hasattr(block, 'exec_params') and '_time_history_' in block.exec_params:
                        block.exec_params['_time_history_'] = np.array(block.exec_params['_time_history_'])
                elif block.block_fn == 'FieldScope2D':
                    if hasattr(block, 'exec_params') and '_field_history_2d_' in block.exec_params:
                        block.exec_params['_field_history_2d_'] = np.array(block.exec_params['_field_history_2d_'])
                    if hasattr(block, 'exec_params') and '_time_history_' in block.exec_params:
                        block.exec_params['_time_history_'] = np.array(block.exec_params['_time_history_'])
            
            return True
            
        except Exception as e:
            logger.error(f"Compiled simulation failed: {e}", exc_info=True)
            return False
