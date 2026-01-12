"""
SimulationEngine - Execution and analysis logic for DiaBloS.
Handles simulation initialization, execution loops, and diagram analysis.
"""

import logging
import time as time_module
from typing import List, Dict, Tuple, Any, Optional, Callable
import numpy as np
from lib.simulation.block import DBlock
from lib.workspace import WorkspaceManager

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

    def check_diagram_integrity(self):
        """
        Verify that all block ports are properly connected.

        Returns:
            bool: True if diagram is valid, False otherwise
        """
        logger.debug("Checking diagram integrity")
        error_trigger = False

        for block in self.model.blocks_list:
            inputs, outputs = self.get_neighbors(block.name)

            # Check input ports
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

            # Check output ports
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

        for line in self.model.line_list:
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
        for line in self.model.line_list:
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
        for block in self.model.blocks_list:
            if block.hierarchy > max_h:
                max_h = block.hierarchy
        return max_h

    def detect_algebraic_loops(self, uncomputed_blocks):
        """
        Detect if there are algebraic loops in uncomputed blocks.

        Args:
            uncomputed_blocks: List of blocks that haven't been computed

        Returns:
            bool: True if algebraic loop detected
        """
        if len(uncomputed_blocks) == 0:
            return False

        logger.debug("Checking for algebraic loops...")
        logger.debug(f"Uncomputed blocks: {[b.name for b in uncomputed_blocks]}")

        # Check if any block can be computed
        for block in uncomputed_blocks:
            inputs, _ = self.get_neighbors(block.name)

            # Count how many inputs are satisfied
            satisfied_inputs = 0
            for input_conn in inputs:
                src_block = self.model.get_block_by_name(input_conn['srcblock'])
                if src_block and src_block.computed_data:
                    satisfied_inputs += 1

            # If all inputs are satisfied, this block can be computed
            if satisfied_inputs == len(inputs):
                logger.debug(f"Block {block.name} can be computed")
                return False

        # If we get here, no block can be computed = algebraic loop
        logger.error("ALGEBRAIC LOOP DETECTED")
        logger.error(f"Blocks involved: {[b.name for b in uncomputed_blocks]}")
        return True

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

    def reset_execution_data(self):
        """Reset execution state for all blocks.
        
        IMPORTANT: Must update global_computed_list AND restore hierarchy from it.
        """
        # Safety check - if global_computed_list isn't populated yet, use simple reset
        if not self.global_computed_list or len(self.global_computed_list) != len(self.model.blocks_list):
            for block in self.model.blocks_list:
                block.computed_data = False
                block.data_recieved = 0
                block.data_sent = 0
                block.hierarchy = -1
                block.input_queue = {}
            return
            
        for i in range(len(self.model.blocks_list)):
            self.global_computed_list[i]['computed_data'] = False
            self.model.blocks_list[i].computed_data = False
            self.model.blocks_list[i].data_recieved = 0
            self.model.blocks_list[i].data_sent = 0
            self.model.blocks_list[i].input_queue = {}
            self.model.blocks_list[i].hierarchy = self.global_computed_list[i]['hierarchy']

    def count_rk45_integrators(self):
        """
        Check if any integrators use RK45 method.

        Returns:
            bool: True if RK45 integrators exist
        """
        for block in self.model.blocks_list:
            if block.block_fn == 'Integrator' and block.params.get('method') == 'RK45':
                return True
            elif block.block_fn == 'External' and block.params.get('method') == 'RK45':
                return True
        return False

    def reset_memblocks(self):
        """Reset memory blocks (integrators, transfer functions, etc.)."""
        for block in self.model.blocks_list:
            if block.block_fn in ['Integrator', 'TranFn']:
                if '_init_start_' in block.params:
                    block.params['_init_start_'] = True

    def update_sim_params(self, sim_time, sim_dt):
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
        """Identify blocks with memory (integrators, strictly proper TFs)."""
        self.memory_blocks = set()
        for block in self.model.blocks_list:
            if block.b_type == 1:
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

    def execute_block(self, block: DBlock, output_only: bool = False) -> Dict[int, Any]:
        """
        Execute a single block.

        Args:
            block: The block to execute
            output_only: If True, only compute output without updating state

        Returns:
            dict: Output values keyed by port number, or {'E': True} on error
        """
        kwargs = {
            'time': self.time_step,
            'inputs': block.input_queue,
            'params': block.exec_params,
        }
        if output_only:
            kwargs['output_only'] = True
        if block.block_fn == 'Integrator':
            kwargs['next_add_in_memory'] = not self.rk45_len or self.rk_counter == 3
            kwargs['dtime'] = self.sim_dt

        try:
            if block.external:
                out_value = getattr(block.file_function, block.fn_name)(**kwargs)
            elif block.block_instance:
                out_value = block.block_instance.execute(**kwargs)
            else:
                out_value = getattr(self.execution_function, block.fn_name)(**kwargs)
            return out_value
        except Exception as e:
            logger.error(f"Error executing block {block.name}: {str(e)}")
            return {'E': True, 'error': str(e)}

    def propagate_outputs(self, block: DBlock, out_value: Dict[int, Any]) -> None:
        """
        Propagate block outputs to connected downstream blocks.

        Args:
            block: Source block
            out_value: Output values from the block
        """
        children = self.get_outputs(block.name)
        for mblock in self.model.blocks_list:
            is_child, tuple_list = self._children_recognition(mblock.name, children)
            if is_child:
                for tuple_child in tuple_list:
                    if tuple_child['dstport'] not in mblock.input_queue:
                        mblock.data_recieved += 1
                    mblock.input_queue[tuple_child['dstport']] = out_value[tuple_child['srcport']]
                    block.data_sent += 1

    def execute_and_propagate(self, block: DBlock, output_only: bool = False) -> Dict[int, Any]:
        """
        Execute a block and propagate its outputs to children.
        
        This is the core execution helper used by execution_init and execution_loop.
        IMPORTANT: Does not create new exec_params dicts - only reads/updates existing ones.
        
        Args:
            block: Block to execute
            output_only: If True, only compute output without updating state
            
        Returns:
            dict: Output values, or {'E': True, 'error': msg} on failure
        """
        out_value = self.execute_block(block, output_only)
        
        if out_value is None:
            return {'E': True, 'error': f'Block {block.name} returned None'}
        
        if 'E' in out_value and out_value['E']:
            return out_value
            
        # Propagate outputs to children (for non-sink blocks)
        if block.b_type not in [1, 3]:  # Not memory-only or sink
            self.propagate_outputs(block, out_value)
        
        return out_value

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

    def update_global_list(self, block_name: str, h_value: int = 0, h_assign: bool = False) -> None:
        """
        Update the global computed list for a block.

        Args:
            block_name: Name of the block
            h_value: Hierarchy value to assign
            h_assign: Whether to assign hierarchy value
        """
        for elem in self.global_computed_list:
            if elem['name'] == block_name:
                if h_assign:
                    elem['hierarchy'] = h_value
                elem['computed_data'] = True
                break

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
