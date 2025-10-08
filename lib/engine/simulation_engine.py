"""
SimulationEngine - Execution and analysis logic for DiaBloS.
Handles simulation initialization, execution loops, and diagram analysis.
"""

import logging
from typing import List, Dict, Tuple, Any
import numpy as np
from lib import functions
from lib.simulation.block import DBlock

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
        self.execution_function = functions

        # Execution state
        self.execution_initialized: bool = False
        self.execution_pause: bool = False
        self.execution_stop: bool = False
        self.error_msg: str = ""

        # Simulation parameters
        self.sim_time: float = 1.0
        self.sim_dt: float = 0.01
        self.real_time: bool = True

        # Execution tracking
        self.global_computed_list: List[Dict[str, Any]] = []
        self.timeline: List[float] = []
        self.outs: List[Any] = []

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
        """Reset execution state for all blocks."""
        for block in self.model.blocks_list:
            block.computed_data = False
            block.data_recieved = 0
            block.data_sent = 0
            block.hierarchy = -1
            block.input_queue = {i: None for i in range(block.in_ports)}

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
