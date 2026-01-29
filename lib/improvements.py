"""
Incremental improvements to DiaBloS - Type hints and utility functions.

This module provides type hints, utility functions, and helpers that can be
gradually integrated with the existing codebase without breaking changes.
"""

from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import logging
import time
from PyQt5.QtGui import QColor

# Set up logging
logger = logging.getLogger(__name__)


@dataclass
class SimulationConfig:
    """Simple configuration for simulation parameters."""
    simulation_time: float = 10.0
    time_step: float = 0.01
    fps: int = 60
    dynamic_plot: bool = False
    plot_time_range: float = 5.0
    solver_method: str = "RK45"
    
    def validate(self) -> Tuple[bool, List[str]]:
        """Validate configuration parameters."""
        errors = []
        
        if self.simulation_time <= 0:
            errors.append("Simulation time must be positive")
        
        if self.time_step <= 0:
            errors.append("Time step must be positive")
        
        if self.time_step >= self.simulation_time:
            errors.append("Time step cannot be larger than simulation time")
        
        if self.fps <= 0:
            errors.append("FPS must be positive")
        
        valid_methods = ["FWD_RECT", "BWD_RECT", "TUSTIN", "RK45"]
        if self.solver_method not in valid_methods:
            errors.append(f"Solver method must be one of {valid_methods}")
        
        return len(errors) == 0, errors


class ValidationHelper:
    """Helper functions for validation that can be used with existing DSim."""
    
    @staticmethod
    def validate_block_connections(blocks_list: List[Any], line_list: List[Any]) -> Tuple[bool, List[str]]:
        logger.debug(f"ValidationHelper.validate_block_connections called.")
        logger.debug(f"  blocks_list: {[b.name for b in blocks_list if hasattr(b, 'name')]}")
        logger.debug(f"  line_list: {[l.name for l in line_list if hasattr(l, 'name')]}")
        errors = []
        warnings = []
        
        # Check for blocks with connections
        connected_blocks = set()
        for line in line_list:
            if hasattr(line, 'srcblock') and hasattr(line, 'dstblock'):
                connected_blocks.add(line.srcblock)
                connected_blocks.add(line.dstblock)
        
        # Only report disconnected blocks as warnings (not errors)
        for block in blocks_list:
            if hasattr(block, 'name') and block.name not in connected_blocks:
                if hasattr(block, 'in_ports') and hasattr(block, 'out_ports'):
                    if block.in_ports > 0 or block.out_ports > 0:
                        warnings.append(f"Block '{block.name}' has no connections")
        
        # Check for multiple connections to same input port (this is an error)
        input_connections = {}
        for line in line_list:
            if hasattr(line, 'dstblock') and hasattr(line, 'dstport'):
                key = (line.dstblock, line.dstport)
                if key in input_connections:
                    errors.append(f"Multiple connections to same input port: block {line.dstblock}, port {line.dstport}")
                input_connections[key] = line
        
        # For now, don't require all input ports to be connected (too strict)
        # This allows partial diagrams and blocks with optional inputs

        # Check for algebraic loops
        no_loops, loop_errors = ValidationHelper.detect_algebraic_loops(blocks_list, line_list)
        if not no_loops:
            errors.extend(loop_errors)
        
        return len(errors) == 0, errors + warnings
    
    @staticmethod
    def detect_algebraic_loops(blocks_list: List[Any], line_list: List[Any]) -> Tuple[bool, List[str]]:
        """
        Simple algebraic loop detection using topological sort.
        
        Memory blocks (Integrator, StateSpace, etc.) break algebraic loops because
        their output depends on previous state, not current input.
        
        Args:
            blocks_list: List of DBlock instances
            line_list: List of DLine instances
            
        Returns:
            Tuple of (no_loops_found, list_of_errors)
        """
        from collections import defaultdict, deque
        
        # Memory blocks break algebraic loops - their output depends on previous state
        MEMORY_BLOCK_TYPES = {
            'Integrator', 'StateSpace', 'DiscreteStateSpace', 
            'DiscreteTranFn', 'ZeroOrderHold', 'Delay'
        }
        
        graph = defaultdict(list)
        in_degree = defaultdict(int)
        block_map = {block.name: block for block in blocks_list}

        for block in blocks_list:
            in_degree[block.name] = 0

        for line in line_list:
            src_block = block_map.get(line.srcblock)
            dst_block = block_map.get(line.dstblock)

            if not src_block or not dst_block:
                continue

            # Check if DESTINATION block is a memory block (breaks the loop)
            # A memory block's output at time t depends on its input at time t-1
            is_memory_block = False

            # Check by b_type: Type 1 blocks are memory blocks
            if dst_block.b_type == 1:
                is_memory_block = True
            
            # Check by block function name
            if dst_block.block_fn in MEMORY_BLOCK_TYPES:
                is_memory_block = True
            
            # TranFn is memory block if strictly proper (num degree < den degree)
            if dst_block.block_fn == 'TranFn':
                num = dst_block.params.get('numerator', [])
                den = dst_block.params.get('denominator', [])
                if len(den) > len(num):
                    is_memory_block = True
                    logger.debug(f"  {dst_block.name} (TranFn) is strictly proper - memory block")

            # DiscreteTranFn is memory block if strictly proper
            if dst_block.block_fn == 'DiscreteTranFn':
                num = dst_block.params.get('numerator', [])
                den = dst_block.params.get('denominator', [])
                if len(den) > len(num):
                    is_memory_block = True
                    logger.debug(f"  {dst_block.name} (DiscreteTranFn) is strictly proper - memory block")

            if is_memory_block:
                logger.debug(f"Memory block {dst_block.name} breaks loop from {src_block.name}")
            else:
                # Only add edge if destination is NOT a memory block
                graph[line.srcblock].append(line.dstblock)
                in_degree[line.dstblock] += 1

        queue = deque([name for name, degree in in_degree.items() if degree == 0])
        
        count = 0
        while queue:
            u = queue.popleft()
            count += 1
            for v in graph[u]:
                in_degree[v] -= 1
                if in_degree[v] == 0:
                    queue.append(v)

        if count < len(blocks_list):
            # Find the nodes in the cycle
            cycle_nodes = [name for name, degree in in_degree.items() if degree > 0]
            return False, [f"Algebraic loop detected involving blocks: {cycle_nodes}"]
        
        return True, []


class PerformanceHelper:
    """Helper for performance monitoring."""
    
    def __init__(self):
        self.start_times: Dict[str, float] = {}
        self.durations: Dict[str, List[float]] = {}
    
    def start_timer(self, operation: str) -> None:
        """Start timing an operation."""
        self.start_times[operation] = time.time()
    
    def end_timer(self, operation: str) -> Optional[float]:
        """End timing and return duration."""
        if operation not in self.start_times:
            return None
        
        duration = time.time() - self.start_times[operation]
        
        if operation not in self.durations:
            self.durations[operation] = []
        self.durations[operation].append(duration)
        
        del self.start_times[operation]
        return duration
    
    def get_stats(self, operation: str) -> Optional[Dict[str, float]]:
        """Get statistics for an operation."""
        if operation not in self.durations or not self.durations[operation]:
            return None
        
        durations = self.durations[operation]
        return {
            'count': len(durations),
            'total': sum(durations),
            'average': sum(durations) / len(durations),
            'min': min(durations),
            'max': max(durations)
        }
    
    def log_stats(self) -> None:
        """Log performance statistics."""
        for operation in self.durations:
            stats = self.get_stats(operation)
            if stats:
                logger.info(
                    f"Performance - {operation}: "
                    f"count={stats['count']}, "
                    f"avg={stats['average']:.4f}s, "
                    f"total={stats['total']:.4f}s"
                )


class SafetyChecks:
    """Safety checks that can be added to existing code."""
    
    @staticmethod
    def check_block_integrity(block: Any) -> Tuple[bool, List[str]]:
        """Check if a block has all required attributes."""
        errors = []
        required_attrs = ['name', 'sid', 'in_ports', 'out_ports', 'b_type', 'fn_name']
        
        for attr in required_attrs:
            if not hasattr(block, attr):
                errors.append(f"Block missing required attribute: {attr}")
        
        if hasattr(block, 'in_ports') and hasattr(block, 'out_ports'):
            if block.in_ports < 0:
                errors.append("Block has negative input ports")
            if block.out_ports < 0:
                errors.append("Block has negative output ports")
        
        if hasattr(block, 'b_type'):
            if block.b_type not in [0, 1, 2, 3]:
                errors.append(f"Invalid block type: {block.b_type}")
        
        return len(errors) == 0, errors
    
    @staticmethod
    def check_simulation_state(dsim_instance: Any) -> Tuple[bool, List[str]]:
        """Check if DSim instance is in valid state for simulation."""
        errors = []
        
        # Check required attributes
        required_attrs = ['blocks_list', 'line_list', 'execution_initialized']
        for attr in required_attrs:
            if not hasattr(dsim_instance, attr):
                errors.append(f"DSim missing required attribute: {attr}")
        
        # Check blocks exist
        if hasattr(dsim_instance, 'blocks_list'):
            if not dsim_instance.blocks_list:
                errors.append("No blocks in simulation")
            else:
                for i, block in enumerate(dsim_instance.blocks_list):
                    is_valid, block_errors = SafetyChecks.check_block_integrity(block)
                    if not is_valid:
                        errors.extend([f"Block {i}: {error}" for error in block_errors])
        
        # Check simulation parameters
        if hasattr(dsim_instance, 'sim_time') and dsim_instance.sim_time <= 0:
            errors.append("Invalid simulation time")
        
        if hasattr(dsim_instance, 'sim_dt') and dsim_instance.sim_dt <= 0:
            errors.append("Invalid simulation time step")
        
        return len(errors) == 0, errors


class LoggingHelper:
    """Enhanced logging for DiaBloS operations."""
    
    @staticmethod
    def setup_logging(level: str = "INFO", log_file: Optional[str] = None) -> None:
        """Setup enhanced logging configuration."""
        numeric_level = getattr(logging, level.upper(), logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        console_handler.setLevel(numeric_level)
        
        # File handler if specified
        handlers = [console_handler]
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            file_handler.setLevel(numeric_level)
            handlers.append(file_handler)
        
        # Configure root logger
        logging.basicConfig(
            level=numeric_level,
            handlers=handlers,
            force=True
        )
    
    @staticmethod
    def log_simulation_start(dsim_instance: Any) -> None:
        """Log simulation start with key parameters."""
        logger.info("=" * 50)
        logger.info("SIMULATION START")
        logger.info("=" * 50)
        
        if hasattr(dsim_instance, 'blocks_list'):
            logger.info(f"Number of blocks: {len(dsim_instance.blocks_list)}")
        
        if hasattr(dsim_instance, 'line_list'):
            logger.info(f"Number of connections: {len(dsim_instance.line_list)}")
        
        if hasattr(dsim_instance, 'sim_time'):
            logger.info(f"Simulation time: {dsim_instance.sim_time}s")
        
        if hasattr(dsim_instance, 'sim_dt'):
            logger.info(f"Time step: {dsim_instance.sim_dt}s")
    
    @staticmethod
    def log_simulation_end(dsim_instance: Any, duration: float) -> None:
        """Log simulation end with statistics."""
        logger.info("=" * 50)
        logger.info("SIMULATION COMPLETE")
        logger.info("=" * 50)
        logger.info(f"Total execution time: {duration:.4f}s")
        
        if hasattr(dsim_instance, 'time_step') and hasattr(dsim_instance, 'sim_dt'):
            steps = int(dsim_instance.time_step / dsim_instance.sim_dt) if dsim_instance.sim_dt > 0 else 0
            logger.info(f"Total simulation steps: {steps}")
            if steps > 0:
                logger.info(f"Average time per step: {duration/steps:.6f}s")


def create_default_colors() -> Dict[str, QColor]:
    """Create the default color palette for DiaBloS."""
    return {
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


def validate_simulation_parameters(sim_time: float, sim_dt: float) -> Tuple[bool, List[str]]:
    """
    Validate simulation parameters.
    
    Args:
        sim_time: Total simulation time
        sim_dt: Time step
        
    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []
    
    if sim_time <= 0:
        errors.append("Simulation time must be positive")
    
    if sim_dt <= 0:
        errors.append("Time step must be positive")
    
    if sim_dt >= sim_time:
        errors.append("Time step cannot be larger than simulation time")
    
    if sim_time / sim_dt > 1000000:
        errors.append("Too many simulation steps (>1M), reduce time or increase step size")
    
    return len(errors) == 0, errors


def safe_execute_block_function(func: Any, *args, **kwargs) -> Tuple[bool, Any, Optional[str]]:
    """
    Safely execute a block function with error handling.
    
    Returns:
        Tuple of (success, result, error_message)
    """
    try:
        result = func(*args, **kwargs)
        
        # Check if result indicates an error
        if isinstance(result, dict) and result.get('E', False):
            return False, result, result.get('error', 'Unknown error from block function')
        
        return True, result, None
        
    except Exception as e:
        error_msg = f"Exception in block function: {str(e)}"
        logger.error(error_msg)
        return False, None, error_msg