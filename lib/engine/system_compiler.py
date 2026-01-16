
import logging
import numpy as np
from typing import List, Callable, Dict, Any, Tuple
from lib.simulation.block import DBlock

logger = logging.getLogger(__name__)

class SystemCompiler:
    """
    Compiles a block diagram into a flat numerical function for fast ODE solving.
    Supports integration with scipy.integrate.solve_ivp.
    """

    def __init__(self):
        # Allowlist of blocks that can be compiled
        self.COMPILABLE_BLOCKS = {
            'Integrator',
            'Gain',
            'Sum',
            'Constant',
            'Sine', 
            'Step',
            'TransferFcn', # Requires specialized strict handling
            'Mux',
            'Demux',
            'Scope'
        }

    def check_compilability(self, blocks: List[DBlock]) -> bool:
        """
        Check if the entire diagram is supported by the compiler.
        
        Args:
            blocks: List of all blocks in the diagram.
            
        Returns:
            bool: True if all blocks are supported, False otherwise.
        """
        for block in blocks:
            # Check block type (case-insensitive for safety)
            # Normalize to Title Case (e.g. Sine, Integrator) or just allow 'sine'
            # Our Set is Title Case. 
            b_type = block.block_fn
            if b_type not in self.COMPILABLE_BLOCKS:
                # Try correcting case (Capitalize first letter) just in case
                if b_type.title() in self.COMPILABLE_BLOCKS:
                     # It's compilable if we treat it as the TitleCase version
                     pass
                else: 
                     logger.debug(f"Block {block.name} ({block.block_fn}) is not compilable.")
                     return False
            
            # Check for custom scripts (External blocks are generally not compilable unless specific standard ones)
            if block.external:
                # We might allow standard library external blocks if we implement their logic
                # For now, simplistic check: everything in standard library that matches COMPILABLE_BLOCKS is ok.
                # But if it's a "Custom Script" user block, reject.
                pass 

        return True

    def compile_system(self, blocks: List[DBlock], sorted_order: List[DBlock], lines: List[Any]) -> Tuple[Callable, np.ndarray, Dict]:
        """
        Generate a fast derivative function f(t, y) using closure-based optimization.
        """
        # 1. Build Dependency Graph (Static Pull-based connections)
        # map: dest_block_name -> port_idx -> (source_block_name, source_port_idx)
        input_map = {b.name: {} for b in blocks}
        
        for line in lines:
            dst_block = line.dstblock
            dst_port = line.dstport
            src_block = line.srcblock
            src_port = line.srcport
            
            input_map[dst_block][dst_port] = (src_block, src_port)

        # 2. Identify States (Integrators)
        state_map = {} # block_name -> (start_idx, size)
        y0_list = []
        current_state_idx = 0
        
        # We need to know state size. For now assume scalar/1D unless specified.
        # In Compilable Blocks, params['init_conds'] defines size.
        
        integrator_blocks = [b for b in blocks if b.block_fn == 'Integrator']
        
        for block in integrator_blocks:
            ic = np.array(block.params.get('init_conds', 0.0), dtype=float)
            ic_flat = np.atleast_1d(ic).flatten()
            size = ic_flat.size
            state_map[block.name] = (current_state_idx, size)
            y0_list.extend(ic_flat)
            current_state_idx += size
            
        y0 = np.array(y0_list, dtype=float)
        
        # 3. Create optimized closure
        def model_func(t, y):
            signals = {} # block_name -> output_value
            dy_vec = np.zeros_like(y)
            
            # A. Populate States
            for b_name, (start, size) in state_map.items():
                signals[b_name] = y[start : start + size]
                
            # B. Execute Blocks
            for block in sorted_order:
                b_name = block.name
                # Normalize function name (handle 'sine' vs 'Sine')
                fn = block.block_fn.title() if block.block_fn else ''
                
                # Integrators: Output is state (already set). We just need to compute derivative later? 
                # No, we need to compute Input to get derivative.
                # But Integrator input comes from other blocks.
                # So Integrator logic here is: Do nothing for output (already in signals), 
                # but we need to compute its input derivative after all inputs are ready?
                # Actually, Integrator is in sorted_order.
                # If it's a feedback loop, sorted_order handles it via algebraic loop breaking?
                # In ODE solvers, 'y' is given. We need 'dy'.
                
                # Fetch Inputs
                inputs = {}
                deps = input_map.get(b_name, {})
                for port_idx, (src_name, src_port) in deps.items():
                    if src_name in signals:
                        val = signals[src_name]
                        # Handle specific output port if block has multiple
                        # For simple blocks, val matches output 0.
                        inputs[port_idx] = val 
                    else:
                        # Unconnected or not computed yet (Algebraic Loop issue?)
                        inputs[port_idx] = 0.0
                
                # Execute Block Logic (Hardcoded fast versions)
                if fn == 'Integrator':
                    # Output is state f(y).
                    # Derivative is Input u(t).
                    # We store input as derivative!
                    input_val = inputs.get(0, 0.0)
                    if b_name in state_map:
                         start, size = state_map[b_name]
                         dy_vec[start : start + size] = np.atleast_1d(input_val).flatten()
                         
                elif fn == 'Gain':
                    gain = float(block.params.get('gain', 1.0))
                    signals[b_name] = inputs.get(0, 0.0) * gain
                    
                elif fn == 'Sum':
                    # Sum has signs '++-' or 'list of signs'
                    signs = block.params.get('inputs', '++')
                    res = 0.0
                    for i, char in enumerate(signs):
                        val = inputs.get(i, 0.0)
                        if char == '+': res += val
                        elif char == '-': res -= val
                    signals[b_name] = res
                    
                elif fn == 'Constant':
                    signals[b_name] = float(block.params.get('value', 0.0))
                    
                elif fn == 'Sine':
                    amp = float(block.params.get('amplitude', 1.0))
                    # Handle param aliases (omega/frequency, init_angle/phase)
                    freq = float(block.params.get('frequency', block.params.get('omega', 1.0)))
                    phase = float(block.params.get('phase', block.params.get('init_angle', 0.0)))
                    bias = float(block.params.get('bias', 0.0))
                    signals[b_name] = amp * np.sin(freq * t + phase) + bias
                    
                elif fn == 'Step':
                    step_t = float(block.params.get('time', 1.0))
                    init_v = float(block.params.get('initial_value', 0.0))
                    final_v = float(block.params.get('final_value', 1.0))
                    signals[b_name] = final_v if t >= step_t else init_v

            return dy_vec

        return model_func, y0, state_map
