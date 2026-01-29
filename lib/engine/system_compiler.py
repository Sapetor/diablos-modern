
import logging
import numpy as np
from typing import List, Callable, Dict, Any, Tuple
from lib.simulation.block import DBlock
from scipy import signal

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
            'TransferFcn',
            'TranFn',
            'StateSpace',
            'Mux',
            'Demux',
            'Scope',
            'SgProd', 'SigProduct',
            'Saturation',
            'Abs', 'AbsBlock',
            'Ramp',
            'Switch',
            'Terminator',
            'Display',
            'Deadband',
            'Exponential',
            'PiD', 'PID',
            'RateLimiter',
            'WaveGenerator',
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
            
            # Special handling for Subsystems (Recursive check)
            if b_type == 'Subsystem':
                 if hasattr(block, 'sub_blocks'):
                     if not self.check_compilability(block.sub_blocks):
                         return False
                 continue

            # Special handling for structural blocks (Inport/Outport)
            # They are flattened away or handled as sources/sinks
            if b_type in ('Inport', 'Outport'):
                continue

            if b_type not in self.COMPILABLE_BLOCKS:
                # Try correcting case (Capitalize first letter) just in case
                if b_type.title() in self.COMPILABLE_BLOCKS:
                     # It's compilable if we treat it as the TitleCase version
                     pass
                elif b_type.upper() in self.COMPILABLE_BLOCKS: # For PID
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

    def _create_block_executor(self, block: DBlock, input_map: Dict, state_map: Dict, block_matrices: Dict) -> Callable[[float, np.ndarray, np.ndarray, Dict], None]:
        """
        Creates a dedicated closure for a specific block's execution.
        Args:
            block: The block to compile.
            input_map: Dependency graph.
            state_map: State allocations.
            block_matrices: Pre-computed matrices.
        Returns:
            function(t, y, dy_vec, signals) -> None
        """
        b_name = block.name
        
        # Normalize Function Name
        fn = block.block_fn.title() if block.block_fn else ''
        if fn == 'Statespace': fn = 'StateSpace'
        if fn in ('Transferfcn', 'Tranfn'): fn = 'TransferFcn'
        if block.block_fn == 'PID': fn = 'PID'
        if fn == 'Ratelimiter': fn = 'RateLimiter'
        
        # Pre-resolve Inputs
        # We need a list of source keys to fetch from 'signals'
        # keys are strings (b_name).
        deps = input_map.get(b_name, {})
        # Sort by port index to ensure order (0, 1, 2...)
        sorted_ports = sorted(deps.keys())
        
        # Optimization: Create a list of source names for each port 0..N
        # If a port is unconnected, we need a default.
        # But signals dict won't have it.
        # So we store [(src_name, src_port), ...] 
        # Actually simplest is to bake the lookup.
        
        input_sources = [] # List of (src_name, default_val) ordered by port
        max_port = max(sorted_ports) if sorted_ports else -1
        for i in range(max_port + 1):
            if i in deps:
                src_name, _ = deps[i]
                input_sources.append(src_name)
            else:
                input_sources.append(None) # None means 0.0 default
        
        # --- Block Specific Closures ---
        
        if fn == 'Gain':
            gain = float(block.params.get('gain', 1.0))
            # Optimization: If only 1 input
            src = input_sources[0] if input_sources else None
            
            def exec_gain(t, y, dy_vec, signals):
                val = signals.get(src, 0.0) if src else 0.0
                signals[b_name] = val * gain
            return exec_gain

        elif fn == 'Sum':
            signs = block.params.get('inputs', '++')
            # Bake signs and sources
            ops = []
            for i, char in enumerate(signs):
                src = input_sources[i] if i < len(input_sources) else None
                ops.append((src, 1.0 if char == '+' else -1.0))
            
            def exec_sum(t, y, dy_vec, signals):
                res = 0.0
                for src, mul in ops:
                    val = signals.get(src, 0.0) if src else 0.0
                    res += val * mul
                signals[b_name] = res
            return exec_sum
            
        elif fn == 'Constant':
            val = float(block.params.get('value', 0.0))
            def exec_constant(t, y, dy_vec, signals):
                signals[b_name] = val
            return exec_constant
            
        elif fn == 'Sine':
            amp = float(block.params.get('amplitude', 1.0))
            freq = float(block.params.get('frequency', block.params.get('omega', 1.0)))
            phase = float(block.params.get('phase', block.params.get('init_angle', 0.0)))
            bias = float(block.params.get('bias', 0.0))

            def exec_sine(t, y, dy_vec, signals):
                signals[b_name] = amp * np.sin(freq * t + phase) + bias
            return exec_sine

        elif fn == 'Wavegenerator':
            waveform = block.params.get('waveform', 'Sine')
            amp = float(block.params.get('amplitude', 1.0))
            freq = float(block.params.get('frequency', 1.0))
            phase = float(block.params.get('phase', 0.0))
            bias = float(block.params.get('bias', 0.0))

            if waveform == 'Sine':
                def exec_wavegen_sine(t, y, dy_vec, signals):
                    arg = 2 * np.pi * freq * t + phase
                    signals[b_name] = bias + amp * np.sin(arg)
                return exec_wavegen_sine
            elif waveform == 'Square':
                def exec_wavegen_square(t, y, dy_vec, signals):
                    arg = 2 * np.pi * freq * t + phase
                    signals[b_name] = bias + amp * signal.square(arg)
                return exec_wavegen_square
            elif waveform == 'Triangle':
                def exec_wavegen_triangle(t, y, dy_vec, signals):
                    arg = 2 * np.pi * freq * t + phase
                    signals[b_name] = bias + amp * signal.sawtooth(arg, width=0.5)
                return exec_wavegen_triangle
            elif waveform == 'Sawtooth':
                def exec_wavegen_sawtooth(t, y, dy_vec, signals):
                    arg = 2 * np.pi * freq * t + phase
                    signals[b_name] = bias + amp * signal.sawtooth(arg, width=1.0)
                return exec_wavegen_sawtooth
            else:
                # Default to sine
                def exec_wavegen_default(t, y, dy_vec, signals):
                    arg = 2 * np.pi * freq * t + phase
                    signals[b_name] = bias + amp * np.sin(arg)
                return exec_wavegen_default

        elif fn == 'Step':
            step_t = float(block.params.get('delay', 0.0))
            val = float(block.params.get('value', 1.0))
            
            def exec_step(t, y, dy_vec, signals):
                signals[b_name] = val if t >= step_t else 0.0
            return exec_step

        elif fn == 'Integrator':
            start, size = state_map[b_name]
            src = input_sources[0] if input_sources else None
            
            def exec_integrator(t, y, dy_vec, signals):
                # Output y = state
                if size == 1:
                    signals[b_name] = y[start]
                    # Derivative dx/dt = input
                    dy_vec[start] = signals.get(src, 0.0) if src else 0.0
                else:
                    signals[b_name] = y[start : start + size]
                    # Vector input?
                    # Assume input is scalar broadcast or vector
                    # For now scalar broadcast if missing logic
                    val = signals.get(src, 0.0) if src else 0.0
                    dy_vec[start : start + size] = np.atleast_1d(val).flatten()
            return exec_integrator

        elif fn in ('StateSpace', 'TransferFcn'):
            if b_name in block_matrices:
                A, B, C, D = block_matrices[b_name]
                start, size = state_map[b_name]
                src = input_sources[0] if input_sources else None
                
                # Pre-calculate dimensions for optimization?
                # B.shape[1] is input dim.
                
                def exec_ss(t, y, dy_vec, signals):
                    x = y[start : start + size].reshape(-1, 1)
                    
                    # Input
                    u_val = signals.get(src, 0.0) if src else 0.0
                    
                    # Handle shapes (simplified for 1D/Scalar)
                    # If strictly scalar system:
                    # y = C*x + D*u
                    # dx = A*x + B*u
                    # Flatten results
                    
                    # Faster scalar path?
                    if size == 1 and B.shape[1] == 1:
                        # Scalar op
                        u = u_val
                        x_s = x[0,0]
                        # A is (1,1), B is (1,1) etc
                        dx = A[0,0]*x_s + B[0,0]*u
                        y_out = C[0,0]*x_s + D[0,0]*u
                        
                        signals[b_name] = y_out
                        dy_vec[start] = dx
                    else:
                        # Vector op
                        u = np.atleast_1d(u_val).reshape(-1, 1)
                        if B.shape[1] > 1: # Broadcast
                             u = np.full((B.shape[1], 1), float(u_val))
                             
                        dx = A @ x + B @ u
                        y_out = C @ x + D @ u
                        
                        signals[b_name] = y_out if y_out.size > 1 else y_out.item()
                        dy_vec[start : start + size] = dx.flatten()
                return exec_ss
        
        elif fn == 'PID':
             start, size = state_map[b_name]
             sp_src = input_sources[0] if len(input_sources)>0 else None
             meas_src = input_sources[1] if len(input_sources)>1 else None
             
             Kp = float(block.params.get('Kp', 1.0))
             Ki = float(block.params.get('Ki', 0.0))
             Kd = float(block.params.get('Kd', 0.0))
             N = float(block.params.get('N', 20.0))
             u_min = float(block.params.get('u_min', -np.inf))
             u_max = float(block.params.get('u_max', np.inf))

             def exec_pid(t, y, dy_vec, signals):
                 sp = signals.get(sp_src, 0.0) if sp_src else 0.0
                 meas = signals.get(meas_src, 0.0) if meas_src else 0.0
                 e = sp - meas
                 
                 # x_i, x_d
                 x_i = y[start]
                 x_d = y[start + 1]
                 
                 dx_i = e
                 dx_d = N * (e - x_d)
                 d_term = Kd * dx_d
                 i_term = Ki * x_i
                 p_term = Kp * e
                 
                 u_unsat = p_term + i_term + d_term
                 u_out = np.clip(u_unsat, u_min, u_max)
                 signals[b_name] = u_out
                 
                 # Anti-windup
                 if (u_unsat > u_max and e > 0) or (u_unsat < u_min and e < 0):
                     dx_i = 0.0
                     
                 dy_vec[start] = dx_i
                 dy_vec[start+1] = dx_d
             return exec_pid

        elif fn == 'Saturation':
            src = input_sources[0] if input_sources else None
            lower = float(block.params.get('min', -np.inf))
            upper = float(block.params.get('max', np.inf))
            def exec_sat(t, y, dy_vec, signals):
                val = signals.get(src, 0.0) if src else 0.0
                signals[b_name] = np.clip(val, lower, upper)
            return exec_sat

        elif fn == 'Deadband':
            src = input_sources[0] if input_sources else None
            start_db = float(block.params.get('start', -0.5))
            end_db = float(block.params.get('end', 0.5))
            def exec_db(t, y, dy_vec, signals):
                val = signals.get(src, 0.0) if src else 0.0
                if val < start_db: out = val - start_db
                elif val > end_db: out = val - end_db
                else: out = 0.0
                signals[b_name] = out
            return exec_db
            
        elif fn == 'Exponential':
            src = input_sources[0] if input_sources else None
            a = float(block.params.get('a', 1.0))
            b_coef = float(block.params.get('b', 1.0)) # avoid local var 'b'
            def exec_exp(t, y, dy_vec, signals):
                x_in = signals.get(src, 0.0) if src else 0.0
                signals[b_name] = a * np.exp(b_coef * x_in)
            return exec_exp

        elif fn == 'Mux':
            # Create a list of inputs
            # Actually optimizations for lists??
            # We can capture 'input_sources' list directly.
            # But we need to resolve it.
            # Using list comp is fast.
            # inputs = [signals.get(src, 0.0) for src in input_sources if src else 0.0] 
            # (careful with None)
            
            # To handle None sources efficiently:
            baked_srcs = [s for s in input_sources] # Copy
            
            def exec_mux(t, y, dy_vec, signals):
                 vals = []
                 for src in baked_srcs:
                     if src: vals.append(signals.get(src, 0.0))
                     else: vals.append(0.0)
                 signals[b_name] = np.array(vals)
            return exec_mux
        
        elif fn == 'Switch':
            ctrl_src = input_sources[0] if input_sources else None
            mode = block.params.get('mode', 'threshold')
            n_inputs = int(block.params.get('n_inputs', 2))
            threshold = float(block.params.get('threshold', 0.0))
            
            def exec_switch(t, y, dy_vec, signals):
                ctrl = signals.get(ctrl_src, 0.0) if ctrl_src else 0.0
                
                if mode == 'index':
                    sel = int(round(ctrl))
                else:
                    sel = 0 if ctrl >= threshold else 1
                
                sel = max(0, min(n_inputs - 1, sel))
                
                # Fetch data input (index sel + 1)
                # input_sources indexing: 0=ctrl, 1=in1, 2=in2...
                # so actual index is sel + 1
                idx = sel + 1
                if idx < len(input_sources):
                    src = input_sources[idx]
                    val = signals.get(src, 0.0) if src else 0.0
                else:
                    val = 0.0
                signals[b_name] = val
            return exec_switch
        
        elif fn in ('SgProd', 'SigProduct'):
             # Logic for product
             baked_srcs = [s for s in input_sources]
             def exec_sgprod(t, y, dy_vec, signals):
                 res = 1.0
                 if baked_srcs:
                    for src in baked_srcs:
                        if src: res *= signals.get(src, 0.0)
                        else: res *= 0.0 # Connected to 0
                 else:
                     res = 1.0 # Or 1.0 for identity
                 signals[b_name] = res
             return exec_sgprod

        elif fn in ('Abs', 'Absblock'):
             src = input_sources[0] if input_sources else None
             def exec_abs(t, y, dy_vec, signals):
                 val = signals.get(src, 0.0) if src else 0.0
                 signals[b_name] = abs(val)
             return exec_abs

        elif fn == 'Ramp':
            slope = float(block.params.get('slope', 1.0))
            delay = float(block.params.get('delay', 0.0))
            def exec_ramp(t, y, dy_vec, signals):
                if slope > 0:
                   val = max(0.0, slope * (t - delay))
                elif slope < 0:
                   val = min(0.0, slope * (t - delay))
                else:
                   val = 0.0
                signals[b_name] = val
            return exec_ramp

        elif fn in ('Terminator', 'Display', 'Scope', 'To', 'From'):
             # Sinks or semantic tags, no runtime math for Solver
             # Scope data is collected by Replay Loop, not Solver Loop (usually)
             # But if Scope is used as input? (Shouldn't happen).
             pass
            
        elif fn == 'RateLimiter':
             start, size = state_map[b_name]
             src = input_sources[0] if input_sources else None
             rising = float(block.params.get('rising', 1.0))
             falling = float(block.params.get('falling', -1.0))
             # Stiffness gain
             K = 1000.0
             
             def exec_ratelimiter(t, y, dy_vec, signals):
                 u = signals.get(src, 0.0) if src else 0.0
                 y_val = y[start]
                 signals[b_name] = y_val
                 
                 # Rate calculation
                 err = u - y_val
                 rate = err * K
                 
                 # Clamp rate
                 dy = np.clip(rate, falling, rising)
                 dy_vec[start] = dy
             return exec_ratelimiter

        # Generic catch-all or pass
        def exec_noop(t, y, dy_vec, signals):
            pass
        return exec_noop

    def compile_system(self, blocks: List[DBlock], sorted_order: List[DBlock], lines: List[Any]) -> Tuple[Callable, np.ndarray, Dict]:
        """
        Generate a fast derivative function f(t, y) using closure-based optimization.
        """
        # 0. Re-order blocks to ensure Sources run first (because Engine hierarchy might put Memory blocks first)
        sources = []
        others = []
        for b in sorted_order:
            fn = b.block_fn.title() if b.block_fn else ''
            if fn in ('Step', 'Sine', 'Constant', 'From', 'Ramp', 'Exponential'): 
                sources.append(b)
            else:
                others.append(b)
        
        sorted_order = sources + others

        # 1. Build Dependency Graph (Static Pull-based connections)
        # map: dest_block_name -> port_idx -> (source_block_name, source_port_idx)
        input_map = {b.name: {} for b in blocks}
        
        for line in lines:
            dst_block = line.dstblock
            dst_port = line.dstport
            src_block = line.srcblock
            src_port = line.srcport
            
            input_map[dst_block][dst_port] = (src_block, src_port)

        # 2. Identify States (Integrators, StateSpace, TransferFcn)
        state_map = {} # block_name -> (start_idx, size)
        block_matrices = {} # block_name -> (A, B, C, D)
        y0_list = []
        current_state_idx = 0
        
        for block in blocks:
            b_name = block.name
            fn = block.block_fn.title() if block.block_fn else ''
            
            # Correction for CamelCase blocks if title() messed them up
            if fn == 'Statespace': fn = 'StateSpace'
            if fn in ('Transferfcn', 'Tranfn'): fn = 'TransferFcn'
            if block.block_fn == 'PID': fn = 'PID' # Keep uppercase
            if fn == 'Ratelimiter': fn = 'RateLimiter'
            
            if fn == 'Integrator':
                ic = np.array(block.params.get('init_conds', 0.0), dtype=float)
                ic_flat = np.atleast_1d(ic).flatten()
                size = ic_flat.size
                state_map[b_name] = (current_state_idx, size)
                y0_list.extend(ic_flat)
                current_state_idx += size
                
            elif fn == 'StateSpace':
                try:
                    A = np.array(block.params['A'], dtype=float)
                    B = np.array(block.params['B'], dtype=float)
                    C = np.array(block.params['C'], dtype=float)
                    D = np.array(block.params['D'], dtype=float)
                    
                    # Fix dimensions
                    n = A.shape[0] if len(A.shape) > 1 else 1 # Basic check
                    A = A.reshape(n, n)
                    
                    # Store matrices
                    block_matrices[b_name] = (A, B, C, D)
                    
                    # Init conditions
                    ic = np.array(block.params.get('init_conds', [0.0]*n), dtype=float)
                    ic_flat = np.atleast_1d(ic).flatten()
                    
                    # Resize/Pad ICs to match n
                    if len(ic_flat) < n:
                        padded = np.zeros(n)
                        padded[:len(ic_flat)] = ic_flat
                        ic_flat = padded
                    elif len(ic_flat) > n:
                        ic_flat = ic_flat[:n]
                        
                    state_map[b_name] = (current_state_idx, n)
                    y0_list.extend(ic_flat)
                    current_state_idx += n
                except Exception as e:
                    logger.error(f"Failed to compile StateSpace {b_name}: {e}")
                    raise e

            elif fn == 'TransferFcn':
                try:
                    num = block.params.get('numerator', [1.0])
                    den = block.params.get('denominator', [1.0, 1.0])
                    
                    # Convert to State Space
                    A, B, C, D = signal.tf2ss(num, den)
                    
                    block_matrices[b_name] = (A, B, C, D)
                    
                    n = A.shape[0]
                    
                     # Init conditions
                    ic = np.array(block.params.get('init_conds', [0.0]*n), dtype=float)
                    ic_flat = np.atleast_1d(ic).flatten()
                    
                    if len(ic_flat) < n:
                        padded = np.zeros(n)
                        padded[:len(ic_flat)] = ic_flat
                        ic_flat = padded
                    elif len(ic_flat) > n:
                        ic_flat = ic_flat[:n]
                        
                    state_map[b_name] = (current_state_idx, n)
                    y0_list.extend(ic_flat)
                    current_state_idx += n
                    
                except Exception as e:
                    logger.error(f"Failed to compile TransferFcn {b_name}: {e}")
                    raise e

            elif fn == 'PID':
                # PID has 2 states: Integrator (1) + Derivative Filter (1)
                # State layout: [x_i, x_d]
                state_map[b_name] = (current_state_idx, 2)
                y0_list.extend([0.0, 0.0]) # Initial conditions for PID usually 0
                current_state_idx += 2

            elif fn == 'RateLimiter':
                # State is the output y.
                state_map[b_name] = (current_state_idx, 1)
                ic = float(block.params.get('init_cond', 0.0))
                y0_list.append(ic)
                current_state_idx += 1
            
        y0 = np.array(y0_list, dtype=float)
        
        # 3. Compile Execution Sequence
        execution_sequence = []
        for block in sorted_order:
             executor = self._create_block_executor(block, input_map, state_map, block_matrices)
             execution_sequence.append(executor)
             
        # 4. Create optimized closure
        def model_func(t, y):
            signals = {} # block_name -> output_value
            dy_vec = np.zeros_like(y)
            
            # Execute all blocks
            for exec_fn in execution_sequence:
                exec_fn(t, y, dy_vec, signals)
                
            return dy_vec

        return model_func, y0, state_map, block_matrices
