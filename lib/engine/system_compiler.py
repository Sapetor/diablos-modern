
import logging
import numpy as np
from typing import List, Callable, Dict, Any, Tuple
from lib.simulation.block import DBlock
from scipy import signal
from lib.engine.pde_helpers import (
    parse_pde_initial_condition,
    parse_pde_2d_initial_condition,
    get_input_source,
    ensure_field_array
)

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
            'Noise',
            'MathFunction',
            'Selector',
            'Hysteresis',
            # PDE Blocks (Method of Lines) - 1D
            'HeatEquation1D',
            'WaveEquation1D',
            'AdvectionEquation1D',
            'DiffusionReaction1D',
            # PDE Blocks (Method of Lines) - 2D
            'HeatEquation2D',
            # Field Processing Blocks - 1D
            'FieldProbe',
            'FieldIntegral',
            'FieldMax',
            'FieldScope',
            'FieldGradient',
            'FieldLaplacian',
            # Field Processing Blocks - 2D
            'FieldProbe2D',
            'FieldScope2D',
            'FieldSlice',
            # Optimization Primitives
            'StateVariable',
            'Product',
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
        # PDE blocks normalization
        if fn == 'Heatequation1d': fn = 'Heatequation1D'
        if fn == 'Waveequation1d': fn = 'Waveequation1D'
        if fn == 'Advectionequation1d': fn = 'Advectionequation1D'
        if fn == 'Diffusionreaction1d': fn = 'Diffusionreaction1D'

        # Use resolved params if available (exec_params), otherwise fall back to params
        # This ensures workspace variables are properly resolved
        params = getattr(block, 'exec_params', None) or block.params

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
            raw_val = block.params.get('value', 0.0)
            # Handle both scalar and array values
            if isinstance(raw_val, (list, tuple)):
                val = np.atleast_1d(raw_val)
            elif hasattr(raw_val, '__iter__') and not isinstance(raw_val, str):
                val = np.atleast_1d(raw_val)
            else:
                val = float(raw_val)
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

        elif fn in ('Product', 'product'):
             # Product block with configurable * and / operations
             baked_srcs = [s for s in input_sources]
             ops = block.params.get('ops', '**')
             def exec_product(t, y, dy_vec, signals, _ops=ops, _srcs=baked_srcs):
                 res = 1.0
                 for i, src in enumerate(_srcs):
                     op = _ops[i] if i < len(_ops) else '*'
                     val = signals.get(src, 0.0) if src else 0.0
                     if op == '*':
                         res *= val
                     elif op == '/':
                         res = res / val if val != 0 else 1e308
                 signals[b_name] = res
             return exec_product

        elif fn in ('StateVariable', 'Statevariable'):
             # State variable for discrete optimization iterations
             # Uses closure-based state storage instead of ODE integration
             src = input_sources[0] if input_sources else None
             initial = block.params.get('initial_value', [1.0])
             if isinstance(initial, str):
                 try:
                     initial = eval(initial)
                 except:
                     initial = [1.0]
             # Preserve full vector state, not just first element
             initial = np.atleast_1d(initial).copy()

             # Closure state - mutable dict to allow updates
             state = {'current': initial, 'prev_t': -1.0}

             def exec_statevariable(t, y, dy_vec, signals, _src=src, _state=state):
                 # Output current state (vector or scalar)
                 current = _state['current']
                 signals[b_name] = current if current.size > 1 else float(current[0])
                 # Check if we've moved to a new time step (discrete update)
                 if t > _state['prev_t'] + 0.5:  # Allow for floating point
                     if _src and _src in signals:
                         # Update state for next iteration - preserve full vector
                         _state['current'] = np.atleast_1d(signals[_src]).copy()
                     _state['prev_t'] = t
             return exec_statevariable

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

        elif fn == 'Noise':
            mu = float(block.params.get('mu', 0.0))
            sigma = float(block.params.get('sigma', 1.0))

            def exec_noise(t, y, dy_vec, signals):
                signals[b_name] = mu + sigma * np.random.randn()
            return exec_noise

        elif fn == 'Mathfunction':
            src = input_sources[0] if input_sources else None
            # Check both 'function' and 'expression' keys for backward compatibility
            func_raw = block.params.get('function', block.params.get('expression', 'sin'))
            func = str(func_raw).lower()

            # Pre-select the function to avoid string comparison at runtime
            np_func = None
            use_expr = False
            if func == 'sin':
                np_func = np.sin
            elif func == 'cos':
                np_func = np.cos
            elif func == 'tan':
                np_func = np.tan
            elif func == 'asin':
                np_func = np.arcsin
            elif func == 'acos':
                np_func = np.arccos
            elif func == 'atan':
                np_func = np.arctan
            elif func == 'exp':
                np_func = np.exp
            elif func == 'log':
                np_func = np.log
            elif func == 'log10':
                np_func = np.log10
            elif func == 'sqrt':
                np_func = np.sqrt
            elif func == 'square':
                np_func = lambda x: x * x
            elif func == 'sign':
                np_func = np.sign
            elif func == 'abs':
                np_func = np.abs
            elif func == 'ceil':
                np_func = np.ceil
            elif func == 'floor':
                np_func = np.floor
            elif func == 'reciprocal':
                np_func = lambda x: 1.0 / x if x != 0 else 0.0
            elif func == 'cube':
                np_func = lambda x: x * x * x
            else:
                # Python expression fallback
                use_expr = True
                expr_str = str(func_raw)  # Use raw string for eval

            if use_expr:
                # Build context for eval
                eval_context = {
                    "sin": np.sin, "cos": np.cos, "tan": np.tan,
                    "asin": np.arcsin, "acos": np.arccos, "atan": np.arctan,
                    "exp": np.exp, "log": np.log, "log10": np.log10,
                    "sqrt": np.sqrt, "abs": np.abs, "sign": np.sign,
                    "ceil": np.ceil, "floor": np.floor,
                    "pi": np.pi, "e": np.e, "np": np
                }

                def exec_mathfunc_expr(t, y, dy_vec, signals, _src=src, _expr=expr_str, _ctx=eval_context):
                    val = signals.get(_src, 0.0) if _src else 0.0
                    try:
                        local_ctx = dict(_ctx)
                        local_ctx['u'] = val
                        local_ctx['t'] = t
                        signals[b_name] = float(eval(_expr, {"__builtins__": None}, local_ctx))
                    except Exception:
                        signals[b_name] = 0.0
                return exec_mathfunc_expr
            else:
                def exec_mathfunc(t, y, dy_vec, signals, _func=np_func, _src=src):
                    val = signals.get(_src, 0.0) if _src else 0.0
                    try:
                        signals[b_name] = _func(val)
                    except (ValueError, ZeroDivisionError):
                        signals[b_name] = 0.0
                return exec_mathfunc

        elif fn == 'Selector':
            src = input_sources[0] if input_sources else None
            indices_str = str(block.params.get('indices', '0'))

            # Pre-parse indices at compile time
            parsed_indices = []
            for part in indices_str.split(','):
                part = part.strip()
                if ':' in part:
                    parts = part.split(':')
                    start_idx = int(parts[0]) if parts[0] else 0
                    end_idx = int(parts[1]) if len(parts) > 1 and parts[1] else None
                    parsed_indices.append(('range', start_idx, end_idx))
                else:
                    try:
                        parsed_indices.append(('idx', int(part)))
                    except ValueError:
                        parsed_indices.append(('idx', 0))

            def exec_selector(t, y, dy_vec, signals, _indices=parsed_indices):
                val = signals.get(src, 0.0) if src else 0.0
                u = np.atleast_1d(val).flatten()
                max_len = len(u)

                result = []
                for item in _indices:
                    if item[0] == 'range':
                        start_i, end_i = item[1], item[2]
                        end_i = end_i if end_i is not None else max_len
                        result.extend(u[start_i:min(end_i, max_len)])
                    else:
                        idx = item[1]
                        if idx < 0:
                            idx = max_len + idx
                        if 0 <= idx < max_len:
                            result.append(u[idx])

                if len(result) == 1:
                    signals[b_name] = result[0]
                else:
                    signals[b_name] = np.array(result) if result else 0.0
            return exec_selector

        elif fn == 'Hysteresis':
            src = input_sources[0] if input_sources else None
            upper = float(block.params.get('upper', 0.5))
            lower = float(block.params.get('lower', -0.5))
            high_val = float(block.params.get('high', 1.0))
            low_val = float(block.params.get('low', 0.0))

            # Use mutable container for state persistence across calls
            state_holder = [low_val]  # Start with low output

            def exec_hysteresis(t, y, dy_vec, signals, _state=state_holder):
                val = signals.get(src, 0.0) if src else 0.0
                # Extract scalar if needed
                if hasattr(val, '__len__'):
                    val = float(np.atleast_1d(val)[0])

                if val >= upper:
                    _state[0] = high_val
                elif val <= lower:
                    _state[0] = low_val
                # else: retain previous state

                signals[b_name] = _state[0]
            return exec_hysteresis

        # ==================== PDE BLOCKS ====================

        elif fn == 'Heatequation1D':
            start, size = state_map[b_name]
            alpha = float(params.get('alpha', 1.0))
            L = float(params.get('L', 1.0))
            N = int(params.get('N', 20))
            dx = L / (N - 1)
            bc_type_left = params.get('bc_type_left', 'Dirichlet')
            bc_type_right = params.get('bc_type_right', 'Dirichlet')
            h_left = float(params.get('h_left', 10.0))
            h_right = float(params.get('h_right', 10.0))
            k_thermal = float(params.get('k_thermal', 1.0))

            q_src_key = input_sources[0] if len(input_sources) > 0 else None
            bc_left_key = input_sources[1] if len(input_sources) > 1 else None
            bc_right_key = input_sources[2] if len(input_sources) > 2 else None

            def exec_heat1d(t, y, dy_vec, signals,
                           _start=start, _N=N, _alpha=alpha, _dx=dx,
                           _bc_type_left=bc_type_left, _bc_type_right=bc_type_right,
                           _h_left=h_left, _h_right=h_right, _k=k_thermal,
                           _q_key=q_src_key, _bc_l_key=bc_left_key, _bc_r_key=bc_right_key):
                T = y[_start:_start + _N]

                # Get inputs
                q_src = signals.get(_q_key, 0.0) if _q_key else 0.0
                bc_left_val = signals.get(_bc_l_key, 0.0) if _bc_l_key else 0.0
                bc_right_val = signals.get(_bc_r_key, 0.0) if _bc_r_key else 0.0

                # Ensure q_src is array
                if isinstance(q_src, (int, float)):
                    q_src = np.full(_N, float(q_src))
                else:
                    q_src = np.atleast_1d(q_src).flatten()
                    if len(q_src) != _N:
                        q_src = np.full(_N, q_src[0] if len(q_src) > 0 else 0.0)

                dT_dt = np.zeros(_N)
                dx_sq = _dx * _dx

                # Interior nodes: central difference
                for i in range(1, _N-1):
                    d2T_dx2 = (T[i+1] - 2*T[i] + T[i-1]) / dx_sq
                    dT_dt[i] = _alpha * d2T_dx2 + q_src[i]

                # Left boundary
                if _bc_type_left == 'Dirichlet':
                    # Force boundary to match input value using penalty method
                    dT_dt[0] = 1000.0 * (bc_left_val - T[0])
                elif _bc_type_left == 'Neumann':
                    d2T_dx2 = (2*T[1] - 2*T[0] - 2*_dx*bc_left_val) / dx_sq
                    dT_dt[0] = _alpha * d2T_dx2 + q_src[0]
                elif _bc_type_left == 'Robin':
                    # Robin BC: k*dT/dx = h*(T_inf - T)
                    dT_dt[0] = 1000.0 * (bc_left_val - T[0])

                # Right boundary
                if _bc_type_right == 'Dirichlet':
                    # Force boundary to match input value using penalty method
                    dT_dt[_N-1] = 1000.0 * (bc_right_val - T[_N-1])
                elif _bc_type_right == 'Neumann':
                    d2T_dx2 = (2*T[_N-2] - 2*T[_N-1] + 2*_dx*bc_right_val) / dx_sq
                    dT_dt[_N-1] = _alpha * d2T_dx2 + q_src[_N-1]
                elif _bc_type_right == 'Robin':
                    dT_dt[_N-1] = 1000.0 * (bc_right_val - T[_N-1])

                # Output: temperature field and average
                signals[b_name] = T
                signals[b_name + '_avg'] = np.mean(T)

                dy_vec[_start:_start + _N] = dT_dt
            return exec_heat1d

        elif fn == 'Waveequation1D':
            start, size = state_map[b_name]
            c = float(params.get('c', 1.0))
            damping = float(params.get('damping', 0.0))
            L = float(params.get('L', 1.0))
            N = int(params.get('N', 50))
            dx = L / (N - 1)
            bc_type_left = params.get('bc_type_left', 'Dirichlet')
            bc_type_right = params.get('bc_type_right', 'Dirichlet')

            force_key = input_sources[0] if len(input_sources) > 0 else None
            bc_left_key = input_sources[1] if len(input_sources) > 1 else None
            bc_right_key = input_sources[2] if len(input_sources) > 2 else None

            def exec_wave1d(t, y, dy_vec, signals,
                           _start=start, _N=N, _c=c, _damping=damping, _dx=dx,
                           _bc_type_left=bc_type_left, _bc_type_right=bc_type_right,
                           _f_key=force_key, _bc_l_key=bc_left_key, _bc_r_key=bc_right_key):
                u = y[_start:_start + _N]
                v = y[_start + _N:_start + 2*_N]

                # Get inputs
                force = signals.get(_f_key, 0.0) if _f_key else 0.0
                bc_left = signals.get(_bc_l_key, 0.0) if _bc_l_key else 0.0
                bc_right = signals.get(_bc_r_key, 0.0) if _bc_r_key else 0.0

                if isinstance(force, (int, float)):
                    force = np.full(_N, float(force))
                else:
                    force = np.atleast_1d(force).flatten()
                    if len(force) != _N:
                        force = np.full(_N, force[0] if len(force) > 0 else 0.0)

                c_sq = _c * _c
                dx_sq = _dx * _dx

                du_dt = v.copy()
                dv_dt = np.zeros(_N)

                # Interior
                for i in range(1, _N-1):
                    d2u_dx2 = (u[i+1] - 2*u[i] + u[i-1]) / dx_sq
                    dv_dt[i] = c_sq * d2u_dx2 - _damping * v[i] + force[i]

                # Boundaries
                if _bc_type_left == 'Dirichlet':
                    du_dt[0] = 0.0
                    dv_dt[0] = 0.0
                elif _bc_type_left == 'Neumann':
                    d2u_dx2 = (2*u[1] - 2*u[0] - 2*_dx*bc_left) / dx_sq
                    dv_dt[0] = c_sq * d2u_dx2 - _damping * v[0] + force[0]

                if _bc_type_right == 'Dirichlet':
                    du_dt[_N-1] = 0.0
                    dv_dt[_N-1] = 0.0
                elif _bc_type_right == 'Neumann':
                    d2u_dx2 = (2*u[_N-2] - 2*u[_N-1] + 2*_dx*bc_right) / dx_sq
                    dv_dt[_N-1] = c_sq * d2u_dx2 - _damping * v[_N-1] + force[_N-1]

                signals[b_name] = u
                signals[b_name + '_v'] = v

                dy_vec[_start:_start + _N] = du_dt
                dy_vec[_start + _N:_start + 2*_N] = dv_dt
            return exec_wave1d

        elif fn == 'Advectionequation1D':
            start, size = state_map[b_name]
            velocity = float(params.get('velocity', 1.0))
            L = float(params.get('L', 1.0))
            N = int(params.get('N', 50))
            dx = L / (N - 1)
            bc_type = params.get('bc_type', 'Dirichlet')

            inlet_key = input_sources[0] if len(input_sources) > 0 else None

            def exec_advection1d(t, y, dy_vec, signals,
                                _start=start, _N=N, _v=velocity, _dx=dx,
                                _bc_type=bc_type, _inlet_key=inlet_key):
                c = y[_start:_start + _N]

                c_inlet = signals.get(_inlet_key, 0.0) if _inlet_key else 0.0

                dc_dt = np.zeros(_N)

                if _v >= 0:
                    # Second-order backward difference (upwind) - reduces numerical diffusion
                    # Interior: (3*c[i] - 4*c[i-1] + c[i-2]) / (2*dx)
                    for i in range(2, _N):
                        dc_dx = (3*c[i] - 4*c[i-1] + c[i-2]) / (2*_dx)
                        dc_dt[i] = -_v * dc_dx
                    # First interior point: first-order fallback
                    if _N > 1:
                        dc_dx = (c[1] - c[0]) / _dx
                        dc_dt[1] = -_v * dc_dx
                    if _bc_type == 'Dirichlet':
                        dc_dt[0] = 1000.0 * (c_inlet - c[0])  # Penalty method for inlet BC
                    elif _bc_type == 'Periodic':
                        dc_dx = (3*c[0] - 4*c[_N-1] + c[_N-2]) / (2*_dx)
                        dc_dt[0] = -_v * dc_dx
                else:
                    # Second-order forward difference (upwind)
                    # Interior: (-3*c[i] + 4*c[i+1] - c[i+2]) / (2*dx)
                    for i in range(_N-2):
                        dc_dx = (-3*c[i] + 4*c[i+1] - c[i+2]) / (2*_dx)
                        dc_dt[i] = -_v * dc_dx
                    # Last interior point: first-order fallback
                    if _N > 1:
                        dc_dx = (c[_N-1] - c[_N-2]) / _dx
                        dc_dt[_N-2] = -_v * dc_dx
                    if _bc_type == 'Dirichlet':
                        dc_dt[_N-1] = 1000.0 * (c_inlet - c[_N-1])  # Penalty method for outlet BC
                    elif _bc_type == 'Periodic':
                        dc_dx = (-3*c[_N-1] + 4*c[0] - c[1]) / (2*_dx)
                        dc_dt[_N-1] = -_v * dc_dx

                signals[b_name] = c
                signals[b_name + '_total'] = np.sum(c) * _dx

                dy_vec[_start:_start + _N] = dc_dt
            return exec_advection1d

        elif fn == 'Diffusionreaction1D':
            start, size = state_map[b_name]
            D = float(params.get('D', 0.01))
            k = float(params.get('k', 0.1))
            n = int(params.get('n', 1))
            L = float(params.get('L', 1.0))
            N = int(params.get('N', 30))
            dx = L / (N - 1)
            bc_type_left = params.get('bc_type_left', 'Dirichlet')
            bc_type_right = params.get('bc_type_right', 'Neumann')

            src_key = input_sources[0] if len(input_sources) > 0 else None
            bc_left_key = input_sources[1] if len(input_sources) > 1 else None
            bc_right_key = input_sources[2] if len(input_sources) > 2 else None

            def exec_diffreact1d(t, y, dy_vec, signals,
                                _start=start, _N=N, _D=D, _k=k, _n=n, _dx=dx,
                                _bc_type_left=bc_type_left, _bc_type_right=bc_type_right,
                                _s_key=src_key, _bc_l_key=bc_left_key, _bc_r_key=bc_right_key):
                c = y[_start:_start + _N]

                source = signals.get(_s_key, 0.0) if _s_key else 0.0
                bc_left = signals.get(_bc_l_key, 0.0) if _bc_l_key else 0.0
                bc_right = signals.get(_bc_r_key, 0.0) if _bc_r_key else 0.0

                if isinstance(source, (int, float)):
                    source = np.full(_N, float(source))
                else:
                    source = np.atleast_1d(source).flatten()
                    if len(source) != _N:
                        source = np.full(_N, source[0] if len(source) > 0 else 0.0)

                dc_dt = np.zeros(_N)
                dx_sq = _dx * _dx

                # Interior
                for i in range(1, _N-1):
                    d2c_dx2 = (c[i+1] - 2*c[i] + c[i-1]) / dx_sq
                    reaction = _k * np.power(max(c[i], 0), _n)
                    dc_dt[i] = _D * d2c_dx2 - reaction + source[i]

                # Boundaries - use penalty method for Dirichlet to force value
                if _bc_type_left == 'Dirichlet':
                    dc_dt[0] = 1000.0 * (bc_left - c[0])  # Force c[0] → bc_left
                elif _bc_type_left == 'Neumann':
                    d2c_dx2 = (2*c[1] - 2*c[0] - 2*_dx*bc_left) / dx_sq
                    reaction = _k * np.power(max(c[0], 0), _n)
                    dc_dt[0] = _D * d2c_dx2 - reaction + source[0]

                if _bc_type_right == 'Dirichlet':
                    dc_dt[_N-1] = 1000.0 * (bc_right - c[_N-1])  # Force c[N-1] → bc_right
                elif _bc_type_right == 'Neumann':
                    d2c_dx2 = (2*c[_N-2] - 2*c[_N-1] + 2*_dx*bc_right) / dx_sq
                    reaction = _k * np.power(max(c[_N-1], 0), _n)
                    dc_dt[_N-1] = _D * d2c_dx2 - reaction + source[_N-1]

                signals[b_name] = c
                signals[b_name + '_total'] = np.sum(c) * _dx

                dy_vec[_start:_start + _N] = dc_dt
            return exec_diffreact1d

        # ==================== 2D PDE BLOCKS ====================

        elif fn == 'Heatequation2D':
            start, size = state_map[b_name]
            alpha = float(params.get('alpha', 0.01))
            Lx = float(params.get('Lx', 1.0))
            Ly = float(params.get('Ly', 1.0))
            Nx = int(params.get('Nx', 20))
            Ny = int(params.get('Ny', 20))
            dx = Lx / (Nx - 1)
            dy = Ly / (Ny - 1)
            bc_type_left = params.get('bc_type_left', 'Dirichlet')
            bc_type_right = params.get('bc_type_right', 'Dirichlet')
            bc_type_bottom = params.get('bc_type_bottom', 'Dirichlet')
            bc_type_top = params.get('bc_type_top', 'Dirichlet')

            q_src_key = input_sources[0] if len(input_sources) > 0 else None
            bc_left_key = input_sources[1] if len(input_sources) > 1 else None
            bc_right_key = input_sources[2] if len(input_sources) > 2 else None
            bc_bottom_key = input_sources[3] if len(input_sources) > 3 else None
            bc_top_key = input_sources[4] if len(input_sources) > 4 else None

            def exec_heat2d(t, y, dy_vec, signals,
                           _start=start, _Nx=Nx, _Ny=Ny, _alpha=alpha, _dx=dx, _dy=dy,
                           _bc_type_left=bc_type_left, _bc_type_right=bc_type_right,
                           _bc_type_bottom=bc_type_bottom, _bc_type_top=bc_type_top,
                           _q_key=q_src_key, _bc_l_key=bc_left_key, _bc_r_key=bc_right_key,
                           _bc_b_key=bc_bottom_key, _bc_t_key=bc_top_key):
                n_states = _Nx * _Ny
                T_flat = y[_start:_start + n_states]
                T = T_flat.reshape((_Ny, _Nx))

                # Get inputs
                q_src = signals.get(_q_key, 0.0) if _q_key else 0.0
                bc_left = signals.get(_bc_l_key, 0.0) if _bc_l_key else 0.0
                bc_right = signals.get(_bc_r_key, 0.0) if _bc_r_key else 0.0
                bc_bottom = signals.get(_bc_b_key, 0.0) if _bc_b_key else 0.0
                bc_top = signals.get(_bc_t_key, 0.0) if _bc_t_key else 0.0

                # Ensure q_src is scalar (simplified)
                if isinstance(q_src, np.ndarray):
                    q_src = float(q_src.flat[0]) if q_src.size > 0 else 0.0

                dT_dt = np.zeros((_Ny, _Nx))
                dx_sq = _dx * _dx
                dy_sq = _dy * _dy
                penalty = 1000.0

                # Interior nodes: 5-point stencil
                for j in range(1, _Ny - 1):
                    for i in range(1, _Nx - 1):
                        d2Tdx2 = (T[j, i+1] - 2*T[j, i] + T[j, i-1]) / dx_sq
                        d2Tdy2 = (T[j+1, i] - 2*T[j, i] + T[j-1, i]) / dy_sq
                        dT_dt[j, i] = _alpha * (d2Tdx2 + d2Tdy2) + q_src

                # Left boundary (i=0)
                if _bc_type_left == 'Dirichlet':
                    for j in range(_Ny):
                        dT_dt[j, 0] = penalty * (bc_left - T[j, 0])
                else:  # Neumann
                    for j in range(1, _Ny - 1):
                        d2Tdx2 = (2*T[j, 1] - 2*T[j, 0] - 2*_dx*bc_left) / dx_sq
                        d2Tdy2 = (T[j+1, 0] - 2*T[j, 0] + T[j-1, 0]) / dy_sq
                        dT_dt[j, 0] = _alpha * (d2Tdx2 + d2Tdy2) + q_src

                # Right boundary (i=Nx-1)
                if _bc_type_right == 'Dirichlet':
                    for j in range(_Ny):
                        dT_dt[j, _Nx-1] = penalty * (bc_right - T[j, _Nx-1])
                else:  # Neumann
                    for j in range(1, _Ny - 1):
                        d2Tdx2 = (2*T[j, _Nx-2] - 2*T[j, _Nx-1] + 2*_dx*bc_right) / dx_sq
                        d2Tdy2 = (T[j+1, _Nx-1] - 2*T[j, _Nx-1] + T[j-1, _Nx-1]) / dy_sq
                        dT_dt[j, _Nx-1] = _alpha * (d2Tdx2 + d2Tdy2) + q_src

                # Bottom boundary (j=0)
                if _bc_type_bottom == 'Dirichlet':
                    for i in range(_Nx):
                        dT_dt[0, i] = penalty * (bc_bottom - T[0, i])
                else:  # Neumann
                    for i in range(1, _Nx - 1):
                        d2Tdx2 = (T[0, i+1] - 2*T[0, i] + T[0, i-1]) / dx_sq
                        d2Tdy2 = (2*T[1, i] - 2*T[0, i] - 2*_dy*bc_bottom) / dy_sq
                        dT_dt[0, i] = _alpha * (d2Tdx2 + d2Tdy2) + q_src

                # Top boundary (j=Ny-1)
                if _bc_type_top == 'Dirichlet':
                    for i in range(_Nx):
                        dT_dt[_Ny-1, i] = penalty * (bc_top - T[_Ny-1, i])
                else:  # Neumann
                    for i in range(1, _Nx - 1):
                        d2Tdx2 = (T[_Ny-1, i+1] - 2*T[_Ny-1, i] + T[_Ny-1, i-1]) / dx_sq
                        d2Tdy2 = (2*T[_Ny-2, i] - 2*T[_Ny-1, i] + 2*_dy*bc_top) / dy_sq
                        dT_dt[_Ny-1, i] = _alpha * (d2Tdx2 + d2Tdy2) + q_src

                # Output: temperature field (2D), average, max
                signals[b_name] = T
                signals[b_name + '_avg'] = np.mean(T)
                signals[b_name + '_max'] = np.max(T)

                dy_vec[_start:_start + n_states] = dT_dt.flatten()
            return exec_heat2d

        # ==================== FIELD PROCESSING BLOCKS ====================

        elif fn == 'Fieldprobe':
            src = input_sources[0] if input_sources else None
            pos_src = input_sources[1] if len(input_sources) > 1 else None
            position = float(block.params.get('position', 0.5))
            mode = block.params.get('position_mode', 'normalized')
            L = float(block.params.get('L', 1.0))

            def exec_fieldprobe(t, y, dy_vec, signals, _src=src, _pos_src=pos_src,
                               _position=position, _mode=mode, _L=L):
                field = signals.get(_src, np.array([0.0])) if _src else np.array([0.0])
                field = np.atleast_1d(field).flatten()

                if len(field) == 0:
                    signals[b_name] = 0.0
                    return

                pos = signals.get(_pos_src, _position) if _pos_src else _position

                if _mode == 'absolute':
                    pos_norm = pos / _L
                else:
                    pos_norm = pos

                pos_norm = np.clip(pos_norm, 0.0, 1.0)
                N = len(field)
                idx_float = pos_norm * (N - 1)
                idx_low = int(np.floor(idx_float))
                idx_high = min(idx_low + 1, N - 1)
                frac = idx_float - idx_low

                value = field[idx_low] * (1 - frac) + field[idx_high] * frac
                signals[b_name] = float(value)
            return exec_fieldprobe

        elif fn == 'Fieldintegral':
            src = input_sources[0] if input_sources else None
            L = float(block.params.get('L', 1.0))
            normalize = block.params.get('normalize', False)

            def exec_fieldintegral(t, y, dy_vec, signals, _src=src, _L=L, _norm=normalize):
                field = signals.get(_src, np.array([0.0])) if _src else np.array([0.0])
                field = np.atleast_1d(field).flatten()

                if len(field) == 0:
                    signals[b_name] = 0.0
                    return

                N = len(field)
                dx = _L / (N - 1) if N > 1 else _L
                integral = np.trapz(field, dx=dx)

                if _norm:
                    integral = integral / _L

                signals[b_name] = float(integral)
            return exec_fieldintegral

        elif fn == 'Fieldmax':
            src = input_sources[0] if input_sources else None
            mode = block.params.get('mode', 'max')
            L = float(block.params.get('L', 1.0))

            def exec_fieldmax(t, y, dy_vec, signals, _src=src, _mode=mode, _L=L):
                field = signals.get(_src, np.array([0.0])) if _src else np.array([0.0])
                field = np.atleast_1d(field).flatten()

                if len(field) == 0:
                    signals[b_name] = 0.0
                    return

                if _mode == 'min':
                    idx = int(np.argmin(field))
                else:
                    idx = int(np.argmax(field))

                value = field[idx]
                N = len(field)
                location = (idx / (N - 1)) * _L if N > 1 else 0.0

                signals[b_name] = float(value)
                signals[b_name + '_loc'] = float(location)
                signals[b_name + '_idx'] = idx
            return exec_fieldmax

        elif fn == 'Fieldgradient':
            src = input_sources[0] if input_sources else None
            L = float(block.params.get('L', 1.0))

            def exec_fieldgradient(t, y, dy_vec, signals, _src=src, _L=L):
                field = signals.get(_src, np.array([0.0])) if _src else np.array([0.0])
                field = np.atleast_1d(field).flatten()

                if len(field) < 2:
                    signals[b_name] = np.array([0.0])
                    return

                N = len(field)
                dx = _L / (N - 1)
                gradient = np.gradient(field, dx)
                signals[b_name] = gradient
            return exec_fieldgradient

        elif fn == 'Fieldlaplacian':
            src = input_sources[0] if input_sources else None
            L = float(block.params.get('L', 1.0))

            def exec_fieldlaplacian(t, y, dy_vec, signals, _src=src, _L=L):
                field = signals.get(_src, np.array([0.0])) if _src else np.array([0.0])
                field = np.atleast_1d(field).flatten()

                if len(field) < 3:
                    signals[b_name] = np.zeros(len(field))
                    return

                N = len(field)
                dx = _L / (N - 1)
                dx_sq = dx * dx

                laplacian = np.zeros(N)
                for i in range(1, N-1):
                    laplacian[i] = (field[i+1] - 2*field[i] + field[i-1]) / dx_sq

                laplacian[0] = (field[2] - 2*field[1] + field[0]) / dx_sq
                laplacian[N-1] = (field[N-1] - 2*field[N-2] + field[N-3]) / dx_sq

                signals[b_name] = laplacian
            return exec_fieldlaplacian

        elif fn == 'Fieldscope':
            # FieldScope is a sink - just pass through
            src = input_sources[0] if input_sources else None

            def exec_fieldscope(t, y, dy_vec, signals, _src=src):
                field = signals.get(_src, np.array([0.0])) if _src else np.array([0.0])
                signals[b_name] = np.atleast_1d(field).flatten()
            return exec_fieldscope

        # ==================== 2D FIELD PROCESSING BLOCKS ====================

        elif fn == 'Fieldprobe2D':
            src = input_sources[0] if input_sources else None
            x_pos_src = input_sources[1] if len(input_sources) > 1 else None
            y_pos_src = input_sources[2] if len(input_sources) > 2 else None
            x_position = float(block.params.get('x_position', 0.5))
            y_position = float(block.params.get('y_position', 0.5))
            position_mode = block.params.get('position_mode', 'normalized')
            Lx = float(block.params.get('Lx', 1.0))
            Ly = float(block.params.get('Ly', 1.0))

            def exec_fieldprobe2d(t, y, dy_vec, signals, _src=src,
                                  _x_pos_src=x_pos_src, _y_pos_src=y_pos_src,
                                  _x_pos=x_position, _y_pos=y_position,
                                  _mode=position_mode, _Lx=Lx, _Ly=Ly):
                field = signals.get(_src, None) if _src else None
                if field is None:
                    signals[b_name] = 0.0
                    return

                field = np.atleast_2d(field)
                Ny, Nx = field.shape

                # Get positions
                x_pos = signals.get(_x_pos_src, _x_pos) if _x_pos_src else _x_pos
                y_pos = signals.get(_y_pos_src, _y_pos) if _y_pos_src else _y_pos

                # Convert to normalized
                if _mode == 'absolute':
                    x_norm = x_pos / _Lx
                    y_norm = y_pos / _Ly
                else:
                    x_norm = x_pos
                    y_norm = y_pos

                x_norm = max(0, min(1, x_norm))
                y_norm = max(0, min(1, y_norm))

                # Bilinear interpolation
                i_float = x_norm * (Nx - 1)
                j_float = y_norm * (Ny - 1)
                i0 = int(np.floor(i_float))
                i1 = min(i0 + 1, Nx - 1)
                j0 = int(np.floor(j_float))
                j1 = min(j0 + 1, Ny - 1)
                di = i_float - i0
                dj = j_float - j0

                val = (field[j0, i0] * (1 - di) * (1 - dj) +
                       field[j0, i1] * di * (1 - dj) +
                       field[j1, i0] * (1 - di) * dj +
                       field[j1, i1] * di * dj)

                signals[b_name] = float(val)
            return exec_fieldprobe2d

        elif fn == 'Fieldscope2D':
            # FieldScope2D is a sink - pass through 2D field
            src = input_sources[0] if input_sources else None

            def exec_fieldscope2d(t, y, dy_vec, signals, _src=src):
                field = signals.get(_src, np.zeros((1, 1))) if _src else np.zeros((1, 1))
                signals[b_name] = np.atleast_2d(field)
            return exec_fieldscope2d

        elif fn == 'Fieldslice':
            src = input_sources[0] if input_sources else None
            pos_src = input_sources[1] if len(input_sources) > 1 else None
            slice_direction = block.params.get('slice_direction', 'x')
            slice_position = float(block.params.get('slice_position', 0.5))

            def exec_fieldslice(t, y, dy_vec, signals, _src=src, _pos_src=pos_src,
                               _direction=slice_direction, _pos=slice_position):
                field = signals.get(_src, None) if _src else None
                if field is None:
                    signals[b_name] = np.array([0.0])
                    return

                field = np.atleast_2d(field)
                Ny, Nx = field.shape

                position = signals.get(_pos_src, _pos) if _pos_src else _pos

                if _direction.lower() == 'x':
                    j = int(position * (Ny - 1))
                    j = max(0, min(Ny - 1, j))
                    slice_arr = field[j, :]
                else:
                    i = int(position * (Nx - 1))
                    i = max(0, min(Nx - 1, i))
                    slice_arr = field[:, i]

                signals[b_name] = slice_arr
            return exec_fieldslice

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
            if fn in ('Step', 'Sine', 'Constant', 'From', 'Ramp', 'Exponential', 'Noise', 'Wavegenerator'):
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
            # PDE blocks normalization
            if fn == 'Heatequation1d': fn = 'Heatequation1D'
            if fn == 'Waveequation1d': fn = 'Waveequation1D'
            if fn == 'Advectionequation1d': fn = 'Advectionequation1D'
            if fn == 'Diffusionreaction1d': fn = 'Diffusionreaction1D'
            
            if fn == 'Integrator':
                ic = np.array(block.params.get('init_conds', 0.0), dtype=float)
                ic_flat = np.atleast_1d(ic).flatten()
                size = ic_flat.size
                state_map[b_name] = (current_state_idx, size)
                y0_list.extend(ic_flat)
                current_state_idx += size

            elif fn in ('StateVariable', 'Statevariable'):
                # StateVariable uses closure-based state, not ODE state
                # State is managed directly in the executor closure
                pass

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

            # ==================== PDE BLOCKS STATE ALLOCATION ====================
            # NOTE: PDE blocks need resolved params (exec_params) for initial conditions
            # because they may be set dynamically or through workspace variables

            elif fn == 'Heatequation1D':
                # HeatEquation1D has N states (one per spatial node)
                pde_params = getattr(block, 'exec_params', None) or block.params
                N = int(pde_params.get('N', 20))
                L = float(pde_params.get('L', 1.0))
                state_map[b_name] = (current_state_idx, N)

                # Get initial conditions using helper
                ic = pde_params.get('init_conds', [0.0])
                ic_flat = parse_pde_initial_condition(ic, N, L, pde_type='heat')

                y0_list.extend(ic_flat)
                current_state_idx += N

            elif fn == 'Waveequation1D':
                # WaveEquation1D has 2N states (N displacement + N velocity)
                pde_params = getattr(block, 'exec_params', None) or block.params
                N = int(pde_params.get('N', 50))
                L = float(pde_params.get('L', 1.0))
                state_map[b_name] = (current_state_idx, 2 * N)

                # Initial displacement using helper
                init_u = pde_params.get('init_displacement', [0.0])
                u0 = parse_pde_initial_condition(init_u, N, L, pde_type='wave')

                # Initial velocity using helper
                init_v = pde_params.get('init_velocity', [0.0])
                v0 = parse_pde_initial_condition(init_v, N, L, pde_type='wave')

                y0_list.extend(u0)
                y0_list.extend(v0)
                current_state_idx += 2 * N

            elif fn == 'Advectionequation1D':
                # AdvectionEquation1D has N states
                pde_params = getattr(block, 'exec_params', None) or block.params
                N = int(pde_params.get('N', 50))
                L = float(pde_params.get('L', 1.0))
                state_map[b_name] = (current_state_idx, N)

                # Get initial conditions using helper
                ic = pde_params.get('init_conds', [0.0])
                c0 = parse_pde_initial_condition(ic, N, L, pde_type='advection')

                y0_list.extend(c0)
                current_state_idx += N

            elif fn == 'Diffusionreaction1D':
                # DiffusionReaction1D has N states
                pde_params = getattr(block, 'exec_params', None) or block.params
                N = int(pde_params.get('N', 30))
                L = float(pde_params.get('L', 1.0))
                state_map[b_name] = (current_state_idx, N)

                # Get initial conditions using helper
                ic = pde_params.get('init_conds', [1.0])
                c0 = parse_pde_initial_condition(ic, N, L, pde_type='diffusion_reaction')

                y0_list.extend(c0)
                current_state_idx += N

            # ==================== 2D PDE BLOCKS STATE ALLOCATION ====================

            elif fn == 'Heatequation2D':
                # HeatEquation2D has Nx*Ny states (one per spatial node)
                pde_params = getattr(block, 'exec_params', None) or block.params
                Nx = int(pde_params.get('Nx', 20))
                Ny = int(pde_params.get('Ny', 20))
                Lx = float(pde_params.get('Lx', 1.0))
                Ly = float(pde_params.get('Ly', 1.0))
                n_states = Nx * Ny
                state_map[b_name] = (current_state_idx, n_states)

                # Get initial temperature using 2D helper
                init_temp = pde_params.get('init_temp', '0.0')
                amplitude = float(pde_params.get('init_amplitude', 1.0))
                T0 = parse_pde_2d_initial_condition(init_temp, Nx, Ny, Lx, Ly, amplitude)

                ic_flat = T0.flatten()
                y0_list.extend(ic_flat)
                current_state_idx += n_states

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
