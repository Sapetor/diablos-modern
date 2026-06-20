
import logging
import numpy as np
from typing import List, Callable, Dict, Any, Tuple
from lib.simulation.block import DBlock
from scipy import signal
from lib.engine.pde_helpers import (
    parse_pde_initial_condition,
    parse_pde_2d_initial_condition
)
from lib.engine.block_names import canonical_fn
from lib.engine.compiler_kernels import BuildContext, get_kernel_builder

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
            'Gain', 'MatrixGain', 'Matrixgain',
            'Sum',
            'Constant',
            'Sine',
            'Step',
            # 'Impulse' — excluded: the compiled path models the impulse as a
            # dt*1e-3-wide rectangular pulse, which adaptive RK45 can step over
            # entirely (the response is then silently lost). Falls back to the
            # interpreted path (blocks/impulse.py), which fires a correct
            # value/dt sample on the fixed-dt grid. Step(type='impulse') is
            # likewise gated out in check_compilability below.
            'TransferFcn',
            'TranFn',
            'StateSpace',
            'Mux',
            'Demux',
            'LogicalOperator',
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
            # 'Noise' — excluded: np.random in the ODE RHS is re-sampled on every
            # solve_ivp stage/rejected step, breaking adaptive error control and
            # reproducibility. Falls back to the interpreted path (blocks/noise.py).
            'PRBS',
            'MathFunction',
            'Selector',
            # 'Hysteresis' — excluded: the relay latch is path-dependent and cannot
            # be a pure function of (t, y). solve_ivp probes the RHS at non-accepted,
            # non-monotonic times, corrupting the latch. Falls back to the
            # interpreted path (blocks/hysteresis.py).
            # PDE Blocks (Method of Lines) - 1D
            'HeatEquation1D',
            'WaveEquation1D',
            'AdvectionEquation1D',
            'DiffusionReaction1D',
            # PDE Blocks (Method of Lines) - 2D
            'HeatEquation2D',
            'WaveEquation2D',
            'AdvectionEquation2D',
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
            # 'StateVariable' — excluded: it performs a discrete state update inside
            # the continuous ODE RHS keyed on a monotonic-time assumption that
            # solve_ivp violates (repeated/rejected/non-monotonic probe times).
            # Falls back to the interpreted path
            # (blocks/optimization_primitives/state_variable.py).
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

            # Step's 'impulse' subtype is a narrow Dirac approximation the
            # adaptive compiled solver can step over; force the interpreter path
            # (same rationale as the excluded Impulse block in COMPILABLE_BLOCKS).
            if b_type == 'Step' and getattr(block, 'params', {}).get('type') == 'impulse':
                logger.debug(f"Block {block.name} (Step/impulse) is not compilable; using interpreter.")
                return False

            # Accept the block if its name matches the allowlist directly or
            # via case correction (TitleCase, e.g. Sine; or UPPER, e.g. PID).
            if (b_type not in self.COMPILABLE_BLOCKS
                    and b_type.title() not in self.COMPILABLE_BLOCKS
                    and b_type.upper() not in self.COMPILABLE_BLOCKS):
                logger.debug(f"Block {block.name} ({block.block_fn}) is not compilable.")
                return False

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
        
        # Normalize Function Name (single source of truth: lib.engine.block_names)
        fn = canonical_fn(block.block_fn)

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
        
        # input_sources holds the signal-dict key to read for each input port,
        # ordered by port index. For a source's port 0 the key is just the
        # source block name; for a secondary output port we use the
        # "{src_name}_out{src_port}" convention (matches the interpreter replay
        # loop and the Demux/PDE secondary outputs). Unconnected -> None (0.0).
        input_sources = []  # List of signal keys (or None) ordered by dst port
        max_port = max(sorted_ports) if sorted_ports else -1
        for i in range(max_port + 1):
            if i in deps:
                src_name, src_port = deps[i]
                if src_port:
                    input_sources.append(f"{src_name}_out{src_port}")
                else:
                    input_sources.append(src_name)
            else:
                input_sources.append(None) # None means 0.0 default

        # --- Block-specific closures ---
        # Every compilable block family is registered in lib.engine.compiler_kernels;
        # dispatch to its builder. Unknown fn-names fall through to the no-op below.
        builder = get_kernel_builder(fn)
        if builder is not None:
            ctx = BuildContext(
                block=block, b_name=b_name, fn=fn, params=params,
                input_sources=input_sources, deps=deps,
                state_map=state_map, block_matrices=block_matrices,
            )
            return builder(ctx)

        # Generic catch-all: blocks with no registered kernel do nothing in the
        # solver loop (sinks/tags, or anything not on the compilable allowlist).
        def exec_noop(t, y, dy_vec, signals):
            pass
        return exec_noop

    def compile_system(self, blocks: List[DBlock], sorted_order: List[DBlock], lines: List[Any]) -> Tuple[Callable, np.ndarray, Dict]:
        """
        Generate a fast derivative function f(t, y) using closure-based optimization.
        """
        # 0. Block ordering is deferred to after state identification (section 2)
        # so we can inspect D matrices to classify D=0 vs D≠0 state blocks.

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
            fn = canonical_fn(block.block_fn)
            
            # Use resolved params if available (exec_params), otherwise fall
            # back to raw params. exec_params is populated by
            # SimulationEngine.run_compiled_simulation before compile_system is
            # called, and contains workspace variables resolved to numeric
            # values. Reading raw params here would misbuild block_matrices
            # (and therefore the D!=0 vs D=0 classification for execution
            # ordering) whenever a state block is parameterised by a workspace
            # variable. Same convention used by the PDE branches below.
            sparams = getattr(block, 'exec_params', None) or block.params

            if fn == 'Integrator':
                ic = np.array(sparams.get('init_conds', 0.0), dtype=float)
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
                    A = np.array(sparams['A'], dtype=float)
                    B = np.array(sparams['B'], dtype=float)
                    C = np.array(sparams['C'], dtype=float)
                    D = np.array(sparams['D'], dtype=float)

                    # Fix dimensions
                    n = A.shape[0] if len(A.shape) > 1 else 1 # Basic check
                    A = A.reshape(n, n)

                    # Store matrices
                    block_matrices[b_name] = (A, B, C, D)

                    # Init conditions
                    ic = np.array(sparams.get('init_conds', [0.0]*n), dtype=float)
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
                    num = sparams.get('numerator', [1.0])
                    den = sparams.get('denominator', [1.0, 1.0])

                    # Convert to State Space
                    A, B, C, D = signal.tf2ss(num, den)

                    block_matrices[b_name] = (A, B, C, D)

                    n = A.shape[0]

                     # Init conditions
                    ic = np.array(sparams.get('init_conds', [0.0]*n), dtype=float)
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
                # State is the output y. RateLimiterBlock has no initial-output
                # parameter (it latches its first input at runtime), so the
                # compiled state starts at 0.0.
                state_map[b_name] = (current_state_idx, 1)
                y0_list.append(0.0)
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

            elif fn == 'Waveequation2D':
                # WaveEquation2D has 2*Nx*Ny states (displacement u + velocity v)
                pde_params = getattr(block, 'exec_params', None) or block.params
                Nx = int(pde_params.get('Nx', 20))
                Ny = int(pde_params.get('Ny', 20))
                n_states = 2 * Nx * Ny
                state_map[b_name] = (current_state_idx, n_states)

                # Use block's own initial state method
                from blocks.pde.wave_equation_2d import WaveEquation2DBlock
                ic = WaveEquation2DBlock().get_initial_state(pde_params)
                y0_list.extend(ic)
                current_state_idx += n_states

            elif fn == 'Advectionequation2D':
                # AdvectionEquation2D has Nx*Ny states (concentration field)
                pde_params = getattr(block, 'exec_params', None) or block.params
                Nx = int(pde_params.get('Nx', 30))
                Ny = int(pde_params.get('Ny', 30))
                n_states = Nx * Ny
                state_map[b_name] = (current_state_idx, n_states)

                # Use block's own initial state method
                from blocks.pde.advection_equation_2d import AdvectionEquation2DBlock
                ic = AdvectionEquation2DBlock().get_initial_state(pde_params)
                y0_list.extend(ic)
                current_state_idx += n_states

        y0 = np.array(y0_list, dtype=float)

        # 3. Re-order blocks into three groups:
        #   (a) Sources — no inputs, always execute first
        #   (b) Middle  — algebraic blocks AND D≠0 state blocks, in original
        #                 topological order.  D≠0 state blocks have direct
        #                 feedthrough (output = C*x + D*u depends on current
        #                 input), so they CANNOT be pre-populated with just C*x.
        #                 They must execute alongside algebraic blocks so their
        #                 input is available when they run.
        #   (c) D=0 state blocks — strictly proper TFs, integrators, PDE blocks.
        #                 Pre-populated with C*x (exact since D*u = 0), execute last.
        #
        # block_matrices is now populated, so we can inspect D to classify.
        source_fns = ('Step', 'Sine', 'Constant', 'From', 'Ramp', 'Exponential',
                      'Noise', 'Wavegenerator', 'Prbs', 'Impulse')
        state_fns = ('Tranfn', 'Transferfcn', 'TransferFcn', 'Statespace', 'StateSpace',
                     'Integrator', 'Pid', 'PID', 'Ratelimiter', 'RateLimiter',
                     'Heatequation1D', 'Waveequation1D', 'Advectionequation1D',
                     'Diffusionreaction1D', 'Heatequation2D', 'Waveequation2D',
                     'Advectionequation2D')

        def _is_d0_state_block(b):
            """True if b is a state block with D=0 (safe to pre-populate)."""
            fn = b.block_fn.title() if b.block_fn else ''
            if fn not in state_fns:
                return False
            if b.name in block_matrices:
                _, _, _, D = block_matrices[b.name]
                return not np.any(D != 0)
            if fn in ('Pid', 'PID'):
                return False  # PID output depends on current error (feedthrough)
            return True  # Integrator, RateLimiter, PDE blocks: D=0

        source_set = set()
        d0_state_set = set()
        for b in sorted_order:
            fn = b.block_fn.title() if b.block_fn else ''
            if fn in source_fns:
                source_set.add(b.name)
            elif _is_d0_state_block(b):
                d0_state_set.add(b.name)

        sources = [b for b in sorted_order if b.name in source_set]
        middle = [b for b in sorted_order if b.name not in source_set and b.name not in d0_state_set]
        state_blocks_d0 = [b for b in sorted_order if b.name in d0_state_set]

        sorted_order = sources + middle + state_blocks_d0

        # 4. Compile Execution Sequence
        execution_sequence = []
        for block in sorted_order:
             executor = self._create_block_executor(block, input_map, state_map, block_matrices)
             execution_sequence.append(executor)

        # 5. Build pre-population list for D=0 state-block outputs ONLY.
        # D≠0 blocks execute in the middle group and are NOT pre-populated.
        state_output_preloads = []
        for b_name, (start, size) in state_map.items():
            if b_name not in d0_state_set:
                continue  # D≠0 or PID: skip pre-population
            if b_name in block_matrices:
                A, B, C, D = block_matrices[b_name]
                state_output_preloads.append((b_name, start, size, C))
            else:
                # Integrator: output = state
                state_output_preloads.append((b_name, start, size, None))

        # 6. Create optimized closure.
        # The executor list is ordered sources-first (sorted_order above), so we
        # can split it at the source boundary. This lets a linearization helper
        # override input-source signals *after* the sources run but *before* the
        # downstream/state blocks consume them, without disturbing normal solves.
        n_sources = len(sources)

        def _evaluate(t, y, input_overrides=None):
            """Run the compiled diagram once; return (dy_vec, signals).

            input_overrides: optional {block_name: value} merged into the signal
            dict immediately after the source blocks execute. Used by the
            numerical linearizer to perturb inputs; pass None for normal solves.
            """
            signals = {}
            dy_vec = np.zeros_like(y)

            # Pre-populate D=0 state-block outputs so feedback loops resolve.
            # Only D=0 blocks are here — their output C*x is exact (D*u = 0).
            for b_name, start, size, C_mat in state_output_preloads:
                if C_mat is not None:
                    # StateSpace/TranFn (D=0): y_out = C * x
                    x = y[start:start + size].reshape(-1, 1)
                    out = C_mat @ x
                    signals[b_name] = out.item() if out.size == 1 else out.flatten()
                else:
                    # Integrator: output = state
                    signals[b_name] = y[start] if size == 1 else y[start:start + size]

            # Sources first, then optional input overrides, then the rest.
            for exec_fn in execution_sequence[:n_sources]:
                exec_fn(t, y, dy_vec, signals)
            if input_overrides:
                signals.update(input_overrides)
            for exec_fn in execution_sequence[n_sources:]:
                exec_fn(t, y, dy_vec, signals)

            return dy_vec, signals

        def model_func(t, y):
            return _evaluate(t, y)[0]

        # Expose the richer evaluator + source names for numerical linearization
        # (lib/analysis/linearizer.py). Attached as attributes so the
        # compile_system return contract is unchanged.
        model_func.evaluate = _evaluate
        model_func.source_names = [b.name for b in sources]
        model_func.state_map = state_map

        return model_func, y0, state_map, block_matrices
