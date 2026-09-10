import logging
import numpy as np
from typing import List, Callable, Dict, Any, Tuple
from lib.simulation.block import DBlock
from scipy import signal
from lib.engine.pde_helpers import (
    companion_seed,
    parse_pde_2d_initial_condition,
    parse_pde_initial_condition,
)
from lib.engine.block_names import canonical_fn
from lib.engine.compiler_kernels import BuildContext, build_events, get_kernel_builder
from lib.engine.block_params import runtime_params

logger = logging.getLogger(__name__)


def _declared_sample_time(block) -> float:
    """Sample time declared on a block: >0 = discrete rate, <=0 = continuous.

    Mirrors ``DBlock.resolve_sample_time`` but tolerates blocks that predate
    it (and unparseable workspace-variable strings), since this runs before
    the engine resolves params.
    """
    resolver = getattr(block, "resolve_sample_time", None)
    if callable(resolver):
        try:
            return resolver()
        except (TypeError, ValueError):
            return -1.0
    params = getattr(block, "params", {}) or {}
    try:
        return float(params.get("sampling_time", params.get("sample_time", -1.0)))
    except (TypeError, ValueError):
        return -1.0


# ---------------------------------------------------------------------------
# compile_system phases (module-level so each is testable on its own)
# ---------------------------------------------------------------------------


def _build_input_map(blocks, lines) -> Dict[str, Dict[int, Tuple[str, int]]]:
    """Static pull-based wiring: ``dst_name -> {dst_port: (src_name, src_port)}``."""
    input_map = {b.name: {} for b in blocks}
    for line in lines:
        input_map[line.dstblock][line.dstport] = (line.srcblock, line.srcport)
    return input_map


def _pad_initial_conditions(ic, n: int) -> np.ndarray:
    """``ic`` as a length-``n`` float vector: zero-padded or truncated."""
    ic_flat = np.atleast_1d(np.array(ic, dtype=float)).flatten()
    if len(ic_flat) < n:
        padded = np.zeros(n)
        padded[: len(ic_flat)] = ic_flat
        return padded
    return ic_flat[:n]


# A state allocator maps (block, resolved params) -> (n_states, y0_slice,
# matrices), where matrices is the (A, B, C, D) tuple for linear state blocks
# and None otherwise. Params are the *resolved* ones (runtime_params): raw
# params may still hold workspace-variable names as strings, which would
# misbuild the matrices and hence the D=0 / D!=0 execution-order classification.


def _alloc_integrator(block, p):
    ic = np.atleast_1d(np.array(p.get("init_conds", 0.0), dtype=float)).flatten()
    return ic.size, ic, None


def _alloc_state_space(block, p):
    A = np.array(p["A"], dtype=float)
    B = np.array(p["B"], dtype=float)
    C = np.array(p["C"], dtype=float)
    D = np.array(p["D"], dtype=float)
    n = A.shape[0] if len(A.shape) > 1 else 1
    A = A.reshape(n, n)
    return n, _pad_initial_conditions(p.get("init_conds", [0.0] * n), n), (A, B, C, D)


def _alloc_transfer_fcn(block, p):
    A, B, C, D = signal.tf2ss(p.get("numerator", [1.0]), p.get("denominator", [1.0, 1.0]))
    n = A.shape[0]
    return n, _pad_initial_conditions(p.get("init_conds", [0.0] * n), n), (A, B, C, D)


def _alloc_pid(block, p):
    # [x_i, x_d]: integrator + derivative-filter states, both starting at 0.
    return 2, np.zeros(2), None


def _alloc_rate_limiter(block, p):
    # State is the output y. The block has no initial-output parameter (it
    # latches its first input at runtime), so the compiled state starts at 0.
    return 1, np.zeros(1), None


def _alloc_heat_1d(block, p):
    N = int(p.get("N", 20))
    ic = parse_pde_initial_condition(
        p.get("init_conds", [0.0]),
        N,
        float(p.get("L", 1.0)),
        pde_type="heat",
        seed=p.get("seed", 0),
    )
    return N, ic, None


def _alloc_wave_1d(block, p):
    # 2N states: N displacement + N velocity.
    N = int(p.get("N", 50))
    L = float(p.get("L", 1.0))
    seed = p.get("seed", 0)
    u0 = parse_pde_initial_condition(
        p.get("init_displacement", [0.0]), N, L, pde_type="wave", seed=seed
    )
    v0 = parse_pde_initial_condition(
        p.get("init_velocity", [0.0]), N, L, pde_type="wave", seed=companion_seed(seed)
    )
    return 2 * N, np.concatenate([np.ravel(u0), np.ravel(v0)]), None


def _alloc_advection_1d(block, p):
    N = int(p.get("N", 50))
    c0 = parse_pde_initial_condition(
        p.get("init_conds", [0.0]), N, float(p.get("L", 1.0)), pde_type="advection"
    )
    return N, c0, None


def _alloc_diffusion_reaction_1d(block, p):
    N = int(p.get("N", 30))
    c0 = parse_pde_initial_condition(
        p.get("init_conds", [1.0]), N, float(p.get("L", 1.0)), pde_type="diffusion_reaction"
    )
    return N, c0, None


def _alloc_heat_2d(block, p):
    Nx = int(p.get("Nx", 20))
    Ny = int(p.get("Ny", 20))
    T0 = parse_pde_2d_initial_condition(
        p.get("init_temp", "0.0"),
        Nx,
        Ny,
        float(p.get("Lx", 1.0)),
        float(p.get("Ly", 1.0)),
        float(p.get("init_amplitude", 1.0)),
        seed=p.get("seed", 0),
    )
    return Nx * Ny, T0.flatten(), None


def _alloc_wave_2d(block, p):
    # 2*Nx*Ny states (displacement u + velocity v); the block owns the layout.
    from blocks.pde.wave_equation_2d import WaveEquation2DBlock

    n_states = 2 * int(p.get("Nx", 20)) * int(p.get("Ny", 20))
    return n_states, WaveEquation2DBlock().get_initial_state(p), None


def _alloc_advection_2d(block, p):
    from blocks.pde.advection_equation_2d import AdvectionEquation2DBlock

    n_states = int(p.get("Nx", 30)) * int(p.get("Ny", 30))
    return n_states, AdvectionEquation2DBlock().get_initial_state(p), None


# canonical_fn -> allocator. Blocks absent here carry no ODE state (algebraic
# blocks, sinks, and StateVariable, whose discrete state lives in its closure).
STATE_ALLOCATORS = {
    "Integrator": _alloc_integrator,
    "StateSpace": _alloc_state_space,
    "TransferFcn": _alloc_transfer_fcn,
    "PID": _alloc_pid,
    "RateLimiter": _alloc_rate_limiter,
    "Heatequation1D": _alloc_heat_1d,
    "Waveequation1D": _alloc_wave_1d,
    "Advectionequation1D": _alloc_advection_1d,
    "Diffusionreaction1D": _alloc_diffusion_reaction_1d,
    "Heatequation2D": _alloc_heat_2d,
    "Waveequation2D": _alloc_wave_2d,
    "Advectionequation2D": _alloc_advection_2d,
}


def _allocate_states(blocks) -> Tuple[Dict[str, Tuple[int, int]], Dict[str, tuple], np.ndarray]:
    """Lay the ODE state vector out block by block, in ``blocks`` order.

    Returns ``(state_map, block_matrices, y0)`` with
    ``state_map[name] = (start_idx, size)`` and ``block_matrices[name] =
    (A, B, C, D)`` for the linear state blocks.
    """
    state_map: Dict[str, Tuple[int, int]] = {}
    block_matrices: Dict[str, tuple] = {}
    y0_parts: List[np.ndarray] = []
    next_idx = 0
    for block in blocks:
        fn = canonical_fn(block.block_fn)
        allocator = STATE_ALLOCATORS.get(fn)
        if allocator is None:
            continue
        try:
            size, y0_slice, matrices = allocator(block, runtime_params(block))
        except Exception as e:
            logger.error(f"Failed to compile {fn} {block.name}: {e}")
            raise
        state_map[block.name] = (next_idx, size)
        if matrices is not None:
            block_matrices[block.name] = matrices
        y0_parts.append(np.asarray(y0_slice, dtype=float).ravel())
        next_idx += size
    y0 = np.concatenate(y0_parts) if y0_parts else np.zeros(0)
    return state_map, block_matrices, y0


# Execution is in three groups (see docs/SOLVER_SEMANTICS.md and CLAUDE.md):
#   (a) sources         -- no inputs, run first;
#   (b) middle          -- algebraic blocks AND D!=0 state blocks (their output
#                          C*x + D*u needs this step's input), in topological
#                          order;
#   (c) D=0 state blocks -- strictly proper TFs, Integrator, RateLimiter, PDEs.
#                          Their output C*x is pre-populated exactly, so they
#                          run last and their derivatives see final inputs.
# Both sets hold canonical spellings only (canonical_fn is the one normalizer).
SOURCE_FNS = frozenset(
    {"Step", "Sine", "Constant", "From", "Ramp", "Noise", "Wavegenerator", "Prbs", "Impulse"}
)
STATE_FNS = frozenset(
    {
        "TransferFcn",
        "StateSpace",
        "Integrator",
        "PID",
        # canonical_fn only maps the exact upstream spelling 'PID' to 'PID';
        # a block_fn of 'pid'/'Pid' canonicalizes to 'Pid'.
        "Pid",
        "RateLimiter",
        "Heatequation1D",
        "Waveequation1D",
        "Advectionequation1D",
        "Diffusionreaction1D",
        "Heatequation2D",
        "Waveequation2D",
        "Advectionequation2D",
    }
)


def _is_d0_state_block(block, fn: str, block_matrices) -> bool:
    """True if ``block`` is a state block with D=0 (safe to pre-populate)."""
    if fn not in STATE_FNS:
        return False
    if block.name in block_matrices:
        _, _, _, D = block_matrices[block.name]
        return not np.any(D != 0)
    if fn in ("Pid", "PID"):
        return False  # PID output depends on the current error (feedthrough)
    return True  # Integrator, RateLimiter, PDE blocks: D=0


def _execution_groups(sorted_order, block_matrices):
    """Split ``sorted_order`` into (sources, middle, d0_state_blocks), each in
    the original topological order."""
    source_names = set()
    d0_names = set()
    for b in sorted_order:
        fn = canonical_fn(b.block_fn)
        if fn in SOURCE_FNS:
            source_names.add(b.name)
        elif _is_d0_state_block(b, fn, block_matrices):
            d0_names.add(b.name)
    sources = [b for b in sorted_order if b.name in source_names]
    middle = [b for b in sorted_order if b.name not in source_names and b.name not in d0_names]
    d0_state_blocks = [b for b in sorted_order if b.name in d0_names]
    return sources, middle, d0_state_blocks


def _state_output_preloads(state_map, block_matrices, d0_names):
    """``(name, start, size, C)`` for every D=0 state block; ``C`` is None for
    an Integrator (output = state). D!=0 blocks run in the middle group and
    are deliberately absent."""
    preloads = []
    for b_name, (start, size) in state_map.items():
        if b_name not in d0_names:
            continue
        matrices = block_matrices.get(b_name)
        preloads.append((b_name, start, size, matrices[2] if matrices is not None else None))
    return preloads


def _make_model_func(execution_sequence, n_sources: int, state_output_preloads):
    """Build the ODE right-hand side ``f(t, y) -> dy`` over the compiled executors.

    ``execution_sequence`` is sources-first, so it is split at ``n_sources``:
    a linearization helper can override input-source signals *after* the
    sources run but *before* downstream/state blocks consume them, without
    disturbing normal solves. The richer ``evaluate`` is exposed as an
    attribute so the ``compile_system`` return contract stays unchanged.
    """

    def _evaluate(t, y, input_overrides=None):
        """Run the compiled diagram once; return (dy_vec, signals)."""
        signals = {}
        dy_vec = np.zeros_like(y)

        # Pre-populate D=0 state-block outputs so feedback loops resolve;
        # their output C*x is exact because D*u = 0.
        for b_name, start, size, C_mat in state_output_preloads:
            if C_mat is not None:
                x = y[start : start + size].reshape(-1, 1)
                out = C_mat @ x
                signals[b_name] = out.item() if out.size == 1 else out.flatten()
            else:
                signals[b_name] = y[start] if size == 1 else y[start : start + size]

        for exec_fn in execution_sequence[:n_sources]:
            exec_fn(t, y, dy_vec, signals)
        if input_overrides:
            signals.update(input_overrides)
        for exec_fn in execution_sequence[n_sources:]:
            exec_fn(t, y, dy_vec, signals)
        return dy_vec, signals

    def model_func(t, y):
        return _evaluate(t, y)[0]

    model_func.evaluate = _evaluate
    return model_func


class SystemCompiler:
    """
    Compiles a block diagram into a flat numerical function for fast ODE solving.
    Supports integration with scipy.integrate.solve_ivp.
    """

    def __init__(self):
        # Per-compile store of latch/mode holders. Populated by _build_context
        # so a block's kernel closure and its event functions close over the
        # SAME mutable holder; reset at the top of every compile_system.
        self._mode_registry: Dict[str, Dict[str, Any]] = {}
        # Whether the caller intends to run with zero-crossing detection. Only
        # affects compilability: blocks whose semantics require located events
        # (Hysteresis) are compilable exactly when events will be available.
        self.zero_crossing_enabled: bool = True
        # Allowlist of blocks that can be compiled
        self.COMPILABLE_BLOCKS = {
            "Integrator",
            "Gain",
            "MatrixGain",
            "Matrixgain",
            "Sum",
            "Constant",
            "Sine",
            "Step",
            # 'Impulse' — excluded: the compiled path models the impulse as a
            # dt*1e-3-wide rectangular pulse, which adaptive RK45 can step over
            # entirely (the response is then silently lost). Falls back to the
            # interpreted path (blocks/impulse.py), which fires a correct
            # value/dt sample on the fixed-dt grid. Step(type='impulse') is
            # likewise gated out in check_compilability below.
            "TransferFcn",
            "TranFn",
            "StateSpace",
            "Mux",
            "Demux",
            "LogicalOperator",
            "Scope",
            "SgProd",
            "SigProduct",
            "Saturation",
            "Abs",
            "AbsBlock",
            "Ramp",
            "Switch",
            "Terminator",
            "Display",
            "Deadband",
            "Exponential",
            "Exp",
            "PiD",
            "PID",
            "RateLimiter",
            "WaveGenerator",
            # 'Noise' — excluded: np.random in the ODE RHS is re-sampled on every
            # solve_ivp stage/rejected step, breaking adaptive error control and
            # reproducibility. Falls back to the interpreted path (blocks/noise.py).
            "PRBS",
            "MathFunction",
            "Selector",
            # 'Hysteresis' — compilable only with zero-crossing detection on (see
            # ZERO_CROSSING_ONLY_BLOCKS below). The relay latch is path-dependent
            # and cannot be a pure function of (t, y) while solve_ivp probes the
            # RHS at non-accepted, non-monotonic times; located terminal events
            # fix that by freezing the mode inside each segment and flipping it
            # only at the switching instant. Without events it still falls back
            # to the interpreted path (blocks/hysteresis.py).
            # PDE Blocks (Method of Lines) - 1D
            "HeatEquation1D",
            "WaveEquation1D",
            "AdvectionEquation1D",
            "DiffusionReaction1D",
            # PDE Blocks (Method of Lines) - 2D
            "HeatEquation2D",
            "WaveEquation2D",
            "AdvectionEquation2D",
            # Field Processing Blocks - 1D
            "FieldProbe",
            "FieldIntegral",
            "FieldMax",
            "FieldScope",
            "FieldGradient",
            "FieldLaplacian",
            # Field Processing Blocks - 2D
            "FieldProbe2D",
            "FieldScope2D",
            "FieldSlice",
            # Optimization Primitives
            # 'StateVariable' — excluded: it performs a discrete state update inside
            # the continuous ODE RHS keyed on a monotonic-time assumption that
            # solve_ivp violates (repeated/rejected/non-monotonic probe times).
            # Falls back to the interpreted path
            # (blocks/optimization_primitives/state_variable.py).
            "Product",
        }

        # Blocks whose compiled kernel is only correct when the runner locates
        # switching instants for it. They join COMPILABLE_BLOCKS when
        # zero_crossing_enabled is True and are otherwise sent to the
        # interpreter.
        self.ZERO_CROSSING_ONLY_BLOCKS = {"Hysteresis"}

        # Why the last check_compilability() said no ("<block> (<fn>): <reason>"),
        # or None when the diagram compiles. The GUI shows it so a silent
        # fall-back to the interpreter is explainable without reading the log.
        self.last_incompatibility = None

    def _not_compilable(self, block, reason: str) -> bool:
        """Record why ``block`` blocks compilation and return False."""
        self.last_incompatibility = "{} ({}): {}".format(
            getattr(block, "name", "?"), getattr(block, "block_fn", "?"), reason
        )
        logger.debug("Not compilable — %s; using interpreter.", self.last_incompatibility)
        return False

    def _compilable_names(self):
        """Canonical allowlist for check_compilability.

        Normalized through the same function the compiler and the replay use,
        so "TranFn"/"Transferfcn"/"tranfn" and friends can never be accepted
        here but spelled differently downstream. Rebuilt per call so a caller
        that edits COMPILABLE_BLOCKS is honoured, and so the event-gated blocks
        (ZERO_CROSSING_ONLY_BLOCKS) follow zero_crossing_enabled.
        """
        names = set(self.COMPILABLE_BLOCKS)
        if self.zero_crossing_enabled:
            names |= self.ZERO_CROSSING_ONLY_BLOCKS
        return {canonical_fn(name) for name in names}

    def check_compilability(self, blocks: List[DBlock], _recursive: bool = False) -> bool:
        """
        Check if the entire diagram is supported by the compiler.

        Records *why* on ``self.last_incompatibility`` when the answer is False,
        so the caller can tell the user which block sent the run to the
        interpreter instead of leaving it to a debug-level log line. Cleared on
        every top-level call; ``_recursive`` is set for the Subsystem walk so an
        inner reason survives back up to the caller.

        Args:
            blocks: List of all blocks in the diagram.
            _recursive: Internal — do not reset the recorded reason.

        Returns:
            bool: True if all blocks are supported, False otherwise.
        """
        if not _recursive:
            self.last_incompatibility = None
        allowed = self._compilable_names()
        for block in blocks:
            b_type = block.block_fn

            # Special handling for Subsystems (Recursive check)
            if b_type == "Subsystem":
                if hasattr(block, "sub_blocks"):
                    if not self.check_compilability(block.sub_blocks, _recursive=True):
                        return False
                continue

            # Special handling for structural blocks (Inport/Outport)
            # They are flattened away or handled as sources/sinks
            if b_type in ("Inport", "Outport"):
                continue

            # Step's 'impulse' subtype is a narrow Dirac approximation the
            # adaptive compiled solver can step over; force the interpreter path
            # (same rationale as the excluded Impulse block in COMPILABLE_BLOCKS).
            if b_type == "Step" and getattr(block, "params", {}).get("type") == "impulse":
                return self._not_compilable(
                    block, "the 'impulse' step shape needs the fixed-step grid"
                )

            # A block gated to a discrete rate (sampling_time > 0) is a
            # sampled-data element: it must hold its output between sample
            # instants and advance its state once per Ts.  The compiled ODE
            # RHS has no notion of sample instants and would silently run the
            # block as a purely continuous one, so the whole diagram falls
            # back to the interpreter, which honours the rate.
            if _declared_sample_time(block) > 0:
                return self._not_compilable(block, "it has a discrete sample time")

            if canonical_fn(b_type) not in allowed:
                if canonical_fn(b_type) in {
                    canonical_fn(n) for n in self.ZERO_CROSSING_ONLY_BLOCKS
                }:
                    return self._not_compilable(
                        block, "it needs zero-crossing detection, which is off"
                    )
                return self._not_compilable(block, "it has no compiled kernel")

        return True

    def _build_context(
        self, block: DBlock, input_map: Dict, state_map: Dict, block_matrices: Dict
    ) -> BuildContext:
        """Assemble the per-block BuildContext shared by the kernel and event
        builders (see ``lib.engine.compiler_kernels``)."""
        b_name = block.name

        # Normalize Function Name (single source of truth: lib.engine.block_names)
        fn = canonical_fn(block.block_fn)

        # Use resolved params if available (exec_params), otherwise fall back to params
        # This ensures workspace variables are properly resolved
        params = runtime_params(block)

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
                input_sources.append(None)  # None means 0.0 default

        return BuildContext(
            block=block,
            b_name=b_name,
            fn=fn,
            params=params,
            input_sources=input_sources,
            deps=deps,
            state_map=state_map,
            block_matrices=block_matrices,
            mode_registry=self._mode_registry,
        )

    def _create_block_executor(
        self, block: DBlock, input_map: Dict, state_map: Dict, block_matrices: Dict
    ) -> Callable[[float, np.ndarray, np.ndarray, Dict], None]:
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
        # --- Block-specific closures ---
        # Every compilable block family is registered in lib.engine.compiler_kernels;
        # dispatch to its builder. Unknown fn-names fall through to the no-op below.
        ctx = self._build_context(block, input_map, state_map, block_matrices)
        builder = get_kernel_builder(ctx.fn)
        if builder is not None:
            return builder(ctx)

        # Generic catch-all: blocks with no registered kernel do nothing in the
        # solver loop (sinks/tags, or anything not on the compilable allowlist).
        def exec_noop(t, y, dy_vec, signals):
            pass

        return exec_noop

    def _collect_event_specs(
        self, blocks: List[DBlock], input_map: Dict, state_map: Dict, block_matrices: Dict
    ) -> List[Any]:
        """Zero-crossing event functions contributed by the compiled blocks.

        Each discontinuous block declares its switching surfaces next to the
        kernel that implements the discontinuity (the ``@events(...)`` registry
        in ``lib.engine.compiler_kernels``); this walks the diagram and gathers
        them.  A block can opt out with its ``zero_crossing`` param set to
        ``"off"`` -- useful when a block switches so fast that locating every
        crossing costs more than it buys.

        The specs are a pure function of the diagram, so they ride along in the
        compiled-system cache; whether they are *used* is the runner's call
        (the simulation-level ``zero_crossing`` setting).
        """
        specs = []
        for block in blocks:
            ctx = self._build_context(block, input_map, state_map, block_matrices)
            if str(ctx.params.get("zero_crossing", "auto")).lower() == "off":
                logger.debug("Zero-crossing disabled on block %s by its own param", ctx.b_name)
                continue
            try:
                specs.extend(build_events(ctx))
            except Exception as e:  # noqa: BLE001 - a bad event must not break compilation
                logger.warning(
                    "Failed to build zero-crossing events for %s (%s): %s", ctx.b_name, ctx.fn, e
                )
        return specs

    def _build_executors(self, sorted_order, input_map, state_map, block_matrices):
        """One kernel closure per block, in execution order.

        Also exposes the ``name -> executor`` map on ``self.block_executors``
        so the post-solve replay reuses these same kernels for pure-function
        blocks (single source of truth for that math).
        """
        execution_sequence = []
        block_executors = {}
        for block in sorted_order:
            executor = self._create_block_executor(block, input_map, state_map, block_matrices)
            execution_sequence.append(executor)
            block_executors[block.name] = executor
        self.block_executors = block_executors
        return execution_sequence

    def compile_system(
        self, blocks: List[DBlock], sorted_order: List[DBlock], lines: List[Any]
    ) -> Tuple[Callable, np.ndarray, Dict]:
        """Compile the diagram into one ODE right-hand side ``f(t, y)``.

        Returns ``(model_func, y0, state_map, block_matrices)``. ``model_func``
        also carries ``evaluate`` (dy and every signal, with optional input
        overrides, for the linearizer), ``source_names``, ``state_map``,
        ``event_specs`` and ``mode_states`` (zero-crossing support).
        """
        # Latch holders belong to this compile only: a stale registry would
        # hand the new closures the previous diagram's relay modes.
        self._mode_registry = {}

        input_map = _build_input_map(blocks, lines)
        state_map, block_matrices, y0 = _allocate_states(blocks)

        # Ordering waits until the states are allocated because the D matrices
        # decide which state blocks are feedthrough (middle) vs D=0 (last).
        sources, middle, d0_state_blocks = _execution_groups(sorted_order, block_matrices)
        sorted_order = sources + middle + d0_state_blocks

        execution_sequence = self._build_executors(
            sorted_order, input_map, state_map, block_matrices
        )
        preloads = _state_output_preloads(
            state_map, block_matrices, {b.name for b in d0_state_blocks}
        )
        model_func = _make_model_func(execution_sequence, len(sources), preloads)
        model_func.source_names = [b.name for b in sources]
        model_func.state_map = state_map

        # Zero-crossing events are collected here (not in the runner) because
        # they need the same input_map / state_map / latch holders the kernels
        # were built from; riding on model_func, the compiled-system cache
        # carries them too. Whether they are used is the runner's call.
        model_func.event_specs = self._collect_event_specs(
            sorted_order, input_map, state_map, block_matrices
        )
        model_func.mode_states = list(self._mode_registry.values())
        if model_func.event_specs:
            logger.debug(
                "Compiled %d zero-crossing event(s): %s",
                len(model_func.event_specs),
                ", ".join(spec.name for spec in model_func.event_specs),
            )

        return model_func, y0, state_map, block_matrices
