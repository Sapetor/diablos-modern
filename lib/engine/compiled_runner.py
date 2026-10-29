"""The compiled (fast-solver) run: ODE solve + post-solve signal replay.

Extracted from :class:`~lib.engine.simulation_engine.SimulationEngine`.  The
two entry points take the engine as their first argument and read/write exactly
the attributes the original methods did; ``SimulationEngine.run_compiled_simulation``
and ``_replay_compiled_signals`` remain as thin delegating methods, so no caller
(GUI, DSim, scripts, tests) had to change.

The split follows the two halves of a fast-solver run:

* :func:`run_compiled_simulation` -- resolve params, compile the diagram into a
  single ODE, integrate it (scipy, or an in-house fixed step), record
  diagnostics.
* :func:`replay_compiled_signals` -- walk the saved state trajectory and
  reconstruct every block's output, so Scope / FieldScope blocks get their
  history (the solve itself only returns the ODE state). The per-block replay
  math lives in :mod:`lib.engine.replay_handlers`.
"""

import logging
import time as time_module
from typing import Any, Dict, List, Tuple

import numpy as np

from lib.engine.block_names import canonical_fn
from lib.engine.block_params import runtime_params
from lib.engine.compile_cache import workspace_fingerprint
from lib.engine.compiled_delays import CompiledDelayError, is_delay_block
from lib.engine.compiler_kernels.discrete import DISCRETE_FNS, is_sampled_block, sample_update
from lib.engine.solver_diagnostics import (
    estimate_stiffness,
    format_diagnostics_for_log,
    format_stiffness_for_log,
)
from lib.engine.replay_handlers import (
    RECORDERS,
    REPLAY_HANDLERS,
    ReplayRun,
    ReplayStep,
    finalize_recorders,
    replay_fallback,
    truncate_recorders,
)
from lib.engine.zero_crossing import (
    DEFAULT_MAX_EVENTS,
    ModeHistoryReplayer,
    solve_fixed_step_with_breakpoints,
    solve_with_events,
)
from lib.simulation.block import DBlock
from lib.engine.topo import kahn_topological_order

logger = logging.getLogger(__name__)

# Compiled-solver methods that scipy.integrate.solve_ivp accepts directly.
SCIPY_SOLVER_METHODS = ("RK45", "RK23", "DOP853", "Radau", "BDF", "LSODA")
# Fixed-step schemes integrated in-house (use the simulation step dt).
FIXED_STEP_METHODS = ("Euler", "RK4")

# "Let DiaBloS pick." Not a scheme of its own: it resolves to LSODA, which
# switches between an Adams (non-stiff) and a BDF (stiff) formula by itself, so
# one setting copes with both regimes. Deliberately NOT the default -- RK45 is,
# so no existing diagram's numbers move -- but it is what the stiffness warning
# points at. Matched case-insensitively; stored verbatim in the .diablos file.
AUTO_SOLVER_METHOD = "auto"
AUTO_RESOLVED_METHOD = "LSODA"


class CompiledRunCancelled(BaseException):
    """The caller's ``cancel_cb`` fired inside a compiled run.

    Raised from the monitored RHS (so from inside ``solve_ivp`` or one of the
    in-house segment loops) and from the replay loop, and caught only by
    :func:`run_compiled_simulation`. A ``BaseException`` for the same reason
    ``asyncio.CancelledError`` is one: the solver stack has several
    ``except Exception`` guards (event callables, discrete updates) that must
    not swallow a cancellation as if it were a block error.

    ``n_done`` is how many output samples were fully replayed before the
    cancel; 0 when the solve itself was interrupted.
    """

    def __init__(self, n_done: int = 0):
        super().__init__("compiled run cancelled")
        self.n_done = int(n_done)


class _RunMonitor:
    """Polls the caller's cancel / progress hooks from inside a compiled run.

    The compiled path used to call neither hook: once ``solve_ivp`` started,
    Stop had to wait for the whole integration and the replay to finish.
    """

    # RHS evaluations between polls. A cancel_cb is a flag read, but the RHS
    # of a small diagram is only a few microseconds, so do not double it.
    STRIDE = 16
    # Replay steps between polls.
    REPLAY_STRIDE = 256

    def __init__(self, cancel_cb=None, progress_cb=None, t_end=0.0):
        self.cancel_cb = cancel_cb
        self.progress_cb = progress_cb
        self.t_end = float(t_end)
        self._t_reported = -np.inf

    @property
    def active(self) -> bool:
        return self.cancel_cb is not None or self.progress_cb is not None

    def check(self, n_done: int = 0) -> None:
        if self.cancel_cb is not None and self.cancel_cb():
            raise CompiledRunCancelled(n_done)

    def poll(self, t) -> None:
        """Cancel check plus progress at solver time ``t`` (from the RHS)."""
        self.check()
        if self.progress_cb is not None:
            # Stages evaluate ahead of the accepted step and a rejected step
            # goes back, so only report forward motion.
            t = float(t)
            if t > self._t_reported:
                self._t_reported = t
                self.progress_cb(min(t, self.t_end), self.t_end)

    def wrap(self, rhs):
        """``rhs`` with a poll every :attr:`STRIDE` calls (unchanged if idle)."""
        return _MonitoredRHS(rhs, self) if self.active else rhs

    def wrap_hook(self, hook):
        """A breakpoint hook that polls before running (``None`` stays None)."""
        if hook is None or not self.active:
            return hook

        def monitored(t, y):
            self.poll(t)
            return hook(t, y)

        return monitored


class _MonitoredRHS:
    """Callable proxy for a compiled RHS that polls a :class:`_RunMonitor`.

    Attribute reads (``evaluate``, ``event_specs``, ``mode_states``, ...) go
    to the wrapped function, so the segment drivers see the same object API.
    """

    def __init__(self, rhs, monitor: _RunMonitor):
        self._rhs = rhs
        self._monitor = monitor
        self._calls = 0

    def __call__(self, t, y):
        self._calls += 1
        if self._calls % _RunMonitor.STRIDE == 0:
            self._monitor.poll(t)
        return self._rhs(t, y)

    def __getattr__(self, name):
        return getattr(self._rhs, name)


def resolve_solver_method(method) -> str:
    """Map the stored solver setting to a scheme the runner can execute.

    Only ``"auto"`` is rewritten (to :data:`AUTO_RESOLVED_METHOD`); everything
    else -- including an unknown name, which the runner reports and degrades to
    RK45 later -- is passed through untouched.
    """
    name = str(method or "").strip()
    if name.lower() == AUTO_SOLVER_METHOD:
        return AUTO_RESOLVED_METHOD
    return name or "RK45"


# Canonical fn-names whose compiled kernel executor is reused verbatim by the
# post-solve replay loop (instead of a duplicated inline computation), so the
# ODE solve and the replay share one source of truth for each block's output
# math. This covers pure-function source/algebraic blocks AND the ODE-state
# blocks whose kernel output is reproducible from the replay's reconstructed
# state -- StateSpace/TransferFcn/PID/RateLimiter read y[start:start+size] (the
# dispatch passes y_step) exactly as the old inline branches read
# current_states, and for multi-separate-input StateSpace the kernel also fixes
# a latent solve/replay divergence the broadcast inline branch had.
# Still excluded -- their replay branches genuinely differ: PDE/Field blocks
# (emit display-only secondary outputs), Mathfunction (domain-guarded math),
# StateVariable (discrete pending-update state) and Demux (secondary-port
# outputs). Hysteresis *is* replayed through its kernel: its latch is driven
# by the mode history the event solve recorded (ModeHistoryReplayer), which is
# the only way a relay's grid samples can switch where the solve did.
# Note that some names below belong to blocks that SystemCompiler excludes from
# COMPILABLE_BLOCKS (Noise): the diagram then never compiles at all, so the
# entry is inert -- it is kept so this set stays a straight mirror of the
# kernel registry.
_KERNEL_REPLAY_FNS = frozenset(
    {
        "Sine",
        "Constant",
        "Gain",
        "Sum",
        "Step",
        "SgProd",
        "Product",
        "Exponential",
        "Exp",
        "Deadband",
        "Hysteresis",
        "Saturation",
        "Abs",
        "Absblock",
        "Ramp",
        "Switch",
        "Wavegenerator",
        "Noise",
        "Mux",
        "Logicaloperator",
        "LogicalOperator",
        "Selector",
        "StateSpace",
        "TransferFcn",
        "PID",
        "RateLimiter",
        # Function: execute() expects the interpreter's array-shaped inputs and
        # fails on the compiled scalars (e.g. "u[0][0]"); the kernel does not.
        "Function",
        # Sampled blocks: the kernel reads the holder ModeHistoryReplayer
        # sets to the output that held at each sample time.
        *DISCRETE_FNS,
    }
)


def integrate_fixed_step(model_func, t_eval, y0, scheme):
    """
    Integrate an ODE system on the fixed grid ``t_eval`` (steps taken from the
    grid spacing, i.e. the simulation dt).

    Args:
        model_func: callable (t, y) -> dy/dt
        t_eval: 1-D array of sample times (monotonic)
        y0: initial state vector
        scheme: 'euler' (explicit Euler) or 'rk4' (classic Runge-Kutta 4)

    Returns:
        y_history: array of shape (n_states, len(t_eval))
    """
    y0 = np.asarray(y0, dtype=float)
    n_states = len(y0)
    n_steps = len(t_eval)
    y_history = np.zeros((n_states, n_steps))
    y = y0.copy()
    y_history[:, 0] = y

    scheme = scheme.lower()
    for idx in range(1, n_steps):
        t_prev = t_eval[idx - 1]
        h = t_eval[idx] - t_prev
        if scheme == "rk4":
            k1 = np.asarray(model_func(t_prev, y), dtype=float)
            k2 = np.asarray(model_func(t_prev + h / 2.0, y + h / 2.0 * k1), dtype=float)
            k3 = np.asarray(model_func(t_prev + h / 2.0, y + h / 2.0 * k2), dtype=float)
            k4 = np.asarray(model_func(t_prev + h, y + h * k3), dtype=float)
            y = y + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        else:  # explicit Euler
            dy = np.asarray(model_func(t_prev, y), dtype=float)
            y = y + h * dy
        y_history[:, idx] = y

    return y_history


def replay_has_feedthrough(block, block_matrices) -> bool:
    """Whether ``block``'s output depends on this step's inputs.

    Only feedthrough edges constrain the replay's execution order: a
    strictly proper state block's output is its already-known state.

    Two sources, in order of authority:

    1. ``block_matrices`` -- the D matrix the compiler actually built for
       this block. Using it keeps the replay order consistent with the
       compiled solve by construction.
    2. ``runtime_params(block)`` -- the resolved params, for a block the
       compiler allocated no matrices for. Reading raw ``params`` here was
       a bug: a TF parameterised by workspace variables still holds the
       variable *names* there, so ``len(den) > len(num)`` compared string
       lengths and could classify a strictly proper TF as feedthrough
       (the compiler itself uses ``runtime_params`` for exactly this
       reason -- see ``SystemCompiler.compile_system``).
    """
    fn = canonical_fn(getattr(block, "block_fn", ""))

    if fn == "Integrator" or is_sampled_block(block) or is_delay_block(block):
        # A sampled block's output between instants is its held value; a
        # transport delay's is recorded history.
        return False
    if fn not in ("TransferFcn", "StateSpace"):
        return True

    matrices = block_matrices.get(block.name) if block_matrices else None
    if matrices is not None:
        return bool(np.any(np.asarray(matrices[3]) != 0))

    sparams = runtime_params(block)
    try:
        if fn == "TransferFcn":
            num = sparams.get("numerator", [])
            den = sparams.get("denominator", [])
            return not len(den) > len(num)
        D = np.asarray(sparams.get("D", [[0.0]]), dtype=float)
        return bool(np.any(D != 0))
    except (TypeError, ValueError):
        # Unresolvable params (still a workspace-variable string, ragged
        # matrix, ...): assume feedthrough, the ordering-safe default.
        return True


def _replay_order(current_blocks, current_lines, block_matrices, precomputed=()):
    """Blocks in replay order: topological on direct-feedthrough edges only.

    A strictly proper state block's output is its (already-known) state, so
    it does not depend on this step's inputs and need not follow its source.
    ``precomputed`` blocks (algebraic-loop members, read from the compiled
    evaluation) depend on nothing either. Kahn's algorithm, stable on block
    name for determinism; any cycle leftovers are appended in current-block
    order so every block still runs.
    """
    block_by_name = {b.name: b for b in current_blocks}
    adj = {name: [] for name in block_by_name}
    # Feedthrough is a property of the destination block: classify each once.
    feedthrough_by_dst: Dict[str, bool] = {}
    for line in current_lines:
        dst = line.dstblock
        dst_block = block_by_name.get(dst)
        if dst_block is None:
            is_feedthrough = True
        else:
            is_feedthrough = feedthrough_by_dst.get(dst)
            if is_feedthrough is None:
                is_feedthrough = replay_has_feedthrough(dst_block, block_matrices)
                feedthrough_by_dst[dst] = is_feedthrough
        if is_feedthrough and line.srcblock in adj and dst not in precomputed:
            adj[line.srcblock].append(dst)

    order_names, leftover_names = kahn_topological_order(
        (b.name for b in current_blocks), adj, key=lambda n: n
    )
    return [block_by_name[n] for n in order_names] + [block_by_name[n] for n in leftover_names]


def _connections_by_destination(current_lines) -> Dict[str, List[Tuple[str, int, int]]]:
    """``dst_name -> [(src_name, src_port, dst_port)]``, built once per replay."""
    inputs_by_dst: Dict[str, List[Tuple[str, int, int]]] = {}
    for line in current_lines:
        src_port = getattr(line, "srcport", 0) or 0
        inputs_by_dst.setdefault(line.dstblock, []).append((line.srcblock, src_port, line.dstport))
    return inputs_by_dst


def _collect_inputs(connections, signals) -> Dict[int, Any]:
    """This step's input values for one block, from the signals computed so far.

    Port 0 of a source is stored under its name; secondary ports under
    ``"{name}_out{port}"`` (falling back to the primary when a block did not
    publish that port).
    """
    inputs = {}
    for srcblock, src_port, dstport in connections:
        if src_port == 0:
            inputs[dstport] = signals.get(srcblock, 0.0)
        else:
            inputs[dstport] = signals.get(f"{srcblock}_out{src_port}", signals.get(srcblock, 0.0))
    return inputs


def replay_compiled_signals(engine, sol, current_blocks, current_lines, state_map, block_matrices):
    """Re-evaluate every block at each saved solver time so Scope /
    FieldScope blocks capture their signal history.

    The compiled solver returns only the ODE state trajectory; this
    post-solve 'replay' reconstructs every block's output from it. Blocks
    in _KERNEL_REPLAY_FNS reuse their compiled kernel executor (one source
    of truth for the output math); the rest go through
    :data:`lib.engine.replay_handlers.REPLAY_HANDLERS` or, unlisted, their
    own ``execute()``. Mutates each Scope / FieldScope block's exec_params.
    """
    num_steps = len(sol.t)
    # Algebraic-loop members cannot be replayed block by block; their outputs
    # come from one compiled evaluation per step (lib/engine/compiled_loops.py).
    loop_model = getattr(engine, "_replay_loop_model", None)
    loop_names = {n for loop in getattr(loop_model, "loops", ()) for n in loop}
    sorted_blocks = _replay_order(current_blocks, current_lines, block_matrices, loop_names)
    inputs_by_dst = _connections_by_destination(current_lines)
    fn_by_name = {b.name: canonical_fn(b.block_fn) for b in sorted_blocks}

    # block_executors was populated by compile_system; pure kernels never
    # write the scratch dy vector.
    block_executors = getattr(engine.compiler, "block_executors", {})
    replay_dy = np.zeros(sol.y.shape[0])
    # The solve wrote its results on the fixed t_eval grid, so the spacing is
    # uniform; fall back to the engine's dt for a degenerate grid.
    replay_dt = float(sol.t[1] - sol.t[0]) if num_steps > 1 else float(engine.sim_dt)
    fallback_failed = set()
    run = ReplayRun(num_steps)

    # Latched blocks take the mode that held at each sample, from the event
    # solve's record; a plain solve has no record and this is a no-op.
    mode_replayer = ModeHistoryReplayer(
        getattr(getattr(engine, "compiler", None), "_mode_registry", {}).values(), sol
    )
    # Set by run_compiled_simulation for the duration of one run (None for a
    # direct call), so a long replay can be cancelled too.
    monitor = getattr(engine, "_compiled_run_monitor", None)

    for i in range(num_steps):
        if monitor is not None and i % _RunMonitor.REPLAY_STRIDE == 0:
            monitor.check(n_done=i)
        t = sol.t[i]
        y_step = sol.y[:, i] if sol.y.ndim > 1 else sol.y
        mode_replayer.at(float(t))
        step = ReplayStep(
            t,
            {name: y_step[start : start + size] for name, (start, size) in state_map.items()},
            {},
        )
        loop_signals = loop_model.evaluate(t, y_step)[1] if loop_names else None

        for block in sorted_blocks:
            b_name = block.name
            fn = fn_by_name[b_name]
            inputs = _collect_inputs(inputs_by_dst.get(b_name, ()), step.signals)

            if b_name in loop_names:
                prefix = f"{b_name}_out"
                for key, value in loop_signals.items():
                    if key.startswith(prefix):
                        step.signals[key] = value
                step.signals[b_name] = loop_signals.get(b_name, 0.0)
                recorder = RECORDERS.get(fn)
                if recorder is not None:
                    recorder(run, block, inputs, i, t)
                continue

            # A rated TranFn/PID/... replays from its sampled holder, like the
            # discrete blocks, not through its continuous handler.
            kernel_replay = (
                fn in _KERNEL_REPLAY_FNS or is_sampled_block(block) or is_delay_block(block)
            )
            executor = block_executors.get(b_name) if kernel_replay else None
            if executor is not None:
                # The kernel reads its inputs from step.signals (same keys the
                # replay uses) and writes its primary output there itself.
                executor(t, y_step, replay_dy, step.signals)
                out_val = step.signals.get(b_name, 0.0)
            else:
                handler = REPLAY_HANDLERS.get(fn)
                if handler is not None:
                    out_val = handler(step, block, inputs)
                else:
                    out_val = replay_fallback(block, inputs, t, replay_dt, fallback_failed)
            step.signals[b_name] = out_val

            recorder = RECORDERS.get(fn)
            if recorder is not None:
                recorder(run, block, inputs, i, t)

    finalize_recorders(current_blocks)


def sample_schedule(sampled, t_span, t_eval):
    """Sample instants in ``(t0, t_end]`` for the sampled holders.

    Instants are ``k * Ts`` (offset 0, as ``DBlock`` schedules them); the
    union over all rates is merged within ``1e-9`` of the span and snapped onto
    the output grid, so a grid point and the instant it represents compare
    equal and the replay shows the post-sample value there, as the
    interpreter does.
    """
    t0, tf = float(t_span[0]), float(t_span[1])
    tol = 1e-9 * (abs(tf - t0) or 1.0)
    times = []
    for holder in sampled:
        ts = float(holder["ts"])
        k = np.arange(int(np.floor(t0 / ts + 1e-9)) + 1, int(np.floor(tf / ts + 1e-9)) + 1)
        times.extend(k * ts)
    if not times:
        return np.zeros(0)
    times = np.sort(np.asarray(times, dtype=float))
    grid = np.asarray(t_eval, dtype=float)
    merged = []
    for t in times:
        if t <= t0 + tol:
            continue
        j = int(np.searchsorted(grid, t))
        for cand in (j - 1, j):
            if 0 <= cand < grid.size and abs(grid[cand] - t) <= tol:
                t = float(grid[cand])
                break
        t = min(t, tf)
        if not merged or t - merged[-1] > tol:
            merged.append(t)
    return np.asarray(merged)


def seed_initial_states(model_func, executors, t0, y0, sampled=()):
    """``y0`` with the slots that depend on t0 signals filled in.

    A kernel whose initial state is a function of its inputs at t0 (the
    RateLimiter latches ``y(0) = u(0)``, as the interpreted block does) exposes
    ``seed_state(y0, signals)`` on its executor; the allocator could only
    write a placeholder. One evaluation at ``(t0, y0)`` supplies the signals.
    Sampled blocks take their t0 sample first, so a seeded block behind a ZOH
    sees the first sample rather than the empty holder; their holders are then
    reset (``holder["reset"]``) and given back their mode, so the solve's own
    t0 sample starts from the same clean state as without seeding.

    Returns ``y0`` itself, without evaluating anything, when no kernel seeds;
    otherwise a seeded copy, so the cached compile's ``y0`` is never mutated.
    The linearizer compiles without going through the runner, so there a
    RateLimiter keeps the allocated 0 as its operating point.
    """
    seeders = [f.seed_state for f in (executors or {}).values() if hasattr(f, "seed_state")]
    if not seeders or not len(y0):
        return y0
    modes = [(h, h.get("mode")) for h in sampled]
    try:
        if modes:
            make_sample_hook(model_func, sampled, (t0, t0 + 1.0))(t0, np.asarray(y0, float))
        _dy, signals = model_func.evaluate(t0, np.asarray(y0, dtype=float))
    finally:
        for holder, mode in modes:
            holder["reset"](holder)
            holder["mode"] = mode
    y0 = np.array(y0, dtype=float, copy=True)
    for seed in seeders:
        seed(y0, signals)
    return y0


def make_sample_hook(model_func, sampled, t_span):
    """``on_breakpoint(t, y)``: update every sampled block due at ``t``.

    Returns whether any block was due (False at a breakpoint that only bounds
    a transport delay's segments).

    Blocks update in ``sampled`` order (feeders first, see
    ``system_compiler._sample_update_order``) and the diagram is re-evaluated
    before each one, so a block downstream of a ZOH latches the ZOH's fresh
    sample, not the one it held over the previous period.
    """
    tol = 1e-9 * (abs(float(t_span[1]) - float(t_span[0])) or 1.0)

    def due(holder, t):
        ts = float(holder["ts"])
        return abs(t - round(t / ts) * ts) <= tol

    def on_breakpoint(t, y):
        fired = False
        for holder in sampled:
            if due(holder, t):
                _dy, signals = model_func.evaluate(t, y)
                sample_update(holder, t, signals)
                fired = True
        return fired

    return on_breakpoint


def _no_fixed_step_tail(model_func, grid, y0, scheme):
    """Chattering-guard fallback for a diagram with transport delays.

    The fixed-step tail records no delay history, so the compiled run stops
    here and ``DSim`` falls through to the interpreter.
    """
    raise CompiledDelayError("the zero-crossing guard tripped in a diagram with transport delays")


class _StatelessRHS:
    """One dummy state for a diagram whose only dynamics are sampled blocks.

    The segmented driver needs a state vector to step; the dummy never moves
    and is dropped from the result.
    """

    def __init__(self, model_func):
        self._f = model_func
        self.event_specs = getattr(model_func, "event_specs", None)
        self.mode_states = getattr(model_func, "mode_states", None)
        self.sampled = getattr(model_func, "sampled", None)

    def __call__(self, t, y):
        return np.zeros(1)

    def evaluate(self, t, y, input_overrides=None):
        _dy, signals = self._f.evaluate(t, np.zeros(0), input_overrides)
        return np.zeros(1), signals


def run_compiled_simulation(
    engine,
    blocks: List[DBlock],
    lines: List[Any],
    t_span: Tuple[float, float],
    dt: float,
    cancel_cb=None,
    progress_cb=None,
) -> bool:
    """
    Run the simulation using the compiled fast solver.

    Args:
        cancel_cb: optional ``() -> bool`` polled from inside the integration
            (every few RHS evaluations, in every backend) and the replay. When
            it returns True the run stops, ``engine.compiled_run_cancelled`` is
            set, Scope histories are cut back to the samples actually replayed
            (none if the solve itself was interrupted), and False is returned.
            Callers must check the flag before falling back to the interpreter.
        progress_cb: optional ``(t_now, t_end)`` called with the solver's time
            as the integration advances.
    """
    run_start = time_module.perf_counter()
    engine.compiled_run_cancelled = False
    monitor = _RunMonitor(cancel_cb, progress_cb, t_span[1])
    engine._compiled_run_monitor = monitor if monitor.active else None
    compile_time = 0.0
    solve_time = 0.0
    replay_time = 0.0
    solve_start = None
    compile_cache_hit = False
    method_requested = getattr(engine, "solver_method", "RK45") or "RK45"
    # "auto" is a stored setting, not a scheme; resolve it once here so every
    # branch below (and the diagnostics) sees the scheme that will actually run.
    method_used = resolve_solver_method(method_requested)
    if method_used != method_requested:
        logger.info("Solver 'auto' resolved to %s for this run.", method_used)

    # This path assembles the whole diagram into a single ODE system and
    # integrates it with one scheme, so an Integrator's per-block "method"
    # has no meaning here — asking for Euler in one integrator and RK4 in
    # another is not expressible in one state vector.  It applies only to
    # the interpreter, which steps each block itself.  The equivalent
    # control for this path is the solver method in Simulation settings,
    # which offers Euler and RK4 too.  Say so rather than silently
    # discarding a setting the user deliberately changed.
    scan_blocks = engine.active_blocks_list or getattr(engine.model, "blocks_list", []) or []
    ignored_methods = sorted(
        block.name
        for block in scan_blocks
        if getattr(block, "block_fn", "") == "Integrator"
        # A rated Integrator runs its own execute() at each instant, which
        # honours the per-block method.
        and not is_sampled_block(block)
        and str((getattr(block, "params", None) or {}).get("method", "SOLVE_IVP"))
        not in ("SOLVE_IVP", "")
    )
    if ignored_methods:
        logger.warning(
            f"Per-block Integrator method ignored by the compiled solver for "
            f"{', '.join(ignored_methods)}: this run integrates the whole diagram "
            f"with '{method_requested}' from Simulation settings. The per-block "
            f"method applies to the interpreted solver only — pick the scheme in "
            f"Simulation settings (it offers Euler and RK4), or turn off the fast "
            f"solver to step each block with its own method."
        )
    backend = None
    fallback_reason = None
    zero_crossing_info = None
    rtol = getattr(engine, "rtol", 1e-9)
    atol = getattr(engine, "atol", 1e-12)
    engine.last_solver_diagnostics = {}

    try:
        from scipy.integrate import solve_ivp

        # Check if already initialized (by DSim.execution_init)
        # Skip redundant initialization to avoid 2x overhead
        if len(engine.active_blocks_list) == 0:
            # Not yet initialized - do it now
            if not engine.initialize_execution(blocks, lines):
                logger.error("Failed to initialize execution: %s", engine.error_msg)
                engine.last_solver_diagnostics = {
                    "success": False,
                    "failure_stage": "initialize",
                    "message": engine.error_msg,
                    "total_wall_time": time_module.perf_counter() - run_start,
                }
                return False
        else:
            logger.debug("Engine already initialized, skipping redundant initialization")

        # Use the FLATTENED lists for checking and compilation
        current_blocks = engine.active_blocks_list
        current_lines = engine.active_line_list if engine.active_line_list else lines

        # Resolve parameters before compilation when exec_params is stale
        # or missing.  initialize_execution above (or DSim.execution_init)
        # already resolves with the same sim_dt, so we skip the per-block
        # pass when exec_params['dtime'] is already current.
        workspace_manager = engine._get_workspace()
        ws_fp = workspace_fingerprint(workspace_manager)
        for block in current_blocks:
            engine._resolve_block_params(block, dt, workspace_manager, ws_fp)

        # Final check on flattened system
        # engine.check_compilability, not the compiler's: it also refuses a
        # diagram whose algebraic loops only the interpreter can solve.
        if not engine.check_compilability(current_blocks):
            logger.error("Flattened system contains uncompilable blocks.")
            engine.last_solver_diagnostics = {
                "success": False,
                "failure_stage": "compilability",
                "message": "Flattened system contains uncompilable blocks.",
                "total_wall_time": time_module.perf_counter() - run_start,
            }
            return False

        # Topological sort via hierarchy
        sorted_blocks = sorted(current_blocks, key=lambda b: b.hierarchy)

        logger.info("Compiling system...")
        compile_start = time_module.perf_counter()
        model_func, y0, state_map, block_matrices, compile_cache_hit = (
            engine._compile_system_cached(current_blocks, sorted_blocks, current_lines, dt)
        )
        compile_time = time_module.perf_counter() - compile_start
        delays = getattr(model_func, "delays", None)
        if delays is not None:
            # Before seeding: a t0 evaluation reads the delays' initial value.
            delays.reset(float(t_span[0]), dt, float(t_span[1]) - float(t_span[0]))
        y0 = seed_initial_states(
            model_func,
            getattr(engine.compiler, "block_executors", {}),
            float(t_span[0]),
            y0,
            list(getattr(model_func, "sampled", None) or []),
        )

        logger.info(f"Solving IVP over {t_span} with {len(y0)} states...")
        t_eval = np.arange(t_span[0], t_span[1] + dt, dt)
        # Clip to avoid floating-point overshoot past t_span[1]
        t_eval = t_eval[t_eval <= t_span[1] + 1e-12]
        t_eval[-1] = min(t_eval[-1], t_span[1])

        # Lightweight container matching scipy solve_ivp result interface
        class _SolverResult:
            __slots__ = ("t", "y", "success", "message", "status", "nfev", "njev", "nlu")

        sampled = list(getattr(model_func, "sampled", None) or [])
        monitor.check()
        solve_start = time_module.perf_counter()
        if len(y0) == 0 and not sampled and delays is None:
            # Purely algebraic system
            backend = "algebraic"
            sol = _SolverResult()
            sol.t = t_eval
            sol.y = np.zeros((0, len(t_eval)))
            sol.success = True
            sol.message = "Algebraic system computed successfully"
            sol.status = 0
            sol.nfev = 0
            sol.njev = 0
            sol.nlu = 0
            logger.info("System is algebraic (0 states). Skipping solver.")
        else:
            method = method_used

            # NOTE: there is deliberately no stochastic-block special case
            # here. Every stochastic source (Noise, PacketLoss,
            # NetworkChannel, RandomSource) is excluded from
            # SystemCompiler.COMPILABLE_BLOCKS, so a diagram containing one
            # never reaches this path at all — check_compilability above
            # sends it to the interpreter, which is also the only path that
            # honours their `seed` params (Monte-Carlo reproducibility).
            # If one is ever made compilable it must force a fixed step:
            # an adaptive solver re-samples the RHS per stage and per
            # rejected step, which destroys its error estimate.
            if method in FIXED_STEP_METHODS and delays is not None:
                # SimulationEngine.check_compilability keeps these on the
                # interpreter; reaching here is a gate bug.
                raise CompiledDelayError("transport delays need an adaptive solver")
            if method in FIXED_STEP_METHODS and sampled:
                # Fixed step on the output grid plus every sample instant, the
                # sampled blocks' own updates at each instant.
                backend = "fixed_step+hybrid"
                scheme = "rk4" if method == "RK4" else "euler"
                rhs = model_func if len(y0) else _StatelessRHS(model_func)
                breakpoints = sample_schedule(sampled, t_span, t_eval)
                logger.info(
                    f"Solving with fixed-step {method} across {len(breakpoints)} sample "
                    f"instant(s) of {len(sampled)} sampled block(s)"
                )
                sol = solve_fixed_step_with_breakpoints(
                    monitor.wrap(rhs),
                    t_span,
                    y0 if len(y0) else np.zeros(1),
                    t_eval,
                    scheme,
                    integrate_fixed_step,
                    breakpoints,
                    monitor.wrap_hook(make_sample_hook(rhs, sampled, t_span)),
                    mode_states=getattr(model_func, "mode_states", None),
                )
                if not len(y0):
                    sol.y = sol.y[:0]
                zero_crossing_info = sol.summary()
                zero_crossing_info["enabled"] = False
            elif method in FIXED_STEP_METHODS:
                backend = "fixed_step"
                scheme = "rk4" if method == "RK4" else "euler"
                y_history = integrate_fixed_step(monitor.wrap(model_func), t_eval, y0, scheme)
                sol = _SolverResult()
                sol.t = t_eval
                sol.y = y_history
                sol.success = True
                sol.message = f"Fixed-step {method}"
                sol.status = 0
                evals_per_step = 4 if method == "RK4" else 1
                sol.nfev = max(0, len(t_eval) - 1) * evals_per_step
                sol.njev = 0
                sol.nlu = 0
            else:
                if method not in SCIPY_SOLVER_METHODS:
                    logger.warning(f"Unknown solver '{method}', falling back to RK45")
                    fallback_reason = f"unknown solver '{method}' fell back to RK45"
                    method = "RK45"
                # Zero-crossing detection: only worth its machinery when the
                # diagram actually contains a discontinuity, so a smooth
                # diagram takes the plain single-shot solve_ivp call below
                # with no `events=` argument and no per-probe signal
                # bookkeeping at all.
                event_specs = list(getattr(model_func, "event_specs", None) or [])
                use_events = bool(getattr(engine, "zero_crossing", True)) and bool(event_specs)
                if sampled or delays is not None:
                    # Hybrid: adaptive between sample instants, the sampled
                    # blocks' own updates at each instant. Transport delays
                    # add every k*tau (method of steps, compiled_delays).
                    backend = "scipy+hybrid"
                    rhs = model_func if len(y0) else _StatelessRHS(model_func)
                    taus = [{"ts": tau} for tau in (delays.taus if delays is not None else ())]
                    breakpoints = sample_schedule(list(sampled) + taus, t_span, t_eval)
                    delay_hooks = {}
                    fallback = integrate_fixed_step
                    if delays is not None:
                        delays.model_func = rhs
                        delay_hooks = {"on_segment_end": delays.finish}
                        fallback = _no_fixed_step_tail
                    logger.info(
                        f"Solving with {method} (rtol={rtol}, atol={atol}) between "
                        f"{len(breakpoints)} breakpoint(s): {len(sampled)} sampled block(s), "
                        f"{len(taus)} transport delay(s)"
                    )
                    sol = solve_with_events(
                        monitor.wrap(rhs),
                        t_span,
                        y0 if len(y0) else np.zeros(1),
                        t_eval,
                        event_specs if use_events else [],
                        method=method,
                        rtol=rtol,
                        atol=atol,
                        max_events=int(getattr(engine, "zero_crossing_max_events", 0) or 0)
                        or DEFAULT_MAX_EVENTS,
                        mode_states=getattr(model_func, "mode_states", None),
                        max_step=dt,
                        fallback_integrator=fallback,
                        breakpoints=breakpoints,
                        on_breakpoint=monitor.wrap_hook(make_sample_hook(rhs, sampled, t_span))
                        if sampled
                        else None,
                        **delay_hooks,
                    )
                    if not len(y0):
                        sol.y = sol.y[:0]
                    zero_crossing_info = sol.summary()
                    zero_crossing_info["enabled"] = use_events
                elif use_events:
                    backend = "scipy+events"
                    logger.info(
                        f"Solving with {method} (rtol={rtol}, atol={atol}) and "
                        f"{len(event_specs)} zero-crossing event(s)"
                    )
                    sol = solve_with_events(
                        monitor.wrap(model_func),
                        t_span,
                        y0,
                        t_eval,
                        event_specs,
                        method=method,
                        rtol=rtol,
                        atol=atol,
                        max_events=int(getattr(engine, "zero_crossing_max_events", 0) or 0)
                        or DEFAULT_MAX_EVENTS,
                        mode_states=getattr(model_func, "mode_states", None),
                        # State-dependent guards can cross and come back
                        # inside one adaptive step; capping at the output
                        # step keeps anything the user can see detectable.
                        max_step=dt,
                        # Chattering systems stall an adaptive solver; the
                        # guard hands the tail to the fixed-step scheme.
                        fallback_integrator=integrate_fixed_step,
                    )
                    zero_crossing_info = sol.summary()
                else:
                    backend = "scipy"
                    logger.info(f"Solving with {method} (rtol={rtol}, atol={atol})")
                    sol = solve_ivp(
                        monitor.wrap(model_func),
                        t_span,
                        y0,
                        t_eval=t_eval,
                        method=method,
                        rtol=rtol,
                        atol=atol,
                    )
                    zero_crossing_info = {
                        "enabled": False,
                        "n_events": 0,
                        "reason": "disabled in settings"
                        if not getattr(engine, "zero_crossing", True)
                        else "no discontinuous blocks",
                    }
            method_used = method
        solve_time = time_module.perf_counter() - solve_start

        if not sol.success:
            logger.error(f"Solver failed: {sol.message}")
            engine._record_solver_diagnostics(
                sol=sol,
                success=False,
                method_requested=method_requested,
                method_used=method_used,
                backend=backend,
                t_span=t_span,
                dt=dt,
                rtol=rtol,
                atol=atol,
                n_states=len(y0),
                n_blocks=len(current_blocks),
                n_lines=len(current_lines),
                compile_cache_hit=compile_cache_hit,
                compile_time=compile_time,
                solve_time=solve_time,
                replay_time=0.0,
                total_time=time_module.perf_counter() - run_start,
                fallback_reason=fallback_reason,
                failure_stage="solve",
                zero_crossing=zero_crossing_info,
            )
            return False

        logger.info("Simulation finished. Processing results...")

        # 4. Distribute results
        engine.outs = sol.y
        engine.timeline = sol.t

        output_range = None
        if sol.y.size > 0:
            output_range = {
                "min": float(np.min(sol.y)),
                "max": float(np.max(sol.y)),
            }
            logger.info(
                f"Solver output range: min={output_range['min']:.6f}, max={output_range['max']:.6f}"
            )

        replay_start = time_module.perf_counter()
        # Through the engine's method (not the module function directly) so an
        # override / monkeypatch of _replay_compiled_signals still takes effect,
        # exactly as before the split.
        engine._replay_loop_model = model_func if getattr(model_func, "loops", None) else None
        engine._replay_compiled_signals(
            sol, current_blocks, current_lines, state_map, block_matrices
        )
        replay_time = time_module.perf_counter() - replay_start

        # Stiffness heuristic. Deliberately last: it re-evaluates the compiled
        # RHS a handful of times for a finite-difference Jacobian, and a block
        # holding a latch must not see those probes before the replay has read
        # its state. Returns None whenever it does not apply (implicit or
        # fixed-step method, no states), and never raises.
        stiffness = estimate_stiffness(
            sol=sol,
            method=method_used,
            dt=dt,
            n_states=len(y0),
            model_func=model_func,
        )
        if stiffness and stiffness.get("suspected"):
            logger.warning(format_stiffness_for_log(stiffness, method_used))

        engine._record_solver_diagnostics(
            stiffness=stiffness,
            sol=sol,
            success=True,
            method_requested=method_requested,
            method_used=method_used,
            backend=backend,
            t_span=t_span,
            dt=dt,
            rtol=rtol,
            atol=atol,
            n_states=len(y0),
            n_blocks=len(current_blocks),
            n_lines=len(current_lines),
            compile_cache_hit=compile_cache_hit,
            compile_time=compile_time,
            solve_time=solve_time,
            replay_time=replay_time,
            total_time=time_module.perf_counter() - run_start,
            fallback_reason=fallback_reason,
            output_range=output_range,
            zero_crossing=zero_crossing_info,
        )
        logger.info(
            "Compiled solver diagnostics: %s",
            format_diagnostics_for_log(engine.last_solver_diagnostics),
        )

        return True

    except CompiledRunCancelled as cancel:
        # Not an error and not a reason to fall back to the interpreter: the
        # user asked to stop. Leave the Scopes holding exactly the samples that
        # were replayed (none when the solve itself was interrupted) so a plot
        # never shows a zero-padded tail or the previous run's trace.
        n_done = cancel.n_done
        logger.info("Compiled simulation cancelled (%d sample(s) replayed).", n_done)
        engine.compiled_run_cancelled = True
        truncate_recorders(engine.active_blocks_list or blocks, n_done)
        if n_done:
            # Cancelled in the replay: the solve set timeline / outs.
            engine.timeline = np.asarray(engine.timeline)[:n_done]
            engine.outs = np.asarray(engine.outs)[..., :n_done]
        else:
            engine.timeline = np.zeros(0)
            engine.outs = []
        if not solve_time and solve_start is not None:
            solve_time = time_module.perf_counter() - solve_start
        engine.last_solver_diagnostics = {
            "success": False,
            "failure_stage": "cancelled",
            "message": "Cancelled",
            "method_requested": method_requested,
            "method_used": method_used,
            "backend": backend,
            "compile_cache_hit": bool(compile_cache_hit),
            "compile_wall_time": float(compile_time),
            "solve_wall_time": float(solve_time),
            "total_wall_time": time_module.perf_counter() - run_start,
        }
        return False

    except Exception as e:
        logger.error(f"Compiled simulation failed: {e}", exc_info=True)
        # Surface the exception type/message so callers and the UI can tell
        # an internal bug (KeyError, shape mismatch, ...) apart from a
        # solver failure, instead of only seeing a generic False.
        engine.error_msg = f"Compiled simulation failed: {type(e).__name__}: {e}"
        engine.last_solver_diagnostics = {
            "success": False,
            "failure_stage": "exception",
            "message": engine.error_msg,
            "method_requested": method_requested,
            "method_used": method_used,
            "backend": backend,
            "compile_cache_hit": bool(compile_cache_hit),
            "compile_wall_time": float(compile_time),
            "solve_wall_time": float(solve_time),
            "replay_wall_time": float(replay_time),
            "total_wall_time": time_module.perf_counter() - run_start,
        }
        return False

    finally:
        engine._compiled_run_monitor = None
