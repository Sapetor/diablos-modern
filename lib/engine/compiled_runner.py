"""The compiled (fast-solver) run: ODE solve + post-solve signal replay.

Extracted verbatim from :class:`~lib.engine.simulation_engine.SimulationEngine`
(2100 lines, with a 570-line replay method inside it).  Every function here
takes the engine as its first argument and reads/writes exactly the attributes
the methods did; ``SimulationEngine.run_compiled_simulation`` and
``_replay_compiled_signals`` remain as thin delegating methods, so no caller
(GUI, DSim, scripts, tests) had to change.

The split follows the two halves of a fast-solver run:

* :func:`run_compiled_simulation` -- resolve params, compile the diagram into a
  single ODE, integrate it (scipy, or an in-house fixed step), record
  diagnostics.
* :func:`replay_compiled_signals` -- walk the saved state trajectory and
  reconstruct every block's output, so Scope / FieldScope blocks get their
  history (the solve itself only returns the ODE state).
"""

import logging
import time as time_module
from typing import Any, Dict, List, Tuple

import numpy as np

from lib.engine.block_names import canonical_fn
from lib.engine.block_params import runtime_params
from lib.engine.pde_ops import wave_energy_1d
from lib.engine.solver_diagnostics import format_diagnostics_for_log
from lib.safe_eval import SafeEvalError, safe_expr, safe_literal
from lib.simulation.block import DBlock
from lib.workspace import WorkspaceManager
from lib.engine.topo import kahn_topological_order

logger = logging.getLogger(__name__)

# Compiled-solver methods that scipy.integrate.solve_ivp accepts directly.
SCIPY_SOLVER_METHODS = ("RK45", "RK23", "DOP853", "Radau", "BDF", "LSODA")
# Fixed-step schemes integrated in-house (use the simulation step dt).
FIXED_STEP_METHODS = ("Euler", "RK4")

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
# StateVariable (discrete pending-update state), Demux (secondary-port outputs),
# and Hysteresis (relay state lives in a kernel closure that the out-of-order
# solve phase pollutes and that has no per-run reset).
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

    if fn == "Integrator":
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


def replay_compiled_signals(engine, sol, current_blocks, current_lines, state_map, block_matrices):
    """Re-evaluate every block at each saved solver time so Scope /
    FieldScope blocks capture their signal history.

    The compiled solver returns only the ODE state trajectory; this
    post-solve 'replay' reconstructs every block's output from it,
    reusing the compiled kernels for the routed blocks (see
    _KERNEL_REPLAY_FNS) and inline branches for the genuinely-divergent
    ones (Integrator, PDE / Field, Mathfunction, StateVariable, Demux,
    Hysteresis). Mutates each Scope / FieldScope block's exec_params.
    """
    # For Scope visualization, we need to populate block outputs.
    # We "replay" the simulation using the solution to capture all signals.
    num_steps = len(sol.t)

    # Replay Sort: topological order respecting Direct Feedthrough.
    # Only feedthrough edges constrain the order — a state block's output
    # is its (already-known) state, so it does not depend on this step's
    # inputs and need not follow its source.
    adj = {b.name: [] for b in current_blocks}
    # name -> block lookup, built once to avoid repeated linear scans
    block_by_name = {b.name: b for b in current_blocks}

    # Feedthrough is a property of the destination block, so classify each
    # one once instead of once per incoming line.
    feedthrough_by_dst: Dict[str, bool] = {}

    for line in current_lines:
        src = line.srcblock
        dst = line.dstblock

        # Check Direct Feedthrough
        dst_block = block_by_name.get(dst)
        if dst_block is None:
            is_feedthrough = True
        else:
            is_feedthrough = feedthrough_by_dst.get(dst)
            if is_feedthrough is None:
                is_feedthrough = replay_has_feedthrough(dst_block, block_matrices)
                feedthrough_by_dst[dst] = is_feedthrough

        if is_feedthrough and src in adj:
            adj[src].append(dst)

    # Kahn's algorithm, stable on block name for determinism (e.g. step
    # before sine if independent). Any cycle leftovers are appended in
    # current-block order (best effort) so every block still runs.
    order_names, leftover_names = kahn_topological_order(
        (b.name for b in current_blocks), adj, key=lambda n: n
    )
    sorted_blocks = [block_by_name[n] for n in order_names] + [
        block_by_name[n] for n in leftover_names
    ]

    # Precompute dst_name -> list of (srcblock, srcport, dstport) once so
    # the per-step replay does not rescan every connection for every
    # block (was O(steps * blocks * lines)).
    inputs_by_dst: Dict[str, List[Tuple[str, int, int]]] = {}
    for line in current_lines:
        src_port = getattr(line, "srcport", 0) or 0
        inputs_by_dst.setdefault(line.dstblock, []).append((line.srcblock, src_port, line.dstport))

    # Pure-function blocks (see _KERNEL_REPLAY_FNS) reuse their compiled
    # kernel executor during replay instead of a duplicated inline
    # computation. block_executors was populated by compile_system above.
    block_executors = getattr(engine.compiler, "block_executors", {})
    replay_dy = np.zeros(sol.y.shape[0])  # scratch dy_vec; pure kernels never write it

    # Step size the replay advances by, for the execute() fallback below.
    # The solve wrote its results on the fixed t_eval grid, so the spacing
    # is uniform; fall back to the engine's dt for a degenerate grid.
    replay_dt = float(sol.t[1] - sol.t[0]) if num_steps > 1 else float(engine.sim_dt)
    # Blocks whose fallback execute() already failed once, so the warning
    # is emitted once per block instead of once per block per step.
    fallback_failed = set()
    # Scopes whose sample width changed mid-run (warned once each).
    scope_width_warned = set()

    # Replay Loop
    for i in range(num_steps):
        t = sol.t[i]
        y_step = sol.y[:, i] if sol.y.ndim > 1 else sol.y

        # 1. State Map - Populate 'current_states' first
        # Output 'signals' populate diffently based on block type.
        current_signals = {}
        current_states = {}  # b_name -> x

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
            # Normalize function name (single source of truth: lib.engine.block_names)
            fn = canonical_fn(block.block_fn)

            # Collect inputs
            inputs = {}
            for srcblock, src_port, dstport in inputs_by_dst.get(b_name, ()):
                # Direct feedthrough lookup - handle multi-output blocks
                if src_port == 0:
                    val = current_signals.get(srcblock, 0.0)
                else:
                    # Secondary output - use suffix naming convention
                    # Check for common suffixes used by multi-output blocks
                    src_key = f"{srcblock}_out{src_port}"
                    val = current_signals.get(src_key, current_signals.get(srcblock, 0.0))
                inputs[dstport] = val

            out_val = 0.0

            executor = block_executors.get(b_name) if fn in _KERNEL_REPLAY_FNS else None
            if executor is not None:
                # Reuse the compiled kernel as the single source of truth
                # for this block's output math. It reads inputs from
                # current_signals (same {src} / {src}_out{p} keys the
                # replay uses) and writes the primary output to
                # current_signals[b_name]; pure-function kernels never
                # touch replay_dy.
                executor(t, y_step, replay_dy, current_signals)
                out_val = current_signals.get(b_name, out_val)
            elif fn == "Integrator":
                # Valid because Integrator state output is just the state
                if b_name in current_states:
                    val = current_states[b_name]
                    out_val = val if val.size > 1 else val.item()

            # ==================== PDE BLOCKS ====================
            elif fn == "Heatequation1D":
                # HeatEquation1D: output is the temperature field (state vector)
                if b_name in current_states:
                    T = current_states[b_name]
                    out_val = T
                    current_signals[b_name + "_out1"] = float(np.mean(T))  # T_avg
                else:
                    out_val = np.zeros(int(block.params.get("N", 20)))

            elif fn == "Waveequation1D":
                # WaveEquation1D: state is [u, v], output primary is u (displacement)
                N = int(block.params.get("N", 50))
                if b_name in current_states:
                    state = current_states[b_name]
                    u = state[:N]  # Displacement field
                    v = state[N:]  # Velocity field
                    out_val = u
                    current_signals[b_name + "_out1"] = v  # v_field
                    # Full kinetic+potential energy, single-sourced with the
                    # block's execute() via pde_ops.wave_energy_1d so the
                    # solve/replay path reports the same energy the
                    # interpreter does (was a kinetic-only "Simplified"
                    # inline that diverged from the block output).
                    current_signals[b_name + "_out2"] = float(wave_energy_1d(u, v, block.params))
                else:
                    out_val = np.zeros(N)

            elif fn == "Advectionequation1D":
                # AdvectionEquation1D: output is concentration field
                if b_name in current_states:
                    c = current_states[b_name]
                    out_val = c
                    L = float(block.params.get("L", 1.0))
                    N = len(c)
                    dx = L / (N - 1) if N > 1 else 1.0
                    current_signals[b_name + "_out1"] = float(np.sum(c) * dx)  # c_total
                else:
                    out_val = np.zeros(int(block.params.get("N", 50)))

            elif fn == "Diffusionreaction1D":
                # DiffusionReaction1D: output is concentration field
                if b_name in current_states:
                    c = current_states[b_name]
                    out_val = c
                    L = float(block.params.get("L", 1.0))
                    N = len(c)
                    dx = L / (N - 1) if N > 1 else 1.0
                    current_signals[b_name + "_out1"] = float(np.sum(c) * dx)  # c_total
                    k = float(block.params.get("k", 0.1))
                    n_order = int(block.params.get("n", 1))
                    reaction = np.sum(k * np.power(np.maximum(c, 0), n_order)) * dx
                    current_signals[b_name + "_out2"] = float(reaction)  # reaction_rate
                else:
                    out_val = np.zeros(int(block.params.get("N", 50)))

            # ==================== 2D PDE BLOCKS ====================
            elif fn == "Heatequation2D":
                # HeatEquation2D: output is 2D temperature field
                Nx = int(block.params.get("Nx", 20))
                Ny = int(block.params.get("Ny", 20))
                if b_name in current_states:
                    state = current_states[b_name]
                    T_field = state.reshape((Ny, Nx))
                else:
                    T_field = np.zeros((Ny, Nx))
                out_val = T_field
                # Store secondary outputs for multi-port access
                current_signals[b_name + "_out1"] = float(np.mean(T_field))  # T_avg
                current_signals[b_name + "_out2"] = float(np.max(T_field))  # T_max

            elif fn == "Fieldprobe2D":
                # FieldProbe2D: bilinear interpolation from 2D field
                field = inputs.get(0, None)
                if field is None or not isinstance(field, np.ndarray) or field.ndim != 2:
                    out_val = 0.0
                else:
                    Ny_f, Nx_f = field.shape
                    x_pos = float(block.params.get("x_position", 0.5))
                    y_pos = float(block.params.get("y_position", 0.5))
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
                    out_val = (
                        field[j0, i0] * (1 - di) * (1 - dj)
                        + field[j0, i1] * di * (1 - dj)
                        + field[j1, i0] * (1 - di) * dj
                        + field[j1, i1] * di * dj
                    )

            elif fn == "Fieldscope2D":
                # FieldScope2D: pass through 2D field
                field = inputs.get(0, np.zeros((1, 1)))
                out_val = np.atleast_2d(field)

            elif fn == "Fieldslice":
                # FieldSlice: extract 1D slice from 2D field
                field = inputs.get(0, None)
                if field is None or not isinstance(field, np.ndarray) or field.ndim != 2:
                    out_val = np.array([0.0])
                else:
                    Ny_f, Nx_f = field.shape
                    direction = block.params.get("slice_direction", "x")
                    position = float(block.params.get("slice_position", 0.5))
                    if direction.lower() == "x":
                        j = int(position * (Ny_f - 1))
                        j = max(0, min(Ny_f - 1, j))
                        out_val = field[j, :]
                    else:
                        i = int(position * (Nx_f - 1))
                        i = max(0, min(Nx_f - 1, i))
                        out_val = field[:, i]

            elif fn in ("Statevariable", "StateVariable"):
                # StateVariable: manage discrete state across iterations
                # State is stored in block.params for persistence across replay steps
                # Key insight: We must update state from PREVIOUS iteration's computed input
                # before outputting, not after.
                # Re-initialize whenever _init_start_ is True (mirroring the
                # Hysteresis branch) so reset_memblocks takes effect across
                # runs; otherwise a second run would keep the previous run's
                # final state instead of resetting to initial_value.
                if block.params.get("_init_start_", True) or "_replay_state_" not in block.params:
                    initial = block.params.get("initial_value", [1.0])
                    if isinstance(initial, str):
                        try:
                            initial = safe_literal(initial)
                        except (SafeEvalError, ValueError, SyntaxError):
                            initial = [1.0]
                    # Preserve full vector state, not just first element
                    block.params["_replay_state_"] = np.atleast_1d(initial).copy()
                    block.params["_replay_pending_"] = None  # Input from previous step
                    block.params["_init_start_"] = False

                # First: Apply pending update from previous iteration
                if block.params["_replay_pending_"] is not None:
                    block.params["_replay_state_"] = block.params["_replay_pending_"]
                    block.params["_replay_pending_"] = None

                # Output current state (preserve vector or return scalar if 1D)
                state = block.params["_replay_state_"]
                out_val = state if np.atleast_1d(state).size > 1 else float(np.atleast_1d(state)[0])

                # Store input for next iteration (will be applied next time step)
                if 0 in inputs:
                    new_val = inputs[0]
                    # Preserve full vector, not just first element
                    block.params["_replay_pending_"] = np.atleast_1d(new_val).copy()

            elif fn == "Mathfunction":
                # Keep the input as an array so vector signals work:
                # float(...) raises on a multi-element array (numpy 2.x).
                val = np.asarray(inputs.get(0, 0.0), dtype=float)
                # Check both 'function' and 'expression' keys for backward compatibility
                func_raw = block.params.get("function", block.params.get("expression", "sin"))
                func = str(func_raw).lower()

                try:
                    if func == "sin":
                        out_val = np.sin(val)
                    elif func == "cos":
                        out_val = np.cos(val)
                    elif func == "tan":
                        out_val = np.tan(val)
                    elif func == "asin":
                        with np.errstate(invalid="ignore"):
                            out_val = np.where(
                                np.abs(val) <= 1, np.arcsin(np.clip(val, -1.0, 1.0)), 0.0
                            )
                    elif func == "acos":
                        with np.errstate(invalid="ignore"):
                            out_val = np.where(
                                np.abs(val) <= 1, np.arccos(np.clip(val, -1.0, 1.0)), 0.0
                            )
                    elif func == "atan":
                        out_val = np.arctan(val)
                    elif func == "exp":
                        out_val = np.exp(val)
                    elif func == "log":
                        with np.errstate(divide="ignore", invalid="ignore"):
                            out_val = np.where(val > 0, np.log(np.where(val > 0, val, 1.0)), 0.0)
                    elif func == "log10":
                        with np.errstate(divide="ignore", invalid="ignore"):
                            out_val = np.where(val > 0, np.log10(np.where(val > 0, val, 1.0)), 0.0)
                    elif func == "sqrt":
                        out_val = np.where(val >= 0, np.sqrt(np.where(val >= 0, val, 0.0)), 0.0)
                    elif func == "square":
                        out_val = val * val
                    elif func == "sign":
                        out_val = np.sign(val)
                    elif func == "abs":
                        out_val = np.abs(val)
                    elif func == "ceil":
                        out_val = np.ceil(val)
                    elif func == "floor":
                        out_val = np.floor(val)
                    elif func == "reciprocal":
                        out_val = np.where(val != 0, 1.0 / np.where(val != 0, val, 1.0), 0.0)
                    elif func == "cube":
                        out_val = val * val * val
                    else:
                        # Python expression fallback (vectorized: no float()).
                        out_val = safe_expr(str(func_raw), variables={"u": val, "t": t})
                except (ValueError, ZeroDivisionError):
                    out_val = 0.0

            elif fn == "Hysteresis":
                # Relay latch state is scalar; reduce vector inputs safely.
                val = float(np.ravel(inputs.get(0, 0.0))[0])
                upper = float(block.params.get("upper", 0.5))
                lower = float(block.params.get("lower", -0.5))
                high_val = float(block.params.get("high", 1.0))
                low_val = float(block.params.get("low", 0.0))

                # Get or initialize persistent state for replay (in exec_params,
                # not on the block instance).
                # Re-initialize whenever _init_start_ is True so reset_memblocks takes effect.
                if (
                    block.exec_params.get("_init_start_", True)
                    or "_replay_hyst_state_" not in block.exec_params
                ):
                    block.exec_params["_replay_hyst_state_"] = low_val
                    block.exec_params["_init_start_"] = False

                if val >= upper:
                    block.exec_params["_replay_hyst_state_"] = high_val
                elif val <= lower:
                    block.exec_params["_replay_hyst_state_"] = low_val

                out_val = block.exec_params["_replay_hyst_state_"]

            elif fn == "Demux":
                # Split the vector input into N consecutive sub-vectors of
                # length output_shape each (mirrors blocks/demux.py). Port
                # 0 is the primary out_val (stored at signals[b_name]);
                # secondary ports use the "{b_name}_out{i}" convention.
                arr = np.atleast_1d(np.asarray(inputs.get(0, 0.0), dtype=float)).flatten()
                output_shape = int(block.params.get("output_shape", 1))
                if output_shape < 1:
                    output_shape = 1
                n_outputs = int(block.params.get("_outputs_", getattr(block, "out_ports", 1)))
                if n_outputs < 1:
                    n_outputs = 1
                out_val = arr[0:output_shape]
                for p in range(1, n_outputs):
                    current_signals[b_name + f"_out{p}"] = arr[
                        p * output_shape : (p + 1) * output_shape
                    ]

            elif fn == "Fieldprobe":
                # FieldProbe: Extract value at position from field array
                field = inputs.get(0, np.array([0.0]))
                field = np.atleast_1d(field).flatten()

                position = float(block.params.get("position", 0.5))
                mode = block.params.get("position_mode", "normalized")
                L = float(block.params.get("L", 1.0))
                N = len(field)

                if N == 0:
                    out_val = 0.0
                else:
                    if mode == "normalized":
                        idx_float = position * (N - 1)
                    else:
                        idx_float = (position / L) * (N - 1)

                    idx_float = max(0, min(N - 1, idx_float))
                    idx_low = int(np.floor(idx_float))
                    idx_high = min(idx_low + 1, N - 1)
                    frac = idx_float - idx_low

                    out_val = field[idx_low] * (1 - frac) + field[idx_high] * frac

            elif fn == "Fieldscope":
                # FieldScope: Store field for 2D visualization
                field = inputs.get(0, np.array([0.0]))
                out_val = np.atleast_1d(field).flatten()

            elif fn in ("Terminator", "Display", "Scope"):
                # Sinks: nothing to compute. A Scope in particular must NOT
                # reach the execute() fallback below -- ScopeBlock.execute
                # writes its own 'vector' into the params dict it is given,
                # which would clobber the replay history recorded further
                # down (the fallback used to be handed raw `params`, so the
                # collision was invisible; it now gets exec_params).
                pass

            else:
                # Fallback: call block.execute() for unhandled block types
                # This handles optimization primitives and custom blocks.
                # Pass the *resolved* params (runtime_params) -- raw params
                # still hold workspace-variable names as strings -- and the
                # replay step, which stateful blocks need to advance.
                if block.block_instance is not None:
                    try:
                        result = block.block_instance.execute(
                            time=t,
                            inputs=inputs,
                            params=runtime_params(block),
                            dtime=replay_dt,
                        )
                        if result and 0 in result:
                            out_val = result[0]
                    except Exception as e:
                        if b_name not in fallback_failed:
                            fallback_failed.add(b_name)
                            logger.warning(
                                "Replay fallback execute() failed for %s (%s): %s. "
                                "Its recorded signal stays 0.0 for this run "
                                "(further failures for this block are not logged).",
                                b_name,
                                block.block_fn,
                                e,
                                exc_info=True,
                            )

            # Store
            current_signals[b_name] = out_val
            # logger.info(f"DEBUG Replay {b_name} t={t:.2f} out={out_val}") # Uncomment for verbose debug

            # Store in Block History for Scopes
            # ScopePlotter expects `block.out_history` list? Or `block.params['vector']`?
            # DSim.execution_loop doesn't seem to append to `out_history` explicitly?
            # Ah, `Scope` blocks have internal `execute` that saves to `vector`.
            # Standard blocks don't save history unless probed.
            # But Scopes DO.
            if fn == "Scope":
                # Scope can have multiple inputs - collect all of them
                # Ensure we write to exec_params as ScopePlotter prioritizes it
                if not hasattr(block, "exec_params"):
                    block.exec_params = block.params.copy()

                # Get number of input ports
                n_inputs = block.in_ports if hasattr(block, "in_ports") else 1

                # Collect and flatten all input values (matching Scope.execute() behavior)
                # Each port value is flattened to 1D so vector signals (e.g. StateSpace
                # with 4 outputs) are properly expanded into individual components.
                combined = []
                for port in range(n_inputs):
                    val = inputs.get(port, 0.0)
                    combined.append(np.atleast_1d(val).flatten())
                new_sample = np.concatenate(combined) if combined else np.array([0.0])
                vec_dim = len(new_sample)

                # Initialize the history buffer and labels on the first
                # timestep.  The sample count is known up front, so the
                # history is preallocated as the (num_steps, vec_dim) array
                # the plotter ultimately wants, instead of growing a Python
                # list of arrays and converting it at the end.
                if i == 0:
                    try:
                        sample_dtype = np.promote_types(new_sample.dtype, np.float64)
                    except TypeError:
                        sample_dtype = float
                    block.exec_params["vector"] = np.zeros((num_steps, vec_dim), dtype=sample_dtype)
                    block.exec_params["vec_dim"] = vec_dim
                    # Set vec_labels from 'labels' param (Scope uses 'labels', plotter reads 'vec_labels')
                    labels_raw = block.params.get("labels", block.exec_params.get("labels", ""))
                    # Guard against non-string labels (e.g. a list/dict):
                    # calling string methods would raise AttributeError
                    # that the broad except would mask as a generic failure.
                    if isinstance(labels_raw, str) and labels_raw and labels_raw != "default":
                        labels_list = [
                            l.strip() for l in labels_raw.replace(" ", "").split(",") if l.strip()
                        ]
                        # Pad or trim to match actual signal dimension
                        while len(labels_list) < vec_dim:
                            labels_list.append(f"{b_name}-{len(labels_list)}")
                        labels_list = labels_list[:vec_dim]
                    else:
                        labels_list = [f"{b_name}-{j}" for j in range(vec_dim)]
                    block.exec_params["vec_labels"] = labels_list

                history = block.exec_params["vector"]
                width = history.shape[1]
                if new_sample.size == width:
                    history[i] = new_sample
                else:
                    # A width change mid-run used to build a ragged list
                    # that np.array() rejected outright at the end of the
                    # replay (losing the whole trace); keep the first
                    # width, pad/truncate, and say so once.
                    n_common = min(new_sample.size, width)
                    history[i, :n_common] = new_sample[:n_common]
                    if b_name not in scope_width_warned:
                        scope_width_warned.add(b_name)
                        logger.warning(
                            "Scope %s: input width changed mid-run (%d -> %d); "
                            "padding/truncating to the first width.",
                            b_name,
                            width,
                            new_sample.size,
                        )

                if i == num_steps - 1:
                    vec = block.exec_params["vector"]
                    logger.info(
                        f"Replay Scope {b_name}: vec_dim={vec_dim}, samples={len(vec)}, labels={block.exec_params.get('vec_labels')}"
                    )

            if fn == "Fieldscope":
                # FieldScope: Store field history for 2D heatmap
                field = inputs.get(0, np.array([0.0]))
                field = np.atleast_1d(field).flatten()

                if not hasattr(block, "exec_params"):
                    block.exec_params = block.params.copy()

                if i == 0:
                    block.exec_params["_field_history_"] = []
                    block.exec_params["_time_history_"] = []

                block.exec_params["_field_history_"].append(field.copy())
                block.exec_params["_time_history_"].append(t)

                if i == num_steps - 1:
                    logger.info(
                        f"DEBUG Replay FieldScope {b_name}: field_len={len(field)}, history_len={len(block.exec_params['_field_history_'])}"
                    )

            if fn == "Fieldscope2D":
                # FieldScope2D: Store 2D field history for animated heatmap
                field = inputs.get(0, np.zeros((1, 1)))
                field = np.atleast_2d(field)

                if not hasattr(block, "exec_params"):
                    block.exec_params = block.params.copy()

                if i == 0:
                    block.exec_params["_field_history_2d_"] = []
                    block.exec_params["_time_history_"] = []

                # Store every N frames to reduce memory
                sample_interval = int(block.params.get("sample_interval", 5))
                if i % sample_interval == 0:
                    block.exec_params["_field_history_2d_"].append(field.copy())
                    block.exec_params["_time_history_"].append(t)

                if i == num_steps - 1:
                    logger.info(
                        f"DEBUG Replay FieldScope2D {b_name}: field_shape={field.shape}, history_len={len(block.exec_params['_field_history_2d_'])}"
                    )

    # Finalize Scope Vectors (convert to numpy)
    for block in current_blocks:
        if block.block_fn == "Scope":
            # Already the preallocated (num_steps, vec_dim) array in the
            # normal path; asarray only matters for a leftover list from a
            # previous interpreted run when the replay recorded no steps.
            if hasattr(block, "exec_params") and "vector" in block.exec_params:
                block.exec_params["vector"] = np.asarray(block.exec_params["vector"])
        elif block.block_fn == "FieldScope":
            if hasattr(block, "exec_params") and "_field_history_" in block.exec_params:
                block.exec_params["_field_history_"] = np.array(
                    block.exec_params["_field_history_"]
                )
            if hasattr(block, "exec_params") and "_time_history_" in block.exec_params:
                block.exec_params["_time_history_"] = np.array(block.exec_params["_time_history_"])
        elif block.block_fn == "FieldScope2D":
            if hasattr(block, "exec_params") and "_field_history_2d_" in block.exec_params:
                block.exec_params["_field_history_2d_"] = np.array(
                    block.exec_params["_field_history_2d_"]
                )
            if hasattr(block, "exec_params") and "_time_history_" in block.exec_params:
                block.exec_params["_time_history_"] = np.array(block.exec_params["_time_history_"])


def run_compiled_simulation(
    engine, blocks: List[DBlock], lines: List[Any], t_span: Tuple[float, float], dt: float
) -> bool:
    """
    Run the simulation using the compiled fast solver.
    """
    run_start = time_module.perf_counter()
    compile_time = 0.0
    solve_time = 0.0
    replay_time = 0.0
    compile_cache_hit = False
    method_requested = getattr(engine, "solver_method", "RK45") or "RK45"
    method_used = method_requested

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
                logger.error("Failed to initialize execution (algebraic loop or error).")
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
        workspace_manager = WorkspaceManager()
        for block in current_blocks:
            engine._resolve_block_params(block, dt, workspace_manager)

        # Final check on flattened system
        if not engine.compiler.check_compilability(current_blocks):
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

        logger.info(f"Solving IVP over {t_span} with {len(y0)} states...")
        t_eval = np.arange(t_span[0], t_span[1] + dt, dt)
        # Clip to avoid floating-point overshoot past t_span[1]
        t_eval = t_eval[t_eval <= t_span[1] + 1e-12]
        t_eval[-1] = min(t_eval[-1], t_span[1])

        # Lightweight container matching scipy solve_ivp result interface
        class _SolverResult:
            __slots__ = ("t", "y", "success", "message", "status", "nfev", "njev", "nlu")

        solve_start = time_module.perf_counter()
        if len(y0) == 0:
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
            method = method_requested

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
            if method in FIXED_STEP_METHODS:
                backend = "fixed_step"
                scheme = "rk4" if method == "RK4" else "euler"
                y_history = integrate_fixed_step(model_func, t_eval, y0, scheme)
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
                backend = "scipy"
                logger.info(f"Solving with {method} (rtol={rtol}, atol={atol})")
                sol = solve_ivp(
                    model_func, t_span, y0, t_eval=t_eval, method=method, rtol=rtol, atol=atol
                )
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
        engine._replay_compiled_signals(
            sol, current_blocks, current_lines, state_map, block_matrices
        )
        replay_time = time_module.perf_counter() - replay_start

        engine._record_solver_diagnostics(
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
        )
        logger.info(
            "Compiled solver diagnostics: %s",
            format_diagnostics_for_log(engine.last_solver_diagnostics),
        )

        return True

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
