"""Per-block handlers for the compiled solver's post-solve signal replay.

:func:`lib.engine.compiled_runner.replay_compiled_signals` walks the saved
state trajectory and re-evaluates every block at each sample so Scope /
FieldScope blocks get their history. Blocks whose compiled kernel is
reproducible from the replayed state are routed through that kernel
(``_KERNEL_REPLAY_FNS`` in ``compiled_runner``); this module holds everything
else:

* :data:`REPLAY_HANDLERS` -- ``canonical_fn -> handler(step, block, inputs)``
  for the blocks whose replay genuinely differs from their kernel (or that
  have none): Integrator, the PDE / Field blocks, StateVariable,
  MathFunction, Hysteresis, Demux and the sinks. A handler returns the
  block's primary output and may write secondary ports into
  ``step.signals`` under the ``"{name}_out{port}"`` convention.
* :data:`RECORDERS` -- ``canonical_fn -> recorder(run, block, inputs, i, t)``
  that append a sample to a Scope / FieldScope / FieldScope2D history, and
  :func:`finalize_recorders` that turns those histories into arrays.
* :func:`replay_fallback` -- ``block.execute()`` for anything unlisted
  (optimization primitives, user blocks).

Every handler reads its parameters from ``block.params`` exactly as the
former inline branches did; state that must persist across replay steps
lives in ``block.params`` / ``block.exec_params``, never on the handler.
"""

import logging
from typing import Any, Callable, Dict, Set

import numpy as np

from lib.engine.block_params import runtime_params
from lib.engine.pde_ops import wave_energy_1d
from lib.safe_eval import SafeEvalError, safe_expr, safe_literal

logger = logging.getLogger(__name__)


class ReplayStep:
    """What one replay sample exposes to the handlers.

    ``t`` is the sample time, ``states`` maps a state block's name to its
    slice of the ODE state at ``t``, and ``signals`` collects every block's
    outputs computed so far this step (primary under the block name,
    secondary ports under ``"{name}_out{port}"``).
    """

    __slots__ = ("t", "states", "signals")

    def __init__(self, t: float, states: Dict[str, np.ndarray], signals: Dict[str, Any]):
        self.t = t
        self.states = states
        self.signals = signals


class ReplayRun:
    """Per-run bookkeeping shared by the recorders."""

    __slots__ = ("num_steps", "scope_width_warned")

    def __init__(self, num_steps: int):
        self.num_steps = num_steps
        # Scopes whose sample width changed mid-run (warned once each).
        self.scope_width_warned: Set[str] = set()


Handler = Callable[[ReplayStep, Any, Dict[int, Any]], Any]
Recorder = Callable[[ReplayRun, Any, Dict[int, Any], int, float], None]


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------


def _replay_sink(step, block, inputs):
    # Sinks compute nothing. A Scope in particular must NOT reach
    # replay_fallback: ScopeBlock.execute writes its own 'vector' into the
    # params dict it is given, which would clobber the replayed history.
    return 0.0


def _replay_integrator(step, block, inputs):
    # Valid because an Integrator's output is its state.
    val = step.states.get(block.name)
    if val is None:
        return 0.0
    return val if val.size > 1 else val.item()


def _replay_heat_1d(step, block, inputs):
    T = step.states.get(block.name)
    if T is None:
        return np.zeros(int(block.params.get("N", 20)))
    step.signals[block.name + "_out1"] = float(np.mean(T))  # T_avg
    return T


def _replay_wave_1d(step, block, inputs):
    # State is [u, v]; the primary output is the displacement u.
    N = int(block.params.get("N", 50))
    state = step.states.get(block.name)
    if state is None:
        return np.zeros(N)
    u = state[:N]
    v = state[N:]
    step.signals[block.name + "_out1"] = v  # v_field
    # Full kinetic+potential energy, single-sourced with the block's
    # execute() via pde_ops.wave_energy_1d so the solve/replay path reports
    # the same energy the interpreter does.
    step.signals[block.name + "_out2"] = float(wave_energy_1d(u, v, block.params))
    return u


def _total_over_domain(c, block):
    L = float(block.params.get("L", 1.0))
    N = len(c)
    dx = L / (N - 1) if N > 1 else 1.0
    return dx, float(np.sum(c) * dx)


def _replay_advection_1d(step, block, inputs):
    c = step.states.get(block.name)
    if c is None:
        return np.zeros(int(block.params.get("N", 50)))
    _dx, c_total = _total_over_domain(c, block)
    step.signals[block.name + "_out1"] = c_total
    return c


def _replay_diffusion_reaction_1d(step, block, inputs):
    c = step.states.get(block.name)
    if c is None:
        return np.zeros(int(block.params.get("N", 50)))
    dx, c_total = _total_over_domain(c, block)
    step.signals[block.name + "_out1"] = c_total
    k = float(block.params.get("k", 0.1))
    n_order = int(block.params.get("n", 1))
    reaction = np.sum(k * np.power(np.maximum(c, 0), n_order)) * dx
    step.signals[block.name + "_out2"] = float(reaction)  # reaction_rate
    return c


def _replay_heat_2d(step, block, inputs):
    Nx = int(block.params.get("Nx", 20))
    Ny = int(block.params.get("Ny", 20))
    state = step.states.get(block.name)
    T_field = state.reshape((Ny, Nx)) if state is not None else np.zeros((Ny, Nx))
    step.signals[block.name + "_out1"] = float(np.mean(T_field))  # T_avg
    step.signals[block.name + "_out2"] = float(np.max(T_field))  # T_max
    return T_field


def _field_2d(inputs):
    field = inputs.get(0, None)
    if field is None or not isinstance(field, np.ndarray) or field.ndim != 2:
        return None
    return field


def _replay_field_probe_2d(step, block, inputs):
    # Bilinear interpolation at a normalized (x, y) position.
    field = _field_2d(inputs)
    if field is None:
        return 0.0
    Ny_f, Nx_f = field.shape
    x_norm = max(0, min(1, float(block.params.get("x_position", 0.5))))
    y_norm = max(0, min(1, float(block.params.get("y_position", 0.5))))
    i_float = x_norm * (Nx_f - 1)
    j_float = y_norm * (Ny_f - 1)
    i0 = int(np.floor(i_float))
    i1 = min(i0 + 1, Nx_f - 1)
    j0 = int(np.floor(j_float))
    j1 = min(j0 + 1, Ny_f - 1)
    di = i_float - i0
    dj = j_float - j0
    return (
        field[j0, i0] * (1 - di) * (1 - dj)
        + field[j0, i1] * di * (1 - dj)
        + field[j1, i0] * (1 - di) * dj
        + field[j1, i1] * di * dj
    )


def _replay_field_scope_2d(step, block, inputs):
    return np.atleast_2d(inputs.get(0, np.zeros((1, 1))))


def _replay_field_slice(step, block, inputs):
    field = _field_2d(inputs)
    if field is None:
        return np.array([0.0])
    Ny_f, Nx_f = field.shape
    direction = block.params.get("slice_direction", "x")
    position = float(block.params.get("slice_position", 0.5))
    if direction.lower() == "x":
        j = max(0, min(Ny_f - 1, int(position * (Ny_f - 1))))
        return field[j, :]
    i = max(0, min(Nx_f - 1, int(position * (Nx_f - 1))))
    return field[:, i]


def _replay_state_variable(step, block, inputs):
    # Discrete state with a one-step delay: the input seen at step k becomes
    # the output at step k+1. State lives in block.params so it persists
    # across replay steps; it is re-initialized whenever _init_start_ is
    # True so reset_memblocks takes effect between runs.
    params = block.params
    if params.get("_init_start_", True) or "_replay_state_" not in params:
        initial = params.get("initial_value", [1.0])
        if isinstance(initial, str):
            try:
                initial = safe_literal(initial)
            except (SafeEvalError, ValueError, SyntaxError):
                initial = [1.0]
        params["_replay_state_"] = np.atleast_1d(initial).copy()
        params["_replay_pending_"] = None
        params["_init_start_"] = False

    if params["_replay_pending_"] is not None:
        params["_replay_state_"] = params["_replay_pending_"]
        params["_replay_pending_"] = None

    state = params["_replay_state_"]
    out_val = state if np.atleast_1d(state).size > 1 else float(np.atleast_1d(state)[0])

    if 0 in inputs:
        params["_replay_pending_"] = np.atleast_1d(inputs[0]).copy()
    return out_val


def _safe_asin(v):
    with np.errstate(invalid="ignore"):
        return np.where(np.abs(v) <= 1, np.arcsin(np.clip(v, -1.0, 1.0)), 0.0)


def _safe_acos(v):
    with np.errstate(invalid="ignore"):
        return np.where(np.abs(v) <= 1, np.arccos(np.clip(v, -1.0, 1.0)), 0.0)


def _safe_log(v):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(v > 0, np.log(np.where(v > 0, v, 1.0)), 0.0)


def _safe_log10(v):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(v > 0, np.log10(np.where(v > 0, v, 1.0)), 0.0)


def _safe_sqrt(v):
    return np.where(v >= 0, np.sqrt(np.where(v >= 0, v, 0.0)), 0.0)


def _safe_reciprocal(v):
    return np.where(v != 0, 1.0 / np.where(v != 0, v, 1.0), 0.0)


# MathFunction's named functions, domain-guarded (out-of-domain samples give
# 0.0 rather than NaN, matching the interpreted block).
MATHFUNCTION_OPS: Dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "sin": np.sin,
    "cos": np.cos,
    "tan": np.tan,
    "asin": _safe_asin,
    "acos": _safe_acos,
    "atan": np.arctan,
    "exp": np.exp,
    "log": _safe_log,
    "log10": _safe_log10,
    "sqrt": _safe_sqrt,
    "square": lambda v: v * v,
    "sign": np.sign,
    "abs": np.abs,
    "ceil": np.ceil,
    "floor": np.floor,
    "reciprocal": _safe_reciprocal,
    "cube": lambda v: v * v * v,
}


def _replay_math_function(step, block, inputs):
    # Keep the input as an array so vector signals work.
    val = np.asarray(inputs.get(0, 0.0), dtype=float)
    # Both 'function' and 'expression' keys are accepted for backward compatibility.
    func_raw = block.params.get("function", block.params.get("expression", "sin"))
    op = MATHFUNCTION_OPS.get(str(func_raw).lower())
    try:
        if op is not None:
            return op(val)
        # Python expression fallback (vectorized: no float()).
        return safe_expr(str(func_raw), variables={"u": val, "t": step.t})
    except (ValueError, ZeroDivisionError):
        return 0.0


def _replay_hysteresis(step, block, inputs):
    # Only reached when the block has no compiled kernel executor (a
    # kernel-driven Hysteresis is latched by the event solve's mode history).
    # The relay latch is scalar; reduce vector inputs safely.
    val = float(np.ravel(inputs.get(0, 0.0))[0])
    params = block.params
    upper = float(params.get("upper", 0.5))
    lower = float(params.get("lower", -0.5))
    high_val = float(params.get("high", 1.0))
    low_val = float(params.get("low", 0.0))

    state = block.exec_params
    if state.get("_init_start_", True) or "_replay_hyst_state_" not in state:
        state["_replay_hyst_state_"] = low_val
        state["_init_start_"] = False
    if val >= upper:
        state["_replay_hyst_state_"] = high_val
    elif val <= lower:
        state["_replay_hyst_state_"] = low_val
    return state["_replay_hyst_state_"]


def _replay_demux(step, block, inputs):
    # Split the vector input into N consecutive sub-vectors of length
    # output_shape (mirrors blocks/demux.py). Port 0 is the primary output;
    # the rest go to "{name}_out{p}".
    arr = np.atleast_1d(np.asarray(inputs.get(0, 0.0), dtype=float)).flatten()
    output_shape = max(1, int(block.params.get("output_shape", 1)))
    n_outputs = max(1, int(block.params.get("_outputs_", getattr(block, "out_ports", 1))))
    for p in range(1, n_outputs):
        step.signals[block.name + f"_out{p}"] = arr[p * output_shape : (p + 1) * output_shape]
    return arr[0:output_shape]


def _replay_field_probe(step, block, inputs):
    # Linear interpolation at a position along a 1-D field.
    field = np.atleast_1d(inputs.get(0, np.array([0.0]))).flatten()
    N = len(field)
    if N == 0:
        return 0.0
    position = float(block.params.get("position", 0.5))
    if block.params.get("position_mode", "normalized") == "normalized":
        idx_float = position * (N - 1)
    else:
        idx_float = (position / float(block.params.get("L", 1.0))) * (N - 1)
    idx_float = max(0, min(N - 1, idx_float))
    idx_low = int(np.floor(idx_float))
    idx_high = min(idx_low + 1, N - 1)
    frac = idx_float - idx_low
    return field[idx_low] * (1 - frac) + field[idx_high] * frac


def _replay_field_scope(step, block, inputs):
    return np.atleast_1d(inputs.get(0, np.array([0.0]))).flatten()


REPLAY_HANDLERS: Dict[str, Handler] = {
    "Integrator": _replay_integrator,
    "Heatequation1D": _replay_heat_1d,
    "Waveequation1D": _replay_wave_1d,
    "Advectionequation1D": _replay_advection_1d,
    "Diffusionreaction1D": _replay_diffusion_reaction_1d,
    "Heatequation2D": _replay_heat_2d,
    "Fieldprobe2D": _replay_field_probe_2d,
    "Fieldscope2D": _replay_field_scope_2d,
    "Fieldslice": _replay_field_slice,
    "Statevariable": _replay_state_variable,
    "StateVariable": _replay_state_variable,
    "Mathfunction": _replay_math_function,
    "Hysteresis": _replay_hysteresis,
    "Demux": _replay_demux,
    "Fieldprobe": _replay_field_probe,
    "Fieldscope": _replay_field_scope,
    "Terminator": _replay_sink,
    "Display": _replay_sink,
    "Scope": _replay_sink,
}


def replay_fallback(block, inputs, t, replay_dt, failed: Set[str]):
    """``block.execute()`` for a block with neither a kernel nor a handler.

    Covers optimization primitives and custom blocks. The block gets its
    *resolved* params (raw params may still hold workspace-variable names)
    and the replay step, which stateful blocks need to advance. A failure is
    logged once per block, and the block's signal stays 0.0.
    """
    if block.block_instance is None:
        return 0.0
    try:
        result = block.block_instance.execute(
            time=t, inputs=inputs, params=runtime_params(block), dtime=replay_dt
        )
    except Exception as e:
        if block.name not in failed:
            failed.add(block.name)
            logger.warning(
                "Replay fallback execute() failed for %s (%s): %s. "
                "Its recorded signal stays 0.0 for this run "
                "(further failures for this block are not logged).",
                block.name,
                block.block_fn,
                e,
                exc_info=True,
            )
        return 0.0
    if result and 0 in result:
        return result[0]
    return 0.0


# ---------------------------------------------------------------------------
# Recorders: Scope / FieldScope histories
# ---------------------------------------------------------------------------


def _ensure_exec_params(block):
    # ScopePlotter reads exec_params first, so the history must land there.
    if not hasattr(block, "exec_params"):
        block.exec_params = block.params.copy()
    return block.exec_params


def _scope_labels(block, vec_dim):
    """Per-channel labels from the 'labels' param, padded/trimmed to vec_dim."""
    labels_raw = block.params.get("labels", block.exec_params.get("labels", ""))
    # A non-string (list/dict) 'labels' falls through to the defaults.
    if isinstance(labels_raw, str) and labels_raw and labels_raw != "default":
        labels = [s.strip() for s in labels_raw.replace(" ", "").split(",") if s.strip()]
        while len(labels) < vec_dim:
            labels.append(f"{block.name}-{len(labels)}")
        return labels[:vec_dim]
    return [f"{block.name}-{j}" for j in range(vec_dim)]


def _record_scope(run, block, inputs, i, t):
    exec_params = _ensure_exec_params(block)
    n_inputs = getattr(block, "in_ports", 1)
    # Flatten every port so vector signals expand into individual channels
    # (matching Scope.execute()).
    combined = [np.atleast_1d(inputs.get(port, 0.0)).flatten() for port in range(n_inputs)]
    new_sample = np.concatenate(combined) if combined else np.array([0.0])
    vec_dim = len(new_sample)

    if i == 0:
        # The sample count is known up front, so the history is preallocated
        # as the (num_steps, vec_dim) array the plotter wants.
        try:
            sample_dtype = np.promote_types(new_sample.dtype, np.float64)
        except TypeError:
            sample_dtype = float
        exec_params["vector"] = np.zeros((run.num_steps, vec_dim), dtype=sample_dtype)
        exec_params["vec_dim"] = vec_dim
        exec_params["vec_labels"] = _scope_labels(block, vec_dim)

    history = exec_params["vector"]
    width = history.shape[1]
    if new_sample.size == width:
        history[i] = new_sample
    else:
        # Keep the first width, pad/truncate, and say so once (a ragged
        # history used to lose the whole trace).
        n_common = min(new_sample.size, width)
        history[i, :n_common] = new_sample[:n_common]
        if block.name not in run.scope_width_warned:
            run.scope_width_warned.add(block.name)
            logger.warning(
                "Scope %s: input width changed mid-run (%d -> %d); "
                "padding/truncating to the first width.",
                block.name,
                width,
                new_sample.size,
            )

    if i == run.num_steps - 1:
        logger.info(
            "Replay Scope %s: vec_dim=%d, samples=%d, labels=%s",
            block.name,
            vec_dim,
            len(history),
            exec_params.get("vec_labels"),
        )


def _record_field_scope(run, block, inputs, i, t):
    exec_params = _ensure_exec_params(block)
    field = np.atleast_1d(inputs.get(0, np.array([0.0]))).flatten()
    if i == 0:
        exec_params["_field_history_"] = []
        exec_params["_time_history_"] = []
    exec_params["_field_history_"].append(field.copy())
    exec_params["_time_history_"].append(t)


def _record_field_scope_2d(run, block, inputs, i, t):
    exec_params = _ensure_exec_params(block)
    field = np.atleast_2d(inputs.get(0, np.zeros((1, 1))))
    if i == 0:
        exec_params["_field_history_2d_"] = []
        exec_params["_time_history_"] = []
    # Store every N frames to reduce memory.
    sample_interval = int(block.params.get("sample_interval", 5))
    if i % sample_interval == 0:
        exec_params["_field_history_2d_"].append(field.copy())
        exec_params["_time_history_"].append(t)


RECORDERS: Dict[str, Recorder] = {
    "Scope": _record_scope,
    "Fieldscope": _record_field_scope,
    "Fieldscope2D": _record_field_scope_2d,
}

# block_fn -> exec_params keys the recorder builds as lists, to be arrays.
_HISTORY_KEYS = {
    "Scope": ("vector",),
    "FieldScope": ("_field_history_", "_time_history_"),
    "FieldScope2D": ("_field_history_2d_", "_time_history_"),
}


def finalize_recorders(blocks) -> None:
    """Convert every recorded Scope / FieldScope history to a numpy array.

    A Scope's 'vector' is already the preallocated array in the normal path;
    asarray only matters for a leftover list from a previous interpreted run
    when the replay recorded no steps.
    """
    for block in blocks:
        keys = _HISTORY_KEYS.get(block.block_fn)
        if not keys or not hasattr(block, "exec_params"):
            continue
        for key in keys:
            if key in block.exec_params:
                block.exec_params[key] = np.asarray(block.exec_params[key])
