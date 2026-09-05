"""Nonlinear / signal-shaping kernels for the compiled path.

Covers Saturation, Deadband, Switch, Selector and Hysteresis. Bodies are
verbatim extractions of the corresponding branches from
``SystemCompiler._create_block_executor``; only the shared locals are unpacked
from the BuildContext at the top.
"""

import logging

import numpy as np

from blocks.selector import normalize_indices_str
from lib.engine.compiler_kernels import EventSpec, events, kernel, signal_scalar

logger = logging.getLogger(__name__)


@kernel("Saturation")
def build_saturation(ctx):
    b_name = ctx.b_name
    params = ctx.params
    input_sources = ctx.input_sources
    src = input_sources[0] if input_sources else None
    lower = float(params.get("min", -np.inf))
    upper = float(params.get("max", np.inf))

    def exec_sat(t, y, dy_vec, signals):
        val = signals.get(src, 0.0) if src else 0.0
        signals[b_name] = np.clip(val, lower, upper)

    return exec_sat


@events("Saturation")
def events_saturation(ctx):
    """Crossings of each finite clip level: ``u - min`` and ``u - max``.

    Below/above a limit the block is the constant limit, between them it is the
    identity -- two kinks the solver should land on exactly.
    """
    src = ctx.input_sources[0] if ctx.input_sources else None
    if not src:
        return []
    specs = []
    for label, key, default in (("lower_limit", "min", -np.inf), ("upper_limit", "max", np.inf)):
        level = float(ctx.params.get(key, default))
        if not np.isfinite(level):
            continue  # an infinite limit is never reached: no discontinuity

        def _g(t, y, signals, _src=src, _level=level):
            return signal_scalar(signals, _src) - _level

        specs.append(EventSpec(block=ctx.b_name, label=label, func=_g))
    return specs


@kernel("Deadband")
def build_deadband(ctx):
    b_name = ctx.b_name
    params = ctx.params
    input_sources = ctx.input_sources
    src = input_sources[0] if input_sources else None
    start_db = float(params.get("start", -0.5))
    end_db = float(params.get("end", 0.5))

    def exec_db(t, y, dy_vec, signals):
        # Element-wise dead zone (works for scalar and vector signals;
        # a bare `if val < start` is ambiguous on a multi-element array).
        val = np.asarray(signals.get(src, 0.0) if src else 0.0, dtype=float)
        signals[b_name] = np.where(
            val < start_db, val - start_db, np.where(val > end_db, val - end_db, 0.0)
        )

    return exec_db


@events("Deadband")
def events_deadband(ctx):
    """Both edges of the dead zone, where the slope jumps between 0 and 1."""
    src = ctx.input_sources[0] if ctx.input_sources else None
    if not src:
        return []
    specs = []
    for label, key, default in (("start_edge", "start", -0.5), ("end_edge", "end", 0.5)):
        edge = float(ctx.params.get(key, default))
        if not np.isfinite(edge):
            continue

        def _g(t, y, signals, _src=src, _edge=edge):
            return signal_scalar(signals, _src) - _edge

        specs.append(EventSpec(block=ctx.b_name, label=label, func=_g))
    return specs


@kernel("Switch")
def build_switch(ctx):
    b_name = ctx.b_name
    params = ctx.params
    input_sources = ctx.input_sources
    ctrl_src = input_sources[0] if input_sources else None
    mode = params.get("mode", "threshold")
    n_inputs = int(params.get("n_inputs", 2))
    threshold = float(params.get("threshold", 0.0))

    def exec_switch(t, y, dy_vec, signals):
        # Control is scalar; reduce safely so a vector control signal
        # doesn't make the comparisons below ambiguous. The selected
        # data input is passed through unchanged (may be a vector).
        ctrl_raw = signals.get(ctrl_src, 0.0) if ctrl_src else 0.0
        ctrl = float(np.ravel(ctrl_raw)[0])

        if mode == "index":
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


@events("Switch")
def events_switch(ctx):
    """The control signal crossing the routing boundary.

    Threshold mode has one boundary (``ctrl - threshold``).  Index mode selects
    ``round(ctrl)``, so every half-integer between the reachable port indices is
    a boundary; those are emitted individually.
    """
    ctrl_src = ctx.input_sources[0] if ctx.input_sources else None
    if not ctrl_src:
        return []
    mode = ctx.params.get("mode", "threshold")

    if mode == "index":
        n_inputs = int(ctx.params.get("n_inputs", 2))
        specs = []
        for k in range(max(0, n_inputs - 1)):
            boundary = k + 0.5

            def _g(t, y, signals, _src=ctrl_src, _b=boundary):
                return signal_scalar(signals, _src) - _b

            specs.append(EventSpec(block=ctx.b_name, label="index_{}_{}".format(k, k + 1), func=_g))
        return specs

    threshold = float(ctx.params.get("threshold", 0.0))

    def _g_threshold(t, y, signals, _src=ctrl_src, _th=threshold):
        return signal_scalar(signals, _src) - _th

    return [EventSpec(block=ctx.b_name, label="threshold", func=_g_threshold)]


@kernel("Selector")
def build_selector(ctx):
    b_name = ctx.b_name
    params = ctx.params
    input_sources = ctx.input_sources
    src = input_sources[0] if input_sources else None
    indices_str = normalize_indices_str(params.get("indices", "0"))

    # Pre-parse indices at compile time
    parsed_indices = []
    for part in indices_str.split(","):
        part = part.strip()
        if ":" in part:
            parts = part.split(":")
            try:
                start_idx = int(parts[0]) if parts[0] else 0
                end_idx = int(parts[1]) if len(parts) > 1 and parts[1] else None
            except ValueError:
                logger.warning(
                    "Selector %s: malformed range '%s'; defaulting to full range.", b_name, part
                )
                start_idx, end_idx = 0, None
            parsed_indices.append(("range", start_idx, end_idx))
        else:
            try:
                parsed_indices.append(("idx", int(part)))
            except ValueError:
                parsed_indices.append(("idx", 0))

    def exec_selector(t, y, dy_vec, signals, _indices=parsed_indices):
        val = signals.get(src, 0.0) if src else 0.0
        u = np.atleast_1d(val).flatten()
        max_len = len(u)

        result = []
        for item in _indices:
            if item[0] == "range":
                start_i, end_i = item[1], item[2]
                end_i = end_i if end_i is not None else max_len
                result.extend(u[start_i : min(end_i, max_len)])
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


# NOTE: Hysteresis is NOT in SystemCompiler.COMPILABLE_BLOCKS (its relay latch
# is path-dependent, and solve_ivp probes the RHS at rejected/non-monotonic
# times), so no user diagram reaches this kernel -- check_compilability sends
# the whole diagram to the interpreter first. It is kept as the compiled-path
# reference implementation and is exercised by the direct-compile tests
# (tests/unit/test_compiler_kernels.py, tests/unit/test_compiler_workspace_vars.py).
@kernel("Hysteresis")
def build_hysteresis(ctx):
    """Relay latch.

    The latch is path-dependent, which is exactly what an ODE right-hand side
    may not be: ``solve_ivp`` probes ``f`` at rejected and non-monotonic times,
    and a latch that updated on every probe would record transitions that never
    happened.  With zero-crossing detection the mode is *frozen* for the whole
    integration segment and flipped only by the event's discrete update at the
    located switching instant, so ``f`` is a pure function of ``(t, y)`` again.

    If the chattering guard has to give up on events, the runner unfreezes the
    holder and the block falls back to the historical probe-order-dependent
    update -- approximate, but it keeps moving.
    """
    b_name = ctx.b_name
    params = ctx.params
    input_sources = ctx.input_sources
    src = input_sources[0] if input_sources else None
    upper = float(params.get("upper", 0.5))
    lower = float(params.get("lower", -0.5))
    high_val = float(params.get("high", 1.0))
    low_val = float(params.get("low", 0.0))

    latch = ctx.latch(mode=low_val, init=low_val)

    def exec_hysteresis(t, y, dy_vec, signals, _latch=latch):
        if not _latch["frozen"]:
            val = signal_scalar(signals, src)
            if val >= upper:
                _latch["mode"] = high_val
            elif val <= lower:
                _latch["mode"] = low_val
        signals[b_name] = _latch["mode"]

    return exec_hysteresis


@events("Hysteresis")
def events_hysteresis(ctx):
    """The single *active* threshold, which depends on the latch's mode.

    Sitting low the relay is waiting for ``u`` to rise through ``upper``;
    sitting high it is waiting for ``u`` to fall through ``lower``.  Watching
    only the active one (with a matching direction) is what makes the loop a
    loop rather than two independent crossings, and it means the freshly
    switched mode never re-triggers: right after a rise through ``upper`` the
    new guard ``u - lower`` is comfortably positive.
    """
    src = ctx.input_sources[0] if ctx.input_sources else None
    if not src:
        return []
    upper = float(ctx.params.get("upper", 0.5))
    lower = float(ctx.params.get("lower", -0.5))
    high_val = float(ctx.params.get("high", 1.0))
    low_val = float(ctx.params.get("low", 0.0))
    latch = ctx.latch(mode=low_val, init=low_val)
    # Claim the latch: the runner may freeze it for the length of the run so
    # the mode changes only at located switching instants. A Hysteresis block
    # whose own zero_crossing param is "off" never gets here, so its latch keeps
    # updating from the RHS as it always did.
    latch["event_driven"] = True

    def _g(t, y, signals, _latch=latch):
        u = signal_scalar(signals, src)
        if _latch["mode"] == high_val:
            return u - lower  # falls to zero on the way down
        return u - upper  # rises to zero on the way up

    def _switch(t, y, signals, _latch=latch):
        _latch["mode"] = low_val if _latch["mode"] == high_val else high_val

    def _seed(t, y, signals, _latch=latch):
        # Reconcile the latch with the actual initial input, so a diagram that
        # starts already past a threshold does not spend its first segment in
        # the wrong mode (the interpreter self-corrects on its first step).
        u = signal_scalar(signals, src)
        if u >= upper:
            _latch["mode"] = high_val
        elif u <= lower:
            _latch["mode"] = low_val

    return [
        EventSpec(
            block=ctx.b_name,
            label="relay",
            func=_g,
            direction=0.0,
            on_event=_switch,
            on_start=_seed,
        )
    ]
