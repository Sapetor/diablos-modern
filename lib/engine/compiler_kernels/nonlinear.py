"""Nonlinear / signal-shaping kernels for the compiled path.

Covers Saturation, Deadband, Switch, Selector and Hysteresis. Bodies are
verbatim extractions of the corresponding branches from
``SystemCompiler._create_block_executor``; only the shared locals are unpacked
from the BuildContext at the top.
"""
import logging

import numpy as np

from lib.engine.compiler_kernels import kernel

logger = logging.getLogger(__name__)


@kernel("Saturation")
def build_saturation(ctx):
    b_name = ctx.b_name
    params = ctx.params
    input_sources = ctx.input_sources
    src = input_sources[0] if input_sources else None
    lower = float(params.get('min', -np.inf))
    upper = float(params.get('max', np.inf))

    def exec_sat(t, y, dy_vec, signals):
        val = signals.get(src, 0.0) if src else 0.0
        signals[b_name] = np.clip(val, lower, upper)
    return exec_sat


@kernel("Deadband")
def build_deadband(ctx):
    b_name = ctx.b_name
    params = ctx.params
    input_sources = ctx.input_sources
    src = input_sources[0] if input_sources else None
    start_db = float(params.get('start', -0.5))
    end_db = float(params.get('end', 0.5))

    def exec_db(t, y, dy_vec, signals):
        # Element-wise dead zone (works for scalar and vector signals;
        # a bare `if val < start` is ambiguous on a multi-element array).
        val = np.asarray(signals.get(src, 0.0) if src else 0.0, dtype=float)
        signals[b_name] = np.where(val < start_db, val - start_db,
                                   np.where(val > end_db, val - end_db, 0.0))
    return exec_db


@kernel("Switch")
def build_switch(ctx):
    b_name = ctx.b_name
    params = ctx.params
    input_sources = ctx.input_sources
    ctrl_src = input_sources[0] if input_sources else None
    mode = params.get('mode', 'threshold')
    n_inputs = int(params.get('n_inputs', 2))
    threshold = float(params.get('threshold', 0.0))

    def exec_switch(t, y, dy_vec, signals):
        # Control is scalar; reduce safely so a vector control signal
        # doesn't make the comparisons below ambiguous. The selected
        # data input is passed through unchanged (may be a vector).
        ctrl_raw = signals.get(ctrl_src, 0.0) if ctrl_src else 0.0
        ctrl = float(np.ravel(ctrl_raw)[0])

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


@kernel("Selector")
def build_selector(ctx):
    b_name = ctx.b_name
    params = ctx.params
    input_sources = ctx.input_sources
    src = input_sources[0] if input_sources else None
    indices_str = str(params.get('indices', '0'))

    # Pre-parse indices at compile time
    parsed_indices = []
    for part in indices_str.split(','):
        part = part.strip()
        if ':' in part:
            parts = part.split(':')
            try:
                start_idx = int(parts[0]) if parts[0] else 0
                end_idx = int(parts[1]) if len(parts) > 1 and parts[1] else None
            except ValueError:
                logger.warning(
                    "Selector %s: malformed range '%s'; defaulting to full range.",
                    b_name, part)
                start_idx, end_idx = 0, None
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


@kernel("Hysteresis")
def build_hysteresis(ctx):
    b_name = ctx.b_name
    params = ctx.params
    input_sources = ctx.input_sources
    src = input_sources[0] if input_sources else None
    upper = float(params.get('upper', 0.5))
    lower = float(params.get('lower', -0.5))
    high_val = float(params.get('high', 1.0))
    low_val = float(params.get('low', 0.0))

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
