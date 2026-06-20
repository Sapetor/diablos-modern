"""Signal routing / logic / sink kernels for the compiled path.

Covers Mux, Demux, LogicalOperator, the sink/tag blocks (Terminator, Display,
Scope, To, From) and StateVariable. Bodies are verbatim extractions of the
corresponding branches from ``SystemCompiler._create_block_executor``; only the
shared locals are unpacked from the BuildContext at the top.
"""
import numpy as np

from lib.engine.compiler_kernels import kernel
from lib.safe_eval import safe_literal, SafeEvalError


@kernel("Mux")
def build_mux(ctx):
    b_name = ctx.b_name
    input_sources = ctx.input_sources
    # Capture 'input_sources' list directly (handling None sources).
    baked_srcs = [s for s in input_sources]  # Copy

    def exec_mux(t, y, dy_vec, signals):
        vals = []
        for src in baked_srcs:
            if src:
                vals.append(signals.get(src, 0.0))
            else:
                vals.append(0.0)
        signals[b_name] = np.array(vals)
    return exec_mux


@kernel("Demux")
def build_demux(ctx):
    # Split the single vector input into N consecutive sub-vectors of
    # length `output_shape` each. Mirrors blocks/demux.py: output port i
    # is input_array[i*output_shape : (i+1)*output_shape]. Port 0 is
    # stored under b_name; secondary ports use the "{b_name}_out{i}"
    # convention (matches the interpreter replay loop and PDE blocks).
    block = ctx.block
    b_name = ctx.b_name
    params = ctx.params
    input_sources = ctx.input_sources
    src = input_sources[0] if input_sources else None
    output_shape = int(params.get('output_shape', 1))
    if output_shape < 1:
        output_shape = 1
    # Number of output ports: prefer the block's resolved port count.
    n_outputs = int(params.get('_outputs_', getattr(block, 'out_ports', 1)))
    if n_outputs < 1:
        n_outputs = 1

    def exec_demux(t, y, dy_vec, signals, _src=src, _shape=output_shape,
                   _n=n_outputs):
        val = signals.get(_src, 0.0) if _src else 0.0
        arr = np.atleast_1d(np.asarray(val, dtype=float)).flatten()
        for i in range(_n):
            seg = arr[i * _shape:(i + 1) * _shape]
            key = b_name if i == 0 else f"{b_name}_out{i}"
            # Match blocks/demux.py which returns the sub-array slice
            # (length output_shape); a scalar-width demux still yields a
            # length-1 array there, so keep the array form for parity.
            signals[key] = seg
    return exec_demux


@kernel("Logicaloperator")
def build_logicaloperator(ctx):
    # Boolean logic over the inputs (nonzero = True), element-wise.
    # Mirrors blocks/logical_operator.py exactly. Single output port.
    b_name = ctx.b_name
    params = ctx.params
    input_sources = ctx.input_sources
    op = str(params.get('operator', 'AND')).upper()
    # Determine the input count the same way the block does: prefer the
    # resolved _inputs_ port count, else the number of wired ports.
    if '_inputs_' in params:
        n_in = int(params['_inputs_'])
    elif input_sources:
        n_in = len(input_sources)
    else:
        n_in = 1
    n_in = max(n_in, 1)
    baked_srcs = [input_sources[i] if i < len(input_sources) else None
                  for i in range(n_in)]

    def exec_logicalop(t, y, dy_vec, signals, _op=op, _srcs=baked_srcs):
        bvals = []
        for src in _srcs:
            raw = signals.get(src, 0.0) if src else 0.0
            bvals.append(np.atleast_1d(np.asarray(raw, dtype=float)) != 0)
        if _op == 'NOT':
            result = np.logical_not(bvals[0])
        elif _op in ('AND', 'NAND'):
            result = bvals[0]
            for b in bvals[1:]:
                result = np.logical_and(result, b)
            if _op == 'NAND':
                result = np.logical_not(result)
        elif _op in ('OR', 'NOR'):
            result = bvals[0]
            for b in bvals[1:]:
                result = np.logical_or(result, b)
            if _op == 'NOR':
                result = np.logical_not(result)
        elif _op == 'XOR':
            result = bvals[0]
            for b in bvals[1:]:
                result = np.logical_xor(result, b)
        else:
            # Unknown operator: emit all-False (defensive; the block
            # itself would return an error dict, but the ODE RHS must
            # stay numeric). Width follows the first input.
            result = np.zeros_like(bvals[0], dtype=bool)
        signals[b_name] = np.atleast_1d(result).astype(float)
    return exec_logicalop


@kernel("Terminator", "Display", "Scope", "To", "From")
def build_sink(ctx):
    # Sinks or semantic tags, no runtime math for the solver. Scope data is
    # collected by the replay loop, not the solver loop. (Formerly a `pass`
    # branch that fell through to the generic no-op executor.)
    def exec_noop(t, y, dy_vec, signals):
        pass
    return exec_noop


@kernel("StateVariable", "Statevariable")
def build_statevariable(ctx):
    # State variable for discrete optimization iterations.
    # Uses closure-based state storage instead of ODE integration.
    b_name = ctx.b_name
    params = ctx.params
    input_sources = ctx.input_sources
    src = input_sources[0] if input_sources else None
    initial = params.get('initial_value', [1.0])
    if isinstance(initial, str):
        try:
            initial = safe_literal(initial)
        except (SafeEvalError, ValueError, SyntaxError):
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
