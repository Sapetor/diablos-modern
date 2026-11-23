"""Scope traces of a finished run, without any plotting.

Run history (``DSim._record_run_history``, read by the Waveform Inspector)
needs each Scope's data and whether it should be drawn as a staircase; the
plot windows need the same. This used to live on the GUI's ``ScopePlotter``,
so recording history required a plotter; it is pure data and belongs to the
engine. ``diablos.gui.plotting.scope_plotter.ScopePlotter`` delegates here.
"""

import numpy as np

#: Blocks whose output is piecewise constant (drawn as steps) or continuous.
DISCRETE_BLOCKS = frozenset({"DiscreteTranFn", "DiscreteStateSpace", "ZeroOrderHold"})
CONTINUOUS_BLOCKS = frozenset({"Integrator", "TranFn", "StateSpace", "Derivative"})


def _src_name(conn):
    return conn.get("srcblock") if isinstance(conn, dict) else getattr(conn, "srcblock", None)


def input_connections(dsim, block_name):
    """Incoming connections of ``block_name`` from the engine's active graph.

    The engine owns the connection query; a dsim without one (plotting a
    diagram that was never run) has no upstream to inspect.
    """
    engine = getattr(dsim, "engine", None)
    if engine is None:
        return []
    inputs, _ = engine.get_neighbors(block_name)
    return inputs


def is_discrete_upstream(dsim, block_name, visited=None):
    """True if ``block_name`` (or what feeds it) is discrete-time / ZOH."""
    if visited is None:
        visited = set()
    if block_name in visited:
        return False
    visited.add(block_name)

    block = None
    try:
        block = dsim.model.get_block_by_name(block_name)
    except Exception:
        # Fallback in case the model is not available for some reason
        block = next((b for b in dsim.blocks_list if b.name == block_name), None)
    if block is None:
        return False

    if block.block_fn in DISCRETE_BLOCKS:
        return True
    if block.block_fn in CONTINUOUS_BLOCKS:
        return False

    for conn in input_connections(dsim, block.name):
        src = _src_name(conn)
        if src and is_discrete_upstream(dsim, src, visited):
            return True
    return False


def _source_blocks(dsim):
    engine = getattr(dsim, "engine", None)
    if engine is not None and engine.active_blocks_list:
        return engine.active_blocks_list
    return dsim.blocks_list


def scope_step_modes(dsim):
    """One step-mode flag per Scope block, in block order."""
    modes = []
    for block in _source_blocks(dsim):
        if block.block_fn != "Scope":
            continue
        inputs = input_connections(dsim, block.name)
        if not inputs:
            modes.append(False)
            continue
        modes.append(any(is_discrete_upstream(dsim, _src_name(c)) for c in inputs if _src_name(c)))
    return modes


def get_scope_traces(dsim):
    """``(timeline, traces)`` with traces ``[{'name', 'y', 'step'}, ...]``.

    ``(None, [])`` when there is no timeline (nothing has run).
    """
    timeline = getattr(dsim, "timeline", None)
    if timeline is None or len(timeline) == 0:
        return None, []

    scopes = [b for b in _source_blocks(dsim) if b.block_fn == "Scope"]
    # One mode per Scope, paired with its own block. (The plotter used to
    # advance the mode index only for Scopes holding data, so a Scope without
    # a buffer shifted every later Scope's step flag onto the wrong trace.)
    step_modes = dict(zip((id(b) for b in scopes), scope_step_modes(dsim)))
    traces = []
    for block in scopes:
        labels = block.exec_params.get("vec_labels", block.params.get("vec_labels"))
        vec = block.exec_params.get("vector", block.params.get("vector"))
        if vec is None:
            continue
        arr = np.array(vec)
        step_flag = step_modes.get(id(block), False)

        if arr.ndim == 1:
            name = labels if isinstance(labels, str) else block.name
            traces.append({"name": name, "y": arr, "step": step_flag})
        elif arr.ndim == 2:
            for i in range(arr.shape[1]):
                if isinstance(labels, (list, tuple)) and i < len(labels):
                    name = labels[i]
                else:
                    name = f"{block.name}[{i}]"
                traces.append({"name": name, "y": arr[:, i], "step": step_flag})
    return timeline, traces
