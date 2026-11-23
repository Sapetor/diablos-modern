"""Scope traces are engine data (diablos/core/plotting/scope_traces.py).

They used to be collected by the GUI's ScopePlotter, so every DSim, even a
headless one, built a plotter just to record run history. And the step-mode
index only advanced for Scopes holding data, so an empty Scope shifted every
later Scope's staircase flag onto the wrong trace.
"""

import os
from types import SimpleNamespace

import numpy as np
import pytest

from diablos.core.plotting.scope_traces import get_scope_traces

pytestmark = pytest.mark.unit

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _block(name, fn, vector=None):
    params = {} if vector is None else {"vector": vector}
    return SimpleNamespace(name=name, block_fn=fn, params=params, exec_params={})


def test_an_empty_scope_does_not_shift_the_next_scopes_step_flag():
    zoh = _block("zoh", "ZeroOrderHold")
    empty = _block("scope_empty", "Scope")  # no buffer (e.g. never reached)
    stepped = _block("scope_zoh", "Scope", vector=[0.0, 1.0, 1.0])
    blocks = [zoh, empty, stepped]
    wires = {"scope_empty": [], "scope_zoh": [{"srcblock": "zoh"}]}
    dsim = SimpleNamespace(
        timeline=np.array([0.0, 0.1, 0.2]),
        blocks_list=blocks,
        engine=SimpleNamespace(
            active_blocks_list=blocks, get_neighbors=lambda n: (wires.get(n, []), [])
        ),
        model=SimpleNamespace(get_block_by_name=lambda n: next(b for b in blocks if b.name == n)),
    )
    timeline, traces = get_scope_traces(dsim)
    assert [(t["name"], t["step"]) for t in traces] == [("scope_zoh", True)]


def test_a_headless_dsim_has_no_plotter_but_records_history(qapp):
    from diablos.core.cli import run_diagram

    dsim = run_diagram(os.path.join(REPO, "examples", "first_order_step_response.diablos"))
    assert dsim.scope_plotter is None
    dsim.plot_again()  # no plotter: a no-op, not an error
    dsim._record_run_history()
    entry = dsim.run_history[-1]
    assert [t["name"] for t in entry["traces"]] and len(entry["timeline"]) > 0
