"""Compiled-vs-interpreted equivalence for the TransportDelay block.

TransportDelay implements a continuous-time transport delay y(t) = u(t - tau)
via linear interpolation over a recorded (time, value) history. A time-varying
input is required to exercise it meaningfully, so the diagram here is

    Sine(A=1, w=3) -> TransportDelay(tau=0.5) -> Scope

which has the closed-form response y(t) = sin(3*(t - 0.5)) for t >= 0.5 and the
initial output 0 before the delay elapses.

Note on the two paths: TransportDelay is *not* in
``SystemCompiler.COMPILABLE_BLOCKS`` -- its output depends on the whole recorded
input history, which cannot be expressed as a pure function of (t, y) inside a
solve_ivp RHS. So a diagram containing it is not compilable, and running with
``use_fast_solver=True`` falls back to the interpreter. This test therefore
guards two invariants at once:

  * the fast-solver path and the interpreter agree on the scope trajectory
    (today via fallback; if a compiled TransportDelay kernel is ever added this
    becomes a genuine RK45-vs-fixed-step comparison and the loose tolerance
    below is the appropriate one -- see tests/regression/test_compiled_golden.py
    for the same rationale), and
  * the interpreted TransportDelay reproduces the analytic delayed sine.

The diagram is built programmatically (there is no shipped example with a
TransportDelay) using the DSim harness from test_compiled_golden and the
DBlock/DLine idiom from tests/unit/test_solver_selection.py.
"""

import numpy as np
import pytest


AMPLITUDE = 1.0
OMEGA = 3.0
DELAY = 0.5
SIM_TIME = 5.0
SIM_DT = 0.01

# Cross-scheme tolerance: loose enough for an RK45-vs-fixed-step comparison if a
# compiled kernel is ever added, tight enough that a real regression trips it.
RTOL = 1e-2
ATOL = 1e-2


def _defaults(block):
    return {
        k: v["default"] if isinstance(v, dict) and "default" in v else v
        for k, v in block.params.items()
    }


def _build_dsim():
    """Build a Sine -> TransportDelay -> Scope DSim from scratch."""
    from PyQt5.QtCore import QRect, QPoint
    from blocks.sine import SineBlock
    from blocks.transport_delay import TransportDelayBlock
    from blocks.scope import ScopeBlock
    from lib.lib import DSim
    from lib.simulation.block import DBlock
    from lib.simulation.connection import DLine
    from lib.workspace import WorkspaceManager

    WorkspaceManager._instance = None
    dsim = DSim()
    model = dsim.model

    sine_b = SineBlock()
    sine = DBlock(
        "Sine",
        0,
        coords=QRect(0, 0, 50, 50),
        color="blue",
        in_ports=0,
        out_ports=1,
        b_type=sine_b.b_type,
        params=_defaults(sine_b),
        block_class=SineBlock,
        category=sine_b.category,
    )
    sine.params["amplitude"] = AMPLITUDE
    sine.params["omega"] = OMEGA
    sine.params["init_angle"] = 0.0

    td_b = TransportDelayBlock()
    td = DBlock(
        "TransportDelay",
        0,
        coords=QRect(100, 0, 50, 50),
        color="cyan",
        in_ports=1,
        out_ports=1,
        b_type=2,
        params=_defaults(td_b),
        block_class=TransportDelayBlock,
        category=td_b.category,
    )
    td.params["delay_time"] = DELAY
    td.params["initial_value"] = 0.0

    scope_b = ScopeBlock()
    scope = DBlock(
        "Scope",
        0,
        coords=QRect(200, 0, 50, 50),
        color="red",
        in_ports=1,
        out_ports=0,
        b_type=scope_b.b_type,
        params=_defaults(scope_b),
        block_class=ScopeBlock,
        category=scope_b.category,
    )

    lines = [
        DLine(
            sid=0,
            srcblock=sine.name,
            srcport=0,
            dstblock=td.name,
            dstport=0,
            points=[QPoint(0, 0), QPoint(1, 1)],
        ),
        DLine(
            sid=1,
            srcblock=td.name,
            srcport=0,
            dstblock=scope.name,
            dstport=0,
            points=[QPoint(1, 1), QPoint(2, 2)],
        ),
    ]
    model.blocks_list.clear()
    model.blocks_list.extend([sine, td, scope])
    model.line_list.clear()
    model.line_list.extend(lines)
    dsim.blocks_list = model.blocks_list
    dsim.line_list = model.line_list
    return dsim


def _scope_trace(dsim):
    """Return (timeline, y) for the single Scope in the diagram."""
    for b in dsim.engine.active_blocks_list:
        if b.block_fn != "Scope":
            continue
        params = getattr(b, "exec_params", b.params)
        vec = params.get("vector")
        if vec is None:
            continue
        y = np.asarray(vec, dtype=float).reshape(-1, params.get("vec_dim", 1))
        return np.asarray(dsim.timeline, dtype=float), y
    raise AssertionError("no Scope trace produced")


def _run(use_fast):
    dsim = _build_dsim()
    compilable = dsim.engine.check_compilability(dsim.blocks_list)
    dsim.use_fast_solver = use_fast
    ok, err = dsim.run_tuning_simulation(SIM_TIME, SIM_DT)
    assert ok, "run (use_fast=%s) failed: %s" % (use_fast, err)
    t, y = _scope_trace(dsim)
    return compilable, t, y


@pytest.mark.regression
class TestTransportDelayEquivalence:
    def test_fast_path_matches_interpreter(self, qapp):
        _, t_interp, y_interp = _run(use_fast=False)
        compilable, t_fast, y_fast = _run(use_fast=True)

        # Documents the current reality: no compiled kernel, so the fast path
        # falls back to the interpreter. If this flips, the assertion below turns
        # into a real cross-scheme comparison rather than an identity check.
        assert compilable is False

        assert y_interp.shape[1] == y_fast.shape[1]
        # Compare at the interpreter's sample times (interpolate the fast trace
        # per output component).
        resampled = np.empty_like(y_interp)
        for c in range(y_interp.shape[1]):
            resampled[:, c] = np.interp(t_interp, t_fast, y_fast[:, c])
        max_abs = float(np.max(np.abs(y_interp - resampled)))
        assert np.allclose(y_interp, resampled, rtol=RTOL, atol=ATOL), (
            "fast-solver trace diverged from interpreter (max|delta|=%.3e)" % max_abs
        )

    def test_interpreted_matches_analytic_delayed_sine(self, qapp):
        _, t, y = _run(use_fast=False)
        y = y[:, 0]

        # Before the delay elapses the block holds its initial output (0).
        pre = t < (DELAY - SIM_DT)
        assert np.max(np.abs(y[pre])) < 1e-9

        # After the delay the output is the delayed sine. Skip a one-step margin
        # around t = DELAY where linear interpolation straddles the switch-on.
        post = t > (DELAY + 2 * SIM_DT)
        analytic = AMPLITUDE * np.sin(OMEGA * (t - DELAY))
        max_abs = float(np.max(np.abs(y[post] - analytic[post])))
        assert np.allclose(y[post], analytic[post], rtol=RTOL, atol=ATOL), (
            "interpreted transport delay does not match sin(3*(t-0.5)) (max|delta|=%.3e)" % max_abs
        )
        # Sanity: the signal actually swings over the full sine amplitude.
        assert np.max(np.abs(y)) > 0.9 * AMPLITUDE
