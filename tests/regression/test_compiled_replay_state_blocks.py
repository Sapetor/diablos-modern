"""Characterization + regression tests for the compiled (fast-solver) REPLAY of
Selector, PID and RateLimiter.

These blocks' post-solve replay output was historically computed by an inline
``elif`` branch in ``SimulationEngine.run_compiled_simulation`` that the
golden-master suite never exercised (no example diagram used them through the
compiled path -- confirmed by a coverage run: those branches were entirely
uncovered). This module pins the compiled-replay output so that migrating these
blocks onto the shared compiler-kernel registry (``lib.engine.compiler_kernels``,
so the ODE solve and the post-solve replay share one source of truth) is
provably behavior-preserving.

``Selector`` is pure-algebraic; ``PID``/``RateLimiter`` are ODE-state blocks
whose compiled replay reads the reconstructed state. All three traces are pinned
golden-master style against values captured from the pre-migration baseline. The
tolerance is loose enough to absorb cross-platform RK45 jitter (the local pins
are generated on the Windows venv; CI also runs them on Linux) while a real
kernel/dispatch regression is O(1) -- far larger.
"""
import numpy as np
import pytest
from PyQt5.QtCore import QRect, QPoint
from PyQt5.QtGui import QColor


def _dsim():
    from lib.lib import DSim
    from lib.workspace import WorkspaceManager
    WorkspaceManager._instance = None
    return DSim()


def _mk(dsim, block_fn, sid, in_ports, out_ports, params):
    """Build a DBlock and attach a real block_instance (mirrors file_service)."""
    from lib.simulation.block import DBlock
    menu = next((mb for mb in dsim.model.menu_blocks if mb.block_fn == block_fn), None)
    assert menu is not None, f"{block_fn} not registered in menu_blocks"
    inst = menu.block_class() if menu.block_class else None
    b_type = getattr(inst, "b_type", 2)
    blk = DBlock(
        block_fn=block_fn, sid=sid, coords=QRect(0, 0, 50, 40),
        color=QColor(150, 150, 150), in_ports=in_ports, out_ports=out_ports,
        params=dict(params), username="", b_type=b_type,
    )
    blk.block_instance = inst
    return blk


def _ln(sid, src, sp, dst, dp):
    from lib.simulation.connection import DLine
    return DLine(sid=sid, srcblock=src, srcport=sp, dstblock=dst, dstport=dp,
                 points=[QPoint(0, 0), QPoint(1, 0)])


def _scope_vec(dsim):
    for b in dsim.engine.active_blocks_list:
        if b.block_fn != "Scope":
            continue
        params = getattr(b, "exec_params", b.params)
        vec = params.get("vector")
        if vec is None:
            continue
        return np.asarray(vec, dtype=float).reshape(-1, params.get("vec_dim", 1))
    raise AssertionError("No Scope vector captured")


def _run_compiled(build, sim_time, sim_dt):
    """Build a FRESH diagram (so per-run block state never leaks) and run it
    through the full compiled path (solve + post-solve replay)."""
    dsim = _dsim()
    blocks, lines = build(dsim)
    dsim.model.blocks_list = blocks
    dsim.model.line_list = lines
    dsim.use_fast_solver = True
    ok, err = dsim.run_tuning_simulation(sim_time, sim_dt)
    assert ok, f"compiled run failed: {err}"
    return _scope_vec(dsim)


# --------------------------------------------------------------------------
# Diagram builders (fresh blocks each call)
# --------------------------------------------------------------------------
def _selector_diagram(dsim):
    blocks = [
        _mk(dsim, "Constant", 0, 0, 1, {"value": [10.0, 20.0, 30.0, 40.0]}),
        _mk(dsim, "Selector", 0, 1, 1, {"indices": "1:3"}),
        _mk(dsim, "Scope", 0, 1, 0, {}),
    ]
    lines = [
        _ln(0, "constant0", 0, "selector0", 0),
        _ln(1, "selector0", 0, "scope0", 0),
    ]
    return blocks, lines


def _pid_diagram(dsim):
    # Step setpoint into PID port 0; constant 0 measurement into port 1 (the
    # integrity check requires every input wired), so error == setpoint and the
    # integral term ramps the output. Kd/N!=0 exercises the derivative filter
    # (the t=0 spike of Kp + Kd*N == 12).
    blocks = [
        _mk(dsim, "Step", 0, 0, 1, {"value": 1.0, "delay": 0.0, "type": "up"}),
        _mk(dsim, "Constant", 0, 0, 1, {"value": 0.0}),
        _mk(dsim, "PID", 0, 2, 1, {"Kp": 2.0, "Ki": 1.0, "Kd": 0.5, "N": 20.0}),
        _mk(dsim, "Scope", 0, 1, 0, {}),
    ]
    lines = [
        _ln(0, "step0", 0, "pid0", 0),
        _ln(1, "constant0", 0, "pid0", 1),
        _ln(2, "pid0", 0, "scope0", 0),
    ]
    return blocks, lines


def _ratelimiter_diagram(dsim):
    blocks = [
        _mk(dsim, "Step", 0, 0, 1, {"value": 1.0, "delay": 0.0, "type": "up"}),
        _mk(dsim, "RateLimiter", 0, 1, 1, {"rising_slew": 2.0, "falling_slew": 2.0}),
        _mk(dsim, "Scope", 0, 1, 0, {}),
    ]
    lines = [
        _ln(0, "step0", 0, "ratelimiter0", 0),
        _ln(1, "ratelimiter0", 0, "scope0", 0),
    ]
    return blocks, lines


SIM_TIME = 2.0
SIM_DT = 0.1

# Pinned compiled-replay traces, captured from the pre-migration inline baseline
# (Windows venv, RK45). Loose tolerance absorbs cross-platform solver jitter.
RTOL = 1e-3
ATOL = 1e-5

_PID_BASELINE = np.array([
    12.0, 3.4533528317144317, 2.383156389452658, 2.324787524282258,
    2.403354626171109, 2.500453998696557, 2.600061442350375, 2.7000083151371603,
    2.80000112652195, 2.9000001516477276, 3.000000019994506, 3.100000001882709,
    3.1999999997036728, 3.3000000019875397, 3.399999989595336, 3.5000000091769348,
    3.599999998342692, 3.7000000055748155, 3.800000001440207, 3.8999999983747666,
    4.000000003021193,
]).reshape(-1, 1)

_RATELIMITER_BASELINE = np.array([
    0.0, 0.2, 0.4, 0.6, 0.8, 0.9992642329695153, 1.0, 1.0, 1.0, 1.0, 1.0,
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
]).reshape(-1, 1)


@pytest.mark.regression
class TestCompiledReplaySelector:
    def test_selector_compiled_replay_pinned(self, qapp):
        """Selector is pure-algebraic: indices '1:3' of [10,20,30,40] -> [20,30]
        at every sample. Pins the compiled-replay output as the migration net."""
        compiled = _run_compiled(_selector_diagram, SIM_TIME, SIM_DT)
        assert compiled.shape == (21, 2), f"unexpected shape {compiled.shape}"
        expected = np.tile([20.0, 30.0], (21, 1))
        np.testing.assert_allclose(compiled, expected, rtol=RTOL, atol=ATOL)


@pytest.mark.regression
class TestCompiledReplayPID:
    def test_pid_compiled_replay_pinned(self, qapp):
        """Pin the PID compiled-replay trace. e==1 constant, so
        u = clip(Kp*e + Ki*x_i + Kd*x_d') with x_i ramping."""
        compiled = _run_compiled(_pid_diagram, SIM_TIME, SIM_DT)
        assert compiled.shape == _PID_BASELINE.shape, (
            f"shape {compiled.shape} != pinned {_PID_BASELINE.shape}")
        np.testing.assert_allclose(compiled, _PID_BASELINE, rtol=RTOL, atol=ATOL)


@pytest.mark.regression
class TestCompiledReplayRateLimiter:
    def test_ratelimiter_compiled_replay_pinned(self, qapp):
        """Pin the RateLimiter compiled-replay trace: output chases the unit step
        at the configured slew rate (0.2 per 0.1 s) and saturates at 1.0."""
        compiled = _run_compiled(_ratelimiter_diagram, SIM_TIME, SIM_DT)
        assert compiled.shape == _RATELIMITER_BASELINE.shape, (
            f"shape {compiled.shape} != pinned {_RATELIMITER_BASELINE.shape}")
        np.testing.assert_allclose(compiled, _RATELIMITER_BASELINE, rtol=RTOL, atol=ATOL)
