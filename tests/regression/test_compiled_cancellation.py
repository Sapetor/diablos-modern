"""
Regression tests: Stop / cancel must interrupt a *compiled* run.

BUG DESCRIPTION
---------------
``DSim.execution_batch`` polled ``cancel_cb`` only in the interpreter loop.
The compiled (fast-solver) path called ``engine.run_compiled_simulation``
without it, so once ``solve_ivp`` (or the in-house event / sampled-data /
transport-delay segment loops) started, the whole integration and the
post-solve replay ran to completion. A probe that set the cancel flag 1 s
into a 9.6 s compiled run saw the run return after 9.65 s, the flag never
polled. ``run_tuning_simulation`` polled it once, before the solve.

THE FIX
-------
``compiled_runner._RunMonitor`` wraps the RHS handed to every integrator
backend and polls the cancel (and progress) hook every few evaluations, and
the replay loop polls it every few hundred steps. A cancel raises the private
``CompiledRunCancelled`` (a ``BaseException`` so the solver's
``except Exception`` guards cannot swallow it), caught at the top of
``run_compiled_simulation``, which sets ``engine.compiled_run_cancelled``,
cuts Scope histories back to what was replayed and returns False. Callers
check the flag so a cancelled run never falls back to the interpreter.
"""

import threading
import time

import numpy as np
import pytest
from PyQt6.QtCore import QPoint

STIFF = [1.0, 10001.0, 1e4]  # (s+1)(s+1e4): RK45 takes ~3e-4 s steps
OSCILLATOR = [1.0, 0.001, 1e4]  # lightly damped, omega = 100 rad/s


def _build(sim_time, sim_dt, method, den=STIFF, extra=None):
    """Step [-> ZOH | TransportDelay] -> TransferFunction -> Scope."""
    from lib.lib import DSim

    dsim = DSim()
    menu = {b.fn_name: b for b in dsim.menu_blocks}
    step = dsim.add_block(menu["step"], QPoint(100, 100))
    tf = dsim.add_block(menu["transfer_function"], QPoint(300, 100))
    scope = dsim.add_block(menu["scope"], QPoint(500, 100))
    tf.params["numerator"] = [den[-1]]
    tf.params["denominator"] = list(den)
    src = step
    if extra == "zoh":
        src = dsim.add_block(menu["zero_order_hold"], QPoint(200, 100))
        src.params["sampling_time"] = 0.05
    elif extra == "delay":
        src = dsim.add_block(menu["transportdelay"], QPoint(200, 100))
        src.params["delay_time"] = 0.3
    if src is not step:
        dsim.add_line((step.name, 0, step.out_coords[0]), (src.name, 0, src.in_coords[0]))
    dsim.add_line((src.name, 0, src.out_coords[0]), (tf.name, 0, tf.in_coords[0]))
    dsim.add_line((tf.name, 0, tf.out_coords[0]), (scope.name, 0, scope.in_coords[0]))
    dsim.use_fast_solver = True
    dsim.solver_method = method
    # The interpreter fallback would pop a SignalPlot window.
    dsim.pyqtPlotScope = lambda: None
    _arm(dsim, sim_time, sim_dt)
    return dsim, scope


def _arm(dsim, sim_time, sim_dt):
    dsim.sim_time = sim_time
    dsim.sim_dt = sim_dt
    dsim.plot_trange = sim_time
    dsim.execution_init_time = lambda: dsim.sim_time
    assert dsim.execution_init() is True


def _scope_vector(scope):
    return np.asarray(scope.exec_params.get("vector", []))


# Each case runs for roughly 10-20 s when left alone (measured on an M-series
# Mac), so a cancel that is not honoured fails the time bound by a wide margin.
LONG_RUNS = [
    pytest.param("RK45", 100.0, 0.01, STIFF, None, "scipy+events", id="rk45-events"),
    pytest.param("LSODA", 300.0, 0.01, OSCILLATOR, None, "scipy+events", id="lsoda"),
    pytest.param("RK45", 60.0, 0.01, STIFF, "zoh", "scipy+hybrid", id="rk45-sampled"),
    pytest.param("RK45", 60.0, 0.01, STIFF, "delay", "scipy+hybrid", id="rk45-delay"),
    pytest.param("RK4", 60.0, 2e-5, OSCILLATOR, None, "fixed_step", id="rk4-fixed"),
    pytest.param("RK4", 60.0, 2e-4, OSCILLATOR, "zoh", "fixed_step+hybrid", id="rk4-sampled"),
]

CANCEL_AFTER = 0.3
# Generous for slow CI runners; an uncancelled run takes 10 s or more.
MAX_ELAPSED = 3.0


def _cancel_from_thread(after):
    flag = threading.Event()
    timer = threading.Timer(after, flag.set)
    timer.daemon = True
    timer.start()
    return flag, timer


@pytest.mark.regression
@pytest.mark.parametrize("method,sim_time,sim_dt,den,extra,backend", LONG_RUNS)
def test_cancel_interrupts_a_running_compiled_solve(
    qapp, method, sim_time, sim_dt, den, extra, backend
):
    dsim, scope = _build(sim_time, sim_dt, method, den, extra)
    flag, timer = _cancel_from_thread(CANCEL_AFTER)
    t0 = time.monotonic()
    try:
        dsim.execution_batch(cancel_cb=flag.is_set, defer_plots=True)
    finally:
        timer.cancel()
    elapsed = time.monotonic() - t0

    assert flag.is_set()
    assert elapsed < MAX_ELAPSED, f"compiled run took {elapsed:.2f}s after cancel"
    assert dsim.last_solver_type == "Fast (Compiled)", "must not fall back to the interpreter"
    assert dsim.engine.compiled_run_cancelled is True
    diag = dsim.engine.last_solver_diagnostics
    assert diag["failure_stage"] == "cancelled"
    assert diag["backend"] == backend
    assert dsim.execution_initialized is False
    # No zero-padded tail and no previous-run trace: the scope and the
    # timeline agree (both empty: the solve itself was interrupted).
    assert len(_scope_vector(scope)) == len(np.asarray(dsim.timeline).ravel()) == 0


@pytest.mark.regression
@pytest.mark.parametrize("extra", [None, "zoh", "delay"])
def test_the_next_run_after_a_cancel_matches_a_fresh_one(qapp, extra):
    dsim, scope = _build(60.0, 0.01, "RK45", STIFF, extra)
    flag, timer = _cancel_from_thread(CANCEL_AFTER)
    try:
        dsim.execution_batch(cancel_cb=flag.is_set, defer_plots=True)
    finally:
        timer.cancel()
    assert dsim.engine.compiled_run_cancelled is True

    _arm(dsim, 1.0, 0.01)
    dsim.execution_batch(defer_plots=True)
    assert dsim.engine.compiled_run_cancelled is False
    assert dsim.engine.last_solver_diagnostics["success"] is True

    fresh, fresh_scope = _build(1.0, 0.01, "RK45", STIFF, extra)
    fresh.execution_batch(defer_plots=True)
    np.testing.assert_allclose(np.asarray(dsim.timeline), np.asarray(fresh.timeline))
    np.testing.assert_allclose(_scope_vector(scope), _scope_vector(fresh_scope), atol=1e-9)
    assert len(_scope_vector(scope)) == 101


@pytest.mark.regression
def test_cancel_during_the_replay_keeps_only_replayed_samples(qapp):
    """Cancelled after the solve: Scope and timeline cut to the same length."""
    from lib.engine.compiled_runner import _RunMonitor

    dsim, scope = _build(20.0, 0.001, "LSODA", [1.0, 1.0])
    engine = dsim.engine
    replaying = {"on": False, "polls": 0}
    original = engine._replay_compiled_signals

    def replay(*args, **kwargs):
        replaying["on"] = True
        return original(*args, **kwargs)

    engine._replay_compiled_signals = replay

    def cancel_cb():
        if replaying["on"]:
            replaying["polls"] += 1
        return replaying["polls"] >= 3  # third replay poll

    dsim.execution_batch(cancel_cb=cancel_cb, defer_plots=True)

    n_done = 2 * _RunMonitor.REPLAY_STRIDE
    assert engine.compiled_run_cancelled is True
    timeline = np.asarray(dsim.timeline).ravel()
    vector = _scope_vector(scope)
    assert len(timeline) == len(vector) == n_done
    # What was kept is the real trace: first-order step response 1 - e^-t.
    np.testing.assert_allclose(vector.ravel(), 1.0 - np.exp(-timeline), atol=1e-6)


@pytest.mark.regression
def test_run_tuning_simulation_cancels_a_compiled_run(qapp):
    dsim, _ = _build(100.0, 0.01, "RK45", STIFF)
    flag, timer = _cancel_from_thread(CANCEL_AFTER)
    t0 = time.monotonic()
    try:
        ok, msg = dsim.run_tuning_simulation(100.0, 0.01, cancel_cb=flag.is_set)
    finally:
        timer.cancel()
    assert (ok, msg) == (False, "cancelled")
    assert time.monotonic() - t0 < MAX_ELAPSED
    assert dsim.execution_initialized is False


@pytest.mark.regression
def test_compiled_run_reports_progress(qapp):
    dsim, _ = _build(5.0, 0.01, "RK45", STIFF)
    seen = []
    dsim.execution_batch(progress_cb=lambda t, t_end: seen.append((t, t_end)), defer_plots=True)
    assert dsim.engine.last_solver_diagnostics["success"] is True
    assert len(seen) > 10
    times = [t for t, _ in seen]
    assert times == sorted(times)
    assert all(t_end == 5.0 for _, t_end in seen)
    assert 0.0 <= times[0] and times[-1] <= 5.0
    assert times[-1] > 4.0
