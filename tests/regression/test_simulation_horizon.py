"""
Regression tests for where a simulation stops.

BUG DESCRIPTION
---------------
The interpreter advanced time, executed the step, and only then tested
``time_step > execution_time``.  That test can only become true on a step
already past the end, so every interpreted run emitted one sample beyond the
horizon:

    sim_time=3.0, dt=0.01   -> 302 samples, last at t=3.01
    sim_time=1.0, dt=0.3    ->   5 samples, last at t=1.20
    sim_time=2.0, dt=0.25   ->  10 samples, last at t=2.25

The compiled solver evaluates on ``arange(0, T+dt, dt)`` clipped to ``<= T``
and stopped at 3.00 / 0.90 / 2.00, so the two solvers disagreed on both the
length and the endpoint of every trace.

THE FIX
-------
``DSim._interpreter_step`` now ends the run when the *next* step would pass
the horizon, matching the compiled grid, and returns early once
``execution_initialized`` has been cleared so a caller driving the loop on a
time comparison cannot append samples on top of re-initialised blocks.
"""

import numpy as np
import pytest
from PyQt5.QtCore import QPoint


def _run(fast, sim_time, sim_dt):
    """Step -> Integrator -> Scope, run through the app's own batch entry point."""
    from lib.lib import DSim

    dsim = DSim()
    dsim.buttons_list = [type("B", (), {"active": False})() for _ in range(20)]
    menu = {b.fn_name: b for b in dsim.menu_blocks}
    step = dsim.add_block(menu["step"], QPoint(100, 100))
    integ = dsim.add_block(menu["integrator"], QPoint(300, 100))
    scope = dsim.add_block(menu["scope"], QPoint(500, 100))
    integ.params["init_conds"] = 0.0
    dsim.add_line((step.name, 0, step.out_coords[0]), (integ.name, 0, integ.in_coords[0]))
    dsim.add_line((integ.name, 0, integ.out_coords[0]), (scope.name, 0, scope.in_coords[0]))
    dsim.sim_time = sim_time
    dsim.sim_dt = sim_dt
    dsim.plot_trange = sim_time
    dsim.execution_init_time = lambda: dsim.sim_time
    dsim.use_fast_solver = fast
    # The interpreter fallback runs the interactive loop, which would pop a
    # SignalPlot window; keep the test headless.
    dsim.pyqtPlotScope = lambda: None
    assert dsim.execution_init() is True
    dsim.execution_batch()
    timeline = np.asarray(dsim.timeline).flatten()
    vector = np.asarray(scope.exec_params["vector"]).flatten()
    return dsim, timeline, vector


# (sim_time, sim_dt, expected sample count, expected last sample time)
HORIZONS = [
    (3.0, 0.01, 301, 3.0),  # exact multiple
    (1.0, 0.3, 4, 0.9),  # horizon is not a multiple of dt
    (2.0, 0.25, 9, 2.0),
]


@pytest.mark.regression
@pytest.mark.parametrize("sim_time,sim_dt,n_expected,t_expected", HORIZONS)
class TestSimulationHorizon:
    def test_interpreter_stops_at_the_horizon(self, qapp, sim_time, sim_dt, n_expected, t_expected):
        dsim, timeline, vector = _run(False, sim_time, sim_dt)
        assert dsim.last_solver_type == "Standard (Interpreter)"
        assert len(timeline) == n_expected
        assert timeline[-1] == pytest.approx(t_expected, abs=1e-9)
        assert timeline[-1] <= sim_time + 1e-9, "run must never step past the horizon"
        assert len(vector) == len(timeline)

    def test_compiled_and_interpreted_grids_agree(
        self, qapp, sim_time, sim_dt, n_expected, t_expected
    ):
        _, t_fast, y_fast = _run(True, sim_time, sim_dt)
        _, t_slow, y_slow = _run(False, sim_time, sim_dt)
        assert len(t_fast) == len(t_slow) == n_expected
        assert t_fast[-1] == pytest.approx(t_slow[-1], abs=1e-9)
        assert len(y_fast) == len(y_slow)
        # Integral of a unit step: both solvers land on the same endpoint.
        assert y_slow[-1] == pytest.approx(t_expected, abs=1e-6)
        assert y_fast[-1] == pytest.approx(t_expected, abs=1e-6)


@pytest.mark.regression
class TestSteppingPastTheEnd:
    def test_run_clears_execution_initialized_at_the_horizon(self, qapp):
        dsim, timeline, _ = _run(False, 1.0, 0.1)
        assert dsim.execution_initialized is False
        assert timeline[-1] == pytest.approx(1.0, abs=1e-9)

    def test_extra_steps_after_the_run_do_no_work(self, qapp):
        """Stepping past the end must not extend or corrupt the trace.

        reset_memblocks() has already re-armed every block by then, so any
        further execution would append samples computed from re-initialised
        state — which is what a caller looping on ``time_step <= sim_time``
        does, since the run now ends at the horizon rather than one step past
        it.  The clock still advances so such a loop terminates.
        """
        dsim, timeline, vector = _run(False, 1.0, 0.1)
        n_t, n_y = len(timeline), len(vector)
        scope = next(b for b in dsim.blocks_list if b.block_fn == "Scope")
        t_before = dsim.time_step

        for _ in range(5):
            dsim.execution_loop_headless()

        assert len(dsim._timeline_list) == n_t
        assert len(np.asarray(scope.exec_params["vector"]).flatten()) == n_y
        assert dsim.time_step > t_before, "clock must advance so time-driven loops end"
