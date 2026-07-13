"""
Integration test: the "Previous run" overlay stash against the REAL engine.

The unit tests in tests/unit/test_scope_plotter_prev_run.py exercise
ScopePlotter._stash_run against a stub dsim whose new_run() fabricates the
assumed timeline behavior. This test instead runs two real simulations
through DSim (both the interpreter path, which builds the timeline stepwise
from _timeline_list, and the compiled fast path, which assigns the solver's
sol.t), calls pyqtPlotScope after each, and asserts that:

- the first run has no previous run,
- after the second run the SignalPlot holds run 1's timeline and data,
- re-plotting run 2 via plot_again() keeps run 1 held (no spurious rotation).

If the engine ever rebuilds the timeline in place (same object identity for
a fresh run), or re-plot produces a new timeline object, this test fails
where the stub-based tests would silently stay green.
"""

import numpy as np
import pytest
from PyQt5.QtCore import QPoint


def _build_sine_scope():
    """Wire Sine -> Scope on a real DSim (both blocks are compilable)."""
    from lib.lib import DSim

    dsim = DSim()
    menu_by_fn = {b.fn_name: b for b in dsim.menu_blocks}
    sine = dsim.add_block(menu_by_fn["sine"], QPoint(100, 100))
    scope = dsim.add_block(menu_by_fn["scope"], QPoint(300, 100))
    dsim.add_line((sine.name, 0, sine.out_coords[0]), (scope.name, 0, scope.in_coords[0]))
    return dsim, sine, scope


def _run_and_plot(dsim, sim_time=1.0, sim_dt=0.01):
    """Run a full headless simulation and plot it (as the GUI does at run end)."""
    ok, msg = dsim.run_tuning_simulation(sim_time, sim_dt)
    assert ok, f"simulation failed: {msg}"
    dsim.scope_plotter.pyqtPlotScope()
    plotty = dsim.scope_plotter.plotty
    assert plotty is not None
    return plotty


@pytest.mark.integration
class TestPrevRunOverlayIntegration:
    @pytest.mark.parametrize("use_fast", [False, True], ids=["interpreter", "compiled"])
    def test_two_real_runs_hold_previous(self, qapp, use_fast):
        dsim, sine, _scope = _build_sine_scope()
        dsim.use_fast_solver = use_fast
        if use_fast:
            # Guard: the diagram must actually take the compiled fast path.
            assert dsim.engine.check_compilability(dsim.blocks_list)

        # Run 1: no held data yet.
        plotty = _run_and_plot(dsim)
        assert plotty.previous_run is None
        assert not plotty.prev_run_checkbox.isEnabled()
        run1_t = np.asarray(dsim.scope_plotter._current_run["t"], dtype=float).copy()
        run1_y = np.asarray(dsim.scope_plotter._current_run["vectors"][0], dtype=float).copy()
        assert len(run1_t) == len(run1_y) > 1

        # Run 2 with a different amplitude: run 1 must be held as previous.
        sine.params["amplitude"] = 3.0
        plotty = _run_and_plot(dsim)
        prev = plotty.previous_run
        assert prev is not None
        np.testing.assert_allclose(prev["t"], run1_t)
        np.testing.assert_allclose(prev["vectors"][0], run1_y)
        assert plotty.prev_run_checkbox.isEnabled()
        # Sanity: run 2 really produced different data than the held run.
        run2_y = np.asarray(dsim.scope_plotter._current_run["vectors"][0], dtype=float)
        assert not np.allclose(run2_y, run1_y)

        # "Plot Again" re-plots run 2 without executing: run 1 stays held.
        dsim.dirty = False
        dsim.scope_plotter.plot_again()
        prev = dsim.scope_plotter.plotty.previous_run
        assert prev is not None
        np.testing.assert_allclose(prev["vectors"][0], run1_y)

        dsim.scope_plotter._close_plotty()
