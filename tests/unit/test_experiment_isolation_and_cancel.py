"""Regression tests for experiment isolation and cancel latency.

BUG DESCRIPTION
---------------
1. ``ExperimentController`` handed the *live* ``window.dsim`` to
   ``MonteCarloWorker`` / ``ParameterSweepWorker``. Both runners drive full
   simulations, rewriting ``blocks_list`` / ``line_list`` / ``timeline`` /
   ``execution_initialized`` from the worker thread, while the window's 60 FPS
   ``safe_update`` iterated the same lists to paint and could itself call
   ``dsim.execution_loop()``.
2. Cancellation was polled only *between* runs, so cancelling cost a whole
   simulation (``run_tuning_simulation`` had no hook at all).

THE FIX
-------
``DSim.clone_for_analysis()`` produces an independent copy for the worker, and
``run_tuning_simulation(..., cancel_cb=...)`` polls the flag inside the step
loop; the runners forward their ``cancel_cb`` into it.
"""

import numpy as np
import pytest
from PyQt5.QtCore import QPoint


def _diagram(sim_time=0.2, sim_dt=0.01):
    """Step -> Integrator -> Scope on a fresh DSim (interpreter path)."""
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
    dsim.use_fast_solver = False  # exercise the interpreter loop (the cancel path)
    dsim.pyqtPlotScope = lambda: None
    return dsim


@pytest.mark.unit
class TestCloneForAnalysis:
    def test_clone_is_independent(self, qapp):
        dsim = _diagram()
        clone = dsim.clone_for_analysis()

        assert clone is not dsim
        assert clone.blocks_list is not dsim.blocks_list
        assert clone.line_list is not dsim.line_list
        assert [b.name for b in clone.blocks_list] == [b.name for b in dsim.blocks_list]
        assert len(clone.line_list) == len(dsim.line_list)
        assert clone.sim_time == dsim.sim_time
        assert clone.sim_dt == dsim.sim_dt

    def test_mutating_the_clone_leaves_the_original_alone(self, qapp):
        dsim = _diagram()
        clone = dsim.clone_for_analysis()

        original = {b.name: b for b in dsim.blocks_list}
        copy_blocks = {b.name: b for b in clone.blocks_list}
        name = next(n for n in copy_blocks if n.startswith("step"))
        copy_blocks[name].params["value"] = 99.0
        assert original[name].params.get("value") != 99.0

    def test_running_the_clone_does_not_touch_the_live_diagram(self, qapp, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        dsim = _diagram()
        clone = dsim.clone_for_analysis()

        ok, err = clone.run_tuning_simulation(0.05, 0.01)
        assert ok, err

        # The live diagram never entered execution and kept its own state.
        assert dsim.execution_initialized is False
        assert dsim.timeline is None or len(np.atleast_1d(dsim.timeline)) <= 1
        assert len(dsim.blocks_list) == 3


@pytest.mark.unit
class TestRunTuningSimulationCancel:
    def test_cancel_before_start_returns_immediately(self, qapp, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        dsim = _diagram(sim_time=5.0, sim_dt=0.001)
        ok, msg = dsim.run_tuning_simulation(5.0, 0.001, cancel_cb=lambda: True)
        assert ok is False
        assert msg == "cancelled"
        assert dsim.execution_initialized is False

    def test_cancel_mid_run_stops_early(self, qapp, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        dsim = _diagram(sim_time=50.0, sim_dt=0.001)

        state = {"polls": 0}

        def cancel_cb():
            state["polls"] += 1
            return state["polls"] > 1  # cancel on the second poll (inside the run)

        ok, msg = dsim.run_tuning_simulation(50.0, 0.001, cancel_cb=cancel_cb, cancel_every=5)
        assert ok is False
        assert msg == "cancelled"
        assert dsim.execution_initialized is False
        # Stopped long before the 50 s horizon.
        assert dsim.time_step < 5.0


class _StubDSim:
    """Minimal DSim stand-in recording how run_tuning_simulation is called."""

    def __init__(self):
        self.sim_time = 1.0
        self.sim_dt = 0.01
        self.blocks_list = []
        self.calls = []
        self.timeline = np.linspace(0.0, 1.0, 3)
        # harvest_scope_signals reads dsim.engine.active_blocks_list first.
        self.engine = type("E", (), {"active_blocks_list": []})()

    def run_tuning_simulation(self, sim_time, sim_dt, cancel_cb=None, cancel_every=200):
        self.calls.append({"cancel_cb": cancel_cb})
        if cancel_cb is not None and cancel_cb():
            return (False, "cancelled")
        return (True, "")


@pytest.mark.unit
class TestRunnersForwardCancel:
    def test_monte_carlo_forwards_cancel_cb_and_stops_mid_run(self):
        from lib.analysis.monte_carlo import MonteCarloRunner

        dsim = _StubDSim()
        flag = {"cancel": False}

        def cancel_cb():
            return flag["cancel"]

        def progress(done, total):
            flag["cancel"] = True  # cancel arrives while run 2 is executing

        result = MonteCarloRunner(dsim).run(
            n_runs=10, sim_time=1.0, sim_dt=0.01, progress_cb=progress, cancel_cb=cancel_cb
        )
        assert dsim.calls, "run_tuning_simulation was never called"
        assert all(c["cancel_cb"] is cancel_cb for c in dsim.calls)
        assert len(dsim.calls) < 10
        assert result["n_runs"] < 10

    def test_parameter_sweep_forwards_cancel_cb(self):
        from lib.analysis.parameter_sweep import ParameterSweepRunner

        dsim = _StubDSim()
        block = type("B", (), {})()
        block.name = "gain0"
        block.block_fn = "Gain"  # not a Scope: harvest_scope_signals skips it
        block.params = {"gain": 1.0}
        block.exec_params = None
        dsim.blocks_list = [block]

        cancel_cb = lambda: False  # noqa: E731 - identity matters, not the body
        ParameterSweepRunner(dsim).run(
            axes=[{"block": "gain0", "param": "gain", "values": [1.0, 2.0]}],
            sim_time=1.0,
            sim_dt=0.01,
            cancel_cb=cancel_cb,
        )
        assert dsim.calls
        assert all(c["cancel_cb"] is cancel_cb for c in dsim.calls)
