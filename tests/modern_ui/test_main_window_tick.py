"""Regression tests for the window's 60 FPS tick (``safe_update``).

BUG DESCRIPTION
---------------
1. The tick advanced an interpreted run by exactly **one** step per frame, so a
   10 s run at dt = 1 ms took ~167 s of wall clock.
2. It called ``canvas.update()`` on every tick even with nothing happening, so
   an idle diagram repainted the whole widget 60 times a second.
3. With the batch run moved onto a worker thread, stepping from the tick as
   well would drive the same DSim from two threads.

THE FIX
-------
``_advance_simulation`` steps until a ~10 ms wall-clock budget is spent (and,
for a paced run, no further than the wall clock), the repaint is gated on
``canvas._animation_should_run()``, and the tick skips stepping while
``batch_simulation_active()``.
"""

import time
import types

import pytest


@pytest.fixture(scope="module")
def window(qapp):
    from modern_ui.main_window import ModernDiaBloSWindow

    w = ModernDiaBloSWindow()
    yield w
    w.dsim.dirty = False
    w.close()


class _StubDSim:
    """Counts execution_loop() calls; never touches the real engine."""

    def __init__(self, steps_to_finish=10**9, dt=0.0, real_time=False):
        self.calls = 0
        self.execution_initialized = True
        self.execution_pause = False
        self.real_time = real_time
        self.execution_time_start = time.time()
        self.time_step = 0.0
        # SafetyChecks.check_simulation_state wants a non-empty, well-formed
        # block list before the tick will step the run.
        self.blocks_list = [
            types.SimpleNamespace(
                name="stub0", sid=0, in_ports=0, out_ports=1, b_type=0, fn_name="stub"
            )
        ]
        self.line_list = []
        self.sim_time = 1.0
        self.sim_dt = 0.01
        self._dt = dt
        self._steps_to_finish = steps_to_finish

    def execution_loop(self):
        self.calls += 1
        self.time_step += self._dt
        if self.calls >= self._steps_to_finish:
            self.execution_initialized = False


@pytest.fixture
def stub_canvas_dsim(window, monkeypatch):
    """Swap in a stub DSim on the canvas for the duration of one test."""

    def _install(stub):
        monkeypatch.setattr(window.canvas, "dsim", stub, raising=False)
        return stub

    return _install


@pytest.mark.qt
class TestAdvanceSimulation:
    def test_spends_the_budget_instead_of_one_step_per_tick(self, window, stub_canvas_dsim):
        stub = stub_canvas_dsim(_StubDSim())
        steps = window._advance_simulation(budget_s=0.02)
        assert steps > 1, "a tick must advance more than a single step"
        assert steps == stub.calls

    def test_respects_the_budget(self, window, stub_canvas_dsim):
        stub_canvas_dsim(_StubDSim())
        t0 = time.perf_counter()
        window._advance_simulation(budget_s=0.005)
        # Generous upper bound: the loop only checks between steps.
        assert time.perf_counter() - t0 < 0.5

    def test_stops_when_the_run_finishes(self, window, stub_canvas_dsim):
        stub = stub_canvas_dsim(_StubDSim(steps_to_finish=3))
        steps = window._advance_simulation(budget_s=1.0)
        assert steps == 3
        assert stub.execution_initialized is False

    def test_stops_when_paused(self, window, stub_canvas_dsim):
        stub = _StubDSim()
        stub_canvas_dsim(stub)
        original = stub.execution_loop

        def loop():
            original()
            stub.execution_pause = True

        stub.execution_loop = loop
        assert window._advance_simulation(budget_s=1.0) == 1

    def test_real_time_pacing_caps_the_burst(self, window, stub_canvas_dsim):
        # Simulated time already far ahead of the wall clock -> one step only.
        stub = _StubDSim(dt=0.0, real_time=True)
        stub.time_step = 10_000.0
        stub_canvas_dsim(stub)
        assert window._advance_simulation(budget_s=1.0) == 1

    def test_no_execution_loop_is_a_noop(self, window, stub_canvas_dsim):
        stub_canvas_dsim(types.SimpleNamespace(execution_initialized=True))
        assert window._advance_simulation() == 0


@pytest.mark.qt
class TestIdleRepaint:
    def _spy_update(self, window, monkeypatch):
        calls = {"n": 0}
        monkeypatch.setattr(window.canvas, "update", lambda *a: calls.__setitem__("n", 1))
        monkeypatch.setattr(window.canvas, "is_simulation_running", lambda: False)
        return calls

    def test_idle_canvas_is_not_repainted(self, window, monkeypatch):
        calls = self._spy_update(window, monkeypatch)
        monkeypatch.setattr(window.canvas, "_animation_should_run", lambda: False)
        window.safe_update()
        assert calls["n"] == 0

    def test_live_canvas_is_repainted(self, window, monkeypatch):
        calls = self._spy_update(window, monkeypatch)
        monkeypatch.setattr(window.canvas, "_animation_should_run", lambda: True)
        window.safe_update()
        assert calls["n"] == 1

    def test_running_canvas_is_repainted(self, window, monkeypatch, stub_canvas_dsim):
        calls = {"n": 0}
        stub_canvas_dsim(_StubDSim(steps_to_finish=1))
        monkeypatch.setattr(window.canvas, "update", lambda *a: calls.__setitem__("n", 1))
        monkeypatch.setattr(window.canvas, "is_simulation_running", lambda: True)
        monkeypatch.setattr(window.canvas, "_animation_should_run", lambda: False)
        window.safe_update()
        assert calls["n"] == 1


@pytest.mark.qt
class TestBatchGuard:
    def test_tick_does_not_step_while_a_batch_worker_runs(
        self, window, monkeypatch, stub_canvas_dsim
    ):
        from modern_ui.controllers import simulation_controller as sc

        stub = stub_canvas_dsim(_StubDSim())
        monkeypatch.setattr(window.canvas, "is_simulation_running", lambda: True)
        monkeypatch.setattr(window.canvas, "update", lambda *a: None)

        sentinel = object()
        sc._ACTIVE_BATCH_WORKERS.add(sentinel)
        try:
            assert sc.batch_simulation_active() is True
            window.safe_update()
        finally:
            sc._ACTIVE_BATCH_WORKERS.discard(sentinel)

        assert stub.calls == 0, "the GUI thread must not step a threaded batch run"
