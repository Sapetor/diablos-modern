"""Smoke + regression tests for :class:`SimulationController`.

The controller had zero coverage while owning validation, start/stop and batch
execution. It also ran ``execution_batch()`` synchronously on the GUI thread
behind one ``processEvents()``, which froze the window for the whole run with
no way to cancel; the batch now runs on a ``BatchSimulationWorker`` thread and
delivers its results back through Qt signals.

Everything here drives stub DSims -- the engine is never run.
"""

import time
import types

import pytest
from PyQt5.QtWidgets import QApplication

from modern_ui.controllers import simulation_controller as sc
from modern_ui.controllers.simulation_controller import SimulationController


def _block(name="stub0"):
    return types.SimpleNamespace(
        name=name,
        sid=0,
        in_ports=0,
        out_ports=1,
        b_type=0,
        fn_name="stub",
        block_fn="Step",
        username=name,
        params={},
        exec_params={},
    )


class _StubDSim:
    """A DSim that satisfies validation and records the calls it receives."""

    def __init__(self, real_time=False, init_ok=True, dynamic_plot=False):
        self.blocks_list = [_block()]
        self.line_list = []
        self.execution_initialized = False
        self.execution_pause = False
        self.real_time = real_time
        self.dynamic_plot = dynamic_plot
        self.sim_time = 1.0
        self.sim_dt = 0.01
        self.error_msg = ""
        self.last_solver_type = "Fast (Compiled)"
        self.engine = None
        self._init_ok = init_ok
        self.batch_calls = []
        self.plot_again_calls = 0
        self.batch_delay = 0.0
        self.raise_in_batch = None

    def execution_init(self):
        self.execution_initialized = self._init_ok
        return self._init_ok

    def execution_batch(self, progress_cb=None, cancel_cb=None, defer_plots=False):
        self.batch_calls.append(
            {"progress_cb": progress_cb, "cancel_cb": cancel_cb, "defer_plots": defer_plots}
        )
        if self.raise_in_batch:
            raise RuntimeError(self.raise_in_batch)
        deadline = time.monotonic() + self.batch_delay
        while time.monotonic() < deadline:
            if cancel_cb is not None and cancel_cb():
                break
            time.sleep(0.005)
        if progress_cb is not None:
            progress_cb(self.sim_time, self.sim_time)
        self.execution_initialized = False

    def plot_again(self):
        self.plot_again_calls += 1


def _pump(predicate, timeout=10.0):
    """Spin the Qt event loop until ``predicate()`` or the timeout expires."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        QApplication.processEvents()
        if predicate():
            return True
        time.sleep(0.005)
    return predicate()


@pytest.fixture
def statuses():
    return []


@pytest.fixture
def make_controller(qapp, statuses):
    created = []

    def _make(dsim):
        ctrl = SimulationController(dsim)
        ctrl.status_changed.connect(statuses.append)
        created.append(ctrl)
        return ctrl

    yield _make
    for ctrl in created:
        ctrl.cancel_batch()


@pytest.mark.qt
class TestStart:
    def test_interactive_run_returns_without_batching(self, make_controller, statuses):
        dsim = _StubDSim(real_time=True)
        ctrl = make_controller(dsim)
        assert ctrl.start() is True
        assert dsim.batch_calls == []
        assert "Simulation started" in statuses

    def test_validation_failure_blocks_the_run(self, make_controller, statuses, monkeypatch):
        monkeypatch.setattr(
            sc.ValidationHelper,
            "validate_block_connections",
            staticmethod(lambda blocks, lines: (False, ["port 0 unconnected"])),
        )
        dsim = _StubDSim()
        ctrl = make_controller(dsim)
        assert ctrl.start() is False
        assert dsim.batch_calls == []
        assert any("Validation failed" in s for s in statuses)

    def test_safety_failure_blocks_the_run(self, make_controller, statuses):
        dsim = _StubDSim()
        dsim.blocks_list = []  # "No blocks in simulation"
        ctrl = make_controller(dsim)
        assert ctrl.start() is False
        assert any("Safety check failed" in s for s in statuses)

    def test_failed_init_is_reported(self, make_controller, statuses):
        dsim = _StubDSim(init_ok=False)
        dsim.error_msg = "boom"
        ctrl = make_controller(dsim)
        assert ctrl.start() is False
        assert any("failed to start" in s for s in statuses)


@pytest.mark.qt
class TestBatchOnAWorkerThread:
    def test_batch_runs_off_the_gui_thread_and_finishes(self, make_controller, statuses):
        dsim = _StubDSim(real_time=False)
        ctrl = make_controller(dsim)

        assert ctrl.start() is True
        assert ctrl.is_batch_running() is True
        assert sc.batch_simulation_active() is True

        assert _pump(lambda: not ctrl.is_batch_running())
        assert sc.batch_simulation_active() is False

        assert len(dsim.batch_calls) == 1
        # Qt objects must not be built off the GUI thread: the run defers them.
        assert dsim.batch_calls[0]["defer_plots"] is True
        assert callable(dsim.batch_calls[0]["cancel_cb"])
        # ...and the plotting happens here instead, on the GUI thread.
        assert dsim.plot_again_calls == 1
        assert any("Simulation finished" in s for s in statuses)

    def test_cancel_stops_a_running_batch(self, make_controller):
        dsim = _StubDSim(real_time=False)
        dsim.batch_delay = 30.0  # would outlive the test if it could not be cancelled
        ctrl = make_controller(dsim)

        assert ctrl.start() is True
        assert ctrl.is_batch_running() is True

        t0 = time.monotonic()
        assert ctrl.cancel_batch(wait_ms=10000) is True
        assert time.monotonic() - t0 < 10.0
        assert ctrl.is_batch_running() is False
        assert sc.batch_simulation_active() is False

    def test_stop_cancels_the_batch_and_clears_the_flag(self, make_controller):
        dsim = _StubDSim(real_time=False)
        dsim.batch_delay = 30.0
        ctrl = make_controller(dsim)
        ctrl.start()

        ctrl.stop()
        assert ctrl.is_batch_running() is False
        assert dsim.execution_initialized is False

    def test_worker_failure_is_reported_not_raised(self, make_controller, statuses):
        dsim = _StubDSim(real_time=False)
        dsim.raise_in_batch = "solver exploded"
        ctrl = make_controller(dsim)

        assert ctrl.start() is True
        assert _pump(lambda: not ctrl.is_batch_running())
        assert any("solver exploded" in s for s in statuses)
        assert dsim.plot_again_calls == 0

    def test_dynamic_plot_stays_on_the_gui_thread(self, make_controller):
        # Live plotting drives pyqtgraph from inside the step loop, so that path
        # must not be moved off the GUI thread.
        dsim = _StubDSim(real_time=False, dynamic_plot=True)
        ctrl = make_controller(dsim)

        assert ctrl.start() is True
        assert ctrl.is_batch_running() is False
        assert len(dsim.batch_calls) == 1
        assert dsim.batch_calls[0]["defer_plots"] is False
        assert dsim.plot_again_calls == 1


@pytest.mark.qt
class TestVerificationReportIsLogged:
    def test_report_goes_to_the_logger_not_stdout(self, make_controller, caplog, capsys):
        dsim = _StubDSim()
        scope = types.SimpleNamespace(
            block_fn="Display",
            username="d0",
            name="display0",
            params={"_display_value_": "42", "label": "answer"},
        )
        dsim.blocks_list = [scope]
        ctrl = make_controller(dsim)

        with caplog.at_level("INFO"):
            ctrl._print_terminal_verification()

        assert capsys.readouterr().out == ""
        assert any("VERIFICATION RESULTS" in r.message for r in caplog.records)
