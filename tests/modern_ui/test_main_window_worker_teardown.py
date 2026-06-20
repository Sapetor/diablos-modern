"""Regression tests for ModernDiaBloSWindow experiment-worker teardown.

closeEvent historically tore down the window without cancelling/joining a
running Monte-Carlo or parameter-sweep QThread, so closing mid-experiment
destroyed a live thread ("QThread: Destroyed while thread is still running"),
aborting the process. These pin the _cancel_experiment_workers helper and its
wiring into closeEvent.
"""
import pytest
from PyQt5.QtGui import QCloseEvent


class _StubWorker:
    """Stand-in for a Monte-Carlo / sweep QThread.

    The real join behaviour is QThread's contract; what needs pinning is our
    helper's control flow (cancel only when running, bounded wait, reset to
    None, swallow a deleted-C++-object RuntimeError).
    """

    def __init__(self, running=True, wait_returns=True, raise_runtime=False):
        self._running = running
        self._wait_returns = wait_returns
        self._raise = raise_runtime
        self.cancel_calls = 0
        self.wait_calls = []

    def isRunning(self):
        if self._raise:
            raise RuntimeError("wrapped C/C++ object of type QThread has been deleted")
        return self._running

    def cancel(self):
        self.cancel_calls += 1
        self._running = False

    def wait(self, ms=None):
        self.wait_calls.append(ms)
        return self._wait_returns


@pytest.fixture(scope="module")
def window(qapp):
    from modern_ui.main_window import ModernDiaBloSWindow
    w = ModernDiaBloSWindow()
    yield w
    w.close()


@pytest.fixture(autouse=True)
def _clear_workers(window):
    """Ensure no worker leaks between tests."""
    window._mc_worker = None
    window._sweep_worker = None
    yield
    window._mc_worker = None
    window._sweep_worker = None


@pytest.mark.qt
class TestCancelExperimentWorkers:
    def test_running_worker_is_cancelled_joined_and_cleared(self, window):
        mc = _StubWorker(running=True)
        sweep = _StubWorker(running=True)
        window._mc_worker = mc
        window._sweep_worker = sweep

        window._cancel_experiment_workers()

        assert mc.cancel_calls == 1
        assert mc.wait_calls == [10000]
        assert sweep.cancel_calls == 1
        assert sweep.wait_calls == [10000]
        # References cleared so the threads can be GC'd / not re-joined.
        assert window._mc_worker is None
        assert window._sweep_worker is None

    def test_no_workers_is_noop(self, window):
        window._mc_worker = None
        window._sweep_worker = None
        # Must not raise even when the attributes are missing entirely.
        del window._mc_worker
        window._cancel_experiment_workers()
        assert getattr(window, '_mc_worker', None) is None

    def test_not_running_worker_is_not_cancelled(self, window):
        idle = _StubWorker(running=False)
        window._mc_worker = idle
        window._cancel_experiment_workers()
        assert idle.cancel_calls == 0
        assert idle.wait_calls == []
        assert window._mc_worker is None

    def test_deleted_cpp_object_runtimeerror_is_swallowed(self, window):
        broken = _StubWorker(raise_runtime=True)
        window._mc_worker = broken
        # Should not propagate the RuntimeError, and should still clear the ref.
        window._cancel_experiment_workers()
        assert window._mc_worker is None

    def test_timeout_logs_warning_but_still_clears(self, window, caplog):
        slow = _StubWorker(running=True, wait_returns=False)
        window._sweep_worker = slow
        with caplog.at_level("WARNING"):
            window._cancel_experiment_workers()
        assert slow.cancel_calls == 1
        assert window._sweep_worker is None
        assert any("did not stop within timeout" in r.message for r in caplog.records)

    def test_closeEvent_invokes_worker_teardown(self, window, monkeypatch):
        called = {"n": 0}
        monkeypatch.setattr(
            window, "_cancel_experiment_workers",
            lambda: called.__setitem__("n", called["n"] + 1),
        )
        window.closeEvent(QCloseEvent())
        assert called["n"] == 1
