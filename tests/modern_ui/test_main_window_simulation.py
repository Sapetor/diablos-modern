"""
Characterization tests for ModernDiaBloSWindow's simulation-facade cluster.

main_window.py historically had zero coverage. These tests build a REAL
ModernDiaBloSWindow under offscreen Qt and pin down the observable behavior of
the window-side simulation handlers before extraction:

  * ``toggle_fast_solver``  (mirror flag onto window + dsim)
  * ``pause_simulation``    (set execution_pause + toolbar state)
  * ``stop_simulation``     (canvas.stop + toolbar state + status)
  * ``start_simulation``    (validate; block on ERROR, else clear + start)
  * ``step_simulation``     (single-step status transitions)

The real simulation engine is never driven; canvas/dsim entry points are spied
or stubbed via monkeypatch so the handler logic is exercised deterministically.

Run with:
    QT_QPA_PLATFORM=offscreen MPLBACKEND=Agg python -m pytest \
        tests/modern_ui/test_main_window_simulation.py -p no:cacheprovider \
        -o addopts="" --timeout=60 --timeout-method=signal
"""

import types

import pytest

from lib.diagram_validator import ErrorSeverity


@pytest.fixture(scope="module")
def window(qapp):
    from modern_ui.main_window import ModernDiaBloSWindow
    w = ModernDiaBloSWindow()
    yield w
    w.close()


@pytest.fixture(autouse=True)
def _restore_sim_state(window):
    dsim = window.dsim
    saved = {
        "pause": getattr(dsim, "execution_pause", None),
        "init": getattr(dsim, "execution_initialized", None),
        "dsim_fast": getattr(dsim, "use_fast_solver", None),
        "win_fast": getattr(window, "use_fast_solver", None),
    }
    yield
    dsim.execution_pause = saved["pause"]
    dsim.execution_initialized = saved["init"]
    if saved["dsim_fast"] is not None:
        dsim.use_fast_solver = saved["dsim_fast"]
    if saved["win_fast"] is not None:
        window.use_fast_solver = saved["win_fast"]


def _err(severity):
    return types.SimpleNamespace(severity=severity)


# ---------------------------------------------------------------------------
# toggle_fast_solver
# ---------------------------------------------------------------------------

class TestToggleFastSolver:
    def test_mirrors_flag_to_window_and_dsim(self, window):
        window.toggle_fast_solver(False)
        assert window.use_fast_solver is False
        assert window.dsim.use_fast_solver is False
        window.toggle_fast_solver(True)
        assert window.use_fast_solver is True
        assert window.dsim.use_fast_solver is True


# ---------------------------------------------------------------------------
# pause_simulation
# ---------------------------------------------------------------------------

class TestPause:
    def test_sets_pause_and_toolbar(self, window, monkeypatch):
        captured = {}
        monkeypatch.setattr(window.toolbar, "set_simulation_state",
                            lambda r, p: captured.__setitem__("state", (r, p)))
        window.pause_simulation()
        assert window.dsim.execution_pause is True
        assert captured["state"] == (True, True)


# ---------------------------------------------------------------------------
# stop_simulation
# ---------------------------------------------------------------------------

class TestStop:
    def test_stops_canvas_and_resets_toolbar(self, window, monkeypatch):
        captured = {}
        monkeypatch.setattr(window.canvas, "stop_simulation",
                            lambda: captured.__setitem__("stopped", True))
        monkeypatch.setattr(window.toolbar, "set_simulation_state",
                            lambda r, p: captured.__setitem__("state", (r, p)))
        window.stop_simulation()
        assert captured.get("stopped") is True
        assert captured["state"] == (False, False)
        assert window.status_message.text() == "Simulation stopped"


# ---------------------------------------------------------------------------
# start_simulation
# ---------------------------------------------------------------------------

class TestStart:
    def test_blocked_by_validation_errors(self, window, monkeypatch):
        started = {}
        monkeypatch.setattr(window.canvas, "run_validation",
                            lambda: [_err(ErrorSeverity.ERROR)])
        monkeypatch.setattr(window.canvas, "start_simulation",
                            lambda: started.__setitem__("ran", True))
        monkeypatch.setattr(window.error_panel, "set_errors", lambda errs: None)
        window.start_simulation()
        # Critical errors block the run entirely.
        assert "ran" not in started
        assert "Cannot start simulation" in window.status_message.text()

    def test_runs_when_validation_clean(self, window, monkeypatch):
        captured = {}
        monkeypatch.setattr(window.canvas, "run_validation", lambda: [])
        monkeypatch.setattr(window.error_panel, "clear",
                            lambda: captured.__setitem__("cleared", True))
        monkeypatch.setattr(window.canvas, "clear_validation", lambda: None)
        monkeypatch.setattr(window.canvas, "start_simulation",
                            lambda: captured.__setitem__("ran", True))
        monkeypatch.setattr(window.canvas, "is_simulation_running", lambda: True)
        window.start_simulation()
        assert captured.get("cleared") is True
        assert captured.get("ran") is True
        assert window.status_message.text() == "Starting simulation..."


# ---------------------------------------------------------------------------
# step_simulation
# ---------------------------------------------------------------------------

class TestStep:
    def test_first_step_initializes(self, window, monkeypatch):
        monkeypatch.setattr(window.dsim, "single_step", lambda: True)
        window.dsim.execution_initialized = False  # not yet initialized
        window.dsim.time_step = 0.1
        monkeypatch.setattr(window.canvas, "update", lambda: None)
        monkeypatch.setattr(window.toolbar, "set_simulation_state", lambda r, p: None)
        window.step_simulation()
        assert "Started stepping at t=0.1000s" == window.status_message.text()

    def test_subsequent_step(self, window, monkeypatch):
        monkeypatch.setattr(window.dsim, "single_step", lambda: True)
        window.dsim.execution_initialized = True  # already running
        window.dsim.time_step = 0.2
        monkeypatch.setattr(window.canvas, "update", lambda: None)
        monkeypatch.setattr(window.toolbar, "set_simulation_state", lambda r, p: None)
        window.step_simulation()
        assert "Stepped to t=0.2000s" == window.status_message.text()

    def test_finished_when_step_fails_after_init(self, window, monkeypatch):
        # single_step returns False AND engine reports de-initialized -> finished.
        monkeypatch.setattr(window.dsim, "single_step", lambda: False)
        window.dsim.execution_initialized = True

        def fake_single_step():
            window.dsim.execution_initialized = False
            return False
        monkeypatch.setattr(window.dsim, "single_step", fake_single_step)
        monkeypatch.setattr(window.toolbar, "set_simulation_state", lambda r, p: None)
        window.step_simulation()
        assert window.status_message.text() == "Simulation finished"
