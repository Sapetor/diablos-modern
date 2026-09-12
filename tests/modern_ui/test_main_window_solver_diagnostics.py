"""
Tests for ModernDiaBloSWindow's solver-diagnostics status-bar hookup.

lib/engine/simulation_engine.py records a compact per-run diagnostics dict and a
one-line summary; DSim exposes it as ``last_solver_diagnostics_summary``. When a
batch run finishes, ``SimulationController._finish_batch`` appends that summary
to the finished status line (and logs it) -- but only when the compiled solver
actually ran. An interpreter-path run records no diagnostics, so the status bar
must keep the plain finished message.

A REAL ModernDiaBloSWindow is built under offscreen Qt (per the module fixture);
the engine is never driven -- diagnostics are stubbed on dsim so the reporting
logic is exercised deterministically.

Run with:
    QT_QPA_PLATFORM=offscreen MPLBACKEND=Agg python -m pytest \
        tests/modern_ui/test_main_window_solver_diagnostics.py -p no:cacheprovider \
        -o addopts="" --timeout=60 --timeout-method=signal
"""

import pytest


@pytest.fixture(scope="module")
def window(qapp):
    from modern_ui.main_window import ModernDiaBloSWindow

    w = ModernDiaBloSWindow()
    yield w
    w.close()


def _finish(window, monkeypatch, summary_property, solver="Fast (Compiled)"):
    """Run the controller's end-of-batch hook against the real window's dsim."""
    monkeypatch.setattr(type(window.dsim), "last_solver_diagnostics_summary", summary_property)
    monkeypatch.setattr(window.dsim, "last_solver_type", solver, raising=False)
    monkeypatch.setattr(window.dsim, "plot_again", lambda: None)
    controller = window.canvas._sim_controller
    monkeypatch.setattr(controller, "_print_terminal_verification", lambda: None)
    controller._finish_batch(True, "")
    return window.status_message.text()


class TestReportSolverDiagnostics:
    def test_finished_appends_summary(self, window, monkeypatch):
        summary = "method=RK45 backend=scipy states=2 points=101 nfev=57 cache=miss"
        text = _finish(window, monkeypatch, property(lambda self: summary))
        assert summary in text
        assert "Simulation finished [Fast (Compiled)]" in text

    def test_finished_without_diagnostics_keeps_plain_message(self, window, monkeypatch):
        # Interpreter path records no diagnostics -> empty summary.
        text = _finish(window, monkeypatch, property(lambda self: ""), solver="Standard")
        assert text == "Simulation finished [Standard]"

    def test_diagnostics_read_failure_is_swallowed(self, window, monkeypatch):
        def _boom(self):
            raise RuntimeError("engine gone")

        # Must not raise; status bar keeps the finished message.
        text = _finish(window, monkeypatch, property(_boom))
        assert text == "Simulation finished [Fast (Compiled)]"

    def test_stop_does_not_report(self, window, monkeypatch):
        called = {"n": 0}

        def _count(self):
            called["n"] += 1
            return "should-not-appear"

        monkeypatch.setattr(type(window.dsim), "last_solver_diagnostics_summary", property(_count))
        window.canvas._sim_controller.stop()
        assert called["n"] == 0
        assert window.status_message.text() == "Simulation stopped"


class TestDiagnosticsSummaryFacade:
    def test_empty_diagnostics_summary_is_blank(self, window):
        window.dsim.engine.last_solver_diagnostics = {}
        assert window.dsim.last_solver_diagnostics_summary == ""

    def test_clear_all_invalidates_compile_cache(self, window):
        engine = window.dsim.engine
        engine._compiled_system_cache_key = ("stale",)
        engine._compiled_system_cache_value = (object(),)
        window.dsim.clear_all()
        assert engine._compiled_system_cache_key is None
        assert engine._compiled_system_cache_value is None

    def test_populated_diagnostics_summary_is_formatted(self, window):
        window.dsim.engine.last_solver_diagnostics = {
            "method_used": "RK45",
            "backend": "scipy",
            "n_states": 2,
            "n_time_points": 101,
            "nfev": 57,
            "compile_cache_hit": False,
            "compile_wall_time": 0.01,
            "solve_wall_time": 0.02,
            "replay_wall_time": 0.03,
            "total_wall_time": 0.06,
        }
        summary = window.dsim.last_solver_diagnostics_summary
        assert "method=RK45" in summary
        assert "cache=miss" in summary
        window.dsim.engine.last_solver_diagnostics = {}
