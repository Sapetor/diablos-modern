"""
Tests for ModernDiaBloSWindow's solver-diagnostics status-bar hookup.

lib/engine/simulation_engine.py records a compact per-run diagnostics dict and a
one-line summary; DSim exposes it as ``last_solver_diagnostics_summary``. When a
batch run finishes, ``_on_simulation_status_changed`` should append that summary
to the status bar (and log it) -- but only when the compiled solver actually ran.
An interpreter-path run records no diagnostics, so the status bar must keep the
plain finished message.

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


class TestReportSolverDiagnostics:
    def test_finished_appends_summary(self, window, monkeypatch):
        summary = "method=RK45 backend=scipy states=2 points=101 nfev=57 cache=miss"
        monkeypatch.setattr(
            type(window.dsim), "last_solver_diagnostics_summary",
            property(lambda self: summary),
        )
        window._on_simulation_status_changed("Simulation finished [Fast (Compiled)]")
        text = window.status_message.text()
        assert summary in text
        assert "Simulation finished" in text

    def test_finished_without_diagnostics_keeps_plain_message(self, window, monkeypatch):
        # Interpreter path records no diagnostics -> empty summary.
        monkeypatch.setattr(
            type(window.dsim), "last_solver_diagnostics_summary",
            property(lambda self: ""),
        )
        status = "Simulation finished [Standard (Interpreter)]"
        window._on_simulation_status_changed(status)
        assert window.status_message.text() == status

    def test_diagnostics_read_failure_is_swallowed(self, window, monkeypatch):
        def _boom(self):
            raise RuntimeError("engine gone")
        monkeypatch.setattr(
            type(window.dsim), "last_solver_diagnostics_summary",
            property(_boom),
        )
        status = "Simulation finished [Fast (Compiled)]"
        # Must not raise; status bar keeps the finished message.
        window._on_simulation_status_changed(status)
        assert window.status_message.text() == status

    def test_non_finished_status_does_not_report(self, window, monkeypatch):
        called = {"n": 0}

        def _count(self):
            called["n"] += 1
            return "should-not-appear"
        monkeypatch.setattr(
            type(window.dsim), "last_solver_diagnostics_summary",
            property(_count),
        )
        window._on_simulation_status_changed("Simulation stopped")
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
