"""
Tests for the Monte-Carlo ensemble results window
(modern_ui/widgets/ensemble_result_window.py).

The window CONSUMES the shared Monte-Carlo result-dict contract (produced by
``lib/analysis/monte_carlo.py`` ``MonteCarloRunner.run``) and only plots the
arrays already present in it -- it performs no statistics of its own. These
tests build hand-crafted sample result dicts and assert the widget constructs
without error and exposes the expected layout.

Run offscreen:
    $env:QT_QPA_PLATFORM="offscreen"
"""

import numpy as np
import pytest

from PyQt5.QtWidgets import QWidget, QComboBox

from modern_ui.widgets.ensemble_result_window import EnsembleResultWindow


def _signal_stats(n_ok, L, base):
    """Build one signal's stats block (runs/mean/std/percentiles/min/max)."""
    rng = np.random.default_rng(int(base * 1000))
    M = base + rng.standard_normal((n_ok, L))
    return {
        "runs": M,
        "mean": M.mean(axis=0),
        "std": M.std(axis=0),
        "p5": np.percentile(M, 5, axis=0),
        "p50": np.percentile(M, 50, axis=0),
        "p95": np.percentile(M, 95, axis=0),
        "min": M.min(axis=0),
        "max": M.max(axis=0),
    }


def _sample_result(n_ok=5, n_runs=5, n_signals=2, L=40):
    """A populated ensemble result with ``n_signals`` signals."""
    timeline = np.linspace(0.0, 1.0, L)
    names = ["Scope_A", "Scope_B", "Scope_C"][:n_signals]
    signals = {nm: _signal_stats(n_ok, L, base=i + 1.0) for i, nm in enumerate(names)}
    return {
        "n_runs": n_runs,
        "n_ok": n_ok,
        "timeline": timeline,
        "signals": signals,
    }


def _empty_result(n_runs=4):
    """An all-failed ensemble: n_ok == 0, no signals, no timeline."""
    return {"n_runs": n_runs, "n_ok": 0, "timeline": None, "signals": {}}


@pytest.mark.unit
class TestEnsembleResultWindow:
    def test_builds_as_qwidget(self, qapp):
        win = EnsembleResultWindow(_sample_result())
        assert isinstance(win, QWidget)

    def test_window_title_set(self, qapp):
        win = EnsembleResultWindow(_sample_result())
        assert win.windowTitle() == "Monte Carlo Ensemble"

    def test_header_reports_run_counts(self, qapp):
        win = EnsembleResultWindow(_sample_result(n_ok=5, n_runs=5))
        assert "5/5" in win.header_label.text()

    def test_two_signals_populate_combo(self, qapp):
        win = EnsembleResultWindow(_sample_result(n_signals=2))
        combo = win.findChild(QComboBox)
        assert combo is not None
        assert combo.count() == 2
        labels = [combo.itemText(i) for i in range(combo.count())]
        assert "Scope_A" in labels and "Scope_B" in labels

    def test_switching_signal_updates_plot(self, qapp):
        win = EnsembleResultWindow(_sample_result(n_signals=2))
        assert win.plot is not None
        # Changing the selection re-plots without error and updates the title.
        win.combo.setCurrentIndex(1)
        assert win.plot.getPlotItem().titleLabel.text == "Scope_B"

    def test_many_members_capped_to_sample(self, qapp):
        """A 500-member ensemble still builds; sample traces are capped."""
        win = EnsembleResultWindow(_sample_result(n_ok=500, n_runs=500, n_signals=1))
        assert isinstance(win, QWidget)
        assert win.plot is not None

    def test_single_signal_hides_combo_row_but_plots(self, qapp):
        win = EnsembleResultWindow(_sample_result(n_signals=1))
        assert win.plot is not None
        # Combo exists for introspection but is not shown for a single signal.
        assert win.combo is not None
        assert win.combo.count() == 1

    def test_no_successful_runs_builds_without_plot(self, qapp):
        win = EnsembleResultWindow(_empty_result())
        assert isinstance(win, QWidget)
        assert win.plot is None
        assert win.combo is None
        assert "0/4" in win.header_label.text()

    def test_none_result_builds(self, qapp):
        """A defensive None should not crash construction."""
        win = EnsembleResultWindow(None)
        assert isinstance(win, QWidget)
        assert win.plot is None
