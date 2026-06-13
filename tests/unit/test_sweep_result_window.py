"""Tests for the parameter-sweep results window
(modern_ui/widgets/sweep_result_window.py).

Builds hand-crafted 1-D and 2-D sweep-result dicts and asserts the widget
constructs and switches views/metrics without error.

Run offscreen:
    $env:QT_QPA_PLATFORM="offscreen"
"""

import numpy as np
import pytest

from PyQt5.QtWidgets import QWidget

from lib.analysis.resim import OUTCOME_METRICS
from modern_ui.widgets.sweep_result_window import SweepResultWindow


def _sweep_1d(n=4, L=20, nsig=1):
    vals = np.linspace(0.0, 3.0, n)
    t = np.linspace(0.0, 1.0, L)
    signals = {}
    for s in range(nsig):
        traces = np.outer(vals, np.ones(L)) + s  # flat row per swept value
        metrics = {m: fn(traces) for m, fn in OUTCOME_METRICS.items()}
        signals[f"S{s}"] = {"traces": traces, "metrics": metrics}
    return {
        "mode": "1d", "n_points": n, "n_ok": n, "timeline": t,
        "axis": {"block": "G", "param": "gain", "values": vals},
        "signals": signals,
    }


def _sweep_2d(nx=2, ny=3, nsig=1):
    xv = np.linspace(1.0, 2.0, nx)
    yv = np.linspace(1.0, 3.0, ny)
    signals = {}
    for s in range(nsig):
        Z = np.outer(xv, yv) + s
        signals[f"S{s}"] = {"metrics": {m: Z.copy() for m in OUTCOME_METRICS}}
    return {
        "mode": "2d", "n_points": nx * ny, "n_ok": nx * ny,
        "timeline": np.linspace(0.0, 1.0, 10),
        "axis_x": {"block": "C", "param": "value", "values": xv},
        "axis_y": {"block": "G", "param": "gain", "values": yv},
        "signals": signals,
    }


def _empty(mode="1d"):
    base = {"mode": mode, "n_points": 4, "n_ok": 0, "timeline": None, "signals": {}}
    if mode == "1d":
        base["axis"] = {"block": "G", "param": "gain", "values": np.zeros(4)}
    return base


@pytest.mark.unit
class TestSweepResultWindow:
    # ------------------------------------------------------------------- 1-D
    def test_1d_builds(self, qapp):
        win = SweepResultWindow(_sweep_1d())
        assert isinstance(win, QWidget)
        assert win.windowTitle() == "Parameter Sweep"
        assert win.combo is not None
        assert win.view_combo is not None and win.metric_combo is not None
        assert win.plot is not None and win.metric_plot is not None

    def test_1d_header_counts(self, qapp):
        win = SweepResultWindow(_sweep_1d(n=5))
        assert "5/5" in win.header_label.text()

    def test_1d_view_toggle_enables_metric(self, qapp):
        win = SweepResultWindow(_sweep_1d())
        assert win.stack.currentIndex() == 0
        assert not win.metric_combo.isEnabled()
        win.view_combo.setCurrentIndex(1)  # Metric vs parameter
        assert win.stack.currentIndex() == 1
        assert win.metric_combo.isEnabled()

    def test_1d_metric_change_rerenders(self, qapp):
        win = SweepResultWindow(_sweep_1d())
        win.view_combo.setCurrentIndex(1)
        win.metric_combo.setCurrentText("max")  # must not raise
        assert "gain" in win.metric_plot.getPlotItem().titleLabel.text

    def test_1d_view_combo_labels(self, qapp):
        win = SweepResultWindow(_sweep_1d())
        labels = [win.view_combo.itemText(i) for i in range(win.view_combo.count())]
        assert labels == ["Response family", "Metric vs parameter"]

    # ------------------------------------------------------------------- 2-D
    def test_2d_builds(self, qapp):
        win = SweepResultWindow(_sweep_2d())
        assert isinstance(win, QWidget)
        assert win.combo is not None and win.metric_combo is not None
        assert win.plot is not None and win.img is not None
        assert win.view_combo is None  # no view toggle in 2-D

    def test_2d_metric_change_rerenders(self, qapp):
        win = SweepResultWindow(_sweep_2d())
        win.metric_combo.setCurrentText("rms")  # must not raise
        assert "rms" in win.plot.getPlotItem().titleLabel.text

    def test_2d_two_signals_switch(self, qapp):
        win = SweepResultWindow(_sweep_2d(nsig=2))
        assert win.combo.count() == 2
        win.combo.setCurrentIndex(1)  # must not raise
        assert "S1" in win.plot.getPlotItem().titleLabel.text

    # ----------------------------------------------------------------- empty
    def test_empty_1d_no_plot(self, qapp):
        win = SweepResultWindow(_empty("1d"))
        assert isinstance(win, QWidget)
        assert win.plot is None and win.combo is None and win.metric_combo is None
        assert "0/4" in win.header_label.text()

    def test_none_result_builds(self, qapp):
        win = SweepResultWindow(None)
        assert isinstance(win, QWidget)
        assert win.plot is None
