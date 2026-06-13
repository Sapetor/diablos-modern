"""Results window for the Monte-Carlo / ensemble feature.

``EnsembleResultWindow`` is a pure *view*: it CONSUMES the shared Monte-Carlo
result-dict contract (produced by ``lib/analysis/monte_carlo.py``
``MonteCarloRunner.run``) and only PLOTS the arrays already present in it. It
performs no statistics of its own -- every band, mean line, and sample trace it
draws comes straight from the supplied ``result`` dict.

Monte-Carlo result-dict contract::

    result = {
      "n_runs": int, "n_ok": int, "timeline": ndarray|None,
      "signals": {
        signal_name: {
          "runs": ndarray(n_ok, L), "mean": ndarray(L), "std": ndarray(L),
          "p5": ndarray(L), "p50": ndarray(L), "p95": ndarray(L),
          "min": ndarray(L), "max": ndarray(L),
        }
      }
    }

When ``n_ok == 0`` the ``signals`` dict is ``{}`` and ``timeline`` is ``None``.
"""

import numpy as np
import pyqtgraph as pg
from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QComboBox,
)
from PyQt5.QtCore import Qt


# How many individual member traces to draw as faint background lines. Capped so
# a 500-member ensemble does not turn the plot into an opaque smear.
_MAX_SAMPLE_RUNS = 30


class EnsembleResultWindow(QWidget):
    """Window presenting a Monte-Carlo ensemble result.

    Layout:
      * header label -- ``Monte Carlo: N_ok/N_runs successful runs``.
      * signal picker -- a ``QComboBox`` shown only when more than one signal is
        present; switching it re-plots.
      * plot -- for the selected signal vs ``result["timeline"]``:
          - a shaded p5-p95 uncertainty band,
          - the ensemble mean as a bold line,
          - a faint sample (<= ~30) of individual member runs.

    When ``result["n_ok"] == 0`` (or no signals are present) a centered
    "No successful runs to display." label replaces the plot.
    """

    def __init__(self, result: dict, parent=None):
        super().__init__(parent)
        self.result = result or {}

        self.setWindowTitle("Monte Carlo Ensemble")
        self.resize(820, 600)

        layout = QVBoxLayout()
        self.setLayout(layout)

        n_ok = int(self.result.get("n_ok", 0) or 0)
        n_runs = int(self.result.get("n_runs", 0) or 0)
        signals = self.result.get("signals") or {}
        self._signal_names = list(signals.keys())

        # Header summarising the run outcome.
        header = QLabel(f"Monte Carlo: {n_ok}/{n_runs} successful runs")
        header.setStyleSheet("font-size: 14px; font-weight: bold; padding: 6px;")
        layout.addWidget(header)
        self.header_label = header

        # Nothing successful (or no harvested signals) -> friendly placeholder.
        if n_ok <= 0 or not self._signal_names:
            empty = QLabel("No successful runs to display.")
            empty.setAlignment(Qt.AlignCenter)
            empty.setStyleSheet("color: #555; font-size: 13px; padding: 48px;")
            layout.addWidget(empty, 1)
            self.combo = None
            self.plot = None
            return

        # Signal picker -- only meaningful when there is more than one signal.
        self.combo = QComboBox()
        self.combo.addItems(self._signal_names)
        if len(self._signal_names) > 1:
            row = QHBoxLayout()
            row.addWidget(QLabel("Signal:"))
            row.addWidget(self.combo, 1)
            layout.addLayout(row)
        # else: keep the combo (so callers/tests can introspect it) but hide it.
        self.combo.currentTextChanged.connect(self._on_signal_changed)

        # The plot canvas, styled to match the other analysis windows.
        self.plot = pg.PlotWidget()
        self.plot.setBackground("w")
        self.plot.showGrid(x=True, y=True, alpha=0.3)
        self.plot.setLabel("bottom", "Time")
        self.plot.setLabel("left", "Value")
        for axis in ("bottom", "left"):
            self.plot.getAxis(axis).setPen("k")
            self.plot.getAxis(axis).setTextPen("k")
        self.plot.addLegend(offset=(10, 10))
        layout.addWidget(self.plot, 1)

        # Initial render.
        self._plot_signal(self._signal_names[0])

    # --------------------------------------------------------------- callbacks
    def _on_signal_changed(self, name):
        if name:
            self._plot_signal(name)

    # ------------------------------------------------------------------ render
    def _plot_signal(self, name):
        """Draw band + mean + sample traces for ``name`` vs the timeline."""
        if self.plot is None:
            return
        self.plot.clear()
        # ``clear`` drops the legend's tracked items; recreate a fresh one.
        self.plot.addLegend(offset=(10, 10))

        sig = (self.result.get("signals") or {}).get(name)
        if not sig:
            return

        t = self._as_1d(self.result.get("timeline"))
        mean = self._as_1d(sig.get("mean"))
        p5 = self._as_1d(sig.get("p5"))
        p95 = self._as_1d(sig.get("p95"))
        runs = sig.get("runs")

        # A consistent x-axis: fall back to sample indices if no timeline given.
        L = mean.size if mean.size else (p5.size or p95.size)
        if t.size != L:
            t = np.arange(L, dtype=float)

        self.plot.setTitle(str(name))

        # Faint sample of individual member runs (capped).
        if runs is not None:
            M = np.atleast_2d(np.asarray(runs, dtype=float))
            n = M.shape[0]
            if n > 0:
                # Evenly spaced selection so the sample spans the ensemble.
                step = max(1, int(np.ceil(n / _MAX_SAMPLE_RUNS)))
                idx = range(0, n, step)
                pen = pg.mkPen(color=(120, 120, 120, 60), width=1)
                for k in idx:
                    row = M[k]
                    xr = t if t.size == row.size else np.arange(row.size, dtype=float)
                    self.plot.plot(xr, row, pen=pen)

        # Shaded p5-p95 uncertainty band.
        if p5.size and p95.size and p5.size == p95.size and t.size == p5.size:
            lower = pg.PlotDataItem(t, p5, pen=pg.mkPen((30, 90, 200, 120), width=1))
            upper = pg.PlotDataItem(t, p95, pen=pg.mkPen((30, 90, 200, 120), width=1))
            band = pg.FillBetweenItem(lower, upper, brush=pg.mkBrush(30, 90, 200, 60))
            self.plot.addItem(lower)
            self.plot.addItem(upper)
            self.plot.addItem(band)

        # Bold mean line on top.
        if mean.size and t.size == mean.size:
            self.plot.plot(
                t, mean, pen=pg.mkPen((200, 30, 30), width=3), name="mean"
            )

    # ----------------------------------------------------------- helpers/util
    @staticmethod
    def _as_1d(arr):
        """Coerce ``arr`` into a 1-D float array (empty if None)."""
        if arr is None:
            return np.empty(0, dtype=float)
        out = np.asarray(arr, dtype=float).ravel()
        return out
