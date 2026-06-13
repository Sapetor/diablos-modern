"""Results window for the Monte-Carlo / ensemble feature.

``EnsembleResultWindow`` is a pure *view*: it CONSUMES the shared Monte-Carlo
result-dict contract (produced by ``lib/analysis/monte_carlo.py``
``MonteCarloRunner.run``) and only PLOTS the arrays already present in it. The
time-series view performs no statistics of its own -- every band, mean line, and
sample trace comes straight from the supplied ``result`` dict. The histogram
view reduces each run to one scalar outcome (final value, peak, ...); it prefers
the runner-supplied ``metrics`` arrays and, for older result dicts that lack
them, derives the same scalars from ``runs`` via ``OUTCOME_METRICS``.

Monte-Carlo result-dict contract::

    result = {
      "n_runs": int, "n_ok": int, "timeline": ndarray|None,
      "signals": {
        signal_name: {
          "runs": ndarray(n_ok, L), "mean": ndarray(L), "std": ndarray(L),
          "p5": ndarray(L), "p50": ndarray(L), "p95": ndarray(L),
          "min": ndarray(L), "max": ndarray(L),
          "metrics": {metric_name: ndarray(n_ok)},   # per-run scalar outcomes
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
    QStackedWidget,
)
from PyQt5.QtCore import Qt

from lib.analysis.monte_carlo import OUTCOME_METRICS


# How many individual member traces to draw as faint background lines. Capped so
# a 500-member ensemble does not turn the plot into an opaque smear.
_MAX_SAMPLE_RUNS = 30


class EnsembleResultWindow(QWidget):
    """Window presenting a Monte-Carlo ensemble result.

    Two views, switched with a ``View:`` picker:
      * Time Series -- for the selected signal vs ``result["timeline"]``: a
        shaded p5-p95 uncertainty band, the ensemble mean as a bold line, and a
        faint sample (<= ~30) of individual member runs.
      * Histogram -- the distribution across runs of the selected outcome metric
        (final / mean / max / min / peak-to-peak / rms) for the selected signal.

    When ``result["n_ok"] == 0`` (or no signals are present) a centered
    "No successful runs to display." label replaces both views and ``plot`` /
    ``hist_plot`` / ``combo`` / ``metric_combo`` are all ``None``.
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
            self.view_combo = None
            self.metric_combo = None
            self.metric_label = None
            self.plot = None
            self.hist_plot = None
            self.stack = None
            return

        # Signal picker -- created/added FIRST so it is the signal combo that
        # ``findChild(QComboBox)`` returns. Only meaningful with >1 signal.
        self.combo = QComboBox()
        self.combo.addItems(self._signal_names)
        if len(self._signal_names) > 1:
            row = QHBoxLayout()
            row.addWidget(QLabel("Signal:"))
            row.addWidget(self.combo, 1)
            layout.addLayout(row)
        # else: keep the combo (so callers/tests can introspect it) but hide it.
        self.combo.currentTextChanged.connect(self._on_signal_changed)

        # View toggle + outcome-metric picker (metric only applies to histograms).
        controls = QHBoxLayout()
        self.view_combo = QComboBox()
        self.view_combo.addItems(["Time Series", "Histogram"])
        controls.addWidget(QLabel("View:"))
        controls.addWidget(self.view_combo)
        controls.addStretch(1)
        self.metric_label = QLabel("Metric:")
        self.metric_combo = QComboBox()
        self.metric_combo.addItems(list(OUTCOME_METRICS.keys()))
        controls.addWidget(self.metric_label)
        controls.addWidget(self.metric_combo)
        layout.addLayout(controls)

        # Stacked time-series / histogram canvases.
        self.stack = QStackedWidget()
        self.plot = self._make_plot("Time", "Value")
        self.hist_plot = self._make_plot("Value", "Runs")
        self.plot.addLegend(offset=(10, 10))
        self.stack.addWidget(self.plot)
        self.stack.addWidget(self.hist_plot)
        layout.addWidget(self.stack, 1)

        self.view_combo.currentIndexChanged.connect(self._on_view_changed)
        self.metric_combo.currentTextChanged.connect(self._on_metric_changed)

        # Initial render of both views; start on the time-series view.
        self._plot_signal(self._signal_names[0])
        self._plot_histogram()
        self._on_view_changed(0)

    # --------------------------------------------------------------- factories
    @staticmethod
    def _make_plot(x_label, y_label):
        """Create a white, grid-styled PlotWidget matching the analysis windows."""
        plot = pg.PlotWidget()
        plot.setBackground("w")
        plot.showGrid(x=True, y=True, alpha=0.3)
        plot.setLabel("bottom", x_label)
        plot.setLabel("left", y_label)
        for axis in ("bottom", "left"):
            plot.getAxis(axis).setPen("k")
            plot.getAxis(axis).setTextPen("k")
        return plot

    # --------------------------------------------------------------- callbacks
    def _on_signal_changed(self, name):
        if not name:
            return
        self._plot_signal(name)
        self._plot_histogram()

    def _on_metric_changed(self, _name):
        self._plot_histogram()

    def _on_view_changed(self, index):
        """Switch the visible canvas; the metric picker only drives histograms."""
        if self.stack is not None:
            self.stack.setCurrentIndex(int(index))
        is_hist = int(index) == 1
        if self.metric_combo is not None:
            self.metric_combo.setEnabled(is_hist)
            self.metric_label.setEnabled(is_hist)

    # ------------------------------------------------------------ time series
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

    # -------------------------------------------------------------- histogram
    def _plot_histogram(self):
        """Draw the distribution of the selected metric across runs."""
        if self.hist_plot is None or self.combo is None or self.metric_combo is None:
            return
        self.hist_plot.clear()

        name = self.combo.currentText()
        metric = self.metric_combo.currentText()
        sig = (self.result.get("signals") or {}).get(name)
        if not sig:
            return

        vals = self._metric_values(sig, metric)
        vals = vals[np.isfinite(vals)] if vals.size else vals
        self.hist_plot.setTitle(f"{name} - {metric} ({vals.size} runs)")
        self.hist_plot.setLabel("bottom", str(metric))
        if vals.size == 0:
            return

        # Bin count from a sqrt rule, clamped to a sensible range.
        bins = int(min(30, max(5, np.sqrt(vals.size))))
        counts, edges = np.histogram(vals, bins=bins)
        centers = (edges[:-1] + edges[1:]) / 2.0
        width = (edges[1] - edges[0]) * 0.9 if edges.size > 1 else 1.0
        bars = pg.BarGraphItem(
            x=centers, height=counts, width=width,
            brush=pg.mkBrush(30, 90, 200, 150), pen=pg.mkPen((30, 90, 200), width=1),
        )
        self.hist_plot.addItem(bars)

        # Mark the ensemble mean of the metric for quick reference.
        mu = float(np.mean(vals))
        self.hist_plot.addItem(
            pg.InfiniteLine(pos=mu, angle=90,
                            pen=pg.mkPen((200, 30, 30), width=2, style=Qt.DashLine))
        )

    def _metric_values(self, sig, metric):
        """Per-run values of ``metric`` for signal stats ``sig`` as a 1-D array.

        Prefers the runner-supplied ``metrics`` array; falls back to deriving the
        scalar from the ``runs`` matrix so the histogram works for any result
        dict carrying ``runs`` (e.g. older results without ``metrics``).
        """
        metrics = sig.get("metrics")
        if isinstance(metrics, dict) and metric in metrics:
            return self._as_1d(metrics[metric])

        runs = sig.get("runs")
        fn = OUTCOME_METRICS.get(metric)
        if runs is None or fn is None:
            return np.empty(0, dtype=float)
        M = np.atleast_2d(np.asarray(runs, dtype=float))
        if M.size == 0:
            return np.empty(0, dtype=float)
        return self._as_1d(fn(M))

    # ----------------------------------------------------------- helpers/util
    @staticmethod
    def _as_1d(arr):
        """Coerce ``arr`` into a 1-D float array (empty if None)."""
        if arr is None:
            return np.empty(0, dtype=float)
        out = np.asarray(arr, dtype=float).ravel()
        return out
