"""Results window for the parameter-sweep feature.

``SweepResultWindow`` is a pure *view*: it consumes the sweep-result dict
produced by ``lib/analysis/parameter_sweep.py`` ``ParameterSweepRunner.run`` and
plots the arrays already in it. It adapts to ``result["mode"]``:

  * **1-D** -- a *response family* (one trace per swept value, colored along the
    parameter) with a ``View`` toggle to a *metric-vs-parameter* line plot.
  * **2-D** -- a *heatmap* of a per-run outcome metric over the (x, y) grid, with
    signal and metric pickers and a colorbar.

It performs no statistics of its own; the per-run metrics come straight from the
result dict. When ``n_ok == 0`` (no successful runs) a centered placeholder
replaces the plot and ``plot`` / ``combo`` / ``metric_combo`` are ``None``.
"""

import numpy as np
import pyqtgraph as pg
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QStackedWidget,
)
from PyQt5.QtCore import Qt, QRectF

from lib.analysis.resim import OUTCOME_METRICS


class SweepResultWindow(QWidget):
    """Window presenting a 1-D/2-D parameter-sweep result."""

    def __init__(self, result: dict, parent=None):
        super().__init__(parent)
        self.result = result or {}
        self.mode = self.result.get("mode", "1d")

        self.setWindowTitle("Parameter Sweep")
        self.resize(860, 620)

        layout = QVBoxLayout()
        self.setLayout(layout)

        n_ok = int(self.result.get("n_ok", 0) or 0)
        n_points = int(self.result.get("n_points", 0) or 0)
        signals = self.result.get("signals") or {}
        self._signal_names = list(signals.keys())

        header = QLabel(
            f"Parameter Sweep ({self.mode.upper()}): "
            f"{n_ok}/{n_points} successful runs")
        header.setStyleSheet("font-size: 14px; font-weight: bold; padding: 6px;")
        layout.addWidget(header)
        self.header_label = header

        # Null out optional attributes; populated below per mode.
        self.combo = None
        self.view_combo = None
        self.metric_combo = None
        self.metric_label = None
        self.plot = None
        self.metric_plot = None
        self.stack = None
        self.img = None
        self.colorbar = None

        if n_ok <= 0 or not self._signal_names:
            empty = QLabel("No successful runs to display.")
            empty.setAlignment(Qt.AlignCenter)
            empty.setStyleSheet("color: #555; font-size: 13px; padding: 48px;")
            layout.addWidget(empty, 1)
            return

        # Signal picker (shared by both modes).
        self.combo = QComboBox()
        self.combo.addItems(self._signal_names)
        if len(self._signal_names) > 1:
            row = QHBoxLayout()
            row.addWidget(QLabel("Signal:"))
            row.addWidget(self.combo, 1)
            layout.addLayout(row)
        self.combo.currentTextChanged.connect(self._on_signal_changed)

        if self.mode == "2d":
            self._build_2d(layout)
        else:
            self._build_1d(layout)

    # --------------------------------------------------------------- factories
    @staticmethod
    def _make_plot(x_label, y_label):
        plot = pg.PlotWidget()
        plot.setBackground("w")
        plot.showGrid(x=True, y=True, alpha=0.3)
        plot.setLabel("bottom", x_label)
        plot.setLabel("left", y_label)
        for axis in ("bottom", "left"):
            plot.getAxis(axis).setPen("k")
            plot.getAxis(axis).setTextPen("k")
        return plot

    def _metric_row(self, layout, with_view):
        """Add the View (1-D only) + Metric picker row; return the layout."""
        controls = QHBoxLayout()
        if with_view:
            self.view_combo = QComboBox()
            self.view_combo.addItems(["Response family", "Metric vs parameter"])
            controls.addWidget(QLabel("View:"))
            controls.addWidget(self.view_combo)
            controls.addStretch(1)
        self.metric_label = QLabel("Metric:")
        self.metric_combo = QComboBox()
        self.metric_combo.addItems(list(OUTCOME_METRICS.keys()))
        controls.addWidget(self.metric_label)
        controls.addWidget(self.metric_combo)
        layout.addLayout(controls)
        self.metric_combo.currentTextChanged.connect(self._on_metric_changed)

    # ------------------------------------------------------------------- 1-D
    def _build_1d(self, layout):
        self._metric_row(layout, with_view=True)

        self.stack = QStackedWidget()
        axis = self.result.get("axis", {})
        pname = axis.get("param", "parameter")
        self.plot = self._make_plot("Time", "Value")
        self.plot.addLegend(offset=(10, 10))
        self.metric_plot = self._make_plot(pname, "Metric")
        self.stack.addWidget(self.plot)
        self.stack.addWidget(self.metric_plot)
        layout.addWidget(self.stack, 1)

        self.view_combo.currentIndexChanged.connect(self._on_view_changed)
        self._plot_family()
        self._plot_metric_1d()
        self._on_view_changed(0)

    def _on_view_changed(self, index):
        if self.stack is not None:
            self.stack.setCurrentIndex(int(index))
        is_metric = int(index) == 1
        if self.metric_combo is not None:
            self.metric_combo.setEnabled(is_metric)
            self.metric_label.setEnabled(is_metric)

    def _plot_family(self):
        """Overlay each swept value's trace, colored along the parameter."""
        if self.plot is None:
            return
        self.plot.clear()
        self.plot.addLegend(offset=(10, 10))
        name = self.combo.currentText()
        sig = (self.result.get("signals") or {}).get(name)
        if not sig:
            return
        traces = np.atleast_2d(np.asarray(sig.get("traces"), dtype=float))
        axis = self.result.get("axis", {})
        vals = np.asarray(axis.get("values"), dtype=float).ravel()
        pname = axis.get("param", "parameter")

        t = self._as_1d(self.result.get("timeline"))
        if t.size != traces.shape[1]:
            t = np.arange(traces.shape[1], dtype=float)

        self.plot.setTitle(f"{name} vs {pname}")
        cmap = pg.colormap.get("viridis")
        n = traces.shape[0]
        for i in range(n):
            row = traces[i]
            if not np.any(np.isfinite(row)):
                continue
            frac = 0.0 if n <= 1 else i / (n - 1)
            color = cmap.map(frac, mode="qcolor")
            label = None
            if i == 0 or i == n - 1:
                v = vals[i] if i < vals.size else i
                label = f"{pname}={v:.3g}"
            self.plot.plot(t, row, pen=pg.mkPen(color, width=1), name=label)

    def _plot_metric_1d(self):
        """Plot the selected outcome metric against the swept parameter value."""
        if self.metric_plot is None or self.metric_combo is None:
            return
        self.metric_plot.clear()
        name = self.combo.currentText()
        metric = self.metric_combo.currentText()
        sig = (self.result.get("signals") or {}).get(name)
        if not sig:
            return
        axis = self.result.get("axis", {})
        vals = np.asarray(axis.get("values"), dtype=float).ravel()
        pname = axis.get("param", "parameter")
        y = self._as_1d((sig.get("metrics") or {}).get(metric))
        self.metric_plot.setLabel("bottom", str(pname))
        self.metric_plot.setLabel("left", str(metric))
        self.metric_plot.setTitle(f"{name}: {metric} vs {pname}")
        n = min(vals.size, y.size)
        if n == 0:
            return
        x, y = vals[:n], y[:n]
        mask = np.isfinite(y)
        if not np.any(mask):
            return
        self.metric_plot.plot(
            x[mask], y[mask],
            pen=pg.mkPen((30, 90, 200), width=2),
            symbol="o", symbolSize=6, symbolBrush=(200, 30, 30))

    # ------------------------------------------------------------------- 2-D
    def _build_2d(self, layout):
        self._metric_row(layout, with_view=False)

        self.plot = self._make_plot(
            self.result.get("axis_x", {}).get("param", "x"),
            self.result.get("axis_y", {}).get("param", "y"))
        self.img = pg.ImageItem()
        self.plot.addItem(self.img)
        try:
            cmap = pg.colormap.get("viridis")
            self.img.setColorMap(cmap)
            self.colorbar = pg.ColorBarItem(values=(0.0, 1.0), colorMap=cmap)
            self.colorbar.setImageItem(self.img, insert_in=self.plot.getPlotItem())
        except Exception:  # noqa: BLE001 - colorbar is a nicety, not essential
            self.colorbar = None
        layout.addWidget(self.plot, 1)
        self._plot_heatmap()

    def _plot_heatmap(self):
        """Render the selected signal+metric as a heatmap over the (x, y) grid."""
        if self.img is None:
            return
        name = self.combo.currentText()
        metric = self.metric_combo.currentText()
        sig = (self.result.get("signals") or {}).get(name)
        if not sig:
            return
        Z = np.asarray((sig.get("metrics") or {}).get(metric), dtype=float)
        if Z.ndim != 2 or Z.size == 0:
            return
        xv = np.asarray(self.result.get("axis_x", {}).get("values"), dtype=float).ravel()
        yv = np.asarray(self.result.get("axis_y", {}).get("values"), dtype=float).ravel()

        self.img.setImage(Z, autoLevels=False)
        if xv.size and yv.size:
            xmin, xmax = float(xv.min()), float(xv.max())
            ymin, ymax = float(yv.min()), float(yv.max())
            self.img.setRect(QRectF(
                xmin, ymin,
                (xmax - xmin) or 1.0, (ymax - ymin) or 1.0))

        finite = Z[np.isfinite(Z)]
        if finite.size:
            lo, hi = float(finite.min()), float(finite.max())
            if lo == hi:
                hi = lo + 1e-9
            self.img.setLevels((lo, hi))
            if self.colorbar is not None:
                try:
                    self.colorbar.setLevels((lo, hi))
                except Exception:  # noqa: BLE001
                    pass
        self.plot.setTitle(f"{name}: {metric}")

    # ----------------------------------------------------------- callbacks
    def _on_signal_changed(self, _name):
        if self.mode == "2d":
            self._plot_heatmap()
        else:
            self._plot_family()
            self._plot_metric_1d()

    def _on_metric_changed(self, _name):
        if self.mode == "2d":
            self._plot_heatmap()
        else:
            self._plot_metric_1d()

    # ----------------------------------------------------------- helpers/util
    @staticmethod
    def _as_1d(arr):
        if arr is None:
            return np.empty(0, dtype=float)
        return np.asarray(arr, dtype=float).ravel()
