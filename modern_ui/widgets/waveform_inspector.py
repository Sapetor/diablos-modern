import numpy as np
import pyqtgraph as pg
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QListWidget, QListWidgetItem,
    QPushButton, QLabel, QSlider, QFileDialog, QMessageBox
)
from PyQt5.QtCore import Qt


class WaveformInspector(QWidget):
    """
    Waveform inspector with run history overlay.
    - Stacked traces with shared time base
    - Toggle runs and individual traces
    - Scrub line shows value at cursor
    - Export selected traces to CSV
    """

    def __init__(self, run_history):
        """
        Args:
            run_history (list[dict]): each dict has keys:
              - name: str
              - timeline: np.ndarray
              - traces: list of dicts {'name','y','step'}
        """
        super().__init__()
        self.run_history = run_history or []
        self.active_runs = set(range(len(self.run_history)))
        # Flatten traces with run tag for plotting
        self.timeline = None
        self.traces = []
        self.active_traces = set()
        self._rebuild_traces()
        self._update_scrub_max()

        self.setWindowTitle("Waveform Inspector")
        self.resize(900, 650)

        root = QHBoxLayout()
        self.setLayout(root)

        # Left: run + trace list
        left = QVBoxLayout()
        root.addLayout(left, 0)

        self.run_list = QListWidget()
        self.run_list.itemChanged.connect(self._on_run_changed)
        for idx, run in enumerate(self.run_history):
            item = QListWidgetItem(run["name"])
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked)
            self.run_list.addItem(item)
        left.addWidget(QLabel("Runs"))
        left.addWidget(self.run_list)

        self.trace_list = QListWidget()
        self.trace_list.itemChanged.connect(self._on_trace_changed)
        self._populate_trace_list()
        left.addWidget(QLabel("Traces"))
        left.addWidget(self.trace_list)

        # Right: plot + controls
        right = QVBoxLayout()
        root.addLayout(right, 1)

        self.plot = pg.PlotWidget(title="Scopes")
        self.plot.showGrid(x=True, y=True)
        self.plot.addLegend()
        self.vline = pg.InfiniteLine(angle=90, movable=True, pen=pg.mkPen(color='r', style=Qt.DashLine))
        self.plot.addItem(self.vline)
        self.vline.sigPositionChanged.connect(self._update_readout)
        self.curves = []
        self._refresh_curves()

        right.addWidget(self.plot, 1)

        # Scrub slider
        slider_row = QHBoxLayout()
        slider_row.addWidget(QLabel("Scrub"))
        self.scrub = QSlider(Qt.Horizontal)
        self.scrub.setMinimum(0)
        self.scrub.setMaximum(len(self.timeline) - 1 if self.timeline is not None and len(self.timeline) else 0)
        self.scrub.valueChanged.connect(self._on_scrub)
        slider_row.addWidget(self.scrub, 1)
        self.readout = QLabel("t = -, values = -")
        slider_row.addWidget(self.readout)
        right.addLayout(slider_row)

        # Buttons
        btns = QHBoxLayout()
        export_btn = QPushButton("Export CSV")
        export_btn.clicked.connect(self._export_csv)
        autoscale_btn = QPushButton("Autoscale")
        autoscale_btn.clicked.connect(self._autoscale)
        btns.addWidget(export_btn)
        btns.addWidget(autoscale_btn)
        btns.addStretch()
        right.addLayout(btns)

        self._update_readout()

    def _rebuild_traces(self):
        """
        Flatten active runs into self.traces and a shared timeline (use longest).
        """
        self.traces = []
        self.timeline = None
        # Use the longest timeline among active runs for scrub slider; per-trace data remains per-run
        for run_idx, run in enumerate(self.run_history):
            if run_idx not in self.active_runs:
                continue
            run_name = run["name"]
            if self.timeline is None or len(run["timeline"]) > len(self.timeline):
                self.timeline = run["timeline"]
            for tr in run["traces"]:
                trace_copy = {
                    "name": f"{run_name}: {tr['name']}",
                    "y": tr["y"],
                    "step": tr.get("step", False),
                    "run_idx": run_idx,
                }
                self.traces.append(trace_copy)

        # rebuild trace list UI
        self.active_traces = set(range(len(self.traces)))
        self._populate_trace_list()
        self._update_scrub_max()

    def _populate_trace_list(self):
        self.trace_list.blockSignals(True)
        self.trace_list.clear()
        for idx, tr in enumerate(self.traces):
            item = QListWidgetItem(tr["name"])
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked if idx in self.active_traces else Qt.Unchecked)
            self.trace_list.addItem(item)
        self.trace_list.blockSignals(False)

    def _update_scrub_max(self):
        if hasattr(self, "scrub"):
            self.scrub.blockSignals(True)
            self.scrub.setMaximum(len(self.timeline) - 1 if self.timeline is not None and len(self.timeline) else 0)
            self.scrub.setValue(0)
            self.scrub.blockSignals(False)

    def _refresh_curves(self):
        self.plot.clear()
        self.plot.addItem(self.vline)
        self.curves = []
        for idx, tr in enumerate(self.traces):
            if idx not in self.active_traces:
                continue
            step_mode = tr.get("step", False)
            # Align lengths
            y = tr["y"]
            x = self.timeline
            if step_mode and len(x) == len(y):
                x = np.append(x, x[-1] + (x[-1] - x[-2] if len(x) > 1 else 1.0))
            elif len(x) > len(y):
                x = x[:len(y)]
            elif len(x) < len(y):
                y = y[:len(x)]
            # Color offset per run
            pen = pg.intColor(tr.get("run_idx", 0), hues=max(len(self.run_history), 8))
            curve = self.plot.plot(x, y, stepMode=step_mode, pen=pen, name=tr["name"])
            self.curves.append(curve)

    def _on_trace_changed(self, item):
        idx = self.trace_list.row(item)
        if item.checkState() == Qt.Checked:
            self.active_traces.add(idx)
        else:
            self.active_traces.discard(idx)
        self._refresh_curves()
        self._update_readout()

    def _on_run_changed(self, item):
        idx = self.run_list.row(item)
        if item.checkState() == Qt.Checked:
            self.active_runs.add(idx)
        else:
            self.active_runs.discard(idx)
        self._rebuild_traces()
        self._refresh_curves()
        self._update_readout()

    def _on_scrub(self, value):
        if self.timeline is None or not len(self.timeline):
            return
        t = self.timeline[min(value, len(self.timeline) - 1)]
        self.vline.setPos(t)

    def _update_readout(self):
        if self.timeline is None or not len(self.timeline):
            self.readout.setText("t = -, values = -")
            return
        t = self.vline.value()
        idx = np.searchsorted(self.timeline, t, side="left")
        idx = min(max(idx, 0), len(self.timeline) - 1)
        vals = []
        for i, tr in enumerate(self.traces):
            if i not in self.active_indices:
                continue
            y = tr["y"]
            if len(y) == 0:
                continue
            yi = y[min(idx, len(y) - 1)]
            vals.append(f"{tr['name']}: {yi:.4g}")
        self.readout.setText(f"t = {self.timeline[idx]:.4g}, " + ("; ".join(vals) if vals else "no active traces"))
        self.scrub.blockSignals(True)
        self.scrub.setValue(idx)
        self.scrub.blockSignals(False)

    def _autoscale(self):
        self.plot.enableAutoRange()

    def _export_csv(self):
        if self.timeline is None or not len(self.timeline):
            QMessageBox.warning(self, "No Data", "No data to export.")
            return
        filepath, _ = QFileDialog.getSaveFileName(self, "Export CSV", "waveforms.csv", "CSV Files (*.csv)")
        if not filepath:
            return
        # Build matrix with selected traces
        active = [i for i in range(len(self.traces)) if i in self.active_indices]
        if not active:
            QMessageBox.warning(self, "No Selection", "Select at least one trace to export.")
            return
        data_cols = [self.timeline]
        header = ["time"]
        for i in active:
            y = self.traces[i]["y"]
            y = y[:len(self.timeline)]
            data_cols.append(y)
            header.append(self.traces[i]["name"])
        mat = np.column_stack(data_cols)
        np.savetxt(filepath, mat, delimiter=",", header=",".join(header), comments='')
