"""
SignalPlot - Dynamic plotting widget for DiaBloS simulation output.

Uses pyqtgraph for high-performance plotting of scope data.
*WARNING: Uses PyQT5 (GPL) via pyqtgraph.*
"""

import csv
import logging
import os
from datetime import datetime

import numpy as np
import pyqtgraph as pg
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import (
    QCheckBox,
    QDialog,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from lib.theming.theme_manager import theme_manager, ThemeType

logger = logging.getLogger(__name__)


class SignalPlot(QWidget):
    """
    Class that manages the display of dynamic plots through the simulation.
    *WARNING: It uses pyqtgraph as base (MIT license, but interacts with PyQT5 (GPL)).*

    :param dt: Sampling time of the system.
    :param labels: List of names of the vectors.
    :param xrange: Maximum number of elements to plot in axis x.
    :param previous_run: Optional held data from the previous completed run,
        a dict with keys "t", "labels", "vectors", "step_modes". When given,
        dimmed dashed overlay curves are created (hidden until the
        "Previous run" checkbox is ticked).
    :type dt: float
    :type labels: list
    :type xrange: int
    :type previous_run: dict or None
    """

    def __init__(self, dt, labels, xrange, step_mode=False, previous_run=None):
        super().__init__()
        self.dt = dt
        self.step_mode = step_mode
        self.curve_step_modes = self._expand_step_modes(step_mode, len(labels))
        self.xrange = xrange * self.dt
        self.plot_items = []
        self.curves = []
        self.previous_run = previous_run
        self.prev_curves = []

        # Store data for export
        self.labels = labels
        self.timeline = None
        self.data_vectors = None

        import math

        self._min_plot_height = 200
        n = len(labels)
        self._columns = max(1, math.ceil(math.sqrt(n))) if n > 0 else 1

        main_layout = QVBoxLayout()
        self.setLayout(main_layout)

        # Toolbar
        toolbar_layout = QHBoxLayout()
        toolbar_layout.addWidget(QLabel("Columns:"))
        self.columns_spin = QSpinBox()
        self.columns_spin.setRange(1, max(1, n))
        self.columns_spin.setValue(self._columns)
        toolbar_layout.addWidget(self.columns_spin)
        self.auto_btn = QPushButton("Auto")
        toolbar_layout.addWidget(self.auto_btn)
        toolbar_layout.addStretch()
        toolbar_layout.addWidget(QLabel("Min height:"))
        self.min_height_spin = QSpinBox()
        self.min_height_spin.setRange(80, 500)
        self.min_height_spin.setSingleStep(20)
        self.min_height_spin.setSuffix(" px")
        self.min_height_spin.setValue(200)
        toolbar_layout.addWidget(self.min_height_spin)
        toolbar_layout.addStretch()

        self.toolbar_widget = QWidget()
        self.toolbar_widget.setLayout(toolbar_layout)
        self.toolbar_widget.setVisible(n > 1)
        main_layout.addWidget(self.toolbar_widget)

        # Plot grid container
        grid_container = QWidget()
        self.plot_grid = QGridLayout()
        self.plot_grid.setSpacing(10)
        self.plot_grid.setContentsMargins(0, 0, 0, 10)
        grid_container.setLayout(self.plot_grid)
        main_layout.addWidget(grid_container)

        for idx, label in enumerate(labels):
            plot_widget = pg.PlotWidget(title=label)
            plot_widget.showGrid(x=True, y=True)
            self._configure_plot_axes(plot_widget)
            pen = pg.mkPen(color=self._curve_color(idx), width=2)
            curve = plot_widget.plot(pen=pen, stepMode=self.curve_step_modes[idx])
            self.plot_items.append(plot_widget)
            self.curves.append(curve)
            self.prev_curves.append(self._create_prev_curve(plot_widget, idx))

        self._reflow_grid()

        self.columns_spin.valueChanged.connect(self._on_columns_changed)
        self.auto_btn.clicked.connect(self._on_auto_columns)
        self.min_height_spin.valueChanged.connect(self._on_min_height_changed)

        main_layout.addSpacing(10)

        bottom_layout = QHBoxLayout()
        self.prev_run_checkbox = QCheckBox("Previous run")
        self.prev_run_checkbox.setToolTip("Overlay the previous run's traces (dimmed, dashed)")
        self.prev_run_checkbox.setChecked(False)
        self.prev_run_checkbox.setEnabled(any(c is not None for c in self.prev_curves))
        self.prev_run_checkbox.toggled.connect(self._on_prev_run_toggled)
        bottom_layout.addWidget(self.prev_run_checkbox)
        bottom_layout.addStretch()

        self.export_button = QPushButton("Export to CSV...")
        self.export_button.setToolTip("Export plot data to CSV file")
        self.export_button.clicked.connect(self.export_to_csv)
        bottom_layout.addWidget(self.export_button)

        self.export_figure_button = QPushButton("Export Figure...")
        self.export_figure_button.setToolTip("Export a publication-quality figure (PDF/PNG/SVG)")
        self.export_figure_button.clicked.connect(self.export_figure)
        bottom_layout.addWidget(self.export_figure_button)
        main_layout.addLayout(bottom_layout)

        self._apply_theme()
        theme_manager.theme_changed.connect(self._apply_theme)

        self.resize(500, 400)

    def closeEvent(self, event):
        """Disconnect from the long-lived theme_manager singleton on close.

        theme_manager is a module-level singleton that retains a bound-method
        reference to this widget via the theme_changed connection. Without
        disconnecting, the widget cannot be garbage-collected and a later theme
        change would invoke _apply_theme on a deleted C++ QWidget (RuntimeError).
        """
        try:
            theme_manager.theme_changed.disconnect(self._apply_theme)
        except (TypeError, RuntimeError):
            logger.debug("No theme_changed connection to disconnect on close", exc_info=True)
        super().closeEvent(event)

    # Curve palette: saturated colors with good contrast on both dark and light
    _CURVE_COLORS_DARK = [
        "#60A5FA",  # blue
        "#34D399",  # emerald
        "#F472B6",  # pink
        "#FBBF24",  # amber
        "#A78BFA",  # violet
        "#FB923C",  # orange
        "#2DD4BF",  # teal
        "#F87171",  # red
        "#818CF8",  # indigo
        "#4ADE80",  # green
    ]
    _CURVE_COLORS_LIGHT = [
        "#2563EB",  # blue
        "#059669",  # emerald
        "#DB2777",  # pink
        "#D97706",  # amber
        "#7C3AED",  # violet
        "#EA580C",  # orange
        "#0D9488",  # teal
        "#DC2626",  # red
        "#4F46E5",  # indigo
        "#16A34A",  # green
    ]

    def _curve_color(self, index):
        is_dark = theme_manager.current_theme == ThemeType.DARK
        palette = self._CURVE_COLORS_DARK if is_dark else self._CURVE_COLORS_LIGHT
        return palette[index % len(palette)]

    def _configure_plot_axes(self, plot_widget):
        plot_widget.setLabel("bottom", "Time")
        plot_widget.getAxis("bottom").setStyle(tickTextOffset=10)
        plot_widget.getAxis("bottom").enableAutoSIPrefix(False)
        plot_widget.setMinimumHeight(self._min_plot_height)

    def _apply_theme(self, *_args):
        is_dark = theme_manager.current_theme == ThemeType.DARK
        bg = "#1C2128" if is_dark else "#FAFBFC"
        fg = "#E5E9EF" if is_dark else "#111827"
        grid_alpha = 0.12 if is_dark else 0.15
        axis_pen = pg.mkPen(color=fg, width=1)
        border = "#2D333D" if is_dark else "#E5E7EB"
        surface = "#1C2128" if is_dark else "#FFFFFF"
        accent = "#60A5FA" if is_dark else "#2563EB"

        self.setStyleSheet(f"""
            SignalPlot {{
                background-color: {bg};
                color: {fg};
            }}
            QLabel {{
                color: {fg};
            }}
            QCheckBox {{
                color: {fg};
            }}
            QPushButton {{
                background-color: {surface};
                color: {fg};
                border: 1px solid {border};
                border-radius: 4px;
                padding: 4px 12px;
            }}
            QPushButton:hover {{
                border-color: {accent};
            }}
            QSpinBox {{
                background-color: {surface};
                color: {fg};
                border: 1px solid {border};
                border-radius: 4px;
                padding: 2px 4px;
            }}
        """)

        for idx, plot_widget in enumerate(self.plot_items):
            plot_widget.setBackground(surface)
            for axis_name in ("bottom", "left", "top", "right"):
                axis = plot_widget.getAxis(axis_name)
                axis.setPen(axis_pen)
                axis.setTextPen(pg.mkPen(color=fg))
            plot_widget.getPlotItem().titleLabel.setAttr("color", fg)
            plot_widget.showGrid(x=True, y=True, alpha=grid_alpha)
            # Update curve color for the new theme
            if idx < len(self.curves):
                self.curves[idx].setPen(pg.mkPen(color=self._curve_color(idx), width=2))
            # Keep previous-run overlays dimmed/dashed in the new theme
            if idx < len(self.prev_curves) and self.prev_curves[idx] is not None:
                self.prev_curves[idx].setPen(self._prev_pen(idx))

    def _reflow_grid(self):
        for plot in self.plot_items:
            self.plot_grid.removeWidget(plot)
        cols = max(1, self._columns)
        for idx, plot in enumerate(self.plot_items):
            row, col = divmod(idx, cols)
            self.plot_grid.addWidget(plot, row, col)

    def _on_columns_changed(self, value):
        self._columns = int(value)
        self._reflow_grid()

    def _on_auto_columns(self):
        import math

        n = len(self.plot_items)
        target = max(1, math.ceil(math.sqrt(n))) if n > 0 else 1
        self.columns_spin.setValue(target)

    def _on_min_height_changed(self, value):
        self._min_plot_height = int(value)
        for plot in self.plot_items:
            plot.setMinimumHeight(self._min_plot_height)

    def _prev_pen(self, idx):
        """Dimmed dashed pen for the previous-run overlay in subplot ``idx``."""
        return pg.mkPen(color=self._curve_color(idx) + "66", width=1, style=Qt.DashLine)

    def _previous_trace(self, idx):
        """Return (y, step) of the held-run trace for subplot ``idx``, or None.

        Matches by label first (the previous run may have signals in a
        different order), falling back to the same index; held traces with no
        match are ignored.
        """
        if not self.previous_run:
            return None
        prev_labels = self.previous_run.get("labels") or []
        prev_vectors = self.previous_run.get("vectors") or []
        prev_steps = self.previous_run.get("step_modes") or []

        label = self.labels[idx] if idx < len(self.labels) else None
        if isinstance(label, str) and label in prev_labels:
            # Duplicate labels (e.g. two scopes left at the same default name)
            # must not all map to the first occurrence: pair the k-th live
            # occurrence of the label with the k-th held occurrence.
            occurrence = self.labels[:idx].count(label)
            matches = [i for i, prev in enumerate(prev_labels) if prev == label]
            match = matches[min(occurrence, len(matches) - 1)]
        elif idx < len(prev_vectors):
            match = idx
        else:
            return None
        if match >= len(prev_vectors):
            return None
        step = bool(prev_steps[match]) if match < len(prev_steps) else False
        return prev_vectors[match], step

    def _create_prev_curve(self, plot_widget, idx):
        """Add the previous-run overlay curve to ``plot_widget`` (hidden).

        Returns the curve, or None when there is no held trace for this
        subplot. The overlay is drawn behind the current curve (z = -1) and
        stays hidden until the "Previous run" checkbox is ticked.
        """
        trace = self._previous_trace(idx)
        if trace is None:
            return None
        y_data, step = trace
        y_data = np.asarray(y_data, dtype=float)
        if y_data.ndim > 1 and y_data.shape[1] == 1:
            y_data = y_data.flatten()
        prev_t = self.previous_run.get("t")
        if prev_t is None or y_data.size == 0:
            return None
        prev_t = np.asarray(prev_t, dtype=float)
        if prev_t.size == 0:
            return None
        curve = plot_widget.plot(pen=self._prev_pen(idx), stepMode=step)
        curve.setZValue(-1)
        curve.setData(self._align_time(prev_t, len(y_data), step), y_data)
        curve.setVisible(False)
        return curve

    def _on_prev_run_toggled(self, checked):
        for curve in self.prev_curves:
            if curve is not None:
                curve.setVisible(checked)

    def _expand_step_modes(self, step_mode, count):
        """Normalize step_mode (bool or list) into a list matching the plot count."""
        if isinstance(step_mode, (list, tuple, np.ndarray)):
            modes = [bool(mode) for mode in step_mode]
        else:
            modes = [bool(step_mode)] * count

        if len(modes) < count:
            modes.extend([modes[-1] if modes else False] * (count - len(modes)))
        elif len(modes) > count:
            modes = modes[:count]
        return modes

    def _align_time(self, t, y_len, step_mode):
        """Return an x-array aligned to ``y_len`` samples.

        stepMode=True curves need len(x) == len(y) + 1; normal curves need
        equal lengths. Shorter time arrays are padded with t[-1] + dt.
        """
        target = y_len + 1 if step_mode else y_len
        if len(t) == target:
            return t
        if len(t) > target:
            return t[:target]
        return np.append(t, [t[-1] + self.dt] * (target - len(t)))

    def loop(self, new_t, new_y):
        """Updates the time and scope vectors and plot them."""
        try:
            self.timeline = new_t
            self.data_vectors = new_y

            for i, curve in enumerate(self.curves):
                if i < len(new_y):
                    # Flatten data if it's (N, 1)
                    y_data = new_y[i]
                    if hasattr(y_data, "ndim") and y_data.ndim > 1 and y_data.shape[1] == 1:
                        y_data = y_data.flatten()

                    step_mode = (
                        self.curve_step_modes[i] if i < len(self.curve_step_modes) else False
                    )
                    curve.setData(self._align_time(new_t, len(y_data), step_mode), y_data)
        except Exception as e:
            logger.error(f"Error updating plot: {e}")

    def _require_export_data(self):
        """Return True when there is run data to export; warn otherwise."""
        if self.timeline is None or self.data_vectors is None:
            QMessageBox.warning(self, "No Data", "No plot data available to export.")
            return False
        return True

    def _flash_export_success(self, button, filepath):
        """Show a transient success message on ``button``, restoring it after 3 s.

        The restore timer is parented to ``button`` so it is destroyed with the
        widget if the plot window closes before it fires — otherwise the lambda
        would touch a deleted C++ QPushButton and raise RuntimeError in the
        event loop. (PyQt5 does not expose QTimer.singleShot's context-object
        overload, so a parented timer is the way to get that guarantee.)
        """
        original_text = button.text()
        button.setText(f"✓ Exported to {os.path.basename(filepath)}")
        button.setEnabled(False)

        timer = QTimer(button)
        timer.setSingleShot(True)
        timer.timeout.connect(
            lambda: (
                button.setText(original_text),
                button.setEnabled(True),
            )
        )
        timer.timeout.connect(timer.deleteLater)
        timer.start(3000)

    def export_to_csv(self):
        """Export plot data to CSV file with user selection of which scopes to include."""
        if not self._require_export_data():
            return

        # Create scope selection dialog
        dialog = QDialog(self)
        dialog.setWindowTitle("Select Scopes to Export")
        dialog_layout = QVBoxLayout()

        instruction_label = QLabel("Select which scope blocks to include in the CSV export:")
        dialog_layout.addWidget(instruction_label)

        # Create checkboxes for each scope
        checkboxes = []
        for i, label in enumerate(self.labels):
            if isinstance(label, str):
                scope_name = label
            elif isinstance(label, list):
                scope_name = f"Scope {i} ({', '.join(label[:3])}{'...' if len(label) > 3 else ''})"
            else:
                scope_name = f"Scope {i}"

            checkbox = QWidget()
            checkbox_layout = QHBoxLayout()
            checkbox_layout.setContentsMargins(0, 0, 0, 0)

            cb = QCheckBox(scope_name)
            cb.setChecked(True)
            checkbox_layout.addWidget(cb)
            checkbox.setLayout(checkbox_layout)

            checkboxes.append(cb)
            dialog_layout.addWidget(checkbox)

        # Buttons
        button_layout = QHBoxLayout()
        select_all_btn = QPushButton("Select All")
        deselect_all_btn = QPushButton("Deselect All")
        ok_btn = QPushButton("Export")
        cancel_btn = QPushButton("Cancel")

        select_all_btn.clicked.connect(lambda: [cb.setChecked(True) for cb in checkboxes])
        deselect_all_btn.clicked.connect(lambda: [cb.setChecked(False) for cb in checkboxes])
        ok_btn.clicked.connect(dialog.accept)
        cancel_btn.clicked.connect(dialog.reject)

        button_layout.addWidget(select_all_btn)
        button_layout.addWidget(deselect_all_btn)
        button_layout.addStretch()
        button_layout.addWidget(ok_btn)
        button_layout.addWidget(cancel_btn)

        dialog_layout.addLayout(button_layout)
        dialog.setLayout(dialog_layout)
        dialog.setMinimumWidth(400)

        if dialog.exec_() != QDialog.Accepted:
            return

        selected_indices = [i for i, cb in enumerate(checkboxes) if cb.isChecked()]

        if not selected_indices:
            QMessageBox.warning(self, "No Selection", "Please select at least one scope to export.")
            return

        # Get save file path
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_filename = f"plot_data_{timestamp}.csv"

        filepath, _ = QFileDialog.getSaveFileName(
            self, "Export Plot Data to CSV", default_filename, "CSV Files (*.csv);;All Files (*)"
        )

        if not filepath:
            return

        # Export data to CSV
        try:
            traces = self._collect_figure_traces(selected_indices)
            header = ["time"] + [trace["name"] for trace in traces]
            column_data = [trace["y"] for trace in traces]

            with open(filepath, "w", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(header)

                num_rows = len(self.timeline)
                for row_idx in range(num_rows):
                    row = [self.timeline[row_idx]]
                    for col_data in column_data:
                        if row_idx < len(col_data):
                            row.append(col_data[row_idx])
                        else:
                            row.append("")
                    writer.writerow(row)

            logger.info(
                f"Plot data exported to {filepath} ({num_rows} rows, {len(header)} columns)"
            )

            self._flash_export_success(self.export_button, filepath)

        except Exception as e:
            QMessageBox.critical(self, "Export Failed", f"Failed to export data:\n{str(e)}")
            logger.error(f"Failed to export plot data: {e}")

    _FIGURE_FILTERS = [
        ("PDF (*.pdf)", ".pdf"),
        ("PNG image (*.png)", ".png"),
        ("SVG (*.svg)", ".svg"),
    ]

    def _collect_figure_traces(self, indices=None):
        """Return the run data for ``indices`` (default: all) as trace dicts.

        Single source for the label flattening shared by export_to_csv and
        export_figure. A list label means one multi-column vector holding
        several signals (live-plot path); a plain string label maps 1:1 to a
        1-D vector, except that a multi-column vector is split per column with
        comma-separated label parts naming the columns (mirroring
        pyqtPlotScope). Each trace is a {"name", "y", "step"} dict with 1-D
        ``y`` (the same shape ScopePlotter.get_scope_traces produces).
        """
        if indices is None:
            indices = range(len(self.labels))
        traces = []
        for idx in indices:
            if idx >= len(self.data_vectors):
                continue
            label = self.labels[idx]
            vector = np.asarray(self.data_vectors[idx], dtype=float)
            step = self.curve_step_modes[idx] if idx < len(self.curve_step_modes) else False
            if isinstance(label, list):
                names = list(label)
            elif vector.ndim > 1 and vector.shape[1] > 1:
                # Multi-signal vector with a plain string label: split per
                # column instead of emitting one interleaved 2-D trace.
                parts = [part.strip() for part in str(label).split(",")]
                names = [
                    parts[i] if i < len(parts) else f"Signal {i}" for i in range(vector.shape[1])
                ]
            else:
                traces.append({"name": label, "y": vector.ravel(), "step": step})
                continue
            for i, name in enumerate(names):
                y = vector[:, i] if vector.ndim > 1 else vector
                traces.append({"name": name, "y": y, "step": step})
        return traces

    def export_figure(self):
        """Export the current run as a publication-quality figure (PDF/PNG/SVG)."""
        if not self._require_export_data():
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_filename = f"plot_figure_{timestamp}.pdf"

        filepath, selected = QFileDialog.getSaveFileName(
            self,
            "Export Publication Figure",
            default_filename,
            ";;".join(f for f, _ in self._FIGURE_FILTERS),
        )
        if not filepath:
            return
        if not os.path.splitext(filepath)[1]:
            filepath += next(
                (ext for f, ext in self._FIGURE_FILTERS if f == selected),
                self._FIGURE_FILTERS[0][1],
            )

        try:
            from lib.plotting.publication_figure import build_publication_figure

            fig = build_publication_figure(
                np.asarray(self.timeline, dtype=float), self._collect_figure_traces()
            )
            if filepath.lower().endswith(".png"):
                fig.savefig(filepath, dpi=300)
            else:
                fig.savefig(filepath)

            logger.info(f"Publication figure exported to {filepath}")

            self._flash_export_success(self.export_figure_button, filepath)

        except Exception as e:
            QMessageBox.critical(self, "Export Failed", f"Failed to export figure:\n{str(e)}")
            logger.error(f"Failed to export figure: {e}")
