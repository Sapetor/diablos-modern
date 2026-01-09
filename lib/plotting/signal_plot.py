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
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import (
    QCheckBox, QDialog, QFileDialog, QHBoxLayout, QLabel,
    QMessageBox, QPushButton, QVBoxLayout, QWidget
)

logger = logging.getLogger(__name__)


class SignalPlot(QWidget):
    """
    Class that manages the display of dynamic plots through the simulation.
    *WARNING: It uses pyqtgraph as base (MIT license, but interacts with PyQT5 (GPL)).*

    :param dt: Sampling time of the system.
    :param labels: List of names of the vectors.
    :param xrange: Maximum number of elements to plot in axis x.
    :type dt: float
    :type labels: list
    :type xrange: int
    """

    def __init__(self, dt, labels, xrange, step_mode=False):
        super().__init__()
        self.dt = dt
        self.step_mode = step_mode
        self.curve_step_modes = self._expand_step_modes(step_mode, len(labels))
        self.xrange = xrange * self.dt
        self.plot_items = []
        self.curves = []

        # Store data for export
        self.labels = labels
        self.timeline = None
        self.data_vectors = None

        main_layout = QVBoxLayout()
        self.setLayout(main_layout)

        # Create plot area
        plot_layout = QVBoxLayout()
        plot_layout.setSpacing(10)  # Add spacing between plots
        plot_layout.setContentsMargins(0, 0, 0, 10)  # Add bottom margin to plot area
        for idx, label in enumerate(labels):
            plot_widget = pg.PlotWidget(title=label)
            plot_widget.showGrid(x=True, y=True)

            # Configure axes for better visibility
            self._configure_plot_axes(plot_widget)

            # Use stepMode=True for Zero-Order Hold visualization (staircase plot)
            curve = plot_widget.plot(pen='y', stepMode=self.curve_step_modes[idx])
            self.plot_items.append(plot_widget)
            self.curves.append(curve)
            plot_layout.addWidget(plot_widget)

        main_layout.addLayout(plot_layout)

        # Add spacing before the export button
        main_layout.addSpacing(20)

        # Add export button at bottom
        self.export_button = QPushButton("Export to CSV...")
        self.export_button.setToolTip("Export plot data to CSV file")
        self.export_button.clicked.connect(self.export_to_csv)
        main_layout.addWidget(self.export_button)

        self.resize(800, 600)

    def _configure_plot_axes(self, plot_widget):
        """Configure plot widget axes for better visibility."""
        plot_widget.setLabel('bottom', 'Time')
        plot_widget.getAxis('bottom').setStyle(tickTextOffset=10)
        plot_widget.getAxis('bottom').setPen(pg.mkPen(color='k', width=1))
        plot_widget.getAxis('bottom').enableAutoSIPrefix(False)
        plot_widget.setMinimumHeight(200)

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

    def pltcolor(self, index, hues=9, hueOff=180, minHue=0, maxHue=360, val=255, sat=255, alpha=255):
        """Assigns a color to a vector for plotting purposes."""
        third = (maxHue - minHue) / 3
        hues = int(hues)
        indc = int(index) // 3
        indr = int(index) % 3

        hsection = indr * third
        hrange = (indc * third / (hues // 3)) % third
        h = (hsection + hrange + hueOff) % 360
        return pg.hsvColor(h / 360, sat / 255, val / 255, alpha / 255)

    def plot_config(self, settings_dict={}):
        return

    def loop(self, new_t, new_y):
        """Updates the time and scope vectors and plot them."""
        try:
            self.timeline = new_t
            self.data_vectors = new_y

            for i, curve in enumerate(self.curves):
                if i < len(new_y):
                    step_mode = self.curve_step_modes[i] if i < len(self.curve_step_modes) else False
                    if step_mode:
                        # For stepMode=True, x must be len(y) + 1
                        if len(new_t) == len(new_y[i]):
                            t_step = np.append(new_t, new_t[-1] + self.dt)
                        elif len(new_t) > len(new_y[i]):
                            t_step = new_t[:len(new_y[i]) + 1]
                        else:
                            t_step = np.append(new_t, [new_t[-1] + self.dt] * (len(new_y[i]) - len(new_t) + 1))
                        curve.setData(t_step, new_y[i])
                    else:
                        # For normal plotting, x and y must have same length
                        if len(new_t) == len(new_y[i]):
                            t_step = new_t
                        elif len(new_t) > len(new_y[i]):
                            t_step = new_t[:len(new_y[i])]
                        else:
                            t_step = np.append(new_t, [new_t[-1] + self.dt] * (len(new_y[i]) - len(new_t)))
                        curve.setData(t_step, new_y[i])
        except Exception as e:
            logger.error(f"Error updating plot: {e}")

    def export_to_csv(self):
        """Export plot data to CSV file with user selection of which scopes to include."""
        if self.timeline is None or self.data_vectors is None:
            QMessageBox.warning(self, "No Data", "No plot data available to export.")
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
            self,
            "Export Plot Data to CSV",
            default_filename,
            "CSV Files (*.csv);;All Files (*)"
        )

        if not filepath:
            return

        # Export data to CSV
        try:
            with open(filepath, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)

                header = ['time']
                column_data = []

                for idx in selected_indices:
                    label = self.labels[idx]
                    vector = self.data_vectors[idx]

                    if isinstance(label, list):
                        for i, sig_label in enumerate(label):
                            header.append(sig_label)
                            if len(vector.shape) > 1:
                                column_data.append(vector[:, i])
                            else:
                                column_data.append(vector)
                    else:
                        header.append(label)
                        column_data.append(vector.flatten() if hasattr(vector, 'flatten') else vector)

                writer.writerow(header)

                num_rows = len(self.timeline)
                for row_idx in range(num_rows):
                    row = [self.timeline[row_idx]]
                    for col_data in column_data:
                        if row_idx < len(col_data):
                            row.append(col_data[row_idx])
                        else:
                            row.append('')
                    writer.writerow(row)

            logger.info(f"Plot data exported to {filepath} ({num_rows} rows, {len(header)} columns)")

            original_text = self.export_button.text()
            self.export_button.setText(f"✓ Exported to {os.path.basename(filepath)}")
            self.export_button.setEnabled(False)

            QTimer.singleShot(3000, lambda: (
                self.export_button.setText(original_text),
                self.export_button.setEnabled(True)
            ))

        except Exception as e:
            QMessageBox.critical(self, "Export Failed", f"Failed to export data:\n{str(e)}")
            logger.error(f"Failed to export plot data: {e}")

    def sort_labels(self, labels):
        """Rearranges the list if some elements are lists too."""
        self.labels = []
        for elem in labels:
            if isinstance(elem, str):
                self.labels += [elem]
            elif isinstance(elem, list):
                self.labels += elem

    def sort_vectors(self, ny):
        """Rearranges all vectors in one matrix."""
        new_vec = ny[0]
        for i in range(1, len(ny)):
            new_vec = np.column_stack((new_vec, ny[i]))
        return new_vec
