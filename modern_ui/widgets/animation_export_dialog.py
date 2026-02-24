"""
Animation Export Dialog for FieldScope visualizations.

Provides a PyQt5 dialog for configuring animation export settings
including format, FPS, quality, and output path.
"""

import os
import logging
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout,
    QLabel, QSpinBox, QComboBox, QLineEdit, QPushButton,
    QFileDialog, QGroupBox, QRadioButton, QButtonGroup,
    QProgressBar, QMessageBox
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal

logger = logging.getLogger(__name__)


class ExportWorker(QThread):
    """Worker thread for animation export to keep UI responsive."""

    progress = pyqtSignal(int, int)  # current frame, total frames
    finished = pyqtSignal(bool, str)  # success, message

    def __init__(self, exporter, filepath, format, fps, dpi):
        super().__init__()
        self.exporter = exporter
        self.filepath = filepath
        self.format = format
        self.fps = fps
        self.dpi = dpi
        self._cancelled = False

    def cancel(self):
        """Request cancellation of the export."""
        self._cancelled = True

    def run(self):
        """Run the export in a separate thread."""
        try:
            def progress_callback(frame, total):
                if self._cancelled:
                    raise InterruptedError("Export cancelled")
                self.progress.emit(frame, total)

            success = self.exporter.export(
                self.filepath,
                format=self.format,
                fps=self.fps,
                dpi=self.dpi,
                progress_callback=progress_callback
            )

            if success:
                self.finished.emit(True, f"Animation exported to {self.filepath}")
            else:
                self.finished.emit(False, "Export failed. Check logs for details.")

        except Exception as e:
            logger.error(f"Export worker error: {e}")
            self.finished.emit(False, str(e))


class AnimationExportDialog(QDialog):
    """
    Dialog for configuring and executing animation export.

    Shows animation info (frames, duration, grid size) and allows
    configuration of format, FPS, quality, and output path.
    """

    def __init__(self, exporter, block_name="FieldScope", parent=None):
        """
        Initialize the export dialog.

        Args:
            exporter: AnimationExporter instance with field data
            block_name: Name of the block being exported
            parent: Parent widget
        """
        super().__init__(parent)
        self.exporter = exporter
        self.block_name = block_name
        self.worker = None

        self.setWindowTitle("Export Animation")
        self.setMinimumWidth(400)
        self.setModal(True)

        self._setup_ui()
        self._update_playback_duration()
        self._check_writers()

    def _setup_ui(self):
        """Set up the dialog UI."""
        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        # Info section
        info_group = QGroupBox("Animation Info")
        info_layout = QFormLayout(info_group)

        self.frames_label = QLabel(f"{self.exporter.n_frames}")
        self.duration_label = QLabel(f"{self.exporter.duration:.2f}s")
        self.grid_label = QLabel(self.exporter.grid_size)

        info_layout.addRow("Frames:", self.frames_label)
        info_layout.addRow("Simulation Duration:", self.duration_label)
        info_layout.addRow("Grid Size:", self.grid_label)

        layout.addWidget(info_group)

        # Format section
        format_group = QGroupBox("Format")
        format_layout = QHBoxLayout(format_group)

        self.format_group = QButtonGroup(self)
        self.gif_radio = QRadioButton("GIF")
        self.mp4_radio = QRadioButton("MP4")
        self.gif_radio.setChecked(True)

        self.format_group.addButton(self.gif_radio, 0)
        self.format_group.addButton(self.mp4_radio, 1)

        format_layout.addWidget(self.gif_radio)
        format_layout.addWidget(self.mp4_radio)
        format_layout.addStretch()

        layout.addWidget(format_group)

        # Settings section
        settings_group = QGroupBox("Settings")
        settings_layout = QFormLayout(settings_group)

        # FPS spinner
        fps_layout = QHBoxLayout()
        self.fps_spin = QSpinBox()
        self.fps_spin.setRange(1, 60)
        self.fps_spin.setValue(self.exporter.get_recommended_fps(5.0))
        self.fps_spin.valueChanged.connect(self._update_playback_duration)

        self.playback_label = QLabel()
        fps_layout.addWidget(self.fps_spin)
        fps_layout.addWidget(QLabel("→"))
        fps_layout.addWidget(self.playback_label)
        fps_layout.addStretch()

        settings_layout.addRow("FPS:", fps_layout)

        # Quality combo
        self.quality_combo = QComboBox()
        self.quality_combo.addItems(["Low (72 dpi)", "Medium (100 dpi)", "High (150 dpi)"])
        self.quality_combo.setCurrentIndex(1)  # Default to medium

        settings_layout.addRow("Quality:", self.quality_combo)

        layout.addWidget(settings_group)

        # Output path section
        path_group = QGroupBox("Output")
        path_layout = QHBoxLayout(path_group)

        self.path_edit = QLineEdit()
        self.path_edit.setPlaceholderText("Select output file...")
        default_name = f"{self.block_name}_animation.gif"
        default_path = os.path.join(os.path.expanduser("~"), default_name)
        self.path_edit.setText(default_path)

        self.browse_btn = QPushButton("Browse...")
        self.browse_btn.clicked.connect(self._browse_file)

        path_layout.addWidget(self.path_edit, 1)
        path_layout.addWidget(self.browse_btn)

        layout.addWidget(path_group)

        # Progress bar (hidden initially)
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        # Buttons
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()

        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.reject)

        self.export_btn = QPushButton("Export")
        self.export_btn.setDefault(True)
        self.export_btn.clicked.connect(self._start_export)

        btn_layout.addWidget(self.cancel_btn)
        btn_layout.addWidget(self.export_btn)

        layout.addLayout(btn_layout)

        # Connect format change to update file extension
        self.format_group.buttonClicked.connect(self._update_file_extension)

    def _check_writers(self):
        """Check available writers and disable unavailable formats."""
        from lib.plotting.animation_exporter import AnimationExporter
        available = AnimationExporter.check_writers()

        if not available.get('gif', False):
            self.gif_radio.setEnabled(False)
            self.gif_radio.setToolTip("Pillow not installed. Install with: pip install Pillow")

        if not available.get('mp4', False):
            self.mp4_radio.setEnabled(False)
            self.mp4_radio.setToolTip("ffmpeg not found. Install ffmpeg to enable MP4 export.")

        # Select first available format
        if not self.gif_radio.isEnabled() and self.mp4_radio.isEnabled():
            self.mp4_radio.setChecked(True)
            self._update_file_extension()
        elif not self.gif_radio.isEnabled() and not self.mp4_radio.isEnabled():
            self.export_btn.setEnabled(False)
            self.export_btn.setToolTip("No export formats available")

    def _update_playback_duration(self):
        """Update the playback duration label based on current FPS."""
        fps = self.fps_spin.value()
        duration = self.exporter.get_playback_duration(fps)
        self.playback_label.setText(f"Playback: {duration:.1f}s")

    def _update_file_extension(self):
        """Update file extension when format changes."""
        current_path = self.path_edit.text()
        if not current_path:
            return

        # Get new extension
        new_ext = '.gif' if self.gif_radio.isChecked() else '.mp4'

        # Replace extension
        base = os.path.splitext(current_path)[0]
        self.path_edit.setText(base + new_ext)

    def _browse_file(self):
        """Open file browser to select output path."""
        ext = "gif" if self.gif_radio.isChecked() else "mp4"
        filter_str = f"{ext.upper()} files (*.{ext})"

        filepath, _ = QFileDialog.getSaveFileName(
            self,
            "Save Animation",
            self.path_edit.text(),
            filter_str
        )

        if filepath:
            # Ensure correct extension
            if not filepath.lower().endswith(f'.{ext}'):
                filepath += f'.{ext}'
            self.path_edit.setText(filepath)

    def _get_dpi(self):
        """Get DPI value from quality selection."""
        quality_map = {0: 72, 1: 100, 2: 150}
        return quality_map.get(self.quality_combo.currentIndex(), 100)

    def _start_export(self):
        """Start the export process."""
        filepath = self.path_edit.text().strip()
        if not filepath:
            QMessageBox.warning(self, "Error", "Please specify an output file path.")
            return

        format = 'gif' if self.gif_radio.isChecked() else 'mp4'
        fps = self.fps_spin.value()
        dpi = self._get_dpi()

        # Disable controls during export
        self.export_btn.setEnabled(False)
        self.browse_btn.setEnabled(False)
        self.fps_spin.setEnabled(False)
        self.quality_combo.setEnabled(False)
        self.gif_radio.setEnabled(False)
        self.mp4_radio.setEnabled(False)

        # Show progress
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, self.exporter.n_frames)
        self.progress_bar.setValue(0)

        # Start worker thread
        self.worker = ExportWorker(self.exporter, filepath, format, fps, dpi)
        self.worker.progress.connect(self._on_progress)
        self.worker.finished.connect(self._on_export_finished)
        self.worker.start()

    def _on_progress(self, current, total):
        """Update progress bar."""
        self.progress_bar.setValue(current)

    def _on_export_finished(self, success, message):
        """Handle export completion."""
        self.progress_bar.setVisible(False)

        if success:
            QMessageBox.information(self, "Export Complete", message)
            self.accept()
        else:
            QMessageBox.critical(self, "Export Failed", message)
            # Re-enable controls
            self.export_btn.setEnabled(True)
            self.browse_btn.setEnabled(True)
            self.fps_spin.setEnabled(True)
            self.quality_combo.setEnabled(True)
            self._check_writers()  # Re-check to set correct enabled states

    def closeEvent(self, event):
        """Handle dialog close - stop any running export."""
        if self.worker and self.worker.isRunning():
            self.worker.cancel()
            self.worker.quit()
            self.worker.wait(3000)
        super().closeEvent(event)
